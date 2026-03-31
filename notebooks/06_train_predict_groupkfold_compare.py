# %%
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    japanize_matplotlib = None

try:
    import mlflow
except ImportError:
    class _MlflowRunStub:
        def __init__(self) -> None:
            self.info = SimpleNamespace(run_id="mlflow-disabled")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class _MlflowStub:
        def set_tracking_uri(self, uri: str) -> None:
            return None

        def set_experiment(self, name: str) -> None:
            return None

        def start_run(self, run_name: str | None = None) -> _MlflowRunStub:
            return _MlflowRunStub()

        def log_params(self, params: dict) -> None:
            return None

        def log_metric(self, name: str, value: float) -> None:
            return None

        def log_artifact(self, path: str) -> None:
            return None

    mlflow = _MlflowStub()

# GroupKFold で樹種リークを避けつつ、樹種特徴量あり/なしを比較する

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_groupkfold_species_compare"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 60
CV = 5
ENCODING = "cp932"


# %%
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding=ENCODING)
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding=ENCODING)

species_col_candidates = ["species number", "樹種", "species"]
group_col = next((c for c in species_col_candidates if c in train_df.columns), None)
if group_col is None:
    raise ValueError(f"樹種列が見つかりません。候補: {species_col_candidates}")

species_feature_cols = [c for c in ["樹種", "species", "species number"] if c in train_df.columns]

exclude_for_spectrum = {ID_COL, TARGET_COL, "乾物率", *species_feature_cols}
wavelength_cols = []
for col in train_df.columns:
    if col in exclude_for_spectrum:
        continue
    if not pd.api.types.is_numeric_dtype(train_df[col]):
        continue
    try:
        float(col)
        wavelength_cols.append(col)
    except (TypeError, ValueError):
        continue

if PREPROCESS_CLIP_NEGATIVE and wavelength_cols:
    train_df.loc[:, wavelength_cols] = train_df[wavelength_cols].clip(lower=0)
    shared_cols = [c for c in wavelength_cols if c in test_df.columns]
    test_df.loc[:, shared_cols] = test_df[shared_cols].clip(lower=0)

if PREPROCESS_DROP_SPECIES:
    drop_col = next((c for c in ["樹種", "species"] if c in train_df.columns), None)
    if drop_col is not None:
        train_df = train_df[~train_df[drop_col].isin(PREPROCESS_DROP_SPECIES)].copy()


# %%
def build_submit_df(test_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    sample_submit_path = os.path.join(DATA_DIR, "sample_submit.csv")
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    if os.path.exists(sample_submit_path):
        submit_df = pd.read_csv(sample_submit_path, header=None, encoding="utf-8-sig")
        if len(submit_df) == len(pred):
            submit_df = submit_df.iloc[:, :2].copy()
            submit_df.columns = ["id", "value"]
            submit_df["id"] = test_ids.to_numpy()
            submit_df["value"] = pred
            return submit_df
    return pd.DataFrame({"id": test_ids, "value": pred})


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_feature_cols(include_species_features: bool) -> list[str]:
    exclude = [ID_COL, TARGET_COL]
    if "乾物率" in train_df.columns:
        exclude.append("乾物率")

    cols = [c for c in train_df.columns if c not in exclude]
    if include_species_features:
        return cols
    return [c for c in cols if c not in species_feature_cols]


def build_model(feature_cols: list[str], alpha: float) -> Pipeline:
    X_ref = train_df[feature_cols]
    forced_categorical = set(species_feature_cols).intersection(feature_cols)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_ref[c]) and c not in forced_categorical]
    categorical_cols = [c for c in feature_cols if c in forced_categorical or not pd.api.types.is_numeric_dtype(X_ref[c])]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE))])


def build_cv_splits(splitter_name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    y_array = train_df[TARGET_COL].to_numpy(dtype=float)
    if splitter_name == "group":
        unique_groups = train_df[group_col].nunique(dropna=True)
        if unique_groups < CV:
            raise ValueError(f"GroupKFold を組むには group のユニーク数が不足しています: {unique_groups} < {CV}")
        splitter = GroupKFold(n_splits=CV)
        splits = list(splitter.split(train_df, y_array, groups=train_df[group_col]))
        return splits, "GroupKFold"

    splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train_df, y_array))
    return splits, "KFold"


def generate_oof_predictions(feature_cols: list[str], alpha: float, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    X_all = train_df[feature_cols].copy()
    y_all = train_df[TARGET_COL].copy()
    oof_pred = np.zeros(len(X_all), dtype=float)
    for fit_idx, valid_idx in splits:
        fold_model = build_model(feature_cols, alpha)
        fold_model.fit(X_all.iloc[fit_idx], y_all.iloc[fit_idx])
        oof_pred[valid_idx] = fold_model.predict(X_all.iloc[valid_idx])
    return oof_pred


def tune_alpha(feature_cols: list[str], splits: list[tuple[np.ndarray, np.ndarray]], n_trials: int) -> tuple[float, float]:
    y_true = train_df[TARGET_COL].to_numpy(dtype=float)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
        oof_pred = generate_oof_predictions(feature_cols, alpha, splits)
        return compute_rmse(y_true, oof_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return float(study.best_params["alpha"]), float(study.best_value)


def plot_comparison_bar(result_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=result_df, x="evaluation_splitter", y="oof_rmse", hue="config_name")
    plt.title("OOF RMSE Comparison")
    plt.xlabel("evaluation_splitter")
    plt.ylabel("OOF RMSE")
    plt.tight_layout()
    plt.show()


def plot_oof_scatter(result_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=result_df,
        x="feature_count",
        y="oof_rmse",
        hue="config_name",
        style="evaluation_splitter",
        s=100,
    )
    plt.title("Feature Count vs OOF RMSE")
    plt.xlabel("feature_count")
    plt.ylabel("OOF RMSE")
    plt.tight_layout()
    plt.show()


def make_prediction_plot_df(group_series: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    plot_df = pd.DataFrame(
        {
            "species": group_series.fillna("NA").astype(str).reset_index(drop=True),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )
    plot_df["sample_order_in_species"] = plot_df.groupby("species", sort=False).cumcount() + 1
    return plot_df


def plot_prediction_trend_by_species(group_series: pd.Series, predictions: np.ndarray, title: str) -> None:
    plot_df = make_prediction_plot_df(group_series, predictions)
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.2,
        alpha=0.85,
    )
    plt.xlabel("樹種内サンプル番号（groupby + cumcount）")
    plt.ylabel("予測含水率")
    plt.title(title)
    plt.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# %%
X_test_id = test_df[ID_COL].copy()
comparison_rows: list[dict[str, float | str | int]] = []
submission_artifacts: list[str] = []
comparison_detail: dict[str, pd.DataFrame] = {}
best_group_plot_payload: dict[str, object] | None = None

config_specs = [
    {"config_name": "with_species_feature", "include_species_features": True},
    {"config_name": "without_species_feature", "include_species_features": False},
]
splitter_specs = [("group", "GroupKFold"), ("kfold", "KFold")]

mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"groupkfold_species_compare_{DATE}") as run:
    for config_spec in config_specs:
        feature_cols = build_feature_cols(config_spec["include_species_features"])
        for splitter_key, splitter_label in splitter_specs:
            splits, splitter_name = build_cv_splits(splitter_key)
            print(f"Running {config_spec['config_name']} with {splitter_name}")
            best_alpha, best_rmse = tune_alpha(feature_cols, splits, n_trials=N_TRIALS)
            oof_pred = generate_oof_predictions(feature_cols, best_alpha, splits)
            full_model = build_model(feature_cols, best_alpha)
            full_model.fit(train_df[feature_cols], train_df[TARGET_COL])
            test_pred = full_model.predict(test_df[feature_cols])

            submit_df = build_submit_df(X_test_id, test_pred)
            submit_path = os.path.join(
                SUBMIT_DIR,
                f"submit_csv_06_{config_spec['config_name']}_{splitter_label.lower()}_{DATE}_{int(round(best_rmse * 10000)):04d}.csv",
            )
            submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
            submission_artifacts.append(submit_path)

            comparison_rows.append(
                {
                    "config_name": config_spec["config_name"],
                    "evaluation_splitter": splitter_name,
                    "best_alpha": best_alpha,
                    "oof_rmse": best_rmse,
                    "feature_count": len(feature_cols),
                    "includes_species_feature": int(config_spec["include_species_features"]),
                }
            )

            comparison_detail[f"{config_spec['config_name']}_{splitter_name}"] = pd.DataFrame(
                {
                    "id": train_df[ID_COL],
                    "group": train_df[group_col].astype(str),
                    "true": train_df[TARGET_COL],
                    "oof_pred": oof_pred,
                }
            )
            if splitter_name == "GroupKFold":
                if best_group_plot_payload is None or float(best_group_plot_payload["oof_rmse"]) > best_rmse:
                    best_group_plot_payload = {
                        "config_name": config_spec["config_name"],
                        "splitter_name": splitter_name,
                        "oof_rmse": best_rmse,
                        "train_pred": oof_pred.copy(),
                        "test_pred": np.asarray(test_pred, dtype=float).copy(),
                    }

    result_df = pd.DataFrame(comparison_rows).sort_values(["evaluation_splitter", "oof_rmse"]).reset_index(drop=True)
    print(result_df)

    summary_path = os.path.join(SUBMIT_DIR, f"comparison_06_groupkfold_species_{DATE}.csv")
    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    submission_artifacts.append(summary_path)

    best_group_row = (
        result_df[result_df["evaluation_splitter"] == "GroupKFold"]
        .sort_values("oof_rmse")
        .iloc[0]
    )
    best_config_name = str(best_group_row["config_name"])
    print(f"Best config under GroupKFold: {best_config_name}")

    mlflow.log_params(
        {
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "group_col": group_col,
            "cv": CV,
            "n_trials": N_TRIALS,
            "species_feature_cols": ",".join(species_feature_cols),
            "best_groupkfold_config": best_config_name,
        }
    )
    for row in comparison_rows:
        prefix = f"{row['config_name']}_{str(row['evaluation_splitter']).lower()}"
        mlflow.log_metric(f"{prefix}_oof_rmse", float(row["oof_rmse"]))
        mlflow.log_metric(f"{prefix}_feature_count", float(row["feature_count"]))
    for artifact_path in submission_artifacts:
        mlflow.log_artifact(artifact_path)
    mlflow.log_artifact(__file__)
    print(f"mlflow run_id: {run.info.run_id}")

plot_comparison_bar(result_df)
plot_oof_scatter(result_df)
if best_group_plot_payload is not None:
    best_plot_name = f"{best_group_plot_payload['config_name']} ({best_group_plot_payload['splitter_name']})"
    plot_prediction_trend_by_species(
        train_df[group_col],
        np.asarray(best_group_plot_payload["train_pred"], dtype=float),
        f"学習データ: 樹種ごとの予測含水率の推移 [{best_plot_name}]",
    )
    plot_prediction_trend_by_species(
        test_df[group_col],
        np.asarray(best_group_plot_payload["test_pred"], dtype=float),
        f"テストデータ: 樹種ごとの予測含水率の推移 [{best_plot_name}]",
    )

# %%
result_df
