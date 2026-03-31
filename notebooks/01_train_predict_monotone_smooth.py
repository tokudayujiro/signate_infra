# %%
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import japanize_matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ridge回帰の生予測を、単調減少かつ滑らかな曲線へ投影して提出用CSVを作る

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_ridge_monotone_smooth"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 100
CV = 6
PREDICTION_LOWER_BOUND = 0.0
SMOOTH_LAMBDA_CANDIDATES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SELECT_ONLY_SMOOTH = True
SMOOTH_PROJECTED_GRADIENT_MAXITER = 300
SMOOTH_PROJECTED_GRADIENT_TOL = 1e-6


# %%
# 学習データとテストデータを読み込む
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")

species_col_candidates = ["樹種", "species", "species number"]
species_col = next((c for c in species_col_candidates if c in train_df.columns), None)
if species_col is None:
    raise ValueError(f"樹種列が見つかりません。候補: {species_col_candidates}")

exclude_for_spectrum = {ID_COL, TARGET_COL, "乾物率", species_col}
wavelength_cols = []
for col in train_df.columns:
    if col in exclude_for_spectrum:
        continue
    if not np.issubdtype(train_df[col].dtype, np.number):
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

if PREPROCESS_DROP_SPECIES and species_col is not None:
    train_df = train_df[~train_df[species_col].isin(PREPROCESS_DROP_SPECIES)].copy()


# %%
def build_sequence_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta_df = pd.DataFrame(
        {
            "group_key": df[species_col].fillna("NA").astype(str).reset_index(drop=True),
            "sample_number": df[ID_COL].reset_index(drop=True),
        }
    )
    meta_df["sample_order_in_species"] = meta_df.groupby("group_key", sort=False).cumcount() + 1
    meta_df["group_size"] = meta_df.groupby("group_key", sort=False)["group_key"].transform("size")
    meta_df["remaining_steps"] = meta_df["group_size"] - meta_df["sample_order_in_species"]
    meta_df["order_ratio"] = meta_df["sample_order_in_species"] / meta_df["group_size"]
    meta_df["remaining_ratio"] = meta_df["remaining_steps"] / meta_df["group_size"]
    return meta_df


def iter_group_indices(group_keys: pd.Series) -> list[np.ndarray]:
    groups = pd.Series(group_keys).reset_index(drop=True)
    return [np.asarray(list(idx)) for idx in groups.groupby(groups, sort=False).groups.values()]


def build_base_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
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


def build_base_model(alpha: float) -> Pipeline:
    return Pipeline([("preprocess", build_base_preprocess()), ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE))])


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def count_monotonic_violations(predictions: np.ndarray, group_keys: pd.Series) -> int:
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    violation_count = 0
    for group_idx in iter_group_indices(group_keys):
        group_pred = pred[group_idx]
        violation_count += int((group_pred[1:] > group_pred[:-1]).sum())
    return violation_count


def compute_mean_squared_second_difference(predictions: np.ndarray, group_keys: pd.Series) -> float:
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    second_diff_squares: list[np.ndarray] = []
    for group_idx in iter_group_indices(group_keys):
        group_pred = pred[group_idx]
        if len(group_pred) >= 3:
            second_diff_squares.append(np.diff(group_pred, n=2) ** 2)

    if not second_diff_squares:
        return 0.0
    return float(np.concatenate(second_diff_squares).mean())


def generate_base_oof_predictions(alpha: float) -> np.ndarray:
    oof_pred = np.zeros(len(X_train), dtype=float)
    for fit_idx, valid_idx in cv_splits:
        fold_model = build_base_model(alpha)
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_pred[valid_idx] = fold_model.predict(X_train.iloc[valid_idx])
    return oof_pred


def apply_groupwise_cummin(predictions: np.ndarray, group_keys: pd.Series) -> np.ndarray:
    adjusted = np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    output = adjusted.copy()
    for group_idx in iter_group_indices(group_keys):
        output[group_idx] = np.minimum.accumulate(output[group_idx])
    return output


def apply_groupwise_isotonic(predictions: np.ndarray, group_keys: pd.Series) -> np.ndarray:
    adjusted = np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    output = adjusted.copy()
    for group_idx in iter_group_indices(group_keys):
        group_pred = adjusted[group_idx]
        if len(group_pred) <= 1:
            output[group_idx] = group_pred
            continue
        order = np.arange(len(group_pred), dtype=float)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        output[group_idx] = iso.fit_transform(order, group_pred)
    return output


@lru_cache(maxsize=None)
def build_second_difference_matrix(n_samples: int) -> np.ndarray:
    if n_samples <= 2:
        return np.zeros((0, n_samples), dtype=float)
    d2 = np.zeros((n_samples - 2, n_samples), dtype=float)
    for i in range(n_samples - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    return d2


def project_to_monotone_decreasing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.clip(values, PREDICTION_LOWER_BOUND, None)

    order = np.arange(len(values), dtype=float)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    projected = iso.fit_transform(order, values)
    projected = np.maximum(projected, PREDICTION_LOWER_BOUND)
    projected = np.minimum.accumulate(projected)
    return projected


def solve_monotone_smooth_projection(raw_values: np.ndarray, smooth_lambda: float) -> np.ndarray:
    raw = np.clip(np.asarray(raw_values, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    n_samples = len(raw)
    if n_samples <= 1:
        return raw.copy()

    isotonic_projection = project_to_monotone_decreasing(raw)
    if smooth_lambda <= 0.0 or n_samples <= 2:
        return isotonic_projection

    second_diff = build_second_difference_matrix(n_samples)
    hessian = np.eye(n_samples, dtype=float) + smooth_lambda * (second_diff.T @ second_diff)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    step_size = 1.0 / max(lipschitz, 1e-8)

    current = isotonic_projection.copy()
    for _ in range(SMOOTH_PROJECTED_GRADIENT_MAXITER):
        gradient = hessian @ current - raw
        next_values = current - step_size * gradient
        next_values = project_to_monotone_decreasing(next_values)

        if np.max(np.abs(next_values - current)) <= SMOOTH_PROJECTED_GRADIENT_TOL:
            current = next_values
            break
        current = next_values

    return current


def apply_groupwise_monotone_smooth(predictions: np.ndarray, group_keys: pd.Series, smooth_lambda: float) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=float).reshape(-1)
    output = np.zeros_like(adjusted, dtype=float)
    for group_idx in iter_group_indices(group_keys):
        output[group_idx] = solve_monotone_smooth_projection(adjusted[group_idx], smooth_lambda)
    return output


def make_method_name_for_lambda(smooth_lambda: float) -> str:
    return f"smooth_{smooth_lambda:g}"


def apply_postprocess_by_name(predictions: np.ndarray, group_keys: pd.Series, method_name: str) -> np.ndarray:
    if method_name == "raw":
        return np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    if method_name == "cummin":
        return apply_groupwise_cummin(predictions, group_keys)
    if method_name == "isotonic":
        return apply_groupwise_isotonic(predictions, group_keys)
    if method_name.startswith("smooth_"):
        smooth_lambda = float(method_name.split("_", 1)[1])
        return apply_groupwise_monotone_smooth(predictions, group_keys, smooth_lambda=smooth_lambda)
    raise ValueError(f"Unsupported method_name: {method_name}")


def evaluate_oof_postprocess_methods(
    raw_predictions: np.ndarray,
    y_true: np.ndarray,
    group_keys: pd.Series,
) -> tuple[str, np.ndarray, dict[str, float], dict[str, float]]:
    method_names = ["raw", "cummin", "isotonic"] + [make_method_name_for_lambda(x) for x in SMOOTH_LAMBDA_CANDIDATES]
    scores: dict[str, float] = {}
    smoothness: dict[str, float] = {}
    adjusted_predictions_by_method: dict[str, np.ndarray] = {}

    for method_name in method_names:
        adjusted = apply_postprocess_by_name(raw_predictions, group_keys, method_name)
        adjusted_predictions_by_method[method_name] = adjusted
        scores[method_name] = compute_rmse(y_true, adjusted)
        smoothness[method_name] = compute_mean_squared_second_difference(adjusted, group_keys)

    if SELECT_ONLY_SMOOTH:
        selectable_method_names = [name for name in method_names if name.startswith("smooth_")]
    else:
        selectable_method_names = method_names
    best_method = min(selectable_method_names, key=scores.get)
    return best_method, adjusted_predictions_by_method[best_method], scores, smoothness


def build_submit_df(test_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    sample_submit_path = os.path.join(DATA_DIR, "sample_submit.csv")
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    if os.path.exists(sample_submit_path):
        submit_df = pd.read_csv(sample_submit_path, header=None, encoding="utf-8-sig")
        if len(submit_df) == len(pred):
            submit_df = submit_df.iloc[:, :2].copy()
            submit_df.iloc[:, 0] = test_ids.to_numpy()
            submit_df.iloc[:, 1] = pred
            return submit_df
    return pd.DataFrame({"id": test_ids, "value": pred})


def make_prediction_plot_df(meta_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": meta_df["group_key"].astype(str).reset_index(drop=True),
            "sample_order_in_species": meta_df["sample_order_in_species"].to_numpy(),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )


def plot_raw_vs_selected(
    meta_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    split_name: str,
    method_name: str,
) -> None:
    raw_plot_df = make_prediction_plot_df(meta_df, raw_predictions)
    adjusted_plot_df = make_prediction_plot_df(meta_df, adjusted_predictions)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    sns.lineplot(
        data=raw_plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.0,
        alpha=0.8,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title(f"{split_name}: raw predicted moisture")
    axes[0].set_xlabel("樹種内サンプル番号")
    axes[0].set_ylabel("予測含水率")

    sns.lineplot(
        data=adjusted_plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.0,
        alpha=0.8,
        ax=axes[1],
    )
    axes[1].set_title(f"{split_name}: selected postprocess ({method_name})")
    axes[1].set_xlabel("樹種内サンプル番号")
    axes[1].set_ylabel("予測含水率")
    axes[1].legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


# %%
# 目的変数やID列を除いて、モデルに使う列を選ぶ
exclude = [ID_COL, TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude]

X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].copy()
X_test = test_df[feature_cols].copy()
test_ids = test_df[ID_COL].copy()
y_train_array = y_train.to_numpy(dtype=float)

forced_categorical = {species_col} if species_col is not None else set()
numeric_cols = [c for c in feature_cols if np.issubdtype(X_train[c].dtype, np.number) and c not in forced_categorical]
categorical_cols = [c for c in feature_cols if c in forced_categorical or not np.issubdtype(X_train[c].dtype, np.number)]
cv_splits = list(KFold(n_splits=CV, shuffle=False).split(X_train, y_train))
train_meta_df = build_sequence_meta(train_df)
test_meta_df = build_sequence_meta(test_df)


# %%
def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    model = build_base_model(alpha)
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


# %%
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"ridge_monotone_smooth_{DATE}") as run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_alpha = float(study.best_params["alpha"])
    base_cv_rmse = float(study.best_value)
    print(f"Best base alpha={best_alpha:.6f}, CV RMSE={base_cv_rmse:.4f}")

    base_oof_pred = np.clip(generate_base_oof_predictions(best_alpha), PREDICTION_LOWER_BOUND, None)
    selected_method, train_pred_selected_oof, oof_scores, oof_smoothness = evaluate_oof_postprocess_methods(
        base_oof_pred,
        y_true=y_train_array,
        group_keys=train_meta_df["group_key"],
    )
    print(
        "OOF RMSE by method:",
        {name: round(score, 4) for name, score in sorted(oof_scores.items(), key=lambda x: x[1])},
    )
    print(
        "OOF smoothness by method:",
        {name: round(oof_smoothness[name], 4) for name, _ in sorted(oof_scores.items(), key=lambda x: x[1])},
    )

    base_model = build_base_model(best_alpha)
    base_model.fit(X_train, y_train)
    train_pred_raw_fit = np.clip(base_model.predict(X_train), PREDICTION_LOWER_BOUND, None)
    test_pred_raw = np.clip(base_model.predict(X_test), PREDICTION_LOWER_BOUND, None)
    train_pred_selected_fit = apply_postprocess_by_name(
        train_pred_raw_fit,
        train_meta_df["group_key"],
        selected_method,
    )
    test_pred_selected = apply_postprocess_by_name(
        test_pred_raw,
        test_meta_df["group_key"],
        selected_method,
    )

    train_raw_rmse = compute_rmse(y_train_array, base_oof_pred)
    train_selected_rmse = compute_rmse(y_train_array, train_pred_selected_oof)
    train_raw_violations = count_monotonic_violations(base_oof_pred, train_meta_df["group_key"])
    train_selected_violations = count_monotonic_violations(train_pred_selected_oof, train_meta_df["group_key"])
    test_raw_violations = count_monotonic_violations(test_pred_raw, test_meta_df["group_key"])
    test_selected_violations = count_monotonic_violations(test_pred_selected, test_meta_df["group_key"])
    print(
        f"Selected method={selected_method}, train raw OOF RMSE={train_raw_rmse:.4f}, "
        f"train selected OOF RMSE={train_selected_rmse:.4f}"
    )
    print(
        f"Violations train raw={train_raw_violations}, train selected={train_selected_violations}, "
        f"test raw={test_raw_violations}, test selected={test_selected_violations}"
    )

    submit_df = build_submit_df(test_ids, test_pred_selected)
    rmse_tag = int(round(train_selected_rmse * 10000))
    submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_monotone_smooth_{DATE}_{rmse_tag:04d}.csv")
    submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
    print(f"saved: {submit_path}")

    detail_df = pd.DataFrame(
        {
            "id": test_ids,
            "species": test_meta_df["group_key"],
            "sample_order_in_species": test_meta_df["sample_order_in_species"],
            "raw_prediction": test_pred_raw,
            "selected_prediction": test_pred_selected,
            "selected_method": selected_method,
        }
    )
    detail_path = os.path.join(SUBMIT_DIR, f"test_prediction_detail_monotone_smooth_{DATE}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    mlflow.log_params(
        {
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "cv": CV,
            "n_trials": N_TRIALS,
            "best_alpha": best_alpha,
            "prediction_lower_bound": PREDICTION_LOWER_BOUND,
            "smooth_lambda_candidates": ",".join(str(x) for x in SMOOTH_LAMBDA_CANDIDATES),
            "select_only_smooth": SELECT_ONLY_SMOOTH,
            "smooth_projected_gradient_maxiter": SMOOTH_PROJECTED_GRADIENT_MAXITER,
            "smooth_projected_gradient_tol": SMOOTH_PROJECTED_GRADIENT_TOL,
            "selected_method": selected_method,
            "feature_count": len(feature_cols),
        }
    )
    mlflow.log_metric("base_cv_rmse", base_cv_rmse)
    mlflow.log_metric("train_raw_oof_rmse", train_raw_rmse)
    mlflow.log_metric("train_selected_oof_rmse", train_selected_rmse)
    mlflow.log_metric("train_raw_violations", train_raw_violations)
    mlflow.log_metric("train_selected_violations", train_selected_violations)
    mlflow.log_metric("test_raw_violations", test_raw_violations)
    mlflow.log_metric("test_selected_violations", test_selected_violations)
    for method_name, method_score in oof_scores.items():
        mlflow.log_metric(f"oof_rmse_{method_name}", float(method_score))
        mlflow.log_metric(f"oof_smoothness_{method_name}", float(oof_smoothness[method_name]))
    mlflow.log_artifact(submit_path)
    mlflow.log_artifact(detail_path)
    mlflow.log_artifact(__file__)
    print(f"mlflow run_id: {run.info.run_id}")

submit_df.head()


# %%
moisture_plot_df = pd.DataFrame(
    {
        "species": train_meta_df["group_key"],
        "sample_order_in_species": train_meta_df["sample_order_in_species"],
        "moisture": y_train_array,
    }
)
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=moisture_plot_df,
    x="sample_order_in_species",
    y="moisture",
    hue="species",
    marker="o",
    linewidth=1.2,
    alpha=0.85,
)
plt.xlabel("樹種内サンプル番号")
plt.ylabel("含水率")
plt.title("Train: 樹種ごとの含水率")
plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

plot_raw_vs_selected(train_meta_df, base_oof_pred, train_pred_selected_oof, "Train", selected_method)
plot_raw_vs_selected(test_meta_df, test_pred_raw, test_pred_selected, "Test", selected_method)

# %%
test_df

# %%
submit_df
# %%
