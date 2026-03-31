# %%
import os
from datetime import datetime, timedelta, timezone

import japanize_matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ridge回帰の生予測を作り、その系列を末尾から前へ補正する2段目モデル

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_ridge_backward_refine"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 100
CV = 6
REFINE_TARGET_METHOD = "isotonic"
PREDICTION_LOWER_BOUND = 0.0
DELTA_UPPER_QUANTILE = 0.995
TAIL_UPPER_QUANTILE = 0.995
REFINE_LEARNING_RATE = 0.05
REFINE_MAX_DEPTH = 3
REFINE_MAX_ITER = 300
REFINE_MIN_SAMPLES_LEAF = 10
REFINE_L2_REGULARIZATION = 0.1


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
    groups = pd.Series(group_keys).reset_index(drop=True)
    violation_count = 0
    for idx in groups.groupby(groups, sort=False).groups.values():
        group_pred = pred[np.asarray(list(idx))]
        violation_count += int((group_pred[1:] > group_pred[:-1]).sum())
    return violation_count


def apply_groupwise_monotone_target(
    values: np.ndarray,
    group_keys: pd.Series,
    method: str = "isotonic",
) -> np.ndarray:
    output = np.asarray(values, dtype=float).reshape(-1).copy()
    groups = pd.Series(group_keys).reset_index(drop=True)
    for idx in groups.groupby(groups, sort=False).groups.values():
        group_idx = np.asarray(list(idx))
        group_values = output[group_idx]

        if method == "raw":
            adjusted = group_values
        elif method == "cummin":
            adjusted = np.minimum.accumulate(group_values)
        elif method == "isotonic":
            order = np.arange(len(group_idx), dtype=float)
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            adjusted = iso.fit_transform(order, group_values)
        else:
            raise ValueError(f"Unsupported REFINE_TARGET_METHOD: {method}")

        output[group_idx] = adjusted
    return output


def generate_base_oof_predictions(alpha: float) -> np.ndarray:
    oof_pred = np.zeros(len(X_train), dtype=float)
    for fit_idx, valid_idx in cv_splits:
        fold_model = build_base_model(alpha)
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_pred[valid_idx] = fold_model.predict(X_train.iloc[valid_idx])
    return oof_pred


def build_group_feature_cache(raw_values: np.ndarray) -> dict[str, np.ndarray | float]:
    raw_values = np.asarray(raw_values, dtype=float).reshape(-1)
    reversed_values = raw_values[::-1]
    suffix_sum = np.cumsum(reversed_values)[::-1]
    suffix_sq_sum = np.cumsum(reversed_values**2)[::-1]
    future_len = np.arange(len(raw_values), 0, -1, dtype=float)
    suffix_mean = suffix_sum / future_len
    suffix_var = np.maximum(suffix_sq_sum / future_len - suffix_mean**2, 0.0)
    return {
        "suffix_min": np.minimum.accumulate(raw_values[::-1])[::-1],
        "suffix_max": np.maximum.accumulate(raw_values[::-1])[::-1],
        "suffix_mean": suffix_mean,
        "suffix_std": np.sqrt(suffix_var),
        "group_raw_start": float(raw_values[0]),
        "group_raw_end": float(raw_values[-1]),
        "group_raw_range": float(raw_values.max() - raw_values.min()),
    }


def make_common_refine_features(
    position: int,
    raw_values: np.ndarray,
    group_meta: pd.DataFrame,
    cache: dict[str, np.ndarray | float],
) -> dict[str, float]:
    n_samples = len(raw_values)
    next_idx = min(position + 1, n_samples - 1)
    next2_idx = min(position + 2, n_samples - 1)
    meta_row = group_meta.iloc[position]
    return {
        "raw_current": float(raw_values[position]),
        "raw_next": float(raw_values[next_idx]),
        "raw_next2": float(raw_values[next2_idx]),
        "raw_delta_next": float(raw_values[position] - raw_values[next_idx]),
        "raw_delta_next2": float(raw_values[position] - raw_values[next2_idx]),
        "future_raw_min": float(cache["suffix_min"][position]),
        "future_raw_max": float(cache["suffix_max"][position]),
        "future_raw_mean": float(cache["suffix_mean"][position]),
        "future_raw_std": float(cache["suffix_std"][position]),
        "group_raw_start": float(cache["group_raw_start"]),
        "group_raw_end": float(cache["group_raw_end"]),
        "group_raw_range": float(cache["group_raw_range"]),
        "sample_order_in_species": float(meta_row["sample_order_in_species"]),
        "group_size": float(meta_row["group_size"]),
        "remaining_steps": float(meta_row["remaining_steps"]),
        "order_ratio": float(meta_row["order_ratio"]),
        "remaining_ratio": float(meta_row["remaining_ratio"]),
    }


def build_refine_training_frames(
    meta_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    teacher_target: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_rows: list[dict[str, float]] = []
    tail_rows: list[dict[str, float]] = []

    groups = meta_df["group_key"].groupby(meta_df["group_key"], sort=False).groups
    raw_predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)
    teacher_target = np.asarray(teacher_target, dtype=float).reshape(-1)

    for idx in groups.values():
        group_idx = np.asarray(list(idx))
        group_raw = raw_predictions[group_idx]
        group_teacher = teacher_target[group_idx]
        group_meta = meta_df.iloc[group_idx].reset_index(drop=True)
        cache = build_group_feature_cache(group_raw)
        last_pos = len(group_idx) - 1

        tail_feature = make_common_refine_features(last_pos, group_raw, group_meta, cache)
        tail_feature["target"] = float(np.clip(group_teacher[last_pos], PREDICTION_LOWER_BOUND, None))
        tail_rows.append(tail_feature)

        for position in range(last_pos - 1, -1, -1):
            delta_feature = make_common_refine_features(position, group_raw, group_meta, cache)
            next_teacher = float(group_teacher[position + 1])
            delta_feature["next_adjusted_value"] = next_teacher
            delta_feature["raw_minus_next_adjusted"] = float(group_raw[position] - next_teacher)
            delta_feature["target"] = float(np.clip(group_teacher[position] - group_teacher[position + 1], 0.0, None))
            delta_rows.append(delta_feature)

    return pd.DataFrame(delta_rows), pd.DataFrame(tail_rows)


def build_refine_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=REFINE_LEARNING_RATE,
        max_depth=REFINE_MAX_DEPTH,
        max_iter=REFINE_MAX_ITER,
        min_samples_leaf=REFINE_MIN_SAMPLES_LEAF,
        l2_regularization=REFINE_L2_REGULARIZATION,
        random_state=RANDOM_STATE,
    )


def apply_backward_refine(
    meta_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    delta_model: HistGradientBoostingRegressor,
    tail_model: HistGradientBoostingRegressor,
    delta_feature_cols: list[str],
    tail_feature_cols: list[str],
    delta_cap: float,
    tail_cap: float,
) -> np.ndarray:
    refined = np.zeros(len(raw_predictions), dtype=float)
    raw_predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)
    groups = meta_df["group_key"].groupby(meta_df["group_key"], sort=False).groups

    for idx in groups.values():
        group_idx = np.asarray(list(idx))
        group_raw = raw_predictions[group_idx]
        group_meta = meta_df.iloc[group_idx].reset_index(drop=True)
        cache = build_group_feature_cache(group_raw)
        local_refined = np.zeros(len(group_idx), dtype=float)
        last_pos = len(group_idx) - 1

        tail_feature = make_common_refine_features(last_pos, group_raw, group_meta, cache)
        tail_df = pd.DataFrame([tail_feature], columns=tail_feature_cols)
        tail_pred = float(tail_model.predict(tail_df)[0])
        tail_pred = float(np.clip(tail_pred, PREDICTION_LOWER_BOUND, tail_cap))
        local_refined[last_pos] = tail_pred

        for position in range(last_pos - 1, -1, -1):
            delta_feature = make_common_refine_features(position, group_raw, group_meta, cache)
            delta_feature["next_adjusted_value"] = float(local_refined[position + 1])
            delta_feature["raw_minus_next_adjusted"] = float(group_raw[position] - local_refined[position + 1])
            delta_df = pd.DataFrame([delta_feature], columns=delta_feature_cols)
            delta_pred = float(delta_model.predict(delta_df)[0])
            delta_pred = float(np.clip(delta_pred, 0.0, delta_cap))
            local_refined[position] = max(local_refined[position + 1] + delta_pred, local_refined[position + 1])

        refined[group_idx] = local_refined

    return refined


def apply_groupwise_cummin(predictions: np.ndarray, group_keys: pd.Series) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=float).reshape(-1).copy()
    adjusted = np.clip(adjusted, PREDICTION_LOWER_BOUND, None)
    groups = pd.Series(group_keys).reset_index(drop=True)
    for idx in groups.groupby(groups, sort=False).groups.values():
        group_idx = np.asarray(list(idx))
        adjusted[group_idx] = np.minimum.accumulate(adjusted[group_idx])
    return adjusted


def make_prediction_plot_df(meta_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": meta_df["group_key"].astype(str).reset_index(drop=True),
            "sample_order_in_species": meta_df["sample_order_in_species"].to_numpy(),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )


def plot_trend_by_species(meta_df: pd.DataFrame, predictions: np.ndarray, title: str) -> None:
    plot_df = make_prediction_plot_df(meta_df, predictions)
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.1,
        alpha=0.85,
    )
    plt.xlabel("樹種内サンプル番号")
    plt.ylabel("予測含水率")
    plt.title(title)
    plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_raw_vs_refined(
    meta_df: pd.DataFrame,
    raw_predictions: np.ndarray,
    refined_predictions: np.ndarray,
    split_name: str,
) -> None:
    raw_plot_df = make_prediction_plot_df(meta_df, raw_predictions)
    refined_plot_df = make_prediction_plot_df(meta_df, refined_predictions)
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
        data=refined_plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.0,
        alpha=0.8,
        ax=axes[1],
    )
    axes[1].set_title(f"{split_name}: backward-refined predicted moisture")
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
with mlflow.start_run(run_name=f"ridge_backward_refine_{DATE}") as run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_alpha = float(study.best_params["alpha"])
    base_cv_rmse = float(study.best_value)
    print(f"Best base alpha={best_alpha:.6f}, CV RMSE={base_cv_rmse:.4f}")

    base_oof_pred = generate_base_oof_predictions(best_alpha)
    base_oof_pred = np.clip(base_oof_pred, PREDICTION_LOWER_BOUND, None)
    base_model = build_base_model(best_alpha)
    base_model.fit(X_train, y_train)
    train_pred_raw = np.clip(base_model.predict(X_train), PREDICTION_LOWER_BOUND, None)
    test_pred_raw = np.clip(base_model.predict(X_test), PREDICTION_LOWER_BOUND, None)

    teacher_target = apply_groupwise_monotone_target(
        y_train_array,
        group_keys=train_meta_df["group_key"],
        method=REFINE_TARGET_METHOD,
    )
    delta_train_df, tail_train_df = build_refine_training_frames(
        train_meta_df,
        raw_predictions=base_oof_pred,
        teacher_target=teacher_target,
    )
    delta_feature_cols = [c for c in delta_train_df.columns if c != "target"]
    tail_feature_cols = [c for c in tail_train_df.columns if c != "target"]

    delta_model = build_refine_model()
    tail_model = build_refine_model()
    delta_model.fit(delta_train_df[delta_feature_cols], delta_train_df["target"])
    tail_model.fit(tail_train_df[tail_feature_cols], tail_train_df["target"])

    delta_cap = float(delta_train_df["target"].quantile(DELTA_UPPER_QUANTILE))
    tail_cap = float(tail_train_df["target"].quantile(TAIL_UPPER_QUANTILE))

    train_pred_backward = apply_backward_refine(
        train_meta_df,
        raw_predictions=base_oof_pred,
        delta_model=delta_model,
        tail_model=tail_model,
        delta_feature_cols=delta_feature_cols,
        tail_feature_cols=tail_feature_cols,
        delta_cap=delta_cap,
        tail_cap=tail_cap,
    )
    test_pred_backward = apply_backward_refine(
        test_meta_df,
        raw_predictions=test_pred_raw,
        delta_model=delta_model,
        tail_model=tail_model,
        delta_feature_cols=delta_feature_cols,
        tail_feature_cols=tail_feature_cols,
        delta_cap=delta_cap,
        tail_cap=tail_cap,
    )
    train_pred_cummin = apply_groupwise_cummin(base_oof_pred, train_meta_df["group_key"])

    train_raw_rmse = compute_rmse(y_train_array, base_oof_pred)
    train_cummin_rmse = compute_rmse(y_train_array, train_pred_cummin)
    train_backward_rmse = compute_rmse(y_train_array, train_pred_backward)
    teacher_backward_rmse = compute_rmse(teacher_target, train_pred_backward)
    print(
        f"Train raw OOF RMSE={train_raw_rmse:.4f}, "
        f"cummin RMSE={train_cummin_rmse:.4f}, "
        f"backward RMSE={train_backward_rmse:.4f}, "
        f"teacher RMSE={teacher_backward_rmse:.4f}"
    )

    train_raw_violations = count_monotonic_violations(base_oof_pred, train_meta_df["group_key"])
    train_cummin_violations = count_monotonic_violations(train_pred_cummin, train_meta_df["group_key"])
    train_backward_violations = count_monotonic_violations(train_pred_backward, train_meta_df["group_key"])
    test_raw_violations = count_monotonic_violations(test_pred_raw, test_meta_df["group_key"])
    test_backward_violations = count_monotonic_violations(test_pred_backward, test_meta_df["group_key"])
    print(
        f"Violations train raw={train_raw_violations}, train cummin={train_cummin_violations}, "
        f"train backward={train_backward_violations}, test raw={test_raw_violations}, "
        f"test backward={test_backward_violations}"
    )

    submit_df = pd.DataFrame({"id": test_ids, "value": test_pred_backward})
    rmse_tag = int(round(train_backward_rmse * 10000))
    submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_backward_refine_{DATE}_{rmse_tag:04d}.csv")
    submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
    print(f"saved: {submit_path}")

    detail_df = pd.DataFrame(
        {
            "id": test_ids,
            "species": test_meta_df["group_key"],
            "sample_order_in_species": test_meta_df["sample_order_in_species"],
            "raw_prediction": test_pred_raw,
            "backward_refined_prediction": test_pred_backward,
        }
    )
    detail_path = os.path.join(SUBMIT_DIR, f"test_prediction_detail_backward_refine_{DATE}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    mlflow.log_params(
        {
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "cv": CV,
            "n_trials": N_TRIALS,
            "best_alpha": best_alpha,
            "refine_target_method": REFINE_TARGET_METHOD,
            "prediction_lower_bound": PREDICTION_LOWER_BOUND,
            "delta_upper_quantile": DELTA_UPPER_QUANTILE,
            "tail_upper_quantile": TAIL_UPPER_QUANTILE,
            "refine_learning_rate": REFINE_LEARNING_RATE,
            "refine_max_depth": REFINE_MAX_DEPTH,
            "refine_max_iter": REFINE_MAX_ITER,
            "refine_min_samples_leaf": REFINE_MIN_SAMPLES_LEAF,
            "refine_l2_regularization": REFINE_L2_REGULARIZATION,
            "delta_feature_count": len(delta_feature_cols),
            "tail_feature_count": len(tail_feature_cols),
        }
    )
    mlflow.log_metric("base_cv_rmse", base_cv_rmse)
    mlflow.log_metric("train_raw_oof_rmse", train_raw_rmse)
    mlflow.log_metric("train_cummin_rmse", train_cummin_rmse)
    mlflow.log_metric("train_backward_rmse", train_backward_rmse)
    mlflow.log_metric("train_backward_teacher_rmse", teacher_backward_rmse)
    mlflow.log_metric("train_raw_violations", train_raw_violations)
    mlflow.log_metric("train_cummin_violations", train_cummin_violations)
    mlflow.log_metric("train_backward_violations", train_backward_violations)
    mlflow.log_metric("test_raw_violations", test_raw_violations)
    mlflow.log_metric("test_backward_violations", test_backward_violations)
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

plot_trend_by_species(train_meta_df, train_pred_cummin, "Train: cummin baseline on base OOF predictions")
plot_raw_vs_refined(train_meta_df, base_oof_pred, train_pred_backward, "Train")
plot_raw_vs_refined(test_meta_df, test_pred_raw, test_pred_backward, "Test")

# %%
test_df

# %%
submit_df
# %%
