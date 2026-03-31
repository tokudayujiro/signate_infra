# %%
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import SimpleNamespace

import matplotlib.pyplot as plt
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

# backward refine と monotone smooth を順序依存で blend する

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_refine_smooth_blend_nested_oof_shape_polish"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 60
CV = 6
PREDICTION_LOWER_BOUND = 0.0

REFINE_TARGET_METHOD = "isotonic"
DELTA_UPPER_QUANTILE = 0.995
TAIL_UPPER_QUANTILE = 0.995
REFINE_LEARNING_RATE = 0.05
REFINE_MAX_DEPTH = 3
REFINE_MAX_ITER = 300
REFINE_MIN_SAMPLES_LEAF = 10
REFINE_L2_REGULARIZATION = 0.1

SMOOTH_LAMBDA_CANDIDATES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SMOOTH_PROJECTED_GRADIENT_MAXITER = 300
SMOOTH_PROJECTED_GRADIENT_TOL = 1e-6

BLEND_LEARNING_RATE = 0.05
BLEND_MAX_DEPTH = 3
BLEND_MAX_ITER = 300
BLEND_MIN_SAMPLES_LEAF = 20
BLEND_L2_REGULARIZATION = 0.1

POLISH_SMOOTH_LAMBDA_CANDIDATES = (0.1, 0.3, 1.0, 3.0)
POLISH_BLEND_RATIO_CANDIDATES = (0.1, 0.2, 0.35)


# %%
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


def count_monotonic_violations(predictions: np.ndarray, group_keys: pd.Series) -> int:
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    violation_count = 0
    for group_idx in iter_group_indices(group_keys):
        group_pred = pred[group_idx]
        violation_count += int((group_pred[1:] > group_pred[:-1]).sum())
    return violation_count


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


def generate_base_oof_predictions(alpha: float) -> np.ndarray:
    oof_pred = np.zeros(len(X_train), dtype=float)
    for fit_idx, valid_idx in cv_splits:
        fold_model = build_base_model(alpha)
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_pred[valid_idx] = fold_model.predict(X_train.iloc[valid_idx])
    return np.clip(oof_pred, PREDICTION_LOWER_BOUND, None)


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
            continue
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        output[group_idx] = iso.fit_transform(np.arange(len(group_pred), dtype=float), group_pred)
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
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    projected = iso.fit_transform(np.arange(len(values), dtype=float), values)
    projected = np.maximum(projected, PREDICTION_LOWER_BOUND)
    projected = np.minimum.accumulate(projected)
    return projected


def solve_monotone_smooth_projection(raw_values: np.ndarray, smooth_lambda: float) -> np.ndarray:
    raw = np.clip(np.asarray(raw_values, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    n_samples = len(raw)
    if n_samples <= 1:
        return raw.copy()

    current = project_to_monotone_decreasing(raw)
    if smooth_lambda <= 0.0 or n_samples <= 2:
        return current

    second_diff = build_second_difference_matrix(n_samples)
    hessian = np.eye(n_samples, dtype=float) + smooth_lambda * (second_diff.T @ second_diff)
    step_size = 1.0 / max(float(np.linalg.eigvalsh(hessian).max()), 1e-8)

    for _ in range(SMOOTH_PROJECTED_GRADIENT_MAXITER):
        gradient = hessian @ current - raw
        next_values = project_to_monotone_decreasing(current - step_size * gradient)
        if np.max(np.abs(next_values - current)) <= SMOOTH_PROJECTED_GRADIENT_TOL:
            current = next_values
            break
        current = next_values
    return current


def apply_groupwise_monotone_smooth(predictions: np.ndarray, group_keys: pd.Series, smooth_lambda: float) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=float).reshape(-1)
    output = np.zeros_like(adjusted)
    for group_idx in iter_group_indices(group_keys):
        output[group_idx] = solve_monotone_smooth_projection(adjusted[group_idx], smooth_lambda)
    return output


def select_best_smooth_lambda(raw_predictions: np.ndarray, y_true: np.ndarray, group_keys: pd.Series) -> tuple[float, np.ndarray, dict[str, float]]:
    scores: dict[str, float] = {}
    adjusted_by_lambda: dict[str, np.ndarray] = {}
    for smooth_lambda in SMOOTH_LAMBDA_CANDIDATES:
        method_name = f"smooth_{smooth_lambda:g}"
        adjusted = apply_groupwise_monotone_smooth(raw_predictions, group_keys, smooth_lambda)
        adjusted_by_lambda[method_name] = adjusted
        scores[method_name] = compute_rmse(y_true, adjusted)
    best_method = min(scores, key=scores.get)
    return float(best_method.split("_", 1)[1]), adjusted_by_lambda[best_method], scores


def apply_groupwise_monotone_target(values: np.ndarray, group_keys: pd.Series, method: str = "isotonic") -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    output = values.copy()
    for group_idx in iter_group_indices(group_keys):
        group_values = output[group_idx]
        if method == "raw":
            adjusted = group_values
        elif method == "cummin":
            adjusted = np.minimum.accumulate(group_values)
        elif method == "isotonic":
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            adjusted = iso.fit_transform(np.arange(len(group_values), dtype=float), group_values)
        else:
            raise ValueError(f"Unsupported REFINE_TARGET_METHOD: {method}")
        output[group_idx] = adjusted
    return output


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


def make_common_refine_features(position: int, raw_values: np.ndarray, group_meta: pd.DataFrame, cache: dict[str, np.ndarray | float]) -> dict[str, float]:
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


def build_refine_training_frames(meta_df: pd.DataFrame, raw_predictions: np.ndarray, teacher_target: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_rows: list[dict[str, float]] = []
    tail_rows: list[dict[str, float]] = []
    raw_predictions = np.asarray(raw_predictions, dtype=float).reshape(-1)
    teacher_target = np.asarray(teacher_target, dtype=float).reshape(-1)

    for group_idx in iter_group_indices(meta_df["group_key"]):
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

    for group_idx in iter_group_indices(meta_df["group_key"]):
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


def build_blend_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=BLEND_LEARNING_RATE,
        max_depth=BLEND_MAX_DEPTH,
        max_iter=BLEND_MAX_ITER,
        min_samples_leaf=BLEND_MIN_SAMPLES_LEAF,
        l2_regularization=BLEND_L2_REGULARIZATION,
        random_state=RANDOM_STATE,
    )


def build_blend_feature_frame(meta_df: pd.DataFrame, raw_pred: np.ndarray, backward_pred: np.ndarray, smooth_pred: np.ndarray) -> pd.DataFrame:
    feature_df = meta_df[["sample_order_in_species", "group_size", "remaining_steps", "order_ratio", "remaining_ratio"]].copy()
    feature_df["raw_pred"] = np.asarray(raw_pred, dtype=float).reshape(-1)
    feature_df["backward_pred"] = np.asarray(backward_pred, dtype=float).reshape(-1)
    feature_df["smooth_pred"] = np.asarray(smooth_pred, dtype=float).reshape(-1)
    feature_df["pred_gap"] = feature_df["backward_pred"] - feature_df["smooth_pred"]
    feature_df["abs_pred_gap"] = feature_df["pred_gap"].abs()
    feature_df["raw_minus_backward"] = feature_df["raw_pred"] - feature_df["backward_pred"]
    feature_df["raw_minus_smooth"] = feature_df["raw_pred"] - feature_df["smooth_pred"]
    feature_df["pred_mean"] = 0.5 * (feature_df["backward_pred"] + feature_df["smooth_pred"])
    return feature_df


def compute_ideal_blend_weight(y_true: np.ndarray, backward_pred: np.ndarray, smooth_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    backward_pred = np.asarray(backward_pred, dtype=float).reshape(-1)
    smooth_pred = np.asarray(smooth_pred, dtype=float).reshape(-1)
    denom = backward_pred - smooth_pred
    weight = np.full_like(y_true, 0.5, dtype=float)
    valid = np.abs(denom) > 1e-8
    weight[valid] = (y_true[valid] - smooth_pred[valid]) / denom[valid]
    return np.clip(weight, 0.0, 1.0)


def apply_final_postprocess(predictions: np.ndarray, group_keys: pd.Series, method_name: str) -> np.ndarray:
    if method_name == "none":
        return np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    if method_name == "cummin":
        return apply_groupwise_cummin(predictions, group_keys)
    if method_name == "isotonic":
        return apply_groupwise_isotonic(predictions, group_keys)
    raise ValueError(f"Unsupported final postprocess: {method_name}")


def select_best_final_postprocess(predictions: np.ndarray, y_true: np.ndarray, group_keys: pd.Series) -> tuple[str, np.ndarray, dict[str, float]]:
    candidates = {}
    for method_name in ["none", "cummin", "isotonic"]:
        adjusted = apply_final_postprocess(predictions, group_keys, method_name)
        candidates[method_name] = adjusted
    scores = {name: compute_rmse(y_true, pred) for name, pred in candidates.items()}
    best_method = min(scores, key=scores.get)
    return best_method, candidates[best_method], scores


def apply_shape_polish(
    predictions: np.ndarray,
    group_keys: pd.Series,
    smooth_lambda: float,
    blend_ratio: float,
) -> np.ndarray:
    """最終予測を軽く滑らかに寄せて、局所的な歪みを弱める。"""
    base = np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    if blend_ratio <= 0.0:
        return base.copy()
    smooth = apply_groupwise_monotone_smooth(base, group_keys, smooth_lambda)
    polished = (1.0 - blend_ratio) * base + blend_ratio * smooth
    polished = np.clip(polished, PREDICTION_LOWER_BOUND, None)
    return apply_groupwise_cummin(polished, group_keys)


def select_best_shape_polish(
    predictions: np.ndarray,
    y_true: np.ndarray,
    group_keys: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray, dict[str, float]]:
    """train OOF 上で軽い shape polish の強さを選ぶ。"""
    candidates: dict[str, np.ndarray] = {"none": np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)}
    configs: dict[str, dict[str, float | str]] = {
        "none": {"name": "none", "smooth_lambda": 0.0, "blend_ratio": 0.0}
    }

    for smooth_lambda in POLISH_SMOOTH_LAMBDA_CANDIDATES:
        for blend_ratio in POLISH_BLEND_RATIO_CANDIDATES:
            method_name = f"polish_l{smooth_lambda:g}_r{blend_ratio:g}"
            candidates[method_name] = apply_shape_polish(
                predictions,
                group_keys=group_keys,
                smooth_lambda=smooth_lambda,
                blend_ratio=blend_ratio,
            )
            configs[method_name] = {
                "name": method_name,
                "smooth_lambda": float(smooth_lambda),
                "blend_ratio": float(blend_ratio),
            }

    scores = {name: compute_rmse(y_true, pred) for name, pred in candidates.items()}
    best_method = min(scores, key=scores.get)
    return configs[best_method], candidates[best_method], scores


def make_prediction_plot_df(meta_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": meta_df["group_key"].astype(str).reset_index(drop=True),
            "sample_order_in_species": meta_df["sample_order_in_species"].to_numpy(),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )


def plot_prediction_trend_by_species(meta_df: pd.DataFrame, predictions: np.ndarray, title: str) -> None:
    plot_df = make_prediction_plot_df(meta_df, predictions)
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
    plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_raw_vs_final(meta_df: pd.DataFrame, raw_predictions: np.ndarray, final_predictions: np.ndarray, split_name: str, final_method: str) -> None:
    raw_plot_df = make_prediction_plot_df(meta_df, raw_predictions)
    final_plot_df = make_prediction_plot_df(meta_df, final_predictions)
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
        data=final_plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.0,
        alpha=0.8,
        ax=axes[1],
    )
    axes[1].set_title(f"{split_name}: blend final ({final_method})")
    axes[1].set_xlabel("樹種内サンプル番号")
    axes[1].set_ylabel("予測含水率")
    axes[1].legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


# %%
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
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train[c]) and c not in forced_categorical]
categorical_cols = [c for c in feature_cols if c in forced_categorical or not pd.api.types.is_numeric_dtype(X_train[c])]
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
def generate_subset_oof_predictions(X_subset: pd.DataFrame, y_subset: pd.Series, alpha: float) -> np.ndarray:
    n_samples = len(X_subset)
    if n_samples < 2:
        single_model = build_base_model(alpha)
        single_model.fit(X_subset, y_subset)
        return np.clip(single_model.predict(X_subset), PREDICTION_LOWER_BOUND, None)

    n_splits = min(CV, n_samples)
    if n_splits < 2:
        raise ValueError(f"inner OOF 用のサンプル数が不足しています: {n_samples}")

    inner_splits = list(KFold(n_splits=n_splits, shuffle=False).split(X_subset, y_subset))
    oof_pred = np.zeros(n_samples, dtype=float)
    for inner_fit_idx, inner_valid_idx in inner_splits:
        inner_model = build_base_model(alpha)
        inner_model.fit(X_subset.iloc[inner_fit_idx], y_subset.iloc[inner_fit_idx])
        oof_pred[inner_valid_idx] = inner_model.predict(X_subset.iloc[inner_valid_idx])
    return np.clip(oof_pred, PREDICTION_LOWER_BOUND, None)


def summarize_choice_counts(values: list[object]) -> str:
    value_counts = pd.Series([str(v) for v in values]).value_counts()
    return ", ".join(f"{label}:{count}" for label, count in value_counts.items())


def run_nested_outer_fold(fold_no: int, fit_idx: np.ndarray, valid_idx: np.ndarray, alpha: float) -> dict[str, object]:
    X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_train.iloc[fit_idx].reset_index(drop=True)
    fit_meta_df = train_meta_df.iloc[fit_idx].reset_index(drop=True)
    y_fit_array = y_fit.to_numpy(dtype=float)

    raw_fit_oof = generate_subset_oof_predictions(X_fit, y_fit, alpha)

    base_model = build_base_model(alpha)
    base_model.fit(X_fit, y_fit)
    raw_train_all = np.clip(base_model.predict(X_train), PREDICTION_LOWER_BOUND, None)
    raw_test = np.clip(base_model.predict(X_test), PREDICTION_LOWER_BOUND, None)

    teacher_target_fit = apply_groupwise_monotone_target(
        y_fit_array,
        group_keys=fit_meta_df["group_key"],
        method=REFINE_TARGET_METHOD,
    )
    delta_train_df, tail_train_df = build_refine_training_frames(
        fit_meta_df,
        raw_predictions=raw_fit_oof,
        teacher_target=teacher_target_fit,
    )
    delta_feature_cols = [c for c in delta_train_df.columns if c != "target"]
    tail_feature_cols = [c for c in tail_train_df.columns if c != "target"]

    delta_model = build_refine_model()
    tail_model = build_refine_model()
    delta_model.fit(delta_train_df[delta_feature_cols], delta_train_df["target"])
    tail_model.fit(tail_train_df[tail_feature_cols], tail_train_df["target"])
    delta_cap = float(delta_train_df["target"].quantile(DELTA_UPPER_QUANTILE))
    tail_cap = float(tail_train_df["target"].quantile(TAIL_UPPER_QUANTILE))

    backward_fit = apply_backward_refine(
        fit_meta_df,
        raw_predictions=raw_fit_oof,
        delta_model=delta_model,
        tail_model=tail_model,
        delta_feature_cols=delta_feature_cols,
        tail_feature_cols=tail_feature_cols,
        delta_cap=delta_cap,
        tail_cap=tail_cap,
    )
    backward_train_all = apply_backward_refine(
        train_meta_df,
        raw_predictions=raw_train_all,
        delta_model=delta_model,
        tail_model=tail_model,
        delta_feature_cols=delta_feature_cols,
        tail_feature_cols=tail_feature_cols,
        delta_cap=delta_cap,
        tail_cap=tail_cap,
    )
    backward_test = apply_backward_refine(
        test_meta_df,
        raw_predictions=raw_test,
        delta_model=delta_model,
        tail_model=tail_model,
        delta_feature_cols=delta_feature_cols,
        tail_feature_cols=tail_feature_cols,
        delta_cap=delta_cap,
        tail_cap=tail_cap,
    )

    best_smooth_lambda, smooth_fit, smooth_scores = select_best_smooth_lambda(
        raw_fit_oof,
        y_true=y_fit_array,
        group_keys=fit_meta_df["group_key"],
    )
    smooth_train_all = apply_groupwise_monotone_smooth(
        raw_train_all,
        train_meta_df["group_key"],
        smooth_lambda=best_smooth_lambda,
    )
    smooth_test = apply_groupwise_monotone_smooth(
        raw_test,
        test_meta_df["group_key"],
        smooth_lambda=best_smooth_lambda,
    )

    fit_meta_for_blend = train_meta_df.iloc[fit_idx].reset_index(drop=True)
    blend_feature_fit = build_blend_feature_frame(
        fit_meta_for_blend,
        raw_pred=raw_train_all[fit_idx],
        backward_pred=backward_train_all[fit_idx],
        smooth_pred=smooth_train_all[fit_idx],
    )
    blend_feature_cols = blend_feature_fit.columns.tolist()
    ideal_blend_weight_fit = compute_ideal_blend_weight(
        y_fit_array,
        backward_pred=backward_train_all[fit_idx],
        smooth_pred=smooth_train_all[fit_idx],
    )

    blend_model = build_blend_model()
    blend_model.fit(blend_feature_fit[blend_feature_cols], ideal_blend_weight_fit)

    blend_feature_train_all = build_blend_feature_frame(
        train_meta_df,
        raw_pred=raw_train_all,
        backward_pred=backward_train_all,
        smooth_pred=smooth_train_all,
    )
    blend_feature_test = build_blend_feature_frame(
        test_meta_df,
        raw_pred=raw_test,
        backward_pred=backward_test,
        smooth_pred=smooth_test,
    )
    weight_train_all = np.clip(blend_model.predict(blend_feature_train_all[blend_feature_cols]), 0.0, 1.0)
    weight_test = np.clip(blend_model.predict(blend_feature_test[blend_feature_cols]), 0.0, 1.0)
    blend_raw_train_all = weight_train_all * backward_train_all + (1.0 - weight_train_all) * smooth_train_all
    blend_raw_test = weight_test * backward_test + (1.0 - weight_test) * smooth_test

    y_valid_array = y_train_array[valid_idx]
    valid_metrics = {
        "fold": fold_no,
        "valid_size": int(len(valid_idx)),
        "smooth_lambda": float(best_smooth_lambda),
        "raw_valid_rmse": compute_rmse(y_valid_array, raw_train_all[valid_idx]),
        "backward_valid_rmse": compute_rmse(y_valid_array, backward_train_all[valid_idx]),
        "smooth_valid_rmse": compute_rmse(y_valid_array, smooth_train_all[valid_idx]),
        "blend_raw_valid_rmse": compute_rmse(y_valid_array, blend_raw_train_all[valid_idx]),
    }

    return {
        "raw_train_all": raw_train_all,
        "backward_train_all": backward_train_all,
        "smooth_train_all": smooth_train_all,
        "weight_train_all": weight_train_all,
        "blend_raw_train_all": blend_raw_train_all,
        "raw_test": raw_test,
        "backward_test": backward_test,
        "smooth_test": smooth_test,
        "weight_test": weight_test,
        "blend_raw_test": blend_raw_test,
        "smooth_lambda": float(best_smooth_lambda),
        "smooth_scores": smooth_scores,
        "valid_metrics": valid_metrics,
        "blend_feature_count": len(blend_feature_cols),
    }


# %%
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"refine_smooth_blend_nested_oof_{DATE}") as run:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_alpha = float(study.best_params["alpha"])
    base_cv_rmse = float(study.best_value)
    print(f"Best base alpha={best_alpha:.6f}, CV RMSE={base_cv_rmse:.4f}")

    train_pred_raw = np.zeros(len(X_train), dtype=float)
    train_pred_backward = np.zeros(len(X_train), dtype=float)
    train_pred_smooth = np.zeros(len(X_train), dtype=float)
    train_weight_pred = np.zeros(len(X_train), dtype=float)
    train_pred_blend_raw = np.zeros(len(X_train), dtype=float)

    test_raw_sum = np.zeros(len(X_test), dtype=float)
    test_backward_sum = np.zeros(len(X_test), dtype=float)
    test_smooth_sum = np.zeros(len(X_test), dtype=float)
    test_weight_sum = np.zeros(len(X_test), dtype=float)
    test_blend_raw_sum = np.zeros(len(X_test), dtype=float)

    fold_rows: list[dict[str, float | int]] = []
    smooth_lambda_choices: list[str] = []
    smooth_metric_summary: dict[str, list[float]] = {}
    blend_feature_count = 0

    for fold_no, (fit_idx, valid_idx) in enumerate(cv_splits, start=1):
        fold_result = run_nested_outer_fold(fold_no, fit_idx, valid_idx, best_alpha)

        train_pred_raw[valid_idx] = np.asarray(fold_result["raw_train_all"], dtype=float)[valid_idx]
        train_pred_backward[valid_idx] = np.asarray(fold_result["backward_train_all"], dtype=float)[valid_idx]
        train_pred_smooth[valid_idx] = np.asarray(fold_result["smooth_train_all"], dtype=float)[valid_idx]
        train_weight_pred[valid_idx] = np.asarray(fold_result["weight_train_all"], dtype=float)[valid_idx]
        train_pred_blend_raw[valid_idx] = np.asarray(fold_result["blend_raw_train_all"], dtype=float)[valid_idx]

        test_raw_sum += np.asarray(fold_result["raw_test"], dtype=float)
        test_backward_sum += np.asarray(fold_result["backward_test"], dtype=float)
        test_smooth_sum += np.asarray(fold_result["smooth_test"], dtype=float)
        test_weight_sum += np.asarray(fold_result["weight_test"], dtype=float)
        test_blend_raw_sum += np.asarray(fold_result["blend_raw_test"], dtype=float)

        fold_rows.append(dict(fold_result["valid_metrics"]))
        smooth_lambda_choices.append(f"{float(fold_result['smooth_lambda']):g}")
        for method_name, score in dict(fold_result["smooth_scores"]).items():
            smooth_metric_summary.setdefault(method_name, []).append(float(score))
        blend_feature_count = int(fold_result["blend_feature_count"])

        print(
            f"Fold {fold_no}: smooth_lambda={float(fold_result['smooth_lambda']):g}, "
            f"valid blend RMSE={float(dict(fold_result['valid_metrics'])['blend_raw_valid_rmse']):.4f}"
        )

    n_folds = len(cv_splits)
    test_pred_raw = test_raw_sum / n_folds
    test_pred_backward = test_backward_sum / n_folds
    test_pred_smooth = test_smooth_sum / n_folds
    test_weight_pred = test_weight_sum / n_folds
    test_pred_blend_raw = test_blend_raw_sum / n_folds

    final_method, train_pred_final, final_scores = select_best_final_postprocess(
        train_pred_blend_raw,
        y_true=y_train_array,
        group_keys=train_meta_df["group_key"],
    )
    test_pred_final = apply_final_postprocess(
        test_pred_blend_raw,
        group_keys=test_meta_df["group_key"],
        method_name=final_method,
    )
    polish_config, train_pred_polished, polish_scores = select_best_shape_polish(
        train_pred_final,
        y_true=y_train_array,
        group_keys=train_meta_df["group_key"],
    )
    test_pred_polished = apply_shape_polish(
        test_pred_final,
        group_keys=test_meta_df["group_key"],
        smooth_lambda=float(polish_config["smooth_lambda"]),
        blend_ratio=float(polish_config["blend_ratio"]),
    )

    fold_summary_df = pd.DataFrame(fold_rows)
    print(fold_summary_df)

    smooth_scores = {
        method_name: float(np.mean(score_list)) for method_name, score_list in smooth_metric_summary.items()
    }
    print("Mean smooth RMSE by lambda:", {k: round(v, 4) for k, v in smooth_scores.items()})
    print(f"Smooth lambda choices by fold: {summarize_choice_counts(smooth_lambda_choices)}")

    train_raw_rmse = compute_rmse(y_train_array, train_pred_raw)
    train_backward_rmse = compute_rmse(y_train_array, train_pred_backward)
    train_smooth_rmse = compute_rmse(y_train_array, train_pred_smooth)
    train_blend_raw_rmse = compute_rmse(y_train_array, train_pred_blend_raw)
    train_final_rmse = compute_rmse(y_train_array, train_pred_final)
    train_polished_rmse = compute_rmse(y_train_array, train_pred_polished)
    print(
        f"Train RMSE raw={train_raw_rmse:.4f}, backward={train_backward_rmse:.4f}, "
        f"smooth={train_smooth_rmse:.4f}, blend_raw={train_blend_raw_rmse:.4f}, "
        f"final={train_final_rmse:.4f}, polished={train_polished_rmse:.4f}"
    )
    print("Final postprocess RMSE:", {k: round(v, 4) for k, v in final_scores.items()})
    print("Shape polish RMSE:", {k: round(v, 4) for k, v in polish_scores.items()})
    print(
        f"Selected shape polish: {str(polish_config['name'])} "
        f"(smooth_lambda={float(polish_config['smooth_lambda']):g}, "
        f"blend_ratio={float(polish_config['blend_ratio']):g})"
    )

    train_raw_violations = count_monotonic_violations(train_pred_raw, train_meta_df["group_key"])
    train_blend_raw_violations = count_monotonic_violations(train_pred_blend_raw, train_meta_df["group_key"])
    train_final_violations = count_monotonic_violations(train_pred_final, train_meta_df["group_key"])
    train_polished_violations = count_monotonic_violations(train_pred_polished, train_meta_df["group_key"])
    test_raw_violations = count_monotonic_violations(test_pred_raw, test_meta_df["group_key"])
    test_blend_raw_violations = count_monotonic_violations(test_pred_blend_raw, test_meta_df["group_key"])
    test_final_violations = count_monotonic_violations(test_pred_final, test_meta_df["group_key"])
    test_polished_violations = count_monotonic_violations(test_pred_polished, test_meta_df["group_key"])
    print(
        f"Violations train raw={train_raw_violations}, train blend_raw={train_blend_raw_violations}, "
        f"train final={train_final_violations}, train polished={train_polished_violations}, "
        f"test raw={test_raw_violations}, test blend_raw={test_blend_raw_violations}, "
        f"test final={test_final_violations}, test polished={test_polished_violations}"
    )

    submit_df = build_submit_df(test_ids, test_pred_polished)
    rmse_tag = int(round(train_polished_rmse * 10000))
    submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_08_refine_smooth_blend_nested_oof_shape_polish_{DATE}_{rmse_tag:04d}.csv")
    submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
    print(f"saved: {submit_path}")

    detail_df = pd.DataFrame(
        {
            "id": test_ids,
            "species": test_meta_df["group_key"],
            "sample_order_in_species": test_meta_df["sample_order_in_species"],
            "raw_prediction": test_pred_raw,
            "backward_prediction": test_pred_backward,
            "smooth_prediction": test_pred_smooth,
            "blend_weight_pred": test_weight_pred,
            "blend_raw_prediction": test_pred_blend_raw,
            "final_prediction": test_pred_final,
            "polished_prediction": test_pred_polished,
            "final_method": final_method,
            "shape_polish_name": str(polish_config["name"]),
            "shape_polish_smooth_lambda": float(polish_config["smooth_lambda"]),
            "shape_polish_blend_ratio": float(polish_config["blend_ratio"]),
            "smooth_lambda_choices": summarize_choice_counts(smooth_lambda_choices),
        }
    )
    detail_path = os.path.join(SUBMIT_DIR, f"test_prediction_detail_08_refine_smooth_blend_nested_oof_shape_polish_{DATE}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    fold_summary_path = os.path.join(SUBMIT_DIR, f"fold_summary_08_refine_smooth_blend_nested_oof_shape_polish_{DATE}.csv")
    fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

    mlflow.log_params(
        {
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "cv": CV,
            "n_trials": N_TRIALS,
            "best_alpha": best_alpha,
            "prediction_lower_bound": PREDICTION_LOWER_BOUND,
            "refine_target_method": REFINE_TARGET_METHOD,
            "final_method": final_method,
            "shape_polish_name": str(polish_config["name"]),
            "shape_polish_smooth_lambda": float(polish_config["smooth_lambda"]),
            "shape_polish_blend_ratio": float(polish_config["blend_ratio"]),
            "blend_feature_count": blend_feature_count,
            "smooth_lambda_choices": summarize_choice_counts(smooth_lambda_choices),
        }
    )
    mlflow.log_metric("base_cv_rmse", base_cv_rmse)
    mlflow.log_metric("train_raw_rmse", train_raw_rmse)
    mlflow.log_metric("train_backward_rmse", train_backward_rmse)
    mlflow.log_metric("train_smooth_rmse", train_smooth_rmse)
    mlflow.log_metric("train_blend_raw_rmse", train_blend_raw_rmse)
    mlflow.log_metric("train_final_rmse", train_final_rmse)
    mlflow.log_metric("train_polished_rmse", train_polished_rmse)
    mlflow.log_metric("train_raw_violations", train_raw_violations)
    mlflow.log_metric("train_blend_raw_violations", train_blend_raw_violations)
    mlflow.log_metric("train_final_violations", train_final_violations)
    mlflow.log_metric("train_polished_violations", train_polished_violations)
    mlflow.log_metric("test_raw_violations", test_raw_violations)
    mlflow.log_metric("test_blend_raw_violations", test_blend_raw_violations)
    mlflow.log_metric("test_final_violations", test_final_violations)
    mlflow.log_metric("test_polished_violations", test_polished_violations)
    for _, row in fold_summary_df.iterrows():
        mlflow.log_metric(f"fold_{int(row['fold'])}_blend_raw_valid_rmse", float(row["blend_raw_valid_rmse"]))
    for method_name, score in smooth_scores.items():
        mlflow.log_metric(f"smooth_rmse_{method_name}", float(score))
    for method_name, score in final_scores.items():
        mlflow.log_metric(f"final_rmse_{method_name}", float(score))
    for method_name, score in polish_scores.items():
        mlflow.log_metric(f"shape_polish_rmse_{method_name}", float(score))
    mlflow.log_artifact(submit_path)
    mlflow.log_artifact(detail_path)
    mlflow.log_artifact(fold_summary_path)
    mlflow.log_artifact(__file__)
    print(f"mlflow run_id: {run.info.run_id}")

submit_df.head()


# %%
plot_raw_vs_final(train_meta_df, train_pred_final, train_pred_polished, "Train", f"shape polish ({str(polish_config['name'])})")
plot_raw_vs_final(test_meta_df, test_pred_final, test_pred_polished, "Test", f"shape polish ({str(polish_config['name'])})")
plot_prediction_trend_by_species(
    train_meta_df,
    train_pred_polished,
    f"学習データ: 樹種ごとの予測含水率の推移 (shape polish)",
)
plot_prediction_trend_by_species(
    test_meta_df,
    test_pred_polished,
    f"テストデータ: 樹種ごとの予測含水率の推移 (shape polish)",
)
# %%
test_pred_polished
# %%
submit_df
# %%
test_meta_df
# %%
