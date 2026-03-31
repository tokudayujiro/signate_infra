# %%
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    japanize_matplotlib = None


# このノートブックの流れ
# 1. スペクトルだけで base 予測を作る
# 2. 樹種 GroupKFold で OOF を作り、未知樹種に近い検証をする
# 3. train の含水率曲線からグローバルな減衰テンプレートを作る
# 4. raw 予測をテンプレートへ射影して、最終 submit を作る


# --- 設定 ---
NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")

TARGET_COL = "含水率"
ID_COL = "sample number"
RANDOM_STATE = 42
N_TRIALS = 60
CV = 6
PREDICTION_LOWER_BOUND = 0.0

REFINE_TARGET_METHOD = "isotonic"
SMOOTH_LAMBDA_CANDIDATES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SMOOTH_PROJECTED_GRADIENT_MAXITER = 300
SMOOTH_PROJECTED_GRADIENT_TOL = 1e-6

TEMPLATE_GRID_SIZE = 201
TEMPLATE_SMOOTH_LAMBDA = 3.0
TEMPLATE_GAMMA_CANDIDATES = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
TEMPLATE_BLEND_WEIGHT_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


# %%
def infer_species_col(df: pd.DataFrame) -> str | None:
    """樹種列名の揺れを吸収して見つける。"""
    candidates = ["樹種", "species", "species number"]
    return next((c for c in candidates if c in df.columns), None)


def infer_target_col(df: pd.DataFrame) -> str:
    """文字化け対策込みで目的変数列を特定する。"""
    if TARGET_COL in df.columns:
        return TARGET_COL
    return df.columns[3]


def build_sequence_meta(df: pd.DataFrame, species_col: str) -> pd.DataFrame:
    """樹種ごとの順序特徴を作る。"""
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
    """同じ group に属する行番号を、元の順序のまま返す。"""
    groups = pd.Series(group_keys).reset_index(drop=True)
    return [np.asarray(list(idx)) for idx in groups.groupby(groups, sort=False).groups.values()]


def compute_group_progress(group_size: int) -> np.ndarray:
    """group 内の進捗を 0..1 に正規化する。"""
    if group_size <= 1:
        return np.zeros(group_size, dtype=float)
    return np.linspace(0.0, 1.0, group_size)


def find_wavelength_cols(train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: set[str]) -> list[str]:
    """波長として読める共有数値列だけを特徴量に使う。"""
    wavelength_cols: list[str] = []
    for col in train_df.columns:
        if col in exclude_cols or col not in test_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        try:
            float(str(col))
            wavelength_cols.append(col)
        except (TypeError, ValueError):
            continue
    return wavelength_cols


def build_submit_df(test_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    """提出形式に合わせて予測結果をまとめる。"""
    sample_submit_path = DATA_DIR / "sample_submit.csv"
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    if sample_submit_path.exists():
        submit_df = pd.read_csv(sample_submit_path, header=None, encoding="utf-8-sig")
        if len(submit_df) == len(pred):
            submit_df = submit_df.iloc[:, :2].copy()
            submit_df.columns = ["id", "value"]
            submit_df["id"] = test_ids.to_numpy()
            submit_df["value"] = pred
            return submit_df
    return pd.DataFrame({"id": test_ids, "value": pred})


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE を返す。"""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def count_monotonic_violations(predictions: np.ndarray, group_keys: pd.Series) -> int:
    """樹種内で増加してしまった回数を数える。"""
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    violation_count = 0
    for group_idx in iter_group_indices(group_keys):
        group_pred = pred[group_idx]
        violation_count += int((group_pred[1:] > group_pred[:-1]).sum())
    return violation_count


def build_base_model(alpha: float) -> Pipeline:
    """スペクトル列だけを使うシンプルな base モデル。"""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def apply_groupwise_cummin(predictions: np.ndarray, group_keys: pd.Series) -> np.ndarray:
    """累積最小で単調減少にそろえる。"""
    adjusted = np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    output = adjusted.copy()
    for group_idx in iter_group_indices(group_keys):
        output[group_idx] = np.minimum.accumulate(output[group_idx])
    return output


def apply_groupwise_isotonic(predictions: np.ndarray, group_keys: pd.Series) -> np.ndarray:
    """isotonic 回帰で単調減少にそろえる。"""
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
    """滑らかさ罰則用の 2 階差分行列を作る。"""
    if n_samples <= 2:
        return np.zeros((0, n_samples), dtype=float)
    d2 = np.zeros((n_samples - 2, n_samples), dtype=float)
    for i in range(n_samples - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    return d2


def project_to_monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """値列を単調減少かつ下限以上へ射影する。"""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.clip(values, PREDICTION_LOWER_BOUND, None)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    projected = iso.fit_transform(np.arange(len(values), dtype=float), values)
    projected = np.maximum(projected, PREDICTION_LOWER_BOUND)
    projected = np.minimum.accumulate(projected)
    return projected


def solve_monotone_smooth_projection(raw_values: np.ndarray, smooth_lambda: float) -> np.ndarray:
    """単調減少と滑らかさを両立する射影を解く。"""
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
    """樹種ごとに単調減少 + 滑らか補正をかける。"""
    adjusted = np.asarray(predictions, dtype=float).reshape(-1)
    output = np.zeros_like(adjusted)
    for group_idx in iter_group_indices(group_keys):
        output[group_idx] = solve_monotone_smooth_projection(adjusted[group_idx], smooth_lambda)
    return output


def apply_groupwise_monotone_target(values: np.ndarray, group_keys: pd.Series, method: str = "isotonic") -> np.ndarray:
    """教師信号側も樹種ごとに単調減少へそろえる。"""
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


def build_global_decay_template(meta_df: pd.DataFrame, teacher_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train 全体から共通の減衰テンプレートを作る。"""
    progress_list: list[np.ndarray] = []
    normalized_list: list[np.ndarray] = []
    target = np.asarray(teacher_target, dtype=float).reshape(-1)

    for group_idx in iter_group_indices(meta_df["group_key"]):
        group_values = target[group_idx]
        head_value = float(group_values[0])
        tail_value = float(group_values[-1])
        amplitude = max(head_value - tail_value, 1e-8)
        normalized = (group_values - tail_value) / amplitude
        normalized = np.clip(normalized, 0.0, 1.0)
        progress_list.append(compute_group_progress(len(group_idx)))
        normalized_list.append(normalized)

    template_progress = np.concatenate(progress_list)
    template_target = np.concatenate(normalized_list)

    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(template_progress, template_target)

    grid = np.linspace(0.0, 1.0, TEMPLATE_GRID_SIZE)
    template_values = iso.predict(grid)
    template_values = solve_monotone_smooth_projection(template_values, TEMPLATE_SMOOTH_LAMBDA)
    template_values = np.clip(template_values, 0.0, 1.0)

    start_value = float(template_values[0])
    end_value = float(template_values[-1])
    if start_value - end_value > 1e-8:
        template_values = (template_values - end_value) / (start_value - end_value)
    template_values = np.clip(template_values, 0.0, 1.0)
    template_values = np.minimum.accumulate(template_values)
    return grid, template_values


def evaluate_template_curve(progress: np.ndarray, template_grid: np.ndarray, template_values: np.ndarray, gamma: float) -> np.ndarray:
    """進捗を warp してテンプレート曲線を評価する。"""
    warped_progress = np.power(np.clip(progress, 0.0, 1.0), gamma)
    evaluated = np.interp(warped_progress, template_grid, template_values)
    return np.clip(evaluated, 0.0, 1.0)


def solve_nonnegative_template_affine(observed_values: np.ndarray, template_curve: np.ndarray) -> tuple[float, float]:
    """observed ≈ amplitude * template + tail を満たす非負係数を求める。"""
    observed = np.asarray(observed_values, dtype=float).reshape(-1)
    curve = np.asarray(template_curve, dtype=float).reshape(-1)
    design = np.column_stack([curve, np.ones_like(curve)])

    candidate_params: list[tuple[float, float]] = []
    coef = np.linalg.lstsq(design, observed, rcond=None)[0]
    candidate_params.append((float(coef[0]), float(coef[1])))

    if float(np.dot(curve, curve)) > 1e-8:
        candidate_params.append((max(float(np.dot(observed, curve) / np.dot(curve, curve)), 0.0), 0.0))
    candidate_params.append((0.0, max(float(np.mean(observed)), 0.0)))
    candidate_params.append((0.0, 0.0))

    best_amplitude = 0.0
    best_tail = 0.0
    best_loss = float("inf")
    for amplitude, tail in candidate_params:
        amplitude = max(float(amplitude), 0.0)
        tail = max(float(tail), 0.0)
        fitted = amplitude * curve + tail
        loss = float(np.mean((observed - fitted) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_amplitude = amplitude
            best_tail = tail
    return best_amplitude, best_tail


def apply_curve_template_projection(
    meta_df: pd.DataFrame,
    predictions: np.ndarray,
    template_grid: np.ndarray,
    template_values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """raw 予測を共通テンプレートへ射影する。"""
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    output = np.zeros_like(pred)

    for group_idx in iter_group_indices(meta_df["group_key"]):
        group_pred = np.clip(pred[group_idx], PREDICTION_LOWER_BOUND, None)
        progress = compute_group_progress(len(group_idx))
        curve = evaluate_template_curve(progress, template_grid, template_values, gamma)
        amplitude, tail = solve_nonnegative_template_affine(group_pred, curve)
        projected = amplitude * curve + tail
        projected = np.clip(projected, PREDICTION_LOWER_BOUND, None)
        projected = np.minimum.accumulate(projected)
        output[group_idx] = projected
    return output


def apply_final_postprocess(predictions: np.ndarray, group_keys: pd.Series, method_name: str) -> np.ndarray:
    """最終の単調化ルールを適用する。"""
    if method_name == "none":
        return np.clip(np.asarray(predictions, dtype=float).reshape(-1), PREDICTION_LOWER_BOUND, None)
    if method_name == "cummin":
        return apply_groupwise_cummin(predictions, group_keys)
    if method_name == "isotonic":
        return apply_groupwise_isotonic(predictions, group_keys)
    if method_name.startswith("smooth_"):
        smooth_lambda = float(method_name.split("_", 1)[1])
        return apply_groupwise_monotone_smooth(predictions, group_keys, smooth_lambda)
    raise ValueError(f"Unsupported final postprocess: {method_name}")


def search_best_curve_projection(
    train_raw_pred: np.ndarray,
    test_raw_pred: np.ndarray,
    y_true: np.ndarray,
    template_grid: np.ndarray,
    template_values: np.ndarray,
    train_meta_df: pd.DataFrame,
    test_meta_df: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """OOF 上で最良のテンプレート投影設定を探す。"""
    result_rows: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None
    candidate_methods = ["none", "cummin", "isotonic"] + [f"smooth_{v:g}" for v in SMOOTH_LAMBDA_CANDIDATES]

    for gamma in TEMPLATE_GAMMA_CANDIDATES:
        projected_train = apply_curve_template_projection(
            train_meta_df,
            predictions=train_raw_pred,
            template_grid=template_grid,
            template_values=template_values,
            gamma=gamma,
        )
        projected_test = apply_curve_template_projection(
            test_meta_df,
            predictions=test_raw_pred,
            template_grid=template_grid,
            template_values=template_values,
            gamma=gamma,
        )

        for template_blend_weight in TEMPLATE_BLEND_WEIGHT_CANDIDATES:
            blended_train_raw = (
                template_blend_weight * np.asarray(train_raw_pred, dtype=float)
                + (1.0 - template_blend_weight) * projected_train
            )
            blended_test_raw = (
                template_blend_weight * np.asarray(test_raw_pred, dtype=float)
                + (1.0 - template_blend_weight) * projected_test
            )
            blended_train_raw = np.clip(blended_train_raw, PREDICTION_LOWER_BOUND, None)
            blended_test_raw = np.clip(blended_test_raw, PREDICTION_LOWER_BOUND, None)

            for method_name in candidate_methods:
                final_train = apply_final_postprocess(blended_train_raw, train_meta_df["group_key"], method_name)
                final_test = apply_final_postprocess(blended_test_raw, test_meta_df["group_key"], method_name)
                train_rmse = compute_rmse(y_true, final_train)
                row = {
                    "gamma": float(gamma),
                    "template_blend_weight": float(template_blend_weight),
                    "method": method_name,
                    "train_rmse": float(train_rmse),
                    "train_violations": int(count_monotonic_violations(final_train, train_meta_df["group_key"])),
                    "test_violations": int(count_monotonic_violations(final_test, test_meta_df["group_key"])),
                }
                result_rows.append(row)
                if best_result is None or float(train_rmse) < float(best_result["train_rmse"]):
                    best_result = {
                        "gamma": float(gamma),
                        "template_blend_weight": float(template_blend_weight),
                        "method": method_name,
                        "train_rmse": float(train_rmse),
                        "projected_train": projected_train.copy(),
                        "projected_test": projected_test.copy(),
                        "train_pred": final_train.copy(),
                        "test_pred": final_test.copy(),
                    }

    if best_result is None:
        raise ValueError("curve projection candidates were empty")

    result_df = pd.DataFrame(result_rows).sort_values("train_rmse", ascending=True).reset_index(drop=True)
    return best_result, result_df


def make_prediction_plot_df(meta_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """描画しやすい形に予測結果を整える。"""
    return pd.DataFrame(
        {
            "species": meta_df["group_key"].astype(str).reset_index(drop=True),
            "sample_order_in_species": meta_df["sample_order_in_species"].to_numpy(),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )


def plot_raw_vs_final(meta_df: pd.DataFrame, raw_predictions: np.ndarray, final_predictions: np.ndarray, split_name: str, title_suffix: str) -> None:
    """補正前後の曲線を並べて比較する。"""
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
        alpha=0.85,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title(f"{split_name}: raw")
    axes[0].set_xlabel("樹種内サンプル番号")
    axes[0].set_ylabel("予測含水率")

    sns.lineplot(
        data=final_plot_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.0,
        alpha=0.85,
        ax=axes[1],
    )
    axes[1].set_title(f"{split_name}: {title_suffix}")
    axes[1].set_xlabel("樹種内サンプル番号")
    axes[1].set_ylabel("予測含水率")
    axes[1].legend(title="species", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_template_curve(template_grid: np.ndarray, template_values: np.ndarray) -> None:
    """グローバル減衰テンプレートを可視化する。"""
    plt.figure(figsize=(8, 5))
    plt.plot(template_grid, template_values, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("group progress")
    plt.ylabel("normalized moisture")
    plt.title("Global decay template")
    plt.tight_layout()
    plt.show()


# %%
train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
test_df = pd.read_csv(DATA_DIR / "test.csv", encoding="cp932")

species_col = infer_species_col(train_df)
if species_col is None:
    raise ValueError("樹種列が見つかりません。")

target_col = infer_target_col(train_df)
species_related_cols = {c for c in ["樹種", "species", "species number"] if c in train_df.columns}
exclude_cols = {ID_COL, target_col, "乾物率"} | species_related_cols
wavelength_cols = find_wavelength_cols(train_df, test_df, exclude_cols)
if not wavelength_cols:
    raise ValueError("波長列が見つかりません。")

train_meta_df = build_sequence_meta(train_df, species_col)
test_meta_df = build_sequence_meta(test_df, species_col)

X_train = train_df[wavelength_cols].copy()
X_test = test_df[wavelength_cols].copy()
y_train = train_df[target_col].copy()
y_train_array = y_train.to_numpy(dtype=float)
test_ids = test_df[ID_COL].copy()
groups = train_meta_df["group_key"]
cv_splits = list(GroupKFold(n_splits=CV).split(X_train, y_train, groups=groups))

print("train species:", sorted(train_meta_df["group_key"].unique()))
print("test species:", sorted(test_meta_df["group_key"].unique()))
print("n_wavelength_cols:", len(wavelength_cols))
print("n_train_groups:", int(groups.nunique()))


# %%
def objective(trial: optuna.Trial) -> float:
    """GroupKFold で base alpha を最適化する。"""
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    model = build_base_model(alpha)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
    )
    return float(-scores.mean())


# %%
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_alpha = float(study.best_params["alpha"])
base_cv_rmse = float(study.best_value)
print(f"Best base alpha={best_alpha:.6f}, GroupKFold RMSE={base_cv_rmse:.4f}")


# %%
train_pred_raw = np.zeros(len(X_train), dtype=float)
test_raw_sum = np.zeros(len(X_test), dtype=float)
fold_rows: list[dict[str, object]] = []

for fold_no, (fit_idx, valid_idx) in enumerate(cv_splits, start=1):
    model = build_base_model(best_alpha)
    model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

    valid_pred = np.clip(model.predict(X_train.iloc[valid_idx]), PREDICTION_LOWER_BOUND, None)
    test_pred = np.clip(model.predict(X_test), PREDICTION_LOWER_BOUND, None)

    train_pred_raw[valid_idx] = valid_pred
    test_raw_sum += test_pred

    valid_species = sorted(train_meta_df.iloc[valid_idx]["group_key"].unique())
    fold_rows.append(
        {
            "fold": fold_no,
            "valid_size": int(len(valid_idx)),
            "valid_species": ", ".join(valid_species),
            "raw_valid_rmse": compute_rmse(y_train_array[valid_idx], valid_pred),
            "raw_valid_violations": count_monotonic_violations(valid_pred, train_meta_df.iloc[valid_idx]["group_key"]),
        }
    )
    print(
        f"Fold {fold_no}: species={valid_species}, "
        f"raw_valid_rmse={float(fold_rows[-1]['raw_valid_rmse']):.4f}"
    )

test_pred_raw = test_raw_sum / len(cv_splits)
fold_summary_df = pd.DataFrame(fold_rows)
print(fold_summary_df)


# %%
teacher_target = apply_groupwise_monotone_target(
    y_train_array,
    group_keys=train_meta_df["group_key"],
    method=REFINE_TARGET_METHOD,
)
template_grid, template_values = build_global_decay_template(train_meta_df, teacher_target)
best_projection_result, projection_search_df = search_best_curve_projection(
    train_raw_pred=train_pred_raw,
    test_raw_pred=test_pred_raw,
    y_true=y_train_array,
    template_grid=template_grid,
    template_values=template_values,
    train_meta_df=train_meta_df,
    test_meta_df=test_meta_df,
)

train_pred_final = np.asarray(best_projection_result["train_pred"], dtype=float)
test_pred_final = np.asarray(best_projection_result["test_pred"], dtype=float)
train_pred_template = np.asarray(best_projection_result["projected_train"], dtype=float)
test_pred_template = np.asarray(best_projection_result["projected_test"], dtype=float)
selected_gamma = float(best_projection_result["gamma"])
selected_template_blend_weight = float(best_projection_result["template_blend_weight"])
selected_method = str(best_projection_result["method"])

train_raw_rmse = compute_rmse(y_train_array, train_pred_raw)
train_template_rmse = compute_rmse(y_train_array, train_pred_template)
train_final_rmse = compute_rmse(y_train_array, train_pred_final)
print(
    f"Train RMSE raw={train_raw_rmse:.4f}, "
    f"template={train_template_rmse:.4f}, "
    f"final={train_final_rmse:.4f}"
)
print(
    f"Selected gamma={selected_gamma:g}, "
    f"template_blend_weight={selected_template_blend_weight:g}, "
    f"method={selected_method}"
)
print(
    f"Violations raw={count_monotonic_violations(train_pred_raw, train_meta_df['group_key'])}, "
    f"template={count_monotonic_violations(train_pred_template, train_meta_df['group_key'])}, "
    f"final={count_monotonic_violations(train_pred_final, train_meta_df['group_key'])}"
)


# %%
submit_df = build_submit_df(test_ids, test_pred_final)
rmse_tag = int(round(train_final_rmse * 10000))

submit_path = OUTPUT_DIR / f"submit_csv_13_groupkfold_curve_template_{DATE}_{rmse_tag:04d}.csv"
submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
print(f"saved: {submit_path}")

detail_df = pd.DataFrame(
    {
        "id": test_ids,
        "species": test_meta_df["group_key"],
        "sample_order_in_species": test_meta_df["sample_order_in_species"],
        "raw_prediction": test_pred_raw,
        "template_prediction": test_pred_template,
        "final_prediction": test_pred_final,
        "selected_gamma": selected_gamma,
        "selected_template_blend_weight": selected_template_blend_weight,
        "selected_method": selected_method,
    }
)
detail_path = OUTPUT_DIR / f"test_prediction_detail_13_groupkfold_curve_template_{DATE}.csv"
detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

fold_summary_path = OUTPUT_DIR / f"fold_summary_13_groupkfold_curve_template_{DATE}.csv"
fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

projection_search_path = OUTPUT_DIR / f"projection_search_13_groupkfold_curve_template_{DATE}.csv"
projection_search_df.to_csv(projection_search_path, index=False, encoding="utf-8-sig")

submit_df.head()


# %%
plot_template_curve(template_grid, template_values)
plot_raw_vs_final(train_meta_df, train_pred_raw, train_pred_final, "Train", f"curve-template final ({selected_method})")
plot_raw_vs_final(test_meta_df, test_pred_raw, test_pred_final, "Test", f"curve-template final ({selected_method})")


# %%
test_pred_final

# %%
