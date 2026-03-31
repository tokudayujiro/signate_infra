# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
import optuna
from datetime import datetime, timedelta, timezone

# Ridge回帰の予測値に樹種内の単調減少制約を加えて提出用CSVを作る

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_ridge_monotonic"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 100
CV = 6
POSTPROCESS_METHOD = "auto"
POSTPROCESS_CANDIDATES = ("cummin", "isotonic")
POSTPROCESS_CLIP_LOWER = 0.0

# %%
# 学習データとテストデータを読み込む
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")

# 樹種列を判定（前処理設定で利用）
species_col_candidates = ["樹種", "species", "species number"]
species_col = next((c for c in species_col_candidates if c in train_df.columns), None)
# %%
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

if not wavelength_cols:
    raise ValueError("波長列が見つかりません。列名が数値(例: 1100, 1101, ...)か確認してください。")

spectrum_long = train_df[[species_col] + wavelength_cols].melt(
    id_vars=species_col,
    var_name="wavelength",
    value_name="absorbance",
)
spectrum_long["wavelength"] = spectrum_long["wavelength"].astype(float)
spectrum_long = spectrum_long.sort_values("wavelength")

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=spectrum_long,
    x="wavelength",
    y="absorbance",
    hue=species_col,
    estimator="mean",
    errorbar=None,
)

# 代表的な近赤外吸収帯の注釈（軸の値を波数 cm^-1 とみなした場合の目安）
nir_band_annotations = [
    (4300, "セルロース/ヘミセルロース（C-H）"),
    (5200, "水（O-H）"),
    (5800, "C-H結合 1次倍音"),
    (6900, "水（O-H）1次倍音"),
    (8400, "セルロース/リグニン（C-H）2次倍音"),
]
ax = plt.gca()
y_min, y_max = spectrum_long["absorbance"].min(), spectrum_long["absorbance"].max()
y_text = y_min + (y_max - y_min) * 0.03
x_min, x_max = spectrum_long["wavelength"].min(), spectrum_long["wavelength"].max()

for x_band, label in nir_band_annotations:
    if x_min <= x_band <= x_max:
        ax.axvline(x=x_band, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.text(
            x_band + 15,
            y_text,
            str(int(x_band)),
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color="dimgray",
            clip_on=True,
        )

band_note = "主要吸収帯（目安, cm-1）\n" + "\n".join(
    [f"{int(x)}: {label}" for x, label in nir_band_annotations if x_min <= x <= x_max]
)
ax.text(
    0.02,
    0.02,
    band_note,
    transform=ax.transAxes,
    va="bottom",
    ha="left",
    fontsize=8.5,
    color="dimgray",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="lightgray"),
)

plt.xlabel("波長")
plt.ylabel("吸光度")
plt.title("樹種ごとの平均近赤外スペクトル（主要吸収帯の注釈付き）")
plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
# %%
# 樹種ごとの含水率の推移を可視化
# 横軸: 各樹種内での出現順（groupby + cumcount）
moisture_df = train_df[[species_col, TARGET_COL]].copy()
moisture_df["sample_order_in_species"] = moisture_df.groupby(species_col).cumcount() + 1

plt.figure(figsize=(12, 7))
sns.lineplot(
    data=moisture_df,
    x="sample_order_in_species",
    y=TARGET_COL,
    hue=species_col,
    marker="o",
    linewidth=1.5,
    alpha=0.9,
)
plt.xlabel("樹種内サンプル番号（groupby + cumcount）")
plt.ylabel("含水率")
plt.title("樹種ごとの含水率の推移")
plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
# %%
# 波長列（吸光度列）を判定
exclude_for_spectrum = {ID_COL, TARGET_COL, "乾物率"}
if species_col is not None:
    exclude_for_spectrum.add(species_col)
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

# 前処理（必要時のみ有効化）
if PREPROCESS_CLIP_NEGATIVE and wavelength_cols:
    train_df.loc[:, wavelength_cols] = train_df[wavelength_cols].clip(lower=0)
    shared_cols = [c for c in wavelength_cols if c in test_df.columns]
    test_df.loc[:, shared_cols] = test_df[shared_cols].clip(lower=0)

if PREPROCESS_DROP_SPECIES and species_col is not None:
    train_df = train_df[~train_df[species_col].isin(PREPROCESS_DROP_SPECIES)].copy()
# %%
train_df

# %%
# 目的変数やID列を除いて、モデルに使う列を選ぶ
exclude = [ID_COL, TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude]
if not feature_cols:
    feature_cols = [c for c in train_df.columns if c not in exclude]

# 学習用の入力と、予測対象の入力を作る
X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].copy()
X_test = test_df[feature_cols].copy()
test_ids = test_df[ID_COL].copy()
train_group_keys = train_df[species_col].fillna("NA").astype(str)
test_group_keys = test_df[species_col].fillna("NA").astype(str)
y_train_array = y_train.to_numpy(dtype=float)

# 数値・カテゴリ列を分ける（樹種列は数値でもカテゴリ扱いにする）
forced_categorical = {species_col} if species_col is not None else set()
numeric_cols = [c for c in feature_cols if np.issubdtype(X_train[c].dtype, np.number) and c not in forced_categorical]
categorical_cols = [c for c in feature_cols if c in forced_categorical or not np.issubdtype(X_train[c].dtype, np.number)]
cv_splits = list(KFold(n_splits=CV, shuffle=False).split(X_train, y_train))

# %%
def build_preprocess() -> ColumnTransformer:
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
    return preprocess


def build_model(alpha: float) -> Pipeline:
    return Pipeline([("preprocess", build_preprocess()), ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE))])


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def apply_monotonic_postprocess(
    predictions: np.ndarray,
    group_keys: pd.Series,
    method: str = "none",
    lower_bound: float | None = 0.0,
) -> np.ndarray:
    adjusted = np.asarray(predictions, dtype=float).reshape(-1).copy()
    if lower_bound is not None:
        adjusted = np.clip(adjusted, lower_bound, None)

    if method == "none":
        return adjusted

    if method not in {"cummin", "isotonic"}:
        raise ValueError(f"Unsupported postprocess method: {method}")

    groups = pd.Series(group_keys).reset_index(drop=True)
    output = adjusted.copy()
    for idx in groups.groupby(groups, sort=False).groups.values():
        group_idx = np.asarray(list(idx))
        group_pred = adjusted[group_idx]

        if method == "cummin":
            output[group_idx] = np.minimum.accumulate(group_pred)
            continue

        if len(group_idx) <= 1:
            output[group_idx] = group_pred
            continue

        order = np.arange(len(group_idx), dtype=float)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        output[group_idx] = iso.fit_transform(order, group_pred)

    return output


def evaluate_postprocess_methods(
    raw_predictions: np.ndarray,
    y_true: np.ndarray,
    group_keys: pd.Series,
    preferred_method: str = "auto",
    candidate_methods: tuple[str, ...] = POSTPROCESS_CANDIDATES,
) -> tuple[str, np.ndarray, dict[str, float]]:
    methods = list(dict.fromkeys(candidate_methods)) if preferred_method == "auto" else [preferred_method]
    if not methods:
        raise ValueError("candidate_methods must not be empty when POSTPROCESS_METHOD='auto'")

    scores: dict[str, float] = {}
    adjusted_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        adjusted = apply_monotonic_postprocess(
            raw_predictions,
            group_keys=group_keys,
            method=method,
            lower_bound=POSTPROCESS_CLIP_LOWER,
        )
        adjusted_by_method[method] = adjusted
        scores[method] = compute_rmse(y_true, adjusted)

    best_method = min(scores, key=scores.get)
    return best_method, adjusted_by_method[best_method], scores


def generate_oof_predictions(alpha: float) -> np.ndarray:
    oof_pred = np.zeros(len(X_train), dtype=float)
    for fit_idx, valid_idx in cv_splits:
        fold_model = build_model(alpha)
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_pred[valid_idx] = fold_model.predict(X_train.iloc[valid_idx])
    return oof_pred


def count_monotonic_violations(predictions: np.ndarray, group_keys: pd.Series) -> int:
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    groups = pd.Series(group_keys).reset_index(drop=True)
    violation_count = 0
    for idx in groups.groupby(groups, sort=False).groups.values():
        group_pred = pred[np.asarray(list(idx))]
        violation_count += int((group_pred[1:] > group_pred[:-1]).sum())
    return violation_count


def make_prediction_plot_df(species_series: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    plot_df = pd.DataFrame(
        {
            "species": species_series.fillna("NA").astype(str).reset_index(drop=True),
            "predicted_moisture": np.asarray(predictions, dtype=float).reshape(-1),
        }
    )
    plot_df["sample_order_in_species"] = plot_df.groupby("species", sort=False).cumcount() + 1
    return plot_df


def plot_count_by_species(species_series: pd.Series, title: str) -> None:
    species_values = species_series.fillna("NA").astype(str)
    species_order = species_values.value_counts().index
    plt.figure(figsize=(12, 5))
    sns.countplot(x=species_values, order=species_order)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("species")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_prediction_box_by_species(species_series: pd.Series, predictions: np.ndarray, title: str) -> None:
    plot_df = make_prediction_plot_df(species_series, predictions)
    plt.figure(figsize=(12, 5))
    order = plot_df.groupby("species")["predicted_moisture"].median().sort_values().index
    sns.boxplot(data=plot_df, x="species", y="predicted_moisture", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_prediction_trend_by_species(species_series: pd.Series, predictions: np.ndarray, title: str) -> None:
    plot_df = make_prediction_plot_df(species_series, predictions)
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


def plot_raw_vs_adjusted_trend_by_species(
    species_series: pd.Series,
    raw_predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    split_name: str,
    method_name: str,
) -> None:
    raw_plot_df = make_prediction_plot_df(species_series, raw_predictions)
    adjusted_plot_df = make_prediction_plot_df(species_series, adjusted_predictions)
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
    axes[1].set_title(f"{split_name}: monotonic-adjusted predicted moisture ({method_name})")
    axes[1].set_xlabel("樹種内サンプル番号")
    axes[1].set_ylabel("予測含水率")
    axes[1].legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def objective(trial: optuna.Trial) -> float:
    # Ridgeの正則化係数alphaを調整し、後処理込みのOOF RMSEを最小化する
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    raw_oof_pred = generate_oof_predictions(alpha)
    raw_oof_rmse = compute_rmse(y_train_array, raw_oof_pred)
    selected_method, adjusted_oof_pred, score_by_method = evaluate_postprocess_methods(
        raw_predictions=raw_oof_pred,
        y_true=y_train_array,
        group_keys=train_group_keys,
        preferred_method=POSTPROCESS_METHOD,
        candidate_methods=POSTPROCESS_CANDIDATES,
    )

    trial.set_user_attr("raw_oof_rmse", raw_oof_rmse)
    trial.set_user_attr("selected_postprocess_method", selected_method)
    for method_name, method_score in score_by_method.items():
        trial.set_user_attr(f"oof_rmse_{method_name}", float(method_score))

    return float(score_by_method[selected_method])


# Optunaでalphaを探索し、MLflowへ記録する
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"ridge_monotonic_{DATE}") as run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_alpha = study.best_params["alpha"]
    raw_oof_pred = generate_oof_predictions(best_alpha)
    raw_oof_rmse = compute_rmse(y_train_array, raw_oof_pred)
    selected_postprocess_method, adjusted_oof_pred, oof_score_by_method = evaluate_postprocess_methods(
        raw_predictions=raw_oof_pred,
        y_true=y_train_array,
        group_keys=train_group_keys,
        preferred_method=POSTPROCESS_METHOD,
        candidate_methods=POSTPROCESS_CANDIDATES,
    )
    best_rmse = float(oof_score_by_method[selected_postprocess_method])
    print(
        f"Best alpha={best_alpha}, raw OOF RMSE={raw_oof_rmse:.4f}, "
        f"adjusted OOF RMSE={best_rmse:.4f}, method={selected_postprocess_method}"
    )
    print(
        "OOF RMSE by method:",
        {method_name: round(method_score, 4) for method_name, method_score in oof_score_by_method.items()},
    )

    # 最適なalphaで学習し、テストデータを予測する
    model = build_model(best_alpha)
    model.fit(X_train, y_train)
    train_pred_raw = model.predict(X_train)
    pred_raw = model.predict(X_test)
    train_pred = apply_monotonic_postprocess(
        train_pred_raw,
        group_keys=train_group_keys,
        method=selected_postprocess_method,
        lower_bound=POSTPROCESS_CLIP_LOWER,
    )
    pred = apply_monotonic_postprocess(
        pred_raw,
        group_keys=test_group_keys,
        method=selected_postprocess_method,
        lower_bound=POSTPROCESS_CLIP_LOWER,
    )
    train_violation_raw = count_monotonic_violations(train_pred_raw, train_group_keys)
    train_violation_adjusted = count_monotonic_violations(train_pred, train_group_keys)
    test_violation_raw = count_monotonic_violations(pred_raw, test_group_keys)
    test_violation_adjusted = count_monotonic_violations(pred, test_group_keys)
    print(
        f"Train violations: raw={train_violation_raw}, adjusted={train_violation_adjusted} | "
        f"Test violations: raw={test_violation_raw}, adjusted={test_violation_adjusted}"
    )

    # 提出形式に合わせて保存する
    submit_df = pd.DataFrame({"id": test_ids, "value": pred})
    rmse_tag = int(round(best_rmse * 10000))
    submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_monotonic_{DATE}_{rmse_tag:04d}.csv")
    submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
    print(f"saved: {submit_path}")

    # 実験再現のための記録
    mlflow.log_params(
        {
            "target_col": TARGET_COL,
            "id_col": ID_COL,
            "n_trials": N_TRIALS,
            "cv": CV,
            "random_state": RANDOM_STATE,
            "feature_count": len(feature_cols),
            "preprocess_clip_negative": PREPROCESS_CLIP_NEGATIVE,
            "preprocess_drop_species": ",".join(PREPROCESS_DROP_SPECIES),
            "best_alpha": best_alpha,
            "postprocess_method_requested": POSTPROCESS_METHOD,
            "postprocess_method_selected": selected_postprocess_method,
            "postprocess_candidates": ",".join(POSTPROCESS_CANDIDATES),
            "postprocess_clip_lower": POSTPROCESS_CLIP_LOWER,
        }
    )
    mlflow.log_metric("oof_rmse_raw", raw_oof_rmse)
    mlflow.log_metric("oof_rmse_selected", best_rmse)
    for method_name, method_score in oof_score_by_method.items():
        mlflow.log_metric(f"oof_rmse_{method_name}", float(method_score))
    mlflow.log_metric("train_monotonic_violations_raw", train_violation_raw)
    mlflow.log_metric("train_monotonic_violations_adjusted", train_violation_adjusted)
    mlflow.log_metric("test_monotonic_violations_raw", test_violation_raw)
    mlflow.log_metric("test_monotonic_violations_adjusted", test_violation_adjusted)
    mlflow.log_artifact(submit_path)
    mlflow.log_artifact(__file__)
    print(f"mlflow run_id: {run.info.run_id}")
submit_df.head()
# %%
# 予測結果の可視化（予測処理とは別セクション）
if species_col is not None:
    train_species_series = train_df[species_col].fillna("NA").astype(str)
    plot_count_by_species(train_species_series, "Train sample count by species")
    plot_prediction_box_by_species(
        train_species_series,
        train_pred,
        f"Train predicted moisture by species ({selected_postprocess_method})",
    )
    plot_prediction_trend_by_species(
        train_species_series,
        train_pred,
        f"学習データ: 樹種ごとの予測含水率の推移 ({selected_postprocess_method})",
    )
    plot_raw_vs_adjusted_trend_by_species(
        train_species_series,
        train_pred_raw,
        train_pred,
        "Train",
        selected_postprocess_method,
    )

    # テストデータ: 樹種列がある場合に可視化
    if species_col in test_df.columns:
        test_species_series = test_df[species_col].fillna("NA").astype(str)
        plot_count_by_species(test_species_series, "Test sample count by species")
        plot_prediction_box_by_species(
            test_species_series,
            pred,
            f"Test predicted moisture by species ({selected_postprocess_method})",
        )
        plot_prediction_trend_by_species(
            test_species_series,
            pred,
            f"テストデータ: 樹種ごとの予測含水率の推移 ({selected_postprocess_method})",
        )
        plot_raw_vs_adjusted_trend_by_species(
            test_species_series,
            pred_raw,
            pred,
            "Test",
            selected_postprocess_method,
        )

# %%
test_df
