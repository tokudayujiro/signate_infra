# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import optuna
from datetime import datetime, timedelta, timezone

# Ridge回帰で含水率を予測し、提出用CSVを作る

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "spectral_moisture_ridge"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []
RANDOM_STATE = 42
N_TRIALS = 100
CV = 6

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

# 数値・カテゴリ列を分ける（樹種列は数値でもカテゴリ扱いにする）
forced_categorical = {species_col} if species_col is not None else set()
numeric_cols = [c for c in feature_cols if np.issubdtype(X_train[c].dtype, np.number) and c not in forced_categorical]
categorical_cols = [c for c in feature_cols if c in forced_categorical or not np.issubdtype(X_train[c].dtype, np.number)]
# cv_splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)

# %%
def objective(trial: optuna.Trial) -> float:
    # Ridgeの正則化係数alphaを調整し、CVのRMSEを最小化する
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
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
    model = Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE))])
    # scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring="neg_root_mean_squared_error")
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_root_mean_squared_error")

    return float(-scores.mean())

# Optunaでalphaを探索し、MLflowへ記録する
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"ridge_{DATE}") as run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_alpha = study.best_params["alpha"]
    best_rmse = float(study.best_value)
    print(f"Best alpha={best_alpha}, CV RMSE={best_rmse:.4f}")

    # 最適なalphaで学習し、テストデータを予測する
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
    model = Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=best_alpha, random_state=RANDOM_STATE))])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    pred = model.predict(X_test)

    # 提出形式に合わせて保存する
    submit_df = pd.DataFrame({"id": test_ids, "value": pred})
    rmse_tag = int(round(best_rmse * 10000))
    submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_{DATE}_{rmse_tag:04d}.csv")
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
        }
    )
    mlflow.log_metric("cv_rmse", best_rmse)
    mlflow.log_artifact(submit_path)
    mlflow.log_artifact(__file__)
    print(f"mlflow run_id: {run.info.run_id}")
submit_df.head()
# %%
# 予測結果の可視化（予測処理とは別セクション）
if species_col is not None:
    # 学習データ: 樹種ごとの件数（count）
    train_species_series = train_df[species_col].astype(str).fillna("NA")
    train_species_order = train_species_series.value_counts().index
    plt.figure(figsize=(12, 5))
    sns.countplot(x=train_species_series, order=train_species_order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Train sample count by species")
    plt.xlabel("species")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # 学習データ: 予測値の樹種別分布
    train_plot_df = pd.DataFrame(
        {
            "species": train_species_series,
            "predicted_moisture": train_pred,
        }
    )
    plt.figure(figsize=(12, 5))
    order_train = train_plot_df.groupby("species")["predicted_moisture"].median().sort_values().index
    sns.boxplot(data=train_plot_df, x="species", y="predicted_moisture", order=order_train)
    plt.xticks(rotation=45, ha="right")
    plt.title("Train predicted moisture by species")
    plt.tight_layout()
    plt.show()

    # 学習データ: 予測値の推移（groupby + cumcount）
    train_trend_df = train_plot_df.copy()
    train_trend_df["sample_order_in_species"] = train_trend_df.groupby("species").cumcount() + 1
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=train_trend_df,
        x="sample_order_in_species",
        y="predicted_moisture",
        hue="species",
        marker="o",
        linewidth=1.2,
        alpha=0.85,
    )
    plt.xlabel("樹種内サンプル番号（groupby + cumcount）")
    plt.ylabel("予測含水率")
    plt.title("学習データ: 樹種ごとの予測含水率の推移")
    plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # テストデータ: 樹種列がある場合に可視化
    if species_col in test_df.columns:
        test_species_series = test_df[species_col].astype(str).fillna("NA")
        test_species_order = test_species_series.value_counts().index

        plt.figure(figsize=(12, 5))
        sns.countplot(x=test_species_series, order=test_species_order)
        plt.xticks(rotation=45, ha="right")
        plt.title("Test sample count by species")
        plt.xlabel("species")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

        test_plot_df = pd.DataFrame(
            {
                "species": test_species_series,
                "predicted_moisture": pred,
            }
        )
        plt.figure(figsize=(12, 5))
        order_test = test_plot_df.groupby("species")["predicted_moisture"].median().sort_values().index
        sns.boxplot(data=test_plot_df, x="species", y="predicted_moisture", order=order_test)
        plt.xticks(rotation=45, ha="right")
        plt.title("Test predicted moisture by species")
        plt.tight_layout()
        plt.show()

        test_trend_df = test_plot_df.copy()
        test_trend_df["sample_order_in_species"] = test_trend_df.groupby("species").cumcount() + 1
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=test_trend_df,
            x="sample_order_in_species",
            y="predicted_moisture",
            hue="species",
            marker="o",
            linewidth=1.2,
            alpha=0.85,
        )
        plt.xlabel("樹種内サンプル番号（groupby + cumcount）")
        plt.ylabel("予測含水率")
        plt.title("テストデータ: 樹種ごとの予測含水率の推移")
        plt.legend(title=species_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

# %%
test_df
# %%
