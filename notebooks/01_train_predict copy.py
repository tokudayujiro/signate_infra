# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
RANDOM_STATE = 42
N_TRIALS = 100
CV = 5

# %%
# 学習データとテストデータを読み込む
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")
# %%
train_df
# %%
# 樹種ごとに近赤外スペクトル（横軸: 波長、縦軸: 吸光度）を可視化
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
# 目的変数やID列を除いて、モデルに使う列を選ぶ
exclude = [ID_COL, "species number", TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
if not feature_cols:
    feature_cols = [c for c in train_df.columns if c not in exclude]

# 学習用の入力と、予測対象の入力を作る
X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].copy()
X_test = test_df[feature_cols].copy()
test_ids = test_df[ID_COL].copy()

# 欠損値は学習データの中央値で埋める
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())
# cv_splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)

# %%
def objective(trial: optuna.Trial) -> float:
    # Ridgeの正則化係数alphaを調整し、CVのRMSEを最小化する
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=RANDOM_STATE))
    # scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring="neg_root_mean_squared_error")
    scores = cross_val_score(model, X_train, y_train, cv=CV, scoring="neg_root_mean_squared_error")

    return float(-scores.mean())

# Optunaでalphaを探索する
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_alpha = study.best_params["alpha"]
print(f"Best alpha={best_alpha}, CV RMSE={study.best_value:.4f}")

# %%
# 最適なalphaで学習し、テストデータを予測する
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, random_state=RANDOM_STATE))
model.fit(X_train, y_train)
pred = model.predict(X_test)

# %%
# 提出形式に合わせて保存する
submit_df = pd.DataFrame({"id": test_ids, "value": pred})
rmse_tag = int(round(study.best_value * 10000))
submit_path = os.path.join(SUBMIT_DIR, f"submit_csv_{DATE}_{rmse_tag:04d}.csv")
submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
print(f"saved: {submit_path}")
submit_df.head()
# %%
test_df
# %%
