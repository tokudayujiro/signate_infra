# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import optuna
import japanize_matplotlib

# Ridge回帰の予測傾向を可視化する

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
SPECIES_COL = "樹種"
RANDOM_STATE = 42
N_TRIALS = 20
CV = 5

# %%
# 学習データを読み込み、01_train_predict.py と同じ特徴量を作る
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")

exclude = [ID_COL, "species number", TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
if not feature_cols:
    feature_cols = [c for c in train_df.columns if c not in exclude]

X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].copy()
X_train = X_train.fillna(X_train.median())
cv_splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)

# %%
# 01_train_predict.py と同じ条件で alpha を探す
def objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=RANDOM_STATE))
    scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring="neg_root_mean_squared_error")
    return float(-scores.mean())


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_alpha = study.best_params["alpha"]
print(f"Best alpha={best_alpha}, CV RMSE={study.best_value:.4f}")

# %%
# 同じ alpha で、学習データに対するCV予測を作る
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, random_state=RANDOM_STATE))
cv_pred = cross_val_predict(model, X_train, y_train, cv=cv_splitter)
residuals = y_train - cv_pred
cv_rmse = np.sqrt(mean_squared_error(y_train, cv_pred))
cv_r2 = r2_score(y_train, cv_pred)

eval_df = pd.DataFrame(
    {
        "true": y_train,
        "pred": cv_pred,
        "residual": residuals,
    }
)
if SPECIES_COL in train_df.columns:
    eval_df[SPECIES_COL] = train_df[SPECIES_COL].fillna("不明")

# %%
# 正解値の分布、正解と予測の比較、残差を並べて見る
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(y_train, bins=30, color="steelblue", edgecolor="black")
axes[0].set_title("True Target Distribution")
axes[0].set_xlabel(TARGET_COL)
axes[0].set_ylabel("count")

line_min = min(y_train.min(), cv_pred.min())
line_max = max(y_train.max(), cv_pred.max())
if SPECIES_COL in eval_df.columns:
    for species_name, group in eval_df.groupby(SPECIES_COL):
        axes[1].scatter(group["true"], group["pred"], alpha=0.6, label=species_name)
    axes[1].legend(title=SPECIES_COL, bbox_to_anchor=(1.02, 1), loc="upper left")
else:
    axes[1].scatter(y_train, cv_pred, alpha=0.6, color="slateblue")
axes[1].plot([line_min, line_max], [line_min, line_max], "--", color="tomato")
axes[1].set_title(f"True vs CV Pred\nRMSE={cv_rmse:.4f}, R2={cv_r2:.4f}")
axes[1].set_xlabel("True")
axes[1].set_ylabel("Predicted")

axes[2].scatter(cv_pred, residuals, alpha=0.6, color="darkorange")
axes[2].axhline(0, linestyle="--", color="black")
axes[2].set_title("Residuals")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True - Predicted")

plt.tight_layout()
plt.show()

# %%
# 樹種ごとの精度とズレをまとめる
if SPECIES_COL in eval_df.columns:
    species_rows = []
    for species_name, group in eval_df.groupby(SPECIES_COL):
        species_rows.append(
            {
                SPECIES_COL: species_name,
                "count": len(group),
                "rmse": np.sqrt(mean_squared_error(group["true"], group["pred"])),
                "mean_residual": group["residual"].mean(),
            }
        )
    species_summary = pd.DataFrame(species_rows).sort_values("rmse")
    print(species_summary)

    plt.figure(figsize=(10, 4))
    eval_df.boxplot(column="residual", by=SPECIES_COL, rot=45)
    plt.suptitle("")
    plt.title("Residuals by Species")
    plt.xlabel(SPECIES_COL)
    plt.ylabel("True - Predicted")
    plt.tight_layout()
    plt.show()

# %%
# モデル全体を学習し、係数を元のスケールで確認する
model.fit(X_train, y_train)
scaler = model.named_steps["standardscaler"]
ridge = model.named_steps["ridge"]
coef_original = ridge.coef_ / scaler.scale_
intercept_original = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)

coef_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "coef": coef_original,
        "abs_coef": np.abs(coef_original),
    }
).sort_values("abs_coef", ascending=False)

top_n = min(20, len(coef_df))
plot_df = coef_df.head(top_n).sort_values("coef")

plt.figure(figsize=(8, max(4, top_n * 0.35)))
plt.barh(plot_df["feature"], plot_df["coef"], color="teal")
plt.title(f"Top {top_n} Coefficients")
plt.xlabel("coefficient")
plt.tight_layout()
plt.show()

print(coef_df.head(10)[["feature", "coef"]])

top_terms = [
    f"({row.coef:.4f}) * {row.feature}"
    for row in coef_df.head(10).itertuples(index=False)
]
formula_text = f"{TARGET_COL} = {intercept_original:.4f} + " + " + ".join(top_terms)
print("式の見方:")
print(formula_text)
print("※ 特徴量が多いので、式は寄与の大きい上位10項だけ表示しています。")

# %%
