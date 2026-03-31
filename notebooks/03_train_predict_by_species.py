# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import optuna
import japanize_matplotlib

# 樹種ごとに Ridge 回帰を学習し、提出用CSVを作る

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
SPECIES_COL = "樹種"
SUBMIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "submit_by_species.csv")
RANDOM_STATE = 42
N_TRIALS = 20
CV = 5
MIN_SAMPLES_PER_SPECIES = 10


def build_model(alpha: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha, random_state=RANDOM_STATE),
    )


def tune_alpha(X: pd.DataFrame, y: pd.Series, n_trials: int, cv: int) -> float:
    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
        model = build_model(alpha)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return float(study.best_params["alpha"])


# %%
# 学習データとテストデータを読み込む
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")

# 01_train_predict.py と同じ特徴量を使う
exclude = [ID_COL, "species number", TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
if not feature_cols:
    feature_cols = [c for c in train_df.columns if c not in exclude]

X_train = train_df[feature_cols].copy()
y_train = train_df[TARGET_COL].copy()
X_test = test_df[feature_cols].copy()
test_ids = test_df[ID_COL].copy()

train_species = train_df[SPECIES_COL].fillna("不明")
test_species = test_df[SPECIES_COL].fillna("不明")
global_cv_splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)

# %%
# まず全体モデルを基準として作る
global_alpha = tune_alpha(X_train, y_train, n_trials=N_TRIALS, cv=global_cv_splitter)
global_model = build_model(global_alpha)
global_oof_pred = cross_val_predict(global_model, X_train, y_train, cv=global_cv_splitter)
global_rmse = np.sqrt(mean_squared_error(y_train, global_oof_pred))

global_model.fit(X_train, y_train)
global_test_pred = global_model.predict(X_test)

print(f"Global best alpha={global_alpha:.6f}, OOF RMSE={global_rmse:.4f}")

# %%
# 樹種ごとに別モデルを作り、OOF予測とテスト予測を作る
species_oof_pred = pd.Series(index=train_df.index, dtype=float)
species_test_pred = pd.Series(index=test_df.index, dtype=float)
species_logs = []

for species_name in sorted(train_species.unique()):
    train_mask = train_species == species_name
    test_mask = test_species == species_name

    X_train_species = X_train.loc[train_mask]
    y_train_species = y_train.loc[train_mask]
    species_count = int(train_mask.sum())
    local_cv = min(CV, species_count)

    if species_count < MIN_SAMPLES_PER_SPECIES or local_cv < 2:
        species_oof_pred.loc[train_mask] = global_oof_pred[train_mask]
        if test_mask.any():
            species_test_pred.loc[test_mask] = global_test_pred[test_mask]
        species_logs.append(
            {
                SPECIES_COL: species_name,
                "train_count": species_count,
                "alpha": global_alpha,
                "mode": "global_fallback",
            }
        )
        continue

    local_cv_splitter = KFold(n_splits=local_cv, shuffle=True, random_state=RANDOM_STATE)
    local_alpha = tune_alpha(X_train_species, y_train_species, n_trials=N_TRIALS, cv=local_cv_splitter)
    local_model = build_model(local_alpha)
    local_oof_pred = cross_val_predict(local_model, X_train_species, y_train_species, cv=local_cv_splitter)
    species_oof_pred.loc[train_mask] = local_oof_pred

    local_model.fit(X_train_species, y_train_species)
    if test_mask.any():
        species_test_pred.loc[test_mask] = local_model.predict(X_test.loc[test_mask])

    local_rmse = np.sqrt(mean_squared_error(y_train_species, local_oof_pred))
    species_logs.append(
        {
            SPECIES_COL: species_name,
            "train_count": species_count,
            "alpha": local_alpha,
            "mode": "species_model",
            "oof_rmse": local_rmse,
        }
    )

# %%
# train にない樹種が test にあれば、全体モデルで補完する
unknown_test_mask = species_test_pred.isna()
if unknown_test_mask.any():
    species_test_pred.loc[unknown_test_mask] = global_test_pred[unknown_test_mask]

species_rmse = np.sqrt(mean_squared_error(y_train, species_oof_pred))
print(f"Species model OOF RMSE={species_rmse:.4f}")
print(f"RMSE improvement={global_rmse - species_rmse:.4f}")

comparison_df = pd.DataFrame(
    {
        SPECIES_COL: train_species,
        "true": y_train,
        "global_pred": global_oof_pred,
        "species_pred": species_oof_pred,
    }
)

species_metric_rows = []
for species_name, group in comparison_df.groupby(SPECIES_COL):
    global_species_rmse = np.sqrt(mean_squared_error(group["true"], group["global_pred"]))
    species_species_rmse = np.sqrt(mean_squared_error(group["true"], group["species_pred"]))
    species_metric_rows.append(
        {
            SPECIES_COL: species_name,
            "global_oof_rmse": global_species_rmse,
            "species_oof_rmse": species_species_rmse,
            "rmse_improvement": global_species_rmse - species_species_rmse,
        }
    )

species_metric_df = pd.DataFrame(species_metric_rows)
species_log_df = pd.DataFrame(species_logs).merge(species_metric_df, on=SPECIES_COL, how="left")
species_log_df = species_log_df.sort_values("rmse_improvement", ascending=False)
print(species_log_df)

# %%
# 全体比較と樹種ごとの差を可視化する
line_min = min(y_train.min(), global_oof_pred.min(), species_oof_pred.min())
line_max = max(y_train.max(), global_oof_pred.max(), species_oof_pred.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(["global", "by_species"], [global_rmse, species_rmse], color=["slateblue", "darkorange"])
axes[0].set_title("Overall OOF RMSE")
axes[0].set_ylabel("RMSE")

axes[1].scatter(y_train, global_oof_pred, alpha=0.5, color="slateblue")
axes[1].plot([line_min, line_max], [line_min, line_max], "--", color="tomato")
axes[1].set_title(f"Global Model\nRMSE={global_rmse:.4f}")
axes[1].set_xlabel("True")
axes[1].set_ylabel("Predicted")

axes[2].scatter(y_train, species_oof_pred, alpha=0.5, color="darkorange")
axes[2].plot([line_min, line_max], [line_min, line_max], "--", color="tomato")
axes[2].set_title(f"By Species Model\nRMSE={species_rmse:.4f}")
axes[2].set_xlabel("True")
axes[2].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

# %%
plot_df = species_log_df.sort_values("rmse_improvement")
y_pos = np.arange(len(plot_df))

plt.figure(figsize=(10, max(4, len(plot_df) * 0.45)))
plt.barh(y_pos - 0.2, plot_df["global_oof_rmse"], height=0.4, label="global")
plt.barh(y_pos + 0.2, plot_df["species_oof_rmse"], height=0.4, label="by_species")
plt.yticks(y_pos, plot_df[SPECIES_COL])
plt.xlabel("OOF RMSE")
plt.title("OOF RMSE by Species")
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, max(4, len(plot_df) * 0.45)))
colors = ["seagreen" if x > 0 else "tomato" for x in plot_df["rmse_improvement"]]
plt.barh(plot_df[SPECIES_COL], plot_df["rmse_improvement"], color=colors)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("global RMSE - by_species RMSE")
plt.title("Positive Means Species Split Improved")
plt.tight_layout()
plt.show()

# %%
# 提出形式に合わせて保存する
submit_df = pd.DataFrame({"id": test_ids, "value": species_test_pred})
submit_df.to_csv(SUBMIT_PATH, index=False, header=False, encoding="utf-8-sig")
submit_df.head()

# %%
