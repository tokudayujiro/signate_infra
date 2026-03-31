# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import optuna
import japanize_matplotlib

# 数値特徴に樹種を追加した Ridge 回帰を比較し、提出用CSVを作る

# --- 設定（ここを編集） ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
SPECIES_COL = "樹種"
SUBMIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "submit_with_species_feature.csv")
RANDOM_STATE = 42
N_TRIALS = 20
CV = 5

# ベイスギを重めに見る例。不要なら {} にする
SPECIES_WEIGHT_MAP = {"ベイスギ": 1.5}


def build_model(alpha: float, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        )
    ]

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
        ]
    )


def make_sample_weight(species: pd.Series, weight_map: dict[str, float]) -> pd.Series | None:
    if not weight_map:
        return None
    return species.map(lambda x: weight_map.get(x, 1.0)).astype(float)


def evaluate_alpha(
    alpha: float,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    sample_weight: pd.Series | None = None,
) -> float:
    rmse_scores = []
    for train_idx, valid_idx in cv_splits:
        X_fit = X.iloc[train_idx]
        y_fit = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = build_model(alpha, numeric_cols, categorical_cols)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["ridge__sample_weight"] = sample_weight.iloc[train_idx].to_numpy()
        model.fit(X_fit, y_fit, **fit_kwargs)

        pred_valid = model.predict(X_valid)
        rmse_scores.append(np.sqrt(mean_squared_error(y_valid, pred_valid)))

    return float(np.mean(rmse_scores))


def tune_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    sample_weight: pd.Series | None = None,
) -> tuple[float, float]:
    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
        return evaluate_alpha(alpha, X, y, numeric_cols, categorical_cols, cv_splits, sample_weight)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return float(study.best_params["alpha"]), float(study.best_value)


def fit_oof_and_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    alpha: float,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    sample_weight: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    oof_pred = np.zeros(len(X_train), dtype=float)

    for train_idx, valid_idx in cv_splits:
        model = build_model(alpha, numeric_cols, categorical_cols)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["ridge__sample_weight"] = sample_weight.iloc[train_idx].to_numpy()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_kwargs)
        oof_pred[valid_idx] = model.predict(X_train.iloc[valid_idx])

    final_model = build_model(alpha, numeric_cols, categorical_cols)
    final_fit_kwargs = {}
    if sample_weight is not None:
        final_fit_kwargs["ridge__sample_weight"] = sample_weight.to_numpy()
    final_model.fit(X_train, y_train, **final_fit_kwargs)
    test_pred = final_model.predict(X_test)

    return oof_pred, test_pred


# %%
# 学習データとテストデータを読み込む
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")

# 01_train_predict.py と同じ数値特徴量を使う
exclude = [ID_COL, "species number", TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
numeric_feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
if not numeric_feature_cols:
    numeric_feature_cols = [c for c in train_df.columns if c not in exclude and c != SPECIES_COL]

train_species = train_df[SPECIES_COL].fillna("不明")
test_species = test_df[SPECIES_COL].fillna("不明")

X_train_numeric = train_df[numeric_feature_cols].copy()
X_test_numeric = test_df[numeric_feature_cols].copy()
X_train_with_species = train_df[numeric_feature_cols + [SPECIES_COL]].copy()
X_test_with_species = test_df[numeric_feature_cols + [SPECIES_COL]].copy()
y_train = train_df[TARGET_COL].copy()
test_ids = test_df[ID_COL].copy()

cv_splits = list(KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE).split(X_train_numeric, y_train))

# %%
# 3パターンを比較する
candidates = [
    {
        "name": "numeric_only",
        "X_train": X_train_numeric,
        "X_test": X_test_numeric,
        "numeric_cols": numeric_feature_cols,
        "categorical_cols": [],
        "sample_weight": None,
    },
    {
        "name": "species_feature",
        "X_train": X_train_with_species,
        "X_test": X_test_with_species,
        "numeric_cols": numeric_feature_cols,
        "categorical_cols": [SPECIES_COL],
        "sample_weight": None,
    },
    {
        "name": "species_feature_weighted",
        "X_train": X_train_with_species,
        "X_test": X_test_with_species,
        "numeric_cols": numeric_feature_cols,
        "categorical_cols": [SPECIES_COL],
        "sample_weight": make_sample_weight(train_species, SPECIES_WEIGHT_MAP),
    },
]

results = []
result_map = {}

for candidate in candidates:
    best_alpha, tuned_rmse = tune_alpha(
        candidate["X_train"],
        y_train,
        candidate["numeric_cols"],
        candidate["categorical_cols"],
        cv_splits,
        candidate["sample_weight"],
    )
    oof_pred, test_pred = fit_oof_and_test(
        candidate["X_train"],
        y_train,
        candidate["X_test"],
        candidate["numeric_cols"],
        candidate["categorical_cols"],
        best_alpha,
        cv_splits,
        candidate["sample_weight"],
    )
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_pred))

    result = {
        "name": candidate["name"],
        "alpha": best_alpha,
        "tuned_rmse": tuned_rmse,
        "oof_rmse": oof_rmse,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
    }
    results.append({"name": candidate["name"], "alpha": best_alpha, "oof_rmse": oof_rmse})
    result_map[candidate["name"]] = result

result_df = pd.DataFrame(results).sort_values("oof_rmse")
print(result_df)

# %%
# 樹種ごとの差も見る
species_metric_rows = []
for species_name, group in train_df.groupby(SPECIES_COL):
    row = {SPECIES_COL: species_name, "count": len(group)}
    group_idx = group.index
    for candidate_name in result_map:
        pred = result_map[candidate_name]["oof_pred"][group_idx]
        row[f"{candidate_name}_rmse"] = np.sqrt(mean_squared_error(y_train.iloc[group_idx], pred))
    species_metric_rows.append(row)

species_metric_df = pd.DataFrame(species_metric_rows)
species_metric_df["species_feature_improvement"] = (
    species_metric_df["numeric_only_rmse"] - species_metric_df["species_feature_rmse"]
)
species_metric_df["weighted_improvement"] = (
    species_metric_df["species_feature_rmse"] - species_metric_df["species_feature_weighted_rmse"]
)
print(species_metric_df.sort_values("weighted_improvement", ascending=False))

# %%
# ベイスギ重み付けが効いたか確認する
if SPECIES_WEIGHT_MAP:
    print("Weight map:", SPECIES_WEIGHT_MAP)
    target_species = list(SPECIES_WEIGHT_MAP.keys())
    print(
        species_metric_df.loc[
            species_metric_df[SPECIES_COL].isin(target_species),
            [
                SPECIES_COL,
                "numeric_only_rmse",
                "species_feature_rmse",
                "species_feature_weighted_rmse",
                "weighted_improvement",
            ],
        ]
    )

# %%
# 全体比較を可視化する
best_model_name = result_df.iloc[0]["name"]
best_result = result_map[best_model_name]
baseline_result = result_map["numeric_only"]

line_min = min(y_train.min(), baseline_result["oof_pred"].min(), best_result["oof_pred"].min())
line_max = max(y_train.max(), baseline_result["oof_pred"].max(), best_result["oof_pred"].max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(result_df["name"], result_df["oof_rmse"], color=["slateblue", "darkorange", "seagreen"])
axes[0].set_title("OOF RMSE Comparison")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=15)

axes[1].scatter(y_train, baseline_result["oof_pred"], alpha=0.5, color="slateblue")
axes[1].plot([line_min, line_max], [line_min, line_max], "--", color="tomato")
axes[1].set_title(f"numeric_only\nRMSE={baseline_result['oof_rmse']:.4f}")
axes[1].set_xlabel("True")
axes[1].set_ylabel("Predicted")

axes[2].scatter(y_train, best_result["oof_pred"], alpha=0.5, color="darkorange")
axes[2].plot([line_min, line_max], [line_min, line_max], "--", color="tomato")
axes[2].set_title(f"{best_model_name}\nRMSE={best_result['oof_rmse']:.4f}")
axes[2].set_xlabel("True")
axes[2].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

# %%
# 樹種ごとの改善量を可視化する
plot_df = species_metric_df.sort_values("species_feature_improvement")
plt.figure(figsize=(10, max(4, len(plot_df) * 0.45)))
colors = ["seagreen" if x > 0 else "tomato" for x in plot_df["species_feature_improvement"]]
plt.barh(plot_df[SPECIES_COL], plot_df["species_feature_improvement"], color=colors)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("numeric_only RMSE - species_feature RMSE")
plt.title("Positive Means Species Feature Improved")
plt.tight_layout()
plt.show()

# %%
plot_df = species_metric_df.sort_values("weighted_improvement")
plt.figure(figsize=(10, max(4, len(plot_df) * 0.45)))
colors = ["seagreen" if x > 0 else "tomato" for x in plot_df["weighted_improvement"]]
plt.barh(plot_df[SPECIES_COL], plot_df["weighted_improvement"], color=colors)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("species_feature RMSE - weighted RMSE")
plt.title("Positive Means Weighting Improved")
plt.tight_layout()
plt.show()

# %%
# OOFが最良のモデルで提出ファイルを作る
submit_df = pd.DataFrame({"id": test_ids, "value": best_result["test_pred"]})
submit_df.to_csv(SUBMIT_PATH, index=False, header=False, encoding="utf-8-sig")
print(f"Selected model: {best_model_name}")
print(f"Saved to: {SUBMIT_PATH}")
submit_df.head()

# %%
