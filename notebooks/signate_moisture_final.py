# %%
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# SIGNATE 木材含水率コンペ向け 完成版
# - 論文ベース: PLS を主軸、Ridge を比較対象として保持
# - 前処理: Raw / SNV / MSC / Savitzky-Golay 微分
# - 帯域: full と水分吸収帯を比較
# - CV: shuffle あり KFold
# - MLflow / Optuna 対応
# ============================================================

# --- 設定（必要に応じて編集） ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()
SUBMIT_DIR = DATA_DIR
MLRUNS_DIR = (BASE_DIR / ".." / "mlruns").resolve()

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_COL = "含水率"
ID_COL = "sample number"
JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
EXPERIMENT_NAME = "spectral_moisture_pls_optuna"
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES: list[str] = []
RANDOM_STATE = 42
N_TRIALS = 80
CV = 5
ENCODING = "cp932"

# SG filter 候補
SG_WINDOW_CANDIDATES = [9, 11, 15, 21]
SG_POLYORDER = 2
MAX_PLS_COMPONENTS = 40
MIN_BAND_COLS = 30


# ============================================================
# Utility
# ============================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def neg_rmse_scorer():
    return make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )


def parse_wavelength(col_name: str) -> float | None:
    try:
        val = float(str(col_name))
    except (TypeError, ValueError):
        return None
    if 300 <= val <= 3000:
        return val
    return None


def infer_species_col(df: pd.DataFrame) -> str | None:
    candidates = ["樹種", "species", "species number"]
    return next((c for c in candidates if c in df.columns), None)


def find_wavelength_cols(train_df: pd.DataFrame, test_df: pd.DataFrame, species_col: str | None) -> list[str]:
    exclude = {ID_COL, TARGET_COL, "乾物率"}
    if species_col is not None:
        exclude.add(species_col)

    cols: list[str] = []
    for col in train_df.columns:
        if col in exclude or col not in test_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        wl = parse_wavelength(col)
        if wl is not None:
            cols.append(col)

    # SG 微分の前提なので必ず波長順でソート
    cols = sorted(cols, key=lambda x: float(x))
    return cols


def fallback_numeric_feature_cols(train_df: pd.DataFrame, test_df: pd.DataFrame, species_col: str | None) -> list[str]:
    exclude = {ID_COL, TARGET_COL, "乾物率"}
    if species_col is not None:
        exclude.add(species_col)

    cols = [
        c
        for c in train_df.columns
        if c in test_df.columns and c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    return cols


def build_band_map(spectral_cols: list[str]) -> tuple[dict[str, tuple[float, float] | None], dict[str, float]]:
    wavelength_map = {c: float(c) for c in spectral_cols if parse_wavelength(c) is not None}

    band_map: dict[str, tuple[float, float] | None] = {"full": None}
    candidate_bands = {
        "water_1400_1470": (1400.0, 1470.0),
        "water_1800_2050": (1800.0, 2050.0),
        "water_1400_2050": (1400.0, 2050.0),
        "water_1400_1900": (1400.0, 1900.0),
    }

    for name, (lo, hi) in candidate_bands.items():
        band_cols = [c for c in spectral_cols if c in wavelength_map and lo <= wavelength_map[c] <= hi]
        if len(band_cols) >= MIN_BAND_COLS:
            band_map[name] = (lo, hi)

    return band_map, wavelength_map


def choose_band_cols(
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    band: tuple[float, float] | None,
) -> list[str]:
    if band is None:
        return spectral_cols
    lo, hi = band
    band_cols = [c for c in spectral_cols if c in wavelength_map and lo <= wavelength_map[c] <= hi]
    if len(band_cols) < MIN_BAND_COLS:
        return spectral_cols
    return band_cols


# ============================================================
# Custom Transformers
# ============================================================
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.cols].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


class SNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, keepdims=True)
        row_std = np.where(row_std < 1e-12, 1.0, row_std)
        return (X - row_mean) / row_std


class MSCTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ref = self.reference_
        X_corr = np.zeros_like(X)

        for i in range(X.shape[0]):
            slope, intercept = np.polyfit(ref, X[i], deg=1)
            if abs(slope) < 1e-12:
                slope = 1.0
            X_corr[i] = (X[i] - intercept) / slope
        return X_corr


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_length: int = 15, polyorder: int = 2, deriv: int = 1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features < 3:
            return X

        window = min(self.window_length, n_features if n_features % 2 == 1 else n_features - 1)
        if window <= self.polyorder or window < 3:
            return X

        return savgol_filter(
            X,
            window_length=window,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


# ============================================================
# Model Builder
# ============================================================
def build_pipeline(
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    params: dict,
) -> tuple[Pipeline, list[str]]:
    band = BAND_MAP[params["band_name"]]
    cols = choose_band_cols(spectral_cols, wavelength_map, band)

    steps: list[tuple[str, object]] = [
        ("select", ColumnSelector(cols)),
        ("impute", SimpleImputer(strategy="median")),
    ]

    preproc = params["preproc"]
    if preproc == "snv":
        steps.append(("snv", SNVTransformer()))
    elif preproc == "msc":
        steps.append(("msc", MSCTransformer()))

    deriv = params["deriv"]
    if deriv > 0:
        steps.append(
            (
                "sg",
                SavitzkyGolayTransformer(
                    window_length=params["sg_window"],
                    polyorder=SG_POLYORDER,
                    deriv=deriv,
                ),
            )
        )

    model_type = params["model_type"]
    if model_type == "pls":
        n_components = max(1, min(int(params["n_components"]), len(cols) - 1))
        steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("model", PLSRegression(n_components=n_components, scale=False)))
    elif model_type == "ridge":
        steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("model", Ridge(alpha=float(params["alpha"]), random_state=RANDOM_STATE)))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps), cols


# ============================================================
# Optuna Objective
# ============================================================
def suggest_params(trial: optuna.Trial) -> dict:
    model_type = trial.suggest_categorical("model_type", ["pls", "ridge"])
    preproc = trial.suggest_categorical("preproc", ["none", "snv", "msc"])
    deriv = trial.suggest_categorical("deriv", [0, 1, 2])
    band_name = trial.suggest_categorical("band_name", list(BAND_MAP.keys()))

    params: dict = {
        "model_type": model_type,
        "preproc": preproc,
        "deriv": deriv,
        "band_name": band_name,
        "sg_window": 0,
    }

    if deriv > 0:
        params["sg_window"] = trial.suggest_categorical("sg_window", SG_WINDOW_CANDIDATES)

    if model_type == "pls":
        selected_cols = choose_band_cols(SPECTRAL_COLS, WAVELENGTH_MAP, BAND_MAP[band_name])
        upper = max(2, min(MAX_PLS_COMPONENTS, len(selected_cols) - 1, len(TRAIN_DF) - (len(TRAIN_DF) // CV + 1)))
        params["n_components"] = trial.suggest_int("n_components", 2, upper)
        params["alpha"] = 0.0
    else:
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 1e4, log=True)
        params["n_components"] = 0

    return params


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    model, used_cols = build_pipeline(SPECTRAL_COLS, WAVELENGTH_MAP, params)

    cv_splitter = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model,
        TRAIN_DF,
        Y_TRAIN,
        cv=cv_splitter,
        scoring=neg_rmse_scorer(),
        error_score="raise",
    )

    score = float(-scores.mean())
    trial.set_user_attr("n_features", len(used_cols))
    trial.set_user_attr("params_json", json.dumps(params, ensure_ascii=False))
    return score


# ============================================================
# Main
# ============================================================
TRAIN_DF: pd.DataFrame
TEST_DF: pd.DataFrame
Y_TRAIN: np.ndarray
SPECTRAL_COLS: list[str]
BAND_MAP: dict[str, tuple[float, float] | None]
WAVELENGTH_MAP: dict[str, float]


def main() -> None:
    global TRAIN_DF, TEST_DF, Y_TRAIN, SPECTRAL_COLS, BAND_MAP, WAVELENGTH_MAP

    TRAIN_DF = pd.read_csv(TRAIN_PATH, encoding=ENCODING)
    TEST_DF = pd.read_csv(TEST_PATH, encoding=ENCODING)

    species_col = infer_species_col(TRAIN_DF)

    if PREPROCESS_DROP_SPECIES and species_col is not None:
        TRAIN_DF = TRAIN_DF[~TRAIN_DF[species_col].isin(PREPROCESS_DROP_SPECIES)].copy()

    SPECTRAL_COLS = find_wavelength_cols(TRAIN_DF, TEST_DF, species_col)
    if not SPECTRAL_COLS:
        SPECTRAL_COLS = fallback_numeric_feature_cols(TRAIN_DF, TEST_DF, species_col)
        if not SPECTRAL_COLS:
            raise ValueError("スペクトル列または数値特徴列を特定できませんでした。")

    if PREPROCESS_CLIP_NEGATIVE:
        TRAIN_DF.loc[:, SPECTRAL_COLS] = TRAIN_DF[SPECTRAL_COLS].clip(lower=0)
        TEST_DF.loc[:, SPECTRAL_COLS] = TEST_DF[SPECTRAL_COLS].clip(lower=0)

    BAND_MAP, WAVELENGTH_MAP = build_band_map(SPECTRAL_COLS)
    Y_TRAIN = TRAIN_DF[TARGET_COL].to_numpy(dtype=float)

    print(f"train shape: {TRAIN_DF.shape}")
    print(f"test shape : {TEST_DF.shape}")
    print(f"target_col : {TARGET_COL}")
    print(f"id_col     : {ID_COL}")
    print(f"species_col: {species_col}")
    print(f"n_features : {len(SPECTRAL_COLS)}")
    print(f"band_names : {list(BAND_MAP.keys())}")

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"pls_optuna_{DATE}") as run:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best_params = suggest_params_from_best_trial(study.best_trial)
        best_rmse = float(study.best_value)
        print("best params:", best_params)
        print(f"best cv rmse: {best_rmse:.6f}")

        best_model, used_cols = build_pipeline(SPECTRAL_COLS, WAVELENGTH_MAP, best_params)
        best_model.fit(TRAIN_DF, Y_TRAIN)
        pred = best_model.predict(TEST_DF)
        pred = np.asarray(pred).reshape(-1)
        pred = np.clip(pred, 0, None)

        test_ids = TEST_DF[ID_COL].copy()
        submit_df = pd.DataFrame({"id": test_ids, "value": pred})
        rmse_tag = int(round(best_rmse * 10000))
        submit_path = SUBMIT_DIR / f"submit_csv_{DATE}_{rmse_tag:04d}.csv"
        submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")
        print(f"saved: {submit_path}")

        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        trials_csv_path = SUBMIT_DIR / f"optuna_trials_{DATE}.csv"
        trials_df.to_csv(trials_csv_path, index=False, encoding="utf-8-sig")

        best_json_path = SUBMIT_DIR / f"best_params_{DATE}.json"
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_params": best_params,
                    "best_cv_rmse": best_rmse,
                    "n_features": len(used_cols),
                    "bands_available": list(BAND_MAP.keys()),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        mlflow.log_params(
            {
                "target_col": TARGET_COL,
                "id_col": ID_COL,
                "n_trials": N_TRIALS,
                "cv": CV,
                "random_state": RANDOM_STATE,
                "feature_count": len(SPECTRAL_COLS),
                "preprocess_clip_negative": PREPROCESS_CLIP_NEGATIVE,
                "preprocess_drop_species": ",".join(PREPROCESS_DROP_SPECIES),
                "best_model_type": best_params["model_type"],
                "best_preproc": best_params["preproc"],
                "best_deriv": best_params["deriv"],
                "best_band_name": best_params["band_name"],
                "best_sg_window": best_params.get("sg_window", 0),
                "best_n_components": best_params.get("n_components", 0),
                "best_alpha": best_params.get("alpha", 0.0),
            }
        )
        mlflow.log_metric("cv_rmse", best_rmse)
        mlflow.log_artifact(str(submit_path))
        mlflow.log_artifact(str(trials_csv_path))
        mlflow.log_artifact(str(best_json_path))
        mlflow.log_artifact(__file__)

        print(f"mlflow run_id: {run.info.run_id}")



def suggest_params_from_best_trial(best_trial: optuna.Trial) -> dict:
    params = {
        "model_type": best_trial.params["model_type"],
        "preproc": best_trial.params["preproc"],
        "deriv": int(best_trial.params["deriv"]),
        "band_name": best_trial.params["band_name"],
        "sg_window": int(best_trial.params.get("sg_window", 0)),
        "n_components": int(best_trial.params.get("n_components", 0)),
        "alpha": float(best_trial.params.get("alpha", 0.0)),
    }
    return params


if __name__ == "__main__":
    main()

# %%
