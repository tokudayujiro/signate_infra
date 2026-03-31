# ============================================================
# SIGNATE 木材含水率コンペ向け
# 論文ベースの PLS 回帰ベースライン
# - 提供スペクトルのみ使用
# - 前処理候補: Raw / SNV / MSC / SG(1次/2次微分)
# - 波長帯候補: full / 1400-1900nm
# - CVで最良設定を選び、submission.csv を出力
# ============================================================
# %%
from __future__ import annotations

import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# =========================
# 設定
# =========================
DATA_DIR = Path("./data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SAMPLE_SUBMISSION_CANDIDATES = [
    DATA_DIR / "sample_submission.csv",
    DATA_DIR / "sample_submit.csv",
    DATA_DIR / "submission_sample.csv",
]

RANDOM_STATE = 42
N_SPLITS = 5
USE_TOPK_BLEND = True   # True なら上位複数モデルを逆RMSE重みでblend
TOPK = 3

# PLS成分候補
N_COMPONENTS_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 40]

# 論文ベースの前処理候補
MODEL_CONFIGS = [
    {"name": "raw_full",              "preproc": None,   "sg_deriv": None, "band": None},
    {"name": "snv_sg1_full",          "preproc": "snv",  "sg_deriv": 1,    "band": None},
    {"name": "snv_sg2_full",          "preproc": "snv",  "sg_deriv": 2,    "band": None},
    {"name": "msc_sg1_full",          "preproc": "msc",  "sg_deriv": 1,    "band": None},
    {"name": "snv_sg1_1400_1900",     "preproc": "snv",  "sg_deriv": 1,    "band": (1400, 1900)},
    {"name": "snv_sg2_1400_1900",     "preproc": "snv",  "sg_deriv": 2,    "band": (1400, 1900)},
    {"name": "msc_sg1_1400_1900",     "preproc": "msc",  "sg_deriv": 1,    "band": (1400, 1900)},
]

# SGフィルタ設定
SG_WINDOW = 15
SG_POLYORDER = 2

# =========================
# ユーティリティ
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def infer_target_col(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    # train にだけ存在する数値列を最優先
    extra_cols = [c for c in train_df.columns if c not in test_df.columns]
    numeric_extra = [c for c in extra_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    if len(numeric_extra) == 1:
        return numeric_extra[0]

    # 名前ヒント
    preferred = [
        "target", "y", "label", "moisture", "moisture_content", "water_content"
    ]
    for c in train_df.columns:
        if c.lower() in preferred:
            return c

    # 最後の列をフォールバック
    return train_df.columns[-1]

def infer_id_col(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> str | None:
    common_cols = [c for c in train_df.columns if c in test_df.columns and c != target_col]

    # 明示的な ID 名
    id_like = [
        c for c in common_cols
        if (
            c.lower() in {"id", "index", "sample_id", "row_id"}
            or c.lower().endswith("_id")
        )
    ]
    if id_like:
        return id_like[0]

    # 先頭列がユニークなら ID とみなす
    if common_cols:
        c0 = common_cols[0]
        if train_df[c0].nunique(dropna=False) == len(train_df) and test_df[c0].nunique(dropna=False) == len(test_df):
            return c0

    return None

def parse_wavelength(col_name: str) -> float | None:
    """
    列名から波長っぽい数値を抜く。
    例:
      "1450" -> 1450
      "wl_1450.5" -> 1450.5
      "x1400" -> 1400
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(col_name))
    if not matches:
        return None

    # いちばん最後の数値を採用
    val = float(matches[-1])

    # 極端に小さい/大きい値は波長でない可能性が高い
    if val < 300 or val > 3000:
        return None
    return val

def infer_spectral_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str | None,
):
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    excluded = {target_col}
    if id_col is not None:
        excluded.add(id_col)

    candidate_cols = [c for c in common_cols if c not in excluded]

    # 数値列のみ
    numeric_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(test_df[c])
    ]

    wavelength_map = {}
    for c in numeric_cols:
        wl = parse_wavelength(c)
        if wl is not None:
            wavelength_map[c] = wl

    # 十分な割合で波長が取れるなら、その順に並べる
    if len(numeric_cols) > 0 and len(wavelength_map) / len(numeric_cols) >= 0.7:
        spectral_cols = sorted(wavelength_map.keys(), key=lambda c: wavelength_map[c])
        return spectral_cols, wavelength_map

    # 取れないときは元の数値列をそのまま使う
    return numeric_cols, {}

def choose_band_cols(
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    band: tuple[float, float] | None,
    min_cols: int = 30,
) -> list[str]:
    if band is None or len(wavelength_map) == 0:
        return spectral_cols

    lo, hi = band
    band_cols = [c for c in spectral_cols if c in wavelength_map and lo <= wavelength_map[c] <= hi]

    # 帯域にほとんど列がないなら full に戻す
    if len(band_cols) < min_cols:
        return spectral_cols
    return band_cols

# =========================
# カスタム前処理
# =========================
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.cols].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate: 行ごとに平均0・分散1へ"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, keepdims=True)
        row_std = np.where(row_std < 1e-12, 1.0, row_std)
        return (X - row_mean) / row_std

class MSCTransformer(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction
    各サンプルを平均スペクトルへ回帰して補正する
    """
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ref = self.reference_
        X_corr = np.zeros_like(X)

        for i in range(X.shape[0]):
            # x_i ≈ slope * ref + intercept
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

        # window_length は奇数かつ polyorder より大きく
        window = min(self.window_length, n_features if n_features % 2 == 1 else n_features - 1)
        if window <= self.polyorder:
            return X
        if window < 3:
            return X

        return savgol_filter(
            X,
            window_length=window,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )

# =========================
# モデル構築
# =========================
def build_pipeline(
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    preproc: str | None,
    sg_deriv: int | None,
    band: tuple[float, float] | None,
    n_components: int,
) -> tuple[Pipeline, list[str]]:
    cols = choose_band_cols(spectral_cols, wavelength_map, band=band)

    steps = [
        ("select", ColumnSelector(cols)),
        ("impute", SimpleImputer(strategy="median")),
    ]

    if preproc == "snv":
        steps.append(("snv", SNVTransformer()))
    elif preproc == "msc":
        steps.append(("msc", MSCTransformer()))

    if sg_deriv is not None:
        steps.append(
            (
                "sg",
                SavitzkyGolayTransformer(
                    window_length=SG_WINDOW,
                    polyorder=SG_POLYORDER,
                    deriv=sg_deriv,
                ),
            )
        )

    steps.extend(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=n_components, scale=False)),
        ]
    )

    return Pipeline(steps), cols

def fit_predict_cv(
    train_df: pd.DataFrame,
    y: np.ndarray,
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    config: dict,
    n_components: int,
):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(train_df), dtype=float)

    # 使用列数から妥当な成分数へ補正
    used_cols = choose_band_cols(
        spectral_cols,
        wavelength_map,
        band=config["band"],
    )
    if len(used_cols) < 2:
        raise ValueError(f"Not enough spectral columns for config={config['name']}")

    for tr_idx, va_idx in cv.split(train_df, y):
        ncomp_eff = max(1, min(n_components, len(used_cols) - 1, len(tr_idx) - 1))

        model, _ = build_pipeline(
            spectral_cols=spectral_cols,
            wavelength_map=wavelength_map,
            preproc=config["preproc"],
            sg_deriv=config["sg_deriv"],
            band=config["band"],
            n_components=ncomp_eff,
        )
        model.fit(train_df.iloc[tr_idx], y[tr_idx])
        oof_pred[va_idx] = model.predict(train_df.iloc[va_idx]).ravel()

    score = rmse(y, oof_pred)
    return oof_pred, score, len(used_cols)

def search_best_models(
    train_df: pd.DataFrame,
    y: np.ndarray,
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
):
    results = []

    for config in MODEL_CONFIGS:
        for n_components in N_COMPONENTS_GRID:
            try:
                oof_pred, score, n_used_cols = fit_predict_cv(
                    train_df=train_df,
                    y=y,
                    spectral_cols=spectral_cols,
                    wavelength_map=wavelength_map,
                    config=config,
                    n_components=n_components,
                )
                results.append(
                    {
                        "config_name": config["name"],
                        "config": config,
                        "n_components": n_components,
                        "rmse": score,
                        "n_used_cols": n_used_cols,
                        "oof_pred": oof_pred,
                    }
                )
                print(
                    f"[OK] {config['name']:22s} | n_comp={n_components:2d} "
                    f"| n_cols={n_used_cols:4d} | RMSE={score:.6f}"
                )
            except Exception as e:
                print(f"[SKIP] {config['name']} | n_comp={n_components} | reason={e}")

    results = sorted(results, key=lambda x: x["rmse"])
    return results

def train_full_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: np.ndarray,
    spectral_cols: list[str],
    wavelength_map: dict[str, float],
    result_row: dict,
):
    config = result_row["config"]
    used_cols = choose_band_cols(spectral_cols, wavelength_map, band=config["band"])
    ncomp_eff = max(1, min(result_row["n_components"], len(used_cols) - 1, len(train_df) - 1))

    model, _ = build_pipeline(
        spectral_cols=spectral_cols,
        wavelength_map=wavelength_map,
        preproc=config["preproc"],
        sg_deriv=config["sg_deriv"],
        band=config["band"],
        n_components=ncomp_eff,
    )
    model.fit(train_df, y)
    pred = model.predict(test_df).ravel()
    return pred

# =========================
# 実行
# =========================
def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_col = infer_target_col(train_df, test_df)
    id_col = infer_id_col(train_df, test_df, target_col)
    spectral_cols, wavelength_map = infer_spectral_cols(train_df, test_df, target_col, id_col)

    print("target_col:", target_col)
    print("id_col    :", id_col)
    print("n_train   :", len(train_df))
    print("n_test    :", len(test_df))
    print("n_spectra :", len(spectral_cols))
    print("wavelength columns parsed:", len(wavelength_map) > 0)

    if len(spectral_cols) == 0:
        raise ValueError("スペクトル列を推定できませんでした。列名を確認してください。")

    y = train_df[target_col].to_numpy(dtype=float)

    # CV で最良設定を探索
    results = search_best_models(
        train_df=train_df,
        y=y,
        spectral_cols=spectral_cols,
        wavelength_map=wavelength_map,
    )

    if len(results) == 0:
        raise RuntimeError("有効なモデルが1つも作れませんでした。")

    print("\n===== Top Results =====")
    for i, row in enumerate(results[:10], start=1):
        print(
            f"{i:2d}. {row['config_name']:22s} | "
            f"n_comp={row['n_components']:2d} | "
            f"n_cols={row['n_used_cols']:4d} | "
            f"RMSE={row['rmse']:.6f}"
        )

    # 予測
    if USE_TOPK_BLEND:
        top_results = results[:TOPK]
        raw_weights = np.array([1.0 / max(r["rmse"] ** 2, 1e-12) for r in top_results], dtype=float)
        weights = raw_weights / raw_weights.sum()

        test_pred = np.zeros(len(test_df), dtype=float)
        for w, row in zip(weights, top_results):
            pred = train_full_and_predict(
                train_df=train_df,
                test_df=test_df,
                y=y,
                spectral_cols=spectral_cols,
                wavelength_map=wavelength_map,
                result_row=row,
            )
            test_pred += w * pred
            print(f"blend: {row['config_name']} (n_comp={row['n_components']}) weight={w:.4f}")
    else:
        best = results[0]
        test_pred = train_full_and_predict(
            train_df=train_df,
            test_df=test_df,
            y=y,
            spectral_cols=spectral_cols,
            wavelength_map=wavelength_map,
            result_row=best,
        )

    # 物理的に負の含水率は不自然なので下限0でクリップ
    test_pred = np.clip(test_pred, 0, None)

    # submission 作成
    sample_sub_path = first_existing(SAMPLE_SUBMISSION_CANDIDATES)

    if sample_sub_path is not None:
        sub = pd.read_csv(sample_sub_path)

        # 予測列を推定
        pred_candidates = [c for c in sub.columns if (id_col is None or c != id_col)]
        if len(pred_candidates) == 0:
            pred_col = sub.columns[-1]
        elif len(pred_candidates) == 1:
            pred_col = pred_candidates[0]
        else:
            # 2列以上あるなら id 以外の最後を使う
            pred_col = pred_candidates[-1]

        if id_col is not None and id_col in sub.columns and id_col in test_df.columns:
            sub[id_col] = test_df[id_col].values

        sub[pred_col] = test_pred
    else:
        # sample_submission がない場合のフォールバック
        if id_col is not None and id_col in test_df.columns:
            sub = pd.DataFrame({id_col: test_df[id_col].values, "prediction": test_pred})
        else:
            sub = pd.DataFrame({"prediction": test_pred})

    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.resolve()}")

    # 学習ログを保存
    log_rows = []
    for row in results:
        log_rows.append(
            {
                "config_name": row["config_name"],
                "n_components": row["n_components"],
                "n_used_cols": row["n_used_cols"],
                "cv_rmse": row["rmse"],
            }
        )
    pd.DataFrame(log_rows).to_csv("cv_results.csv", index=False)
    print("Saved: cv_results.csv")

    meta = {
        "target_col": target_col,
        "id_col": id_col,
        "n_spectral_cols": len(spectral_cols),
        "wavelength_parsed": len(wavelength_map) > 0,
        "top_result": {
            "config_name": results[0]["config_name"],
            "n_components": int(results[0]["n_components"]),
            "cv_rmse": float(results[0]["rmse"]),
        },
    }
    Path("run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved: run_meta.json")

if __name__ == "__main__":
    main()
# %%
