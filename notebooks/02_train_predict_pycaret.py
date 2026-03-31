# %%
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    from pycaret.regression import create_model, finalize_model, predict_model, pull, setup, tune_model
except Exception as e:
    raise ImportError(
        "pycaret の import に失敗しました。Python 3.12 環境で `uv add pycaret` を実行してください。"
    ) from e


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TARGET_COL = "含水率"
ID_COL = "sample number"
RANDOM_STATE = 42
CV = 5
N_TRIALS = 50
PREPROCESS_CLIP_NEGATIVE = False
PREPROCESS_DROP_SPECIES = []

JST = timezone(timedelta(hours=9))
DATE = datetime.now(JST).strftime("%Y%m%d%H%M")
SUBMIT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# %%
def get_species_col(df: pd.DataFrame) -> str | None:
    for c in ["樹種", "species", "species number"]:
        if c in df.columns:
            return c
    return None


def get_wavelength_cols(df: pd.DataFrame, species_col: str | None) -> list[str]:
    exclude = {ID_COL, TARGET_COL, "乾物率"}
    if species_col:
        exclude.add(species_col)
    cols = []
    for c in df.columns:
        if c in exclude or not np.issubdtype(df[c].dtype, np.number):
            continue
        try:
            float(c)
            cols.append(c)
        except (TypeError, ValueError):
            continue
    return cols


# %%
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="cp932")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="cp932")

species_col = get_species_col(train_df)
wavelength_cols = get_wavelength_cols(train_df, species_col)

if PREPROCESS_CLIP_NEGATIVE and wavelength_cols:
    train_df.loc[:, wavelength_cols] = train_df[wavelength_cols].clip(lower=0)
    shared_cols = [c for c in wavelength_cols if c in test_df.columns]
    test_df.loc[:, shared_cols] = test_df[shared_cols].clip(lower=0)

if PREPROCESS_DROP_SPECIES and species_col:
    train_df = train_df[~train_df[species_col].isin(PREPROCESS_DROP_SPECIES)].copy()

exclude = [ID_COL, "species number", TARGET_COL]
if "乾物率" in train_df.columns:
    exclude.append("乾物率")
feature_cols = [c for c in train_df.columns if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)]
if not feature_cols:
    feature_cols = [c for c in train_df.columns if c not in exclude]

train_for_pc = train_df[feature_cols + [TARGET_COL]].copy()
test_for_pc = test_df[feature_cols].copy()
test_ids = test_df[ID_COL].copy()

# %%
setup(
    data=train_for_pc,
    target=TARGET_COL,
    session_id=RANDOM_STATE,
    fold=CV,
    html=False,
    verbose=False,
)

base_model = create_model("ridge", verbose=False)
tuned_model = tune_model(
    base_model,
    optimize="RMSE",
    n_iter=N_TRIALS,
    search_library="optuna",
    choose_better=True,
    verbose=False,
)
cv_table = pull()
if "Mean" in cv_table.index and "RMSE" in cv_table.columns:
    cv_rmse = float(cv_table.loc["Mean", "RMSE"])
else:
    cv_rmse = float(cv_table["RMSE"].iloc[0])

# %%
final_model = finalize_model(tuned_model)
pred_df = predict_model(final_model, data=test_for_pc, verbose=False)
pred_col = "prediction_label" if "prediction_label" in pred_df.columns else "Label"
submit_df = pd.DataFrame({"id": test_ids, "value": pred_df[pred_col]})

rmse_tag = int(round(cv_rmse * 10000))
submit_path = os.path.join(SUBMIT_DIR, f"submit_pycaret_{DATE}_{rmse_tag:04d}.csv")
submit_df.to_csv(submit_path, index=False, header=False, encoding="utf-8-sig")

print(f"CV RMSE: {cv_rmse:.5f}")
print(f"saved: {submit_path}")

# %%
