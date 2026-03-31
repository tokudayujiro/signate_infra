# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# %%
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
# %%
train_df = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    encoding="cp932",  # Windows 日本語（Shift-JIS）
)
test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    encoding="cp932",  # Windows 日本語（Shift-JIS）
)
# %%
train_df
# %%
test_df
# %%
