"""
data_loader.py
Shared utility — load, label, split, and scale the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = r"..\..\data\final\musk_events_k10_replies_True.csv"

TARGET_COL = "max_z_next5"

FEATURE_COLS = [
    "mentions_tesla", "is_reply", "is_quote", "is_retweet",
    "positive", "negative", "neutral",
    "close_delta_z", "volume_delta_z",
    "price_cv", "volume_cv", "close_position"
]


def make_labels(series, low=-1.0, high=1.0):
    """Bin continuous z-score into 3 classes: down / neutral / up."""
    return pd.cut(
        series,
        bins=[-np.inf, low, high, np.inf],
        labels=["down", "neutral", "up"]
    ).astype(str)


def load_data():
    """
    Returns a dict with train/val/test splits for both tasks.

    Keys:
        X_train, X_val, X_test          — scaled feature arrays
        y_train_cls, y_val_cls, y_test_cls  — classification labels (down/neutral/up)
        y_train_reg, y_val_reg, y_test_reg  — regression targets (raw z-score)
        feature_cols                    — list of feature names
        scaler                          — fitted StandardScaler
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    df["label"] = make_labels(df[TARGET_COL])

    X     = df[FEATURE_COLS].values
    y_cls = df["label"].values
    y_reg = df[TARGET_COL].values

    # 70 / 15 / 15 split
    X_tr, X_tmp, y_tr_cls, y_tmp_cls, y_tr_reg, y_tmp_reg = train_test_split(
        X, y_cls, y_reg, test_size=0.30, random_state=42, stratify=y_cls
    )
    X_val, X_te, y_val_cls, y_te_cls, y_val_reg, y_te_reg = train_test_split(
        X_tmp, y_tmp_cls, y_tmp_reg, test_size=0.50, random_state=42, stratify=y_tmp_cls
    )

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_te)

    print(f"[data_loader] Train: {len(X_tr_sc)} | Val: {len(X_val_sc)} | Test: {len(X_te_sc)}")
    print(f"[data_loader] Class distribution (train):\n"
          f"  {dict(zip(*np.unique(y_tr_cls, return_counts=True)))}\n")

    return {
        "X_train":     X_tr_sc,
        "X_val":       X_val_sc,
        "X_test":      X_te_sc,
        "y_train_cls": y_tr_cls,
        "y_val_cls":   y_val_cls,
        "y_test_cls":  y_te_cls,
        "y_train_reg": y_tr_reg,
        "y_val_reg":   y_val_reg,
        "y_test_reg":  y_te_reg,
        "feature_cols": FEATURE_COLS,
        "scaler":      scaler,
    }
