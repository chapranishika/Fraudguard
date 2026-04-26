"""
src/preprocess.py
-----------------
Shared preprocessing utilities for the PaySim financial dataset.

Raw PaySim columns:
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Engineered features (17 total):
  Balance error features (key fraud signal):
    balance_delta_orig : newbalanceOrig + amount - oldbalanceOrg  (should be ~0 in honest tx)
    balance_delta_dest : oldbalanceDest + amount - newbalanceDest (accounting error at dest)
    balance_zero_orig  : 1 if sender's balance was wiped to exactly 0
    balance_zero_dest  : 1 if recipient had zero opening balance

  Temporal velocity features (sequential fraud signal):
    orig_tx_count_so_far   : cumulative number of transactions sent by this nameOrig
    orig_amount_sum_so_far : cumulative amount sent by this nameOrig up to this step
    Computed on the full dataframe sorted by step BEFORE any train/test split.
    These are expanding window features — each row sees only its own past,
    not future transactions (no future leakage within the feature itself).

  Transaction type one-hot:
    type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    Fraud only occurs on TRANSFER and CASH_OUT — the type encoding captures this.

  Raw financial fields:
    amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step

IMPORTANT — split-before-scale contract:
  fit_scaler() MUST be called only on X_train_raw (post-split).
  load_raw() intentionally does NOT fit or apply the scaler — the caller (train.py)
  is responsible for splitting first, then fitting the scaler on the train partition only.
  This prevents test-set distribution leakage into StandardScaler.mean_ and .scale_.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns to standard-scale (continuous, wide numeric range)
SCALE_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_delta_orig", "balance_delta_dest",
    "orig_tx_count_so_far", "orig_amount_sum_so_far",
    "step",
]

# Final feature set — must match exactly what train.py builds and api.py receives
FEATURE_NAMES = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_delta_orig", "balance_delta_dest",
    "balance_zero_orig", "balance_zero_dest",
    "orig_tx_count_so_far", "orig_amount_sum_so_far",
    "step",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER",
]

N_FEATURES = len(FEATURE_NAMES)   # 17


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a raw PaySim DataFrame.

    Works on both training data (has isFraud) and inference data (no label).
    Returns a new DataFrame with only the model feature columns.

    For velocity features (orig_tx_count_so_far, orig_amount_sum_so_far):
    - During TRAINING: call this on the full dataframe sorted by step BEFORE splitting.
      The expanding groupby produces a cumulative count/sum that only uses each row's
      own past — no future leakage within the feature itself.
    - During INFERENCE on a single transaction: velocity features cannot be computed
      from a single row. Pass the account's known cumulative count/sum from your
      feature store, or set them to 0 for demo purposes.
    """
    # Sort by step to ensure temporal order for velocity features
    df = df.sort_values("step").copy()
    out = pd.DataFrame(index=df.index)

    # Raw financial values
    out["amount"]        = df["amount"]
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["newbalanceOrig"]= df["newbalanceOrig"]
    out["oldbalanceDest"]= df["oldbalanceDest"]
    out["newbalanceDest"]= df["newbalanceDest"]
    out["step"]          = df["step"]

    # Balance error features
    # In a legitimate transaction: newbalanceOrig = oldbalanceOrg - amount → delta ≈ 0
    # Fraud often drains accounts to exactly 0 or skips updating destination balances.
    out["balance_delta_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    out["balance_delta_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Binary flags
    out["balance_zero_orig"] = (df["newbalanceOrig"] == 0).astype(int)
    out["balance_zero_dest"] = (df["oldbalanceDest"] == 0).astype(int)

    # Temporal velocity features — expanding window per sender account
    # expanding().count() gives the cumulative number of prior txns including current row
    # expanding().sum()   gives the cumulative amount sent up to and including this step
    # Using transform() preserves the original row index for correct alignment after sort
    out["orig_tx_count_so_far"] = (
        df.groupby("nameOrig")["amount"]
        .transform(lambda x: x.expanding().count())
    )
    out["orig_amount_sum_so_far"] = (
        df.groupby("nameOrig")["amount"]
        .transform(lambda x: x.expanding().sum())
    )

    # Transaction type one-hot
    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
                "type_PAYMENT", "type_TRANSFER"]:
        out[col] = type_dummies[col].astype(int) if col in type_dummies.columns \
                   else pd.Series(0, index=df.index)

    return out[FEATURE_NAMES]


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on SCALE_COLS of X.
    MUST be called on X_train_raw only — never on the full dataset.
    """
    scaler = StandardScaler()
    scaler.fit(X[SCALE_COLS])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler to X. Never fits — safe to call on test set."""
    X = X.copy()
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
    return X


def make_dataframe(raw: list[float]) -> pd.DataFrame:
    """Convert a flat list of N_FEATURES floats into a properly named DataFrame."""
    if len(raw) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {len(raw)}")
    return pd.DataFrame([raw], columns=FEATURE_NAMES)


def load_raw(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load PaySim CSV and engineer features.
    Returns (X_raw, y) — unscaled features and labels.

    Intentionally does NOT fit or apply the scaler.
    The caller (train.py) must:
      1. Call this function
      2. Split X_raw and y with train_test_split
      3. Call fit_scaler(X_train_raw) — train split ONLY
      4. Call apply_scaler() on both partitions

    This contract prevents StandardScaler from seeing test-set distribution
    during fit, which would constitute preprocessing data leakage.
    """
    df = pd.read_csv(csv_path)
    y  = df["isFraud"].astype(int)
    X  = engineer_features(df)
    return X, y
