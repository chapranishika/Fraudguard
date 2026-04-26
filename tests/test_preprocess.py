"""
tests/test_preprocess.py
Unit tests for the PaySim preprocessing pipeline (17 features).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocess import (
    FEATURE_NAMES, N_FEATURES, SCALE_COLS,
    apply_scaler, engineer_features, fit_scaler, make_dataframe,
)


def _make_paysim_rows(n=5, fraud=False):
    """Create n synthetic PaySim rows with the same nameOrig for velocity testing."""
    rows = []
    for i in range(n):
        rows.append({
            "step": i + 1,
            "type": "TRANSFER",
            "amount": float((i + 1) * 1000),
            "nameOrig": "C123",
            "oldbalanceOrg": float((n - i) * 1000),
            "newbalanceOrig": 0.0 if fraud else float((n - i - 1) * 1000),
            "nameDest": "C456",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0 if fraud else float((i + 1) * 1000),
            "isFraud": int(fraud),
            "isFlaggedFraud": 0,
        })
    return pd.DataFrame(rows)


def _make_single_row(**kwargs):
    defaults = {
        "step": 1, "type": "TRANSFER", "amount": 1000.0,
        "nameOrig": "C123", "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0,
        "nameDest": "C456", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": 1, "isFlaggedFraud": 0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestEngineerFeatures:
    def test_output_columns_match_feature_names(self):
        df = engineer_features(_make_single_row())
        assert list(df.columns) == FEATURE_NAMES

    def test_correct_row_count(self):
        df = _make_single_row()
        assert engineer_features(df).shape[0] == 1

    def test_balance_delta_orig_computed(self):
        row = _make_single_row(oldbalanceOrg=1000, newbalanceOrig=0, amount=1000)
        out = engineer_features(row)
        # newbalanceOrig + amount - oldbalanceOrg = 0 + 1000 - 1000 = 0
        assert abs(out["balance_delta_orig"].iloc[0] - 0.0) < 1e-6

    def test_balance_zero_orig_flag_set(self):
        row = _make_single_row(newbalanceOrig=0)
        assert engineer_features(row)["balance_zero_orig"].iloc[0] == 1

    def test_balance_zero_orig_flag_not_set(self):
        row = _make_single_row(newbalanceOrig=500)
        assert engineer_features(row)["balance_zero_orig"].iloc[0] == 0

    def test_transfer_one_hot(self):
        out = engineer_features(_make_single_row(type="TRANSFER"))
        assert out["type_TRANSFER"].iloc[0] == 1
        assert out["type_CASH_OUT"].iloc[0] == 0

    def test_cash_out_one_hot(self):
        out = engineer_features(_make_single_row(type="CASH_OUT"))
        assert out["type_CASH_OUT"].iloc[0] == 1
        assert out["type_TRANSFER"].iloc[0] == 0

    def test_unknown_type_all_zeros(self):
        out = engineer_features(_make_single_row(type="UNKNOWN_TYPE"))
        for col in ["type_CASH_IN","type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"]:
            assert out[col].iloc[0] == 0


class TestVelocityFeatures:
    def test_tx_count_increments_per_sender(self):
        df  = _make_paysim_rows(n=5)
        out = engineer_features(df)
        counts = out["orig_tx_count_so_far"].tolist()
        # Sorted by step → expanding count should go 1, 2, 3, 4, 5
        assert counts == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_amount_sum_accumulates(self):
        df  = _make_paysim_rows(n=3)
        out = engineer_features(df)
        # amounts are 1000, 2000, 3000 → cumulative: 1000, 3000, 6000
        sums = out["orig_amount_sum_so_far"].tolist()
        assert sums == [1000.0, 3000.0, 6000.0]

    def test_different_senders_independent(self):
        df1 = _make_paysim_rows(n=2)           # nameOrig = C123
        df2 = _make_paysim_rows(n=2)
        df2["nameOrig"] = "C999"               # different sender
        df2["step"] = [3, 4]
        combined = pd.concat([df1, df2], ignore_index=True)
        out = engineer_features(combined)
        # Each sender's count should reset at 1
        c123 = out[combined["nameOrig"] == "C123"]["orig_tx_count_so_far"]
        c999 = out[combined["nameOrig"] == "C999"]["orig_tx_count_so_far"]
        assert c123.min() == 1.0
        assert c999.min() == 1.0

    def test_velocity_features_in_feature_names(self):
        assert "orig_tx_count_so_far"   in FEATURE_NAMES
        assert "orig_amount_sum_so_far" in FEATURE_NAMES

    def test_velocity_features_in_scale_cols(self):
        assert "orig_tx_count_so_far"   in SCALE_COLS
        assert "orig_amount_sum_so_far" in SCALE_COLS


class TestMakeDataframe:
    def test_correct_shape(self):
        assert make_dataframe([0.0] * N_FEATURES).shape == (1, N_FEATURES)

    def test_column_names(self):
        assert list(make_dataframe([0.0] * N_FEATURES).columns) == FEATURE_NAMES

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            make_dataframe([0.0] * (N_FEATURES - 1))

    def test_too_many_raises(self):
        with pytest.raises(ValueError):
            make_dataframe([0.0] * (N_FEATURES + 1))


class TestScaler:
    @pytest.fixture
    def sample_df(self):
        rows = [_make_single_row(amount=float(i) * 100, oldbalanceOrg=float(i) * 200)
                for i in range(1, 51)]
        raw = pd.concat(rows, ignore_index=True)
        return engineer_features(raw)

    def test_fit_returns_standard_scaler(self, sample_df):
        assert isinstance(fit_scaler(sample_df), StandardScaler)

    def test_scale_cols_are_scaled(self, sample_df):
        sc     = fit_scaler(sample_df)
        scaled = apply_scaler(sample_df, sc)
        for col in SCALE_COLS:
            assert abs(scaled[col].mean()) < 0.15

    def test_binary_flags_unchanged(self, sample_df):
        sc     = fit_scaler(sample_df)
        scaled = apply_scaler(sample_df, sc)
        for col in ["balance_zero_orig", "balance_zero_dest"]:
            assert scaled[col].isin([0, 1]).all()

    def test_does_not_mutate_input(self, sample_df):
        orig_vals = sample_df["amount"].copy()
        sc = fit_scaler(sample_df)
        apply_scaler(sample_df, sc)
        pd.testing.assert_series_equal(sample_df["amount"], orig_vals)

    def test_apply_only_transforms_no_fit(self, sample_df):
        sc1 = fit_scaler(sample_df)
        sc2 = fit_scaler(sample_df)
        r1  = apply_scaler(sample_df, sc1)
        r2  = apply_scaler(sample_df, sc2)
        pd.testing.assert_frame_equal(r1, r2)


class TestFeatureList:
    def test_exactly_n_features(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_n_features_is_17(self):
        assert N_FEATURES == 17

    def test_no_duplicates(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_all_type_cols_present(self):
        for t in ["type_CASH_IN","type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"]:
            assert t in FEATURE_NAMES

    def test_scale_cols_subset_of_features(self):
        assert all(c in FEATURE_NAMES for c in SCALE_COLS)
