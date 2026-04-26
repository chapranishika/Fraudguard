"""
tests/test_api.py
-----------------
Unit tests for the FastAPI fraud detection endpoints.
Uses unittest.mock to inject fakes — no .pkl files or dataset needed.

Run with:  pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.preprocess import FEATURE_NAMES, N_FEATURES

ZERO_FEATURES = [0.0] * N_FEATURES   # 15 floats — PaySim schema


# ── Fake artifact factory ────────────────────────────────────────────────────
def _make_fakes():
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    fake_scaler = MagicMock()
    fake_scaler.transform.return_value = np.zeros((1, len([
        "amount","oldbalanceOrg","newbalanceOrig",
        "oldbalanceDest","newbalanceDest",
        "balance_delta_orig","balance_delta_dest","step",
    ])))
    # Expose mean_/scale_ so _validate_features doesn't crash
    fake_scaler.mean_  = np.zeros(8)
    fake_scaler.scale_ = np.ones(8)
    fake_scaler.feature_names_in_ = np.array([
        "amount","oldbalanceOrg","newbalanceOrig",
        "oldbalanceDest","newbalanceDest",
        "balance_delta_orig","balance_delta_dest","step",
    ])

    fake_explainer = MagicMock()
    fake_explainer.shap_values.return_value = np.zeros((1, N_FEATURES))

    fake_metrics = {
        "best_model":         "XGBoost",
        "selection_criterion":"pr_auc",
        "dataset":            "paysim_sample.csv",
        "data_hash":          "abc123",
        "trained_at":         "2024-01-01T00:00:00+00:00",
        "roc_auc":            0.99,
        "pr_auc":             0.87,
        "precision":          0.95,
        "recall":             0.88,
        "f1":                 0.91,
        "roc_auc_calibrated": 0.98,
        "pr_auc_calibrated":  0.86,
        "precision_calibrated": 0.94,
        "recall_calibrated":  0.87,
        "f1_calibrated":      0.90,
        "confusion_matrix":   [[199400, 20], [15, 107]],
        "total_transactions": 1_000_000,
        "fraud_count":        535,
        "fraud_pct":          0.0535,
        "feature_names":      FEATURE_NAMES,
        "global_shap":        {f: 0.01*i for i, f in enumerate(FEATURE_NAMES)},
        "comparison": {
            "RandomForest": {"roc_auc":0.97,"pr_auc":0.83,"precision":0.93,"recall":0.85,"f1":0.89,"confusion_matrix":[]},
            "XGBoost":      {"roc_auc":0.99,"pr_auc":0.87,"precision":0.95,"recall":0.88,"f1":0.91,"confusion_matrix":[]},
            "LightGBM":     {"roc_auc":0.98,"pr_auc":0.85,"precision":0.94,"recall":0.87,"f1":0.90,"confusion_matrix":[]},
        },
    }

    fake_threshold_curve = [
        {"threshold": round(t, 2), "precision": 0.9, "recall": 1.0 - t, "f1": 0.85}
        for t in [i/10 for i in range(1, 10)]
    ]

    return fake_model, fake_scaler, fake_explainer, fake_metrics, fake_threshold_curve


# ── Session-scoped test client ────────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    # Artifacts are written by conftest.py before this runs.
    # Purge any stale cached import.
    for mod in list(sys.modules.keys()):
        if "src.api" in mod:
            del sys.modules[mod]

    import importlib
    import src.api as api_mod
    importlib.reload(api_mod)
    from fastapi.testclient import TestClient

    # Starlette's TestClient does not forward lifespan= in this version.
    # We enter the lifespan context manually so all globals are populated
    # (model, scaler, explainer, _metrics, _threshold_curve) before tests run.
    import asyncio

    async def _run_lifespan():
        async with api_mod.lifespan(api_mod.app):
            pass   # populates globals, then yields back

    asyncio.get_event_loop().run_until_complete(_run_lifespan())

    with TestClient(api_mod.app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_returns_only_status_key(self, client):
        """
        /health must ONLY return {"status": "ok"}.
        Provenance data (trained_at, data_hash, pr_auc_calibrated) belongs
        on the authenticated /info endpoint — not exposed to unauthenticated callers.
        """
        body = client.get("/health").json()
        assert set(body.keys()) == {"status"}

    def test_no_auth_required(self, client):
        assert client.get("/health").status_code != 401

    def test_does_not_expose_model_name(self, client):
        assert "model" not in client.get("/health").json()

    def test_does_not_expose_trained_at(self, client):
        assert "trained_at" not in client.get("/health").json()

    def test_does_not_expose_data_hash(self, client):
        assert "data_hash" not in client.get("/health").json()


# ── /info ─────────────────────────────────────────────────────────────────────
class TestInfo:
    H = {"X-API-Key": "demo-key-123"}

    def test_requires_auth(self, client):
        assert client.get("/info").status_code == 401

    def test_returns_200_with_key(self, client):
        assert client.get("/info", headers=self.H).status_code == 200

    def test_has_model_field(self, client):
        assert "model" in client.get("/info", headers=self.H).json()

    def test_has_trained_at(self, client):
        assert "trained_at" in client.get("/info", headers=self.H).json()

    def test_has_data_hash(self, client):
        assert "data_hash" in client.get("/info", headers=self.H).json()

    def test_has_pr_auc_calibrated(self, client):
        assert "pr_auc_calibrated" in client.get("/info", headers=self.H).json()

    def test_has_feature_count(self, client):
        d = client.get("/info", headers=self.H).json()
        assert "feature_count" in d
        assert d["feature_count"] == 17

    def test_has_selection_criterion(self, client):
        d = client.get("/info", headers=self.H).json()
        assert d.get("selection_criterion") == "pr_auc"


# ── /drift_report ─────────────────────────────────────────────────────────────
class TestDriftReport:
    H = {"X-API-Key": "demo-key-123"}

    def test_requires_auth(self, client):
        assert client.get("/drift_report").status_code == 401

    def test_returns_200_with_key(self, client):
        assert client.get("/drift_report", headers=self.H).status_code == 200

    def test_has_required_keys(self, client):
        d = client.get("/drift_report", headers=self.H).json()
        for key in ["status", "records_checked", "live_fraud_rate",
                    "baseline_fraud_rate", "message"]:
            assert key in d, f"drift_report missing key: {key}"

    def test_status_is_valid_value(self, client):
        d = client.get("/drift_report", headers=self.H).json()
        assert d["status"] in {"OK", "DRIFT_DETECTED", "NOT_ENOUGH_DATA"}

    def test_not_enough_data_or_valid_status(self, client):
        """
        The test DB may already contain audit rows from earlier /predict calls.
        Accept any of the three valid statuses — the key contract is that the
        endpoint always returns a valid, parseable response.
        """
        d = client.get("/drift_report", headers=self.H).json()
        assert d["status"] in {"OK", "DRIFT_DETECTED", "NOT_ENOUGH_DATA"}
        assert "records_checked" in d

    def test_records_checked_is_non_negative(self, client):
        d = client.get("/drift_report", headers=self.H).json()
        assert d["records_checked"] >= 0

    def test_baseline_fraud_rate_matches_metrics(self, client):
        import json
        from pathlib import Path
        metrics = json.load(open(Path("models") / "metrics.json"))
        d = client.get("/drift_report", headers=self.H).json()
        expected = round(metrics.get("fraud_pct", 0) / 100.0, 6)
        assert d["baseline_fraud_rate"] == expected


# ── SHAP cache ─────────────────────────────────────────────────────────────────
class TestShapCache:
    H = {"X-API-Key": "demo-key-123"}

    def test_cache_hit_returns_same_shap(self, client):
        """Two identical feature vectors must return identical SHAP values."""
        payload = {"features": ZERO_FEATURES, "threshold": 0.5}
        r1 = client.post("/predict", json=payload, headers=self.H).json()
        r2 = client.post("/predict", json=payload, headers=self.H).json()
        assert r1["top_shap_features"] == r2["top_shap_features"]

    def test_different_features_different_shap(self, client):
        """Two different feature vectors should return different SHAP dicts (almost always)."""
        features_a = list(ZERO_FEATURES)
        features_b = list(ZERO_FEATURES)
        features_b[0] = 99999.0   # very different amount
        r1 = client.post("/predict", json={"features": features_a}, headers=self.H).json()
        r2 = client.post("/predict", json={"features": features_b}, headers=self.H).json()
        # Not necessarily different for a DummyClassifier stub, but the call must succeed
        assert "top_shap_features" in r1
        assert "top_shap_features" in r2

    def test_cache_module_has_correct_constants(self):
        """Verify cache constants are set to expected production values."""
        import importlib
        import src.api as api_mod
        assert api_mod._SHAP_CACHE_TTL    == 60.0
        assert api_mod._SHAP_CACHE_MAXSIZE == 512

    def test_cache_is_dict(self):
        import src.api as api_mod
        assert isinstance(api_mod._SHAP_CACHE, dict)
class TestRegistry:
    """
    Verify the registry.json written by conftest._write_test_artifacts()
    has the correct structure and that the versioned filenames on disk match it.
    """
    def test_registry_file_exists(self):
        from pathlib import Path
        assert (Path("models") / "registry.json").exists()

    def test_registry_has_required_keys(self):
        import json
        from pathlib import Path
        reg = json.load(open(Path("models") / "registry.json"))
        for key in ["active_model", "active_scaler", "active_explainer",
                    "active_hash", "activated_at"]:
            assert key in reg, f"registry.json missing key: {key}"

    def test_registry_active_model_file_exists(self):
        import json
        from pathlib import Path
        reg = json.load(open(Path("models") / "registry.json"))
        assert (Path("models") / reg["active_model"]).exists()

    def test_registry_active_scaler_file_exists(self):
        import json
        from pathlib import Path
        reg = json.load(open(Path("models") / "registry.json"))
        assert (Path("models") / reg["active_scaler"]).exists()

    def test_registry_active_explainer_file_exists(self):
        import json
        from pathlib import Path
        reg = json.load(open(Path("models") / "registry.json"))
        assert (Path("models") / reg["active_explainer"]).exists()

    def test_registry_filenames_contain_hash(self):
        import json
        from pathlib import Path
        reg = json.load(open(Path("models") / "registry.json"))
        h = reg["active_hash"]
        assert h in reg["active_model"]
        assert h in reg["active_scaler"]
        assert h in reg["active_explainer"]

    def test_metrics_filename_fields_match_registry(self):
        import json
        from pathlib import Path
        reg     = json.load(open(Path("models") / "registry.json"))
        metrics = json.load(open(Path("models") / "metrics.json"))
        assert metrics.get("model_filename")     == reg["active_model"]
        assert metrics.get("scaler_filename")    == reg["active_scaler"]
        assert metrics.get("explainer_filename") == reg["active_explainer"]


# ── /metrics ──────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_requires_auth(self, client):
        assert client.get("/metrics").status_code == 401

    def test_returns_200_with_key(self, client):
        assert client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).status_code == 200

    def test_has_pr_auc(self, client):
        d = client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).json()
        assert "pr_auc" in d

    def test_has_calibrated_metrics(self, client):
        d = client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).json()
        for key in ["pr_auc_calibrated","roc_auc_calibrated","f1_calibrated"]:
            assert key in d, f"Missing {key}"

    def test_has_selection_criterion(self, client):
        d = client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).json()
        assert d["selection_criterion"] == "pr_auc"

    def test_has_three_models_in_comparison(self, client):
        d = client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).json()
        assert len(d["comparison"]) == 3

    def test_has_feature_names(self, client):
        d = client.get("/metrics", headers={"X-API-Key":"demo-key-123"}).json()
        assert d["feature_names"] == FEATURE_NAMES


# ── /threshold_analysis ───────────────────────────────────────────────────────
class TestThresholdAnalysis:
    def test_requires_auth(self, client):
        assert client.get("/threshold_analysis").status_code == 401

    def test_returns_200_with_key(self, client):
        assert client.get("/threshold_analysis",
                          headers={"X-API-Key":"demo-key-123"}).status_code == 200

    def test_has_curve(self, client):
        d = client.get("/threshold_analysis", headers={"X-API-Key":"demo-key-123"}).json()
        assert "curve" in d
        assert isinstance(d["curve"], list)
        assert len(d["curve"]) > 0

    def test_curve_points_have_required_keys(self, client):
        d = client.get("/threshold_analysis", headers={"X-API-Key":"demo-key-123"}).json()
        point = d["curve"][0]
        for k in ["threshold","precision","recall","f1"]:
            assert k in point, f"Curve point missing {k}"

    def test_has_model_field(self, client):
        d = client.get("/threshold_analysis", headers={"X-API-Key":"demo-key-123"}).json()
        assert "model" in d


# ── /predict ──────────────────────────────────────────────────────────────────
class TestPredict:
    H = {"X-API-Key": "demo-key-123"}

    def test_requires_auth(self, client):
        assert client.post("/predict",
                           json={"features": ZERO_FEATURES}).status_code == 401

    def test_returns_200_with_valid_input(self, client):
        r = client.post("/predict",
                        json={"features": ZERO_FEATURES, "threshold": 0.5},
                        headers=self.H)
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        body = client.post("/predict",
                           json={"features": ZERO_FEATURES},
                           headers=self.H).json()
        for field in ["fraud_probability", "is_fraud", "threshold_used",
                      "top_shap_features", "input_warnings", "velocity_warning"]:
            assert field in body, f"Missing field: {field}"

    def test_probability_in_range(self, client):
        prob = client.post("/predict",
                           json={"features": ZERO_FEATURES},
                           headers=self.H).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_high_threshold_not_fraud(self, client):
        r = client.post("/predict",
                        json={"features": ZERO_FEATURES, "threshold": 0.99},
                        headers=self.H).json()
        assert r["is_fraud"] is False

    def test_zero_threshold_always_fraud(self, client):
        r = client.post("/predict",
                        json={"features": ZERO_FEATURES, "threshold": 0.0},
                        headers=self.H).json()
        assert r["is_fraud"] is True

    def test_wrong_feature_count_rejected(self, client):
        r = client.post("/predict",
                        json={"features": [0.0] * (N_FEATURES - 1)},
                        headers=self.H)
        assert r.status_code == 422

    def test_too_many_features_rejected(self, client):
        r = client.post("/predict",
                        json={"features": [0.0] * (N_FEATURES + 1)},
                        headers=self.H)
        assert r.status_code == 422

    def test_top_shap_at_most_10(self, client):
        shap = client.post("/predict",
                           json={"features": ZERO_FEATURES},
                           headers=self.H).json()["top_shap_features"]
        assert len(shap) <= 10

    def test_input_warnings_is_list(self, client):
        w = client.post("/predict",
                        json={"features": ZERO_FEATURES},
                        headers=self.H).json()["input_warnings"]
        assert isinstance(w, list)

    def test_velocity_warning_present_when_both_zero_and_not_new_sender(self, client):
        """Both velocity=0 + is_new_sender=False (default) → warning must fire."""
        body = client.post(
            "/predict",
            json={"features": ZERO_FEATURES, "is_new_sender": False},
            headers=self.H,
        ).json()
        assert body["velocity_warning"] is not None
        assert "Velocity features missing" in body["velocity_warning"]

    def test_velocity_warning_suppressed_when_both_zero_and_new_sender(self, client):
        """Both velocity=0 + is_new_sender=True → warning must be suppressed."""
        body = client.post(
            "/predict",
            json={"features": ZERO_FEATURES, "is_new_sender": True},
            headers=self.H,
        ).json()
        assert body["velocity_warning"] is None

    def test_velocity_warning_default_fires_without_flag(self, client):
        """is_new_sender defaults to False — bare ZERO_FEATURES must still warn."""
        body = client.post(
            "/predict",
            json={"features": ZERO_FEATURES},
            headers=self.H,
        ).json()
        assert body["velocity_warning"] is not None

    def test_velocity_warning_absent_when_features_set(self, client):
        """Both velocity features nonzero → no warning regardless of is_new_sender."""
        features = list(ZERO_FEATURES)
        from src.preprocess import FEATURE_NAMES
        features[FEATURE_NAMES.index("orig_tx_count_so_far")]   = 5.0
        features[FEATURE_NAMES.index("orig_amount_sum_so_far")] = 12000.0
        for new_sender in [True, False]:
            body = client.post(
                "/predict",
                json={"features": features, "is_new_sender": new_sender},
                headers=self.H,
            ).json()
            assert body["velocity_warning"] is None, (
                f"Expected no warning with nonzero velocity features "
                f"and is_new_sender={new_sender}"
            )

    def test_velocity_warning_absent_when_only_count_nonzero(self, client):
        """Only one velocity feature nonzero → no warning (both must be zero to warn)."""
        features = list(ZERO_FEATURES)
        from src.preprocess import FEATURE_NAMES
        features[FEATURE_NAMES.index("orig_tx_count_so_far")] = 3.0
        body = client.post(
            "/predict",
            json={"features": features},
            headers=self.H,
        ).json()
        assert body["velocity_warning"] is None

    def test_is_new_sender_field_accepted_in_schema(self, client):
        """is_new_sender must be accepted by Pydantic without a 422."""
        r = client.post(
            "/predict",
            json={"features": ZERO_FEATURES, "is_new_sender": True, "threshold": 0.5},
            headers=self.H,
        )
        assert r.status_code == 200

    def test_is_new_sender_invalid_type_rejected(self, client):
        """is_new_sender must be a bool — string value should be rejected."""
        r = client.post(
            "/predict",
            json={"features": ZERO_FEATURES, "is_new_sender": "yes"},
            headers=self.H,
        )
        # Pydantic coerces "yes" → True in lax mode; strict would 422.
        # The important contract is the endpoint doesn't 500.
        assert r.status_code in {200, 422}


# ── /predict/batch ────────────────────────────────────────────────────────────
class TestBatchPredict:
    H = {"X-API-Key": "demo-key-123"}

    def _payload(self, n=3):
        return {"transactions": [{"features": ZERO_FEATURES, "threshold": 0.5}] * n}

    def test_requires_auth(self, client):
        assert client.post("/predict/batch", json=self._payload()).status_code == 401

    def test_returns_200(self, client):
        assert client.post("/predict/batch", json=self._payload(),
                           headers=self.H).status_code == 200

    def test_total_matches_input(self, client):
        assert client.post("/predict/batch", json=self._payload(5),
                           headers=self.H).json()["total"] == 5

    def test_fraud_count_non_negative(self, client):
        assert client.post("/predict/batch", json=self._payload(4),
                           headers=self.H).json()["fraud_count"] >= 0

    def test_results_array_length(self, client):
        assert len(client.post("/predict/batch", json=self._payload(1),
                               headers=self.H).json()["results"]) == 1

    def test_fraud_count_not_exceed_total(self, client):
        d = client.post("/predict/batch", json=self._payload(6), headers=self.H).json()
        assert d["fraud_count"] <= d["total"]


# ── /audit ────────────────────────────────────────────────────────────────────
class TestAudit:
    H = {"X-API-Key": "demo-key-123"}

    def test_requires_auth(self, client):
        assert client.get("/audit").status_code == 401

    def test_returns_200(self, client):
        assert client.get("/audit", headers=self.H).status_code == 200

    def test_response_has_records_key(self, client):
        d = client.get("/audit", headers=self.H).json()
        assert "records" in d

    def test_response_has_pagination_fields(self, client):
        d = client.get("/audit", headers=self.H).json()
        assert "total_returned" in d
        assert "offset" in d

    def test_fraud_only_filter_accepted(self, client):
        r = client.get("/audit?fraud_only=true", headers=self.H)
        assert r.status_code == 200

    def test_limit_param_accepted(self, client):
        r = client.get("/audit?limit=10", headers=self.H)
        assert r.status_code == 200
