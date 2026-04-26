"""
conftest.py
-----------
Session-scoped setup: writes real picklable sklearn artifacts to models/
so api.py's module-level _load() calls succeed without any patching.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocess import FEATURE_NAMES, N_FEATURES, SCALE_COLS


# ── Must be at MODULE scope so pickle can serialise it ────────────────────────
class _StubExplainer:
    """Minimal SHAP-compatible explainer that returns zeros."""
    def shap_values(self, X):
        return np.zeros((len(X), N_FEATURES))


def _write_test_artifacts() -> None:
    models = Path("models")
    models.mkdir(exist_ok=True)

    # Use a fixed test hash so filenames are deterministic across test runs
    TEST_HASH = "testabcd1234"

    # Real DummyClassifier — predict_proba returns [[0.8, 0.2]] always
    clf = DummyClassifier(strategy="constant", constant=0)
    clf.fit(np.zeros((2, N_FEATURES)), np.array([0, 1]))

    # Real StandardScaler fitted on random data matching SCALE_COLS width
    sc = StandardScaler()
    sc.fit(np.random.randn(20, len(SCALE_COLS)))
    sc.feature_names_in_ = np.array(SCALE_COLS)

    model_f    = f"model_{TEST_HASH}.pkl"
    scaler_f   = f"scaler_{TEST_HASH}.pkl"
    explainer_f = f"explainer_{TEST_HASH}.pkl"

    metrics = {
        "best_model": "XGBoost",
        "selection_criterion": "pr_auc",
        "dataset": "paysim_sample.csv",
        "data_hash": TEST_HASH,
        "trained_at": "2024-01-15T10:00:00+00:00",
        "model_filename":     model_f,
        "scaler_filename":    scaler_f,
        "explainer_filename": explainer_f,
        "roc_auc": 0.99, "pr_auc": 0.87,
        "precision": 0.95, "recall": 0.88, "f1": 0.91,
        "roc_auc_calibrated": 0.98, "pr_auc_calibrated": 0.86,
        "precision_calibrated": 0.94, "recall_calibrated": 0.87,
        "f1_calibrated": 0.90,
        "confusion_matrix": [[199400, 20], [15, 107]],
        "total_transactions": 1_000_000,
        "fraud_count": 535,
        "fraud_pct": 0.0535,
        "feature_names": FEATURE_NAMES,
        "global_shap": {f: round(0.01 * i, 4) for i, f in enumerate(FEATURE_NAMES)},
        "comparison": {
            "RandomForest": {"roc_auc":0.97,"pr_auc":0.83,"precision":0.93,
                             "recall":0.85,"f1":0.89,"confusion_matrix":[]},
            "XGBoost":      {"roc_auc":0.99,"pr_auc":0.87,"precision":0.95,
                             "recall":0.88,"f1":0.91,"confusion_matrix":[]},
            "LightGBM":     {"roc_auc":0.98,"pr_auc":0.85,"precision":0.94,
                             "recall":0.87,"f1":0.90,"confusion_matrix":[]},
        },
    }

    curve = [
        {
            "threshold": round(t, 2),
            "precision": round(0.95 - t * 0.3, 4),
            "recall":    round(1.0 - t, 4),
            "f1":        round(0.85 - t * 0.1, 4),
        }
        for t in [i / 10 for i in range(1, 10)]
    ]

    # Write versioned pkl files
    pickle.dump(clf,              open(models / model_f,    "wb"))
    pickle.dump(sc,               open(models / scaler_f,   "wb"))
    pickle.dump(_StubExplainer(), open(models / explainer_f,"wb"))

    # Write registry.json pointing to this test version
    registry = {
        "active_model":     model_f,
        "active_scaler":    scaler_f,
        "active_explainer": explainer_f,
        "active_hash":      TEST_HASH,
        "activated_at":     "2024-01-15T10:00:00+00:00",
    }
    json.dump(registry, open(models / "registry.json",       "w"), indent=2)
    json.dump(metrics,  open(models / "metrics.json",        "w"), indent=2)
    json.dump(curve,    open(models / "threshold_curve.json","w"))


_write_test_artifacts()
