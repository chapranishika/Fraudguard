"""
train.py
--------
Trains and compares RandomForest, XGBoost (tuned), and LightGBM on the PaySim
financial transactions dataset (Synthetic_Financial_datasets_log.csv).

Feature engineering (src/preprocess.py — 17 features total):
  Balance error signals:
    balance_delta_orig / dest : accounting discrepancy (key fraud signal)
    balance_zero_orig / dest  : binary flags for wiped/empty balances
  Temporal velocity signals:
    orig_tx_count_so_far      : cumulative send count for this nameOrig
    orig_amount_sum_so_far    : cumulative send amount for this nameOrig
  Raw financials + type one-hot (5 types)

Pipeline (leakage-free):
  1. Load raw CSV → engineer_features() on full dataset sorted by step
     (velocity features computed here — expanding window, no future leakage)
  2. train_test_split FIRST (stratified 80/20) — split before any scaling
  3. fit_scaler(X_train_raw) — external scaler for API inference; sees train only
  4. apply_scaler() to both partitions — test set scaled with train statistics
  5. Optuna hyperparameter tuning for XGBoost (20 trials):
     - Objective operates on raw X_train / y_train (pre-ADASYN)
     - imblearn Pipeline([ADASYN → StandardScaler → XGBClassifier]) evaluated
       via cross_val_score with StratifiedKFold — ADASYN is applied inside each
       fold's training split only; validation folds are always pristine real data
     - Search space: learning_rate (log), max_depth, subsample, colsample_bytree
  6. 5-fold StratifiedKFold CV on tuned pipeline — reports mean ± std PR-AUC
  7. Fit full pipeline on X_train (post-split, pre-ADASYN); ADASYN handled inside
  8. Compare XGBoost vs RF vs LightGBM on held-out test set
  9. Calibrate best XGBoost estimator (CalibratedClassifierCV, isotonic, cv=5)
  10. Build SHAP TreeExplainer on underlying estimator (not calibration wrapper)
  11. Save versioned artifacts (model_<hash>.pkl, scaler_<hash>.pkl,
      explainer_<hash>.pkl) and write models/registry.json.

Usage:
    python train.py
    python train.py --csv path/to/paysim.csv
    python train.py --trials 40          # more Optuna trials
    python train.py --no-tune            # skip Optuna, use XGBoost defaults
"""

import argparse
import hashlib
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_raw, fit_scaler, apply_scaler, FEATURE_NAMES

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial logs

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv", default="paysim_sample.csv",
    help="Path to PaySim CSV (default: paysim_sample.csv)"
)
parser.add_argument(
    "--trials", type=int, default=20,
    help="Number of Optuna trials for XGBoost tuning (default: 20)"
)
parser.add_argument(
    "--no-tune", action="store_true",
    help="Skip Optuna tuning and use XGBoost default hyperparameters"
)
args = parser.parse_args()

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = args.csv

# ── 1. Load & engineer features (no scaling yet) ───────────────────────────────
# load_raw() returns unscaled X_raw and y.
# Velocity features (orig_tx_count_so_far, orig_amount_sum_so_far) are computed
# inside engineer_features() on the full dataset sorted by step, so each row's
# cumulative count/sum only reflects its own chronological past — no future leakage
# within the feature. The scaler is NOT fitted here.
print(f"Loading dataset: {CSV_PATH}")
X_raw, y = load_raw(CSV_PATH)

# Dataset fingerprint for audit trail (computed before scaling)
data_hash = hashlib.md5(pd.util.hash_pandas_object(X_raw).values.tobytes()).hexdigest()[:12]
train_ts   = datetime.now(timezone.utc).isoformat()

print(f"  Rows: {len(X_raw):,} | Fraud: {int(y.sum()):,} ({y.mean()*100:.4f}%)")
print(f"  Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")

# ── 2. Split BEFORE scaling — prevents StandardScaler leakage ──────────────────
# Fitting the scaler on the full dataset before splitting exposes test-set
# distribution (mean, std) to the scaler during training — a form of data leakage.
# Correct contract: split first, fit scaler on X_train_raw only, transform both.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# Fit scaler on train partition ONLY — test set is invisible at this point
scaler  = fit_scaler(X_train_raw)
X_train = apply_scaler(X_train_raw, scaler)
X_test  = apply_scaler(X_test_raw,  scaler)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Scaler fitted on train split only — no test-set leakage")

# ── 3. Optuna hyperparameter tuning for XGBoost ────────────────────────────────
# The objective function operates on raw X_train / y_train (pre-ADASYN,
# pre-scaling). An imblearn Pipeline containing ADASYN → StandardScaler →
# XGBClassifier is evaluated with cross_val_score and StratifiedKFold.
#
# Why Pipeline + cross_val_score instead of manual fold loop on X_train_res:
#   The previous approach called ADASYN on the full X_train_res before CV, so
#   synthetic minority samples generated from fold-k's training rows could appear
#   in fold-k's validation set — inflating every CV score. imblearn Pipeline
#   applies fit_resample() only inside each fold's fit() call; the validation
#   split is always pristine real transactions. This is the correct contract.
#
# Search space (expanded from previous version):
#   learning_rate    : log-uniform 0.01–0.3 — small values win on imbalanced data
#   max_depth        : 3–8 integer — depth controls overfitting on minority class
#   subsample        : 0.6–1.0 — row subsampling adds regularisation
#   colsample_bytree : 0.5–1.0 — column subsampling, previously fixed at 0.8

N_TRIALS  = args.trials
TUNE_XGB  = not args.no_tune
OPTUNA_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

optuna_results: dict = {}

if TUNE_XGB:
    print(f"\n── Optuna XGBoost tuning ({N_TRIALS} trials, Pipeline + 3-fold CV on X_train) ──")
    print("   ADASYN runs inside each fold — validation folds are always real data.")

    def _xgb_objective(trial: optuna.Trial) -> float:
        """
        Build a fresh imblearn Pipeline per trial and score it with cross_val_score.
        X_train / y_train are raw (pre-ADASYN). The pipeline handles resampling
        and scaling inside each CV fold, keeping validation folds clean.
        """
        params = {
            "n_estimators":      300,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric":       "logloss",
            "random_state":      42,
            "n_jobs":            -1,
        }

        pipe = ImbPipeline([
            ("adasyn",  ADASYN(random_state=42, n_neighbors=5)),
            ("scaler",  StandardScaler()),
            ("xgb",     XGBClassifier(**params)),
        ])

        # scoring="average_precision" computes PR-AUC using predict_proba[:,1]
        # needs_proba is handled automatically by cross_val_score for pipelines
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=OPTUNA_CV,
            scoring="average_precision",
            n_jobs=1,   # outer parallelism; XGB already uses n_jobs=-1
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(_xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params       = study.best_params
    best_trial_score  = round(study.best_value, 4)

    print(f"  Best trial PR-AUC (3-fold CV, Pipeline): {best_trial_score}")
    print(f"  Best params: {best_params}")

    optuna_results = {
        "n_trials":         N_TRIALS,
        "best_trial_score": best_trial_score,
        "best_params":      best_params,
        "all_trial_scores": sorted(
            [round(t.value, 4) for t in study.trials if t.value is not None],
            reverse=True,
        ),
    }
else:
    print("\n── Skipping Optuna tuning (--no-tune flag set) ──")
    best_params = {}

# ── 4. 5-fold CV variance report on tuned pipeline ────────────────────────────
# Uses the same Pipeline structure as the Optuna objective.
# Operates on X_train / y_train (pre-ADASYN) — ADASYN fires inside each fold.
# Reports mean ± std PR-AUC to quantify metric variance honestly.
# ~107 real fraud cases per validation fold → variance matters.

print(f"\n── 5-fold StratifiedKFold CV variance report (tuned XGBoost Pipeline) ──")
REPORT_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_cv_params = {
    "n_estimators":      300,
    "learning_rate":     best_params.get("learning_rate", 0.05),
    "max_depth":         best_params.get("max_depth", 6),
    "subsample":         best_params.get("subsample", 0.8),
    "colsample_bytree":  best_params.get("colsample_bytree", 0.8),
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}

report_pipe = ImbPipeline([
    ("adasyn", ADASYN(random_state=42, n_neighbors=5)),
    ("scaler", StandardScaler()),
    ("xgb",    XGBClassifier(**xgb_cv_params)),
])

cv_scores = cross_val_score(
    report_pipe, X_train, y_train,
    cv=REPORT_CV,
    scoring="average_precision",
    n_jobs=1,
)

for fold_num, s in enumerate(cv_scores, start=1):
    print(f"  Fold {fold_num}: PR-AUC = {s:.4f}")

cv_mean = float(np.mean(cv_scores))
cv_std  = float(np.std(cv_scores))
print(f"\n  ✅ CV PR-AUC: {cv_mean:.4f} ± {cv_std:.4f}  (mean ± std, 5-fold)")
print(f"     ADASYN applied inside each fold — validation folds are real data only.")

cv_report = {
    "n_folds":     5,
    "fold_scores": [round(float(s), 4) for s in cv_scores],
    "mean_pr_auc": round(cv_mean, 4),
    "std_pr_auc":  round(cv_std, 4),
    "params_used": xgb_cv_params,
    "pipeline":    "ADASYN → StandardScaler → XGBClassifier (inside each fold)",
}

# ── 5. Fit final XGBoost on full training set ──────────────────────────────────
# Apply ADASYN once to the full training split, then fit the tuned XGBoost.
# This is separate from the pipeline because:
#   a) CalibratedClassifierCV needs the raw fitted XGBClassifier as its estimator
#   b) The external scaler artifact (saved separately) is used by the API at
#      inference time — it must be fitted outside the pipeline
print("\nFitting final XGBoost on full training set...")
adasyn_final = ADASYN(random_state=42, n_neighbors=5)
X_train_res, y_train_res = adasyn_final.fit_resample(X_train, y_train)
print(f"  Train: {len(y_train):,} → {len(y_train_res):,} after ADASYN")

xgb_final = XGBClassifier(**xgb_cv_params)
xgb_final.fit(X_train_res, y_train_res)

# ── 6. Model comparison on held-out test set ───────────────────────────────────
# XGBoost uses best tuned params. RF and LightGBM use fixed defaults for comparison.
# All models evaluated on X_test (scaled with train-only statistics, no leakage).
print(f"\n{'─'*66}")
print(f"{'Model':<20} {'ROC-AUC':>8} {'PR-AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"{'─'*66}")

candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    ),
    "XGBoost (tuned)": xgb_final,   # already fitted above
    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    ),
}

results       = {}
trained_models = {}

for name, clf in candidates.items():
    if name != "XGBoost (tuned)":
        clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    roc  = round(roc_auc_score(y_test, y_prob), 4)
    pr   = round(average_precision_score(y_test, y_prob), 4)
    prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
    rec  = round(recall_score(y_test, y_pred, zero_division=0), 4)
    f1   = round(f1_score(y_test, y_pred, zero_division=0), 4)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "roc_auc": roc, "pr_auc": pr,
        "precision": prec, "recall": rec,
        "f1": f1, "confusion_matrix": cm,
    }
    trained_models[name] = clf
    print(f"{name:<20} {roc:>8} {pr:>8} {prec:>10} {rec:>8} {f1:>8}")

# ── 7. Select best by PR-AUC ───────────────────────────────────────────────────
best_name      = max(results, key=lambda m: results[m]["pr_auc"])
best_model_raw = trained_models[best_name]
print(
    f"\n✅ Best model: {best_name} "
    f"(PR-AUC={results[best_name]['pr_auc']}, "
    f"ROC-AUC={results[best_name]['roc_auc']})"
)

# ── 7b. Calibrate ─────────────────────────────────────────────────────────────
# Isotonic regression corrects tree-ensemble overconfidence.
# cv=5: calibration is fit on held-out folds → no leakage.
# The raw fitted XGBClassifier (xgb_final or best comparison model) is the
# estimator — not the imblearn Pipeline. CalibratedClassifierCV needs a bare
# classifier, not a pipeline, so it can access .calibrated_classifiers_[i].estimator
# for SHAP TreeExplainer unwrapping later.
print("Calibrating (CalibratedClassifierCV, isotonic, cv=5)...")
best_model = CalibratedClassifierCV(
    estimator=best_model_raw, method="isotonic", cv=5
)
best_model.fit(X_train_res, y_train_res)

y_pred_cal = best_model.predict(X_test)
y_prob_cal = best_model.predict_proba(X_test)[:, 1]

cal = {
    "roc_auc_calibrated":    round(roc_auc_score(y_test, y_prob_cal), 4),
    "pr_auc_calibrated":     round(average_precision_score(y_test, y_prob_cal), 4),
    "precision_calibrated":  round(precision_score(y_test, y_pred_cal, zero_division=0), 4),
    "recall_calibrated":     round(recall_score(y_test, y_pred_cal, zero_division=0), 4),
    "f1_calibrated":         round(f1_score(y_test, y_pred_cal, zero_division=0), 4),
}
results[best_name].update(cal)
print(
    f"  Calibrated → PR-AUC: {cal['pr_auc_calibrated']}  "
    f"F1: {cal['f1_calibrated']}  "
    f"Recall: {cal['recall_calibrated']}"
)

# ── 7c. Threshold curve ────────────────────────────────────────────────────────
# Expose the full precision-recall curve so ops teams can choose a threshold
# based on their actual false-positive budget — not a hardcoded 0.5.
print("Computing PR curve for /threshold_analysis endpoint...")
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, y_prob_cal)

# Store as compact list of dicts (skip last point where recall=0)
threshold_curve = [
    {
        "threshold": round(float(t), 4),
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(2*p*r/(p+r+1e-9), 4),
    }
    for t, p, r in zip(thresh_curve, prec_curve[:-1], rec_curve[:-1])
    if t > 0.0
]

with open(MODELS_DIR / "threshold_curve.json", "w") as f:
    json.dump(threshold_curve, f)
print(f"  Saved {len(threshold_curve)} threshold points")

# ── 8. SHAP explainer ─────────────────────────────────────────────────────────
# CalibratedClassifierCV wraps estimators in .calibrated_classifiers_[i].estimator
# TreeExplainer needs the raw tree object, not the calibration wrapper.
print("Building SHAP TreeExplainer...")
_raw_for_shap = best_model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(_raw_for_shap)

sample_idx  = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
X_sample    = X_test.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

shap_fraud  = shap_values[1] if isinstance(shap_values, list) else shap_values
mean_abs_shap = np.abs(shap_fraud).mean(axis=0)
global_shap   = dict(zip(FEATURE_NAMES, mean_abs_shap.tolist()))

# ── 9. Persist artifacts ───────────────────────────────────────────────────────
# Use data_hash as the version tag — content-addressed, deterministic.
# Retraining on the same dataset produces the same hash; different data → new files.
# The artifact triplet (model, scaler, explainer) shares one hash so they are
# unambiguously linked. Old versions are preserved; nothing is overwritten.
model_filename    = f"model_{data_hash}.pkl"
scaler_filename   = f"scaler_{data_hash}.pkl"
explainer_filename = f"explainer_{data_hash}.pkl"

with open(MODELS_DIR / model_filename,    "wb") as f: pickle.dump(best_model, f)
with open(MODELS_DIR / scaler_filename,   "wb") as f: pickle.dump(scaler, f)
with open(MODELS_DIR / explainer_filename,"wb") as f: pickle.dump(explainer, f)

best_r = results[best_name]
all_metrics = {
    # Provenance
    "dataset":            CSV_PATH,
    "data_hash":          data_hash,
    "trained_at":         train_ts,
    "best_model":         best_name,
    "selection_criterion":"pr_auc",
    "feature_names":      FEATURE_NAMES,
    # Artifact filenames — stored in metrics so registry and API can verify alignment
    "model_filename":     model_filename,
    "scaler_filename":    scaler_filename,
    "explainer_filename": explainer_filename,
    # Dataset stats
    "total_transactions": int(len(X_raw)),
    "fraud_count":        int(y.sum()),
    "fraud_pct":          round(float(y.mean()) * 100, 4),
    # Hyperparameter tuning results
    "optuna_tuning":      optuna_results,   # empty dict if --no-tune
    # Cross-validation variance report (key for interview credibility)
    "cross_validation":   cv_report,
    # Model comparison
    "comparison": results,
    # Global SHAP
    "global_shap": global_shap,
    # Best model — raw (pre-calibration)
    "roc_auc":   best_r["roc_auc"],
    "pr_auc":    best_r["pr_auc"],
    "precision": best_r["precision"],
    "recall":    best_r["recall"],
    "f1":        best_r["f1"],
    "confusion_matrix": best_r["confusion_matrix"],
    # Best model — calibrated (what the deployed model actually achieves)
    "roc_auc_calibrated":   cal["roc_auc_calibrated"],
    "pr_auc_calibrated":    cal["pr_auc_calibrated"],
    "precision_calibrated": cal["precision_calibrated"],
    "recall_calibrated":    cal["recall_calibrated"],
    "f1_calibrated":        cal["f1_calibrated"],
}

with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# ── 10. Write registry.json ────────────────────────────────────────────────────
# The registry is the single source of truth for which version is active.
# To roll back: edit registry.json "active_model" to any previous hash.
# api.py reads this at startup — never loads a hardcoded "model.pkl".
registry = {
    "active_model":    model_filename,
    "active_scaler":   scaler_filename,
    "active_explainer": explainer_filename,
    "active_hash":     data_hash,
    "activated_at":    train_ts,
}
with open(MODELS_DIR / "registry.json", "w") as f:
    json.dump(registry, f, indent=2)

print(
    f"\n✅ Saved versioned artifacts:\n"
    f"   models/{model_filename}\n"
    f"   models/{scaler_filename}\n"
    f"   models/{explainer_filename}\n"
    f"   models/metrics.json\n"
    f"   models/threshold_curve.json\n"
    f"   models/registry.json  ← active version: {data_hash}"
)
print(f"\n   Model: {best_name} | PR-AUC (calibrated): {cal['pr_auc_calibrated']}")
print(f"   To roll back: edit models/registry.json 'active_hash' to a previous hash.")
