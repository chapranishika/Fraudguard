"""
src/api.py
----------
FastAPI inference service for the FraudGuard fraud detection pipeline.

Startup (lifespan):
  Reads models/registry.json → loads the active versioned artifact triplet
  (model_<hash>.pkl, scaler_<hash>.pkl, explainer_<hash>.pkl).
  Falls back to unversioned model.pkl for backwards compatibility.
  Initialises SQLite DB with WAL mode.

Endpoints (all except /health require X-API-Key header):
  POST /predict             → single transaction prediction + SHAP values
                              async; model inference + SHAP run in thread pool
                              Rate limited: 60 requests/minute per IP
  POST /predict/batch       → bulk prediction
                              async; each prediction offloaded to thread pool
                              Rate limited: 10 requests/minute per IP
  GET  /health              → liveness only — {"status": "ok"} — no auth
  GET  /info                → model provenance (trained_at, data_hash, pr_auc) — auth required
  GET  /metrics             → full model metrics from metrics.json
  GET  /threshold_analysis  → full PR curve for ops threshold selection
  GET  /audit               → paginated SQLite audit log
  GET  /audit?fraud_only=true → only flagged predictions
  GET  /random_sample       → random row from paysim_sample.csv

Async concurrency:
  /predict and /predict/batch are async def. All blocking work (model.predict_proba,
  explainer.shap_values, SQLite writes) is offloaded via run_in_threadpool() so the
  event loop is never blocked. Multiple concurrent requests can be served without
  one SHAP computation stalling all others.

Velocity feature schema:
  TransactionInput accepts orig_tx_count_so_far and orig_amount_sum_so_far with
  default=0.0. A model_validator detects when both are zero and attaches a warning
  to the response: predictions are still served, but callers are notified that the
  sequential fraud signal is absent.

Security:
  API key is read from the API_KEY environment variable.
  Loaded from .env via python-dotenv before os.environ is evaluated.
  Falls back to "demo-key-123" if not set (local dev only).

Database:
  SQLite with WAL (Write-Ahead Logging) mode and check_same_thread=False.
  DB writes run in thread pool — never block the event loop.
  For multi-process production scale, migrate to PostgreSQL + asyncpg.

Feature schema (17 floats, in order):
  amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
  balance_delta_orig, balance_delta_dest, balance_zero_orig, balance_zero_dest,
  orig_tx_count_so_far, orig_amount_sum_so_far, step,
  type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER

Run locally:
    uvicorn src.api:app --reload --port 8000
"""

# ── load .env FIRST — must happen before os.environ.get("API_KEY") below ──────
from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ; silently no-ops if file is absent

import asyncio
import json
import os
import pickle
import sqlite3
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from src.preprocess import (
    FEATURE_NAMES, N_FEATURES, SCALE_COLS,
    apply_scaler, engineer_features, make_dataframe,
)

warnings.filterwarnings("ignore")

MODELS_DIR  = Path("models")
DB_PATH     = Path("transactions.db")
SAMPLE_CSV  = Path("paysim_sample.csv")

# ── Database ───────────────────────────────────────────────────────────────────
def _init_db() -> None:
    """Create the predictions table and enable WAL mode for concurrent writers."""
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id          TEXT PRIMARY KEY,
                amount      REAL,
                probability REAL,
                is_fraud    INTEGER,
                timestamp   DATETIME,
                tx_type     TEXT
            )
            """
        )
        conn.commit()

# _init_db() is called inside lifespan() at startup — not at module level.
# This ensures the DB is initialised after FastAPI's startup sequence, not during import.

@contextmanager
def _get_db():
    """
    Yield a WAL-enabled sqlite3 connection safe for background threads.
    check_same_thread=False: required when BackgroundTasks runs writes in a
    thread-pool worker separate from the main request thread.
    WAL mode: allows concurrent readers + one writer without full table locks.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _write_prediction(amount: float, probability: float, is_fraud: bool, tx_type: str) -> None:
    with _get_db() as conn:
        conn.execute(
            "INSERT INTO predictions (id, amount, probability, is_fraud, timestamp, tx_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), amount, round(probability, 6),
             int(is_fraud), datetime.now(timezone.utc).isoformat(), tx_type),
        )

# ── Model artifacts ────────────────────────────────────────────────────────────
def _load(filename: str):
    """Load a pickle file from MODELS_DIR. Raises RuntimeError if missing."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise RuntimeError(
            f"Artifact '{filename}' not found in {MODELS_DIR}/. "
            "Run `python train.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_artifacts() -> tuple[str, str, str]:
    """
    Determine which artifact filenames to load by reading registry.json.

    Resolution order:
      1. models/registry.json  →  use active_model / active_scaler / active_explainer
      2. Fallback               →  models/model.pkl, scaler.pkl, explainer.pkl
         (backwards-compatible with runs of train.py before versioning was added)
      3. Neither present        →  raise RuntimeError with clear instructions

    Returns (model_filename, scaler_filename, explainer_filename).
    """
    registry_path = MODELS_DIR / "registry.json"

    if registry_path.exists():
        with open(registry_path) as f:
            reg = json.load(f)

        model_f    = reg.get("active_model")
        scaler_f   = reg.get("active_scaler")
        explainer_f = reg.get("active_explainer")

        # Validate all three keys are present and non-empty
        if not all([model_f, scaler_f, explainer_f]):
            raise RuntimeError(
                "registry.json is malformed — expected keys: "
                "active_model, active_scaler, active_explainer. "
                f"Got: {reg}"
            )

        active_hash = reg.get("active_hash", "unknown")
        print(f"  Registry found → active version: {active_hash}")
        print(f"  Loading: {model_f}, {scaler_f}, {explainer_f}")
        return model_f, scaler_f, explainer_f

    # Fallback: unversioned artifacts from a pre-registry training run
    fallback = ("model.pkl", "scaler.pkl", "explainer.pkl")
    fallback_present = all((MODELS_DIR / f).exists() for f in fallback)

    if fallback_present:
        print(
            "  ⚠ models/registry.json not found — "
            "falling back to unversioned model.pkl / scaler.pkl / explainer.pkl. "
            "Run `python train.py` to generate a versioned registry."
        )
        return fallback

    raise RuntimeError(
        "No model artifacts found. "
        "Run `python train.py` to train and register a model."
    )


# Module-level globals — populated by lifespan() before any request is served.
# Kept as globals so _predict_one() and _validate_features() can reference them
# without dependency injection threading through every function call.
model     = None
scaler    = None
explainer = None
_metrics  = {}
_threshold_curve = []
_df_sample = None

# Semaphore limiting concurrent thread-pool slots used by /predict/batch.
# Starlette's default thread pool has 40 workers. Without a semaphore, a single
# batch of 100 transactions fires 100 run_in_threadpool() calls simultaneously,
# consuming 100% of the pool and starving /predict and other endpoints.
# Semaphore(10) caps batch concurrency at 10 threads, leaving headroom for
# concurrent single-predict requests on the same server.
_BATCH_SEMAPHORE = asyncio.Semaphore(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: load all artifacts before accepting requests.

    Replaces the deprecated @app.on_event("startup") pattern.
    Reads registry.json to determine which versioned artifact files to load,
    with a fallback to unversioned model.pkl for backwards compatibility.
    All errors surface clearly at startup rather than silently at first request.
    """
    global model, scaler, explainer, _metrics, _threshold_curve, _df_sample

    print("FraudGuard API — loading model artifacts...")

    model_f, scaler_f, explainer_f = _resolve_artifacts()

    model     = _load(model_f)
    scaler    = _load(scaler_f)
    explainer = _load(explainer_f)

    with open(MODELS_DIR / "metrics.json") as f:
        _metrics = json.load(f)

    with open(MODELS_DIR / "threshold_curve.json") as f:
        _threshold_curve = json.load(f)

    try:
        _df_sample = pd.read_csv(SAMPLE_CSV)
    except Exception:
        _df_sample = None
        print("  ⚠ paysim_sample.csv not found — /random_sample endpoint disabled.")

    active_hash = _metrics.get("data_hash", "unknown")
    best        = _metrics.get("best_model", "unknown")
    pr          = _metrics.get("pr_auc_calibrated", "unknown")
    print(f"  ✅ Loaded {best} | hash={active_hash} | PR-AUC (cal)={pr}")

    _init_db()

    yield  # ← application runs here; everything after yield is teardown

    print("FraudGuard API — shutting down.")

# ── App ────────────────────────────────────────────────────────────────────────
# Rate limiter — keyed on client IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="FraudGuard API",
    description="Real-time fraud detection on PaySim financial data. SHAP explainability included.",
    version="2.0.0",
    lifespan=lifespan,
)

# Attach limiter to app state and register the 429 exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── API key auth ───────────────────────────────────────────────────────────────
# Read from environment — never hardcode secrets in source.
# Local dev: set API_KEY in a .env file (see .env.example).
# Production: inject via your secrets manager / environment variables.
_API_KEY       = os.environ.get("API_KEY", "demo-key-123")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)):
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return key

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

# Indices of the velocity features within FEATURE_NAMES — resolved once at import.
# Used by TransactionInput.validate_velocity() to avoid hardcoded magic numbers.
_VELOCITY_INDICES = {
    "orig_tx_count_so_far":   FEATURE_NAMES.index("orig_tx_count_so_far"),
    "orig_amount_sum_so_far": FEATURE_NAMES.index("orig_amount_sum_so_far"),
}
_VELOCITY_WARNING = (
    "Velocity features missing (orig_tx_count_so_far=0 and orig_amount_sum_so_far=0); "
    "prediction precision may be degraded. In production, supply running counters "
    "from a feature store keyed on the sender account."
)
_VELOCITY_NEW_SENDER_NOTE = (
    "New sender confirmed (is_new_sender=True); velocity features are correctly zero. "
    "No feature store signal is missing."
)


class TransactionInput(BaseModel):
    """
    17-feature PaySim transaction vector.

    Velocity features (orig_tx_count_so_far, orig_amount_sum_so_far) default to 0.0
    for callers that do not have access to a feature store. The model_validator
    distinguishes two cases when both are zero:

      is_new_sender=True  → zeros are semantically correct (first-ever transaction).
                            No warning is issued; the caller has confirmed the account
                            history is genuinely empty.

      is_new_sender=False → zeros may mean a missing feature store lookup. A warning
                            is attached to the response so the calling system can log
                            it or route the transaction for manual review.

    is_new_sender is a caller-supplied signal, not a model feature — it is not
    included in the 17-feature vector sent to the model.
    """
    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=(
            f"{N_FEATURES} values in order: "
            + ", ".join(FEATURE_NAMES)
        ),
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    is_new_sender: bool = Field(
        default=False,
        description=(
            "Set to True when the calling system has confirmed this is the sender's "
            "first transaction and velocity features are legitimately zero. "
            "When True and both velocity features are 0.0, the degraded-precision "
            "warning is suppressed. When False (default), zero velocity features "
            "trigger a warning indicating a possible missing feature store lookup."
        ),
    )

    # Internal field: set by model_validator, consumed by _predict_one.
    # exclude=True keeps it out of the request schema shown in Swagger.
    velocity_warning: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_velocity(self) -> "TransactionInput":
        """
        Resolve velocity warning based on feature values and is_new_sender flag.

        Decision matrix:
          both zero + is_new_sender=True  → no warning (zeros are correct)
          both zero + is_new_sender=False → warning (possible missing feature store)
          either nonzero                  → no warning (values are populated)
        """
        count_idx  = _VELOCITY_INDICES["orig_tx_count_so_far"]
        amount_idx = _VELOCITY_INDICES["orig_amount_sum_so_far"]
        both_zero  = (
            self.features[count_idx]  == 0.0 and
            self.features[amount_idx] == 0.0
        )

        if both_zero and not self.is_new_sender:
            self.velocity_warning = _VELOCITY_WARNING
        # both_zero + is_new_sender=True → velocity_warning stays None (no warning)
        # either nonzero                 → velocity_warning stays None (no warning)
        return self


class BatchInput(BaseModel):
    transactions: list[TransactionInput] = Field(
        ...,
        max_length=100,
        description="Maximum 100 transactions per request. Use multiple requests for larger batches.",
    )


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    top_shap_features: dict[str, float]
    input_warnings: list[str]
    velocity_warning: str | None = Field(
        default=None,
        description=(
            "Present when orig_tx_count_so_far and orig_amount_sum_so_far are both 0. "
            "Indicates the sequential fraud signal is absent from this prediction."
        ),
    )


class BatchOutput(BaseModel):
    results: list[PredictionOutput]
    total: int
    fraud_count: int

# ── Input validation ───────────────────────────────────────────────────────────
def _validate_features(features: list[float]) -> list[str]:
    """
    Warn (don't reject) if continuous inputs are far outside training distribution.
    Uses scaler mean/std to detect potential out-of-distribution inputs.
    Returns list of warning strings (empty = all clear).
    """
    warnings_out = []
    df = make_dataframe(features)

    for i, col in enumerate(SCALE_COLS):
        if col not in df.columns:
            continue
        col_idx = list(scaler.feature_names_in_).index(col) if hasattr(scaler, 'feature_names_in_') else None
        if col_idx is None:
            continue
        mean = scaler.mean_[col_idx]
        std  = scaler.scale_[col_idx]
        val  = df[col].iloc[0]
        if std > 0 and abs(val - mean) > 5 * std:
            warnings_out.append(
                f"{col}={val:.2f} is {abs(val-mean)/std:.1f} std devs from training mean ({mean:.2f})"
            )
    return warnings_out

# ── SHAP explanation cache ─────────────────────────────────────────────────────
# explainer.shap_values() takes 50–200ms per call on a 300-tree ensemble.
# Identical feature vectors (e.g. repeated /predict calls from a retry loop or
# a dashboard polling the same transaction) recompute SHAP unnecessarily.
#
# Cache design:
#   key   : tuple(features) — tuples are hashable, lists are not
#   value : (top_shap_dict, inserted_at_timestamp)
#   TTL   : 60 seconds — stale enough to be useful, fresh enough to be safe
#   size  : max 512 entries — each entry is ~17 floats + 10 SHAP floats ≈ 1KB
#           512 entries ≈ 512KB maximum memory footprint, negligible
#
# Thread safety: the cache dict is accessed from run_in_threadpool() worker threads.
# Python's GIL makes individual dict read/write operations atomic, so no explicit
# lock is needed for this simple cache pattern.

_SHAP_CACHE: dict[tuple, tuple[dict, float]] = {}
_SHAP_CACHE_TTL     = 60.0    # seconds
_SHAP_CACHE_MAXSIZE = 512
_SHAP_LOCK          = threading.Lock()
# Lock protects concurrent cache reads/writes from run_in_threadpool() workers.
# Without a lock, two threads can simultaneously miss the cache, both compute
# SHAP, and race to write — wasting compute. The lock is brief (dict op only)
# and does not wrap the expensive explainer.shap_values() call.


def _get_cached_shap(features_key: tuple) -> dict | None:
    """Return cached top_shap dict if present and not expired, else None."""
    with _SHAP_LOCK:
        entry = _SHAP_CACHE.get(features_key)
        if entry is None:
            return None
        top_shap, inserted_at = entry
        if time.monotonic() - inserted_at > _SHAP_CACHE_TTL:
            _SHAP_CACHE.pop(features_key, None)
            return None
        return top_shap


def _set_cached_shap(features_key: tuple, top_shap: dict) -> None:
    """Insert into cache, evicting oldest entries if at max size."""
    with _SHAP_LOCK:
        if len(_SHAP_CACHE) >= _SHAP_CACHE_MAXSIZE:
            oldest_key = next(iter(_SHAP_CACHE))
            _SHAP_CACHE.pop(oldest_key, None)
        _SHAP_CACHE[features_key] = (top_shap, time.monotonic())


def _predict_one(
    features: list[float],
    threshold: float,
    velocity_warning: str | None = None,
) -> PredictionOutput:
    """
    Synchronous prediction + SHAP computation.
    Called exclusively via run_in_threadpool() from the async route handlers
    so this blocking work never stalls the event loop.

    SHAP values are cached by feature vector (tuple key) for up to 60 seconds.
    Cache hits skip the 50–200ms explainer.shap_values() call entirely.
    """
    ood_warnings = _validate_features(features)

    df        = make_dataframe(features)
    df_scaled = apply_scaler(df, scaler)

    prob     = float(model.predict_proba(df_scaled)[0][1])
    is_fraud = prob >= threshold

    # SHAP cache lookup — tuple(features) is the hashable key
    features_key = tuple(features)
    top_shap     = _get_cached_shap(features_key)

    if top_shap is None:
        # Cache miss — compute SHAP and store
        shap_vals = explainer.shap_values(df_scaled)
        arr       = np.array(
            shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        ).flatten()[:N_FEATURES]
        shap_map  = {name: round(float(v), 6) for name, v in zip(FEATURE_NAMES, arr)}
        top_shap  = dict(sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        _set_cached_shap(features_key, top_shap)

    return PredictionOutput(
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        threshold_used=threshold,
        top_shap_features=top_shap,
        input_warnings=ood_warnings,
        velocity_warning=velocity_warning,
    )

def _tx_type_from_features(features: list[float]) -> str:
    """Decode which type_* one-hot is active (for audit log)."""
    type_cols = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
    start_idx = FEATURE_NAMES.index("type_CASH_IN")
    for i, name in enumerate(type_cols):
        if features[start_idx + i] == 1:
            return name.replace("type_", "")
    return "UNKNOWN"

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Liveness probe — no authentication required.
    Returns only {"status": "ok"} so Docker/k8s health checks work without
    credentials, and so unauthenticated callers learn nothing about the model.
    Internal provenance data is available via the authenticated /info endpoint.
    """
    return {"status": "ok"}


@app.get("/info", dependencies=[Depends(require_api_key)])
def info():
    """
    Model provenance — requires X-API-Key authentication.
    Returns operational metadata: which model is deployed, when it was trained,
    the data fingerprint it was trained on, and its calibrated PR-AUC.
    Kept separate from /health so liveness probes never expose internal state.
    """
    return {
        "model":              _metrics.get("best_model", "unknown"),
        "trained_at":         _metrics.get("trained_at", "unknown"),
        "data_hash":          _metrics.get("data_hash", "unknown"),
        "pr_auc_calibrated":  _metrics.get("pr_auc_calibrated"),
        "feature_count":      len(_metrics.get("feature_names", [])),
        "selection_criterion": _metrics.get("selection_criterion", "pr_auc"),
    }

@app.get("/metrics", dependencies=[Depends(require_api_key)])
def get_metrics():
    return _metrics

@app.get("/threshold_analysis", dependencies=[Depends(require_api_key)])
def threshold_analysis():
    """
    Returns the full precision-recall curve computed at training time.
    Use this to pick an operating threshold based on your acceptable
    false-positive rate rather than the default 0.5.

    Each point: { threshold, precision, recall, f1 }
    """
    return {
        "model": _metrics.get("best_model"),
        "pr_auc_calibrated": _metrics.get("pr_auc_calibrated"),
        "note": (
            "Select threshold where recall meets your ops team's capacity. "
            "Lower threshold = higher recall (catch more fraud), more false positives."
        ),
        "curve": _threshold_curve,
    }


@app.get("/drift_report", dependencies=[Depends(require_api_key)])
def drift_report():
    """
    Model drift detection — compares live prediction distribution to training baseline.

    Method:
      1. Query the last 1,000 predictions from the SQLite audit log.
      2. Compute the live fraud flag rate (fraction of predictions marked is_fraud=1).
      3. Compare to the training baseline fraud rate stored in metrics.json.
      4. If live_rate > 2× baseline OR live_rate < 0.5× baseline, flag DRIFT_DETECTED.

    Interpretation:
      DRIFT_DETECTED does not mean the model is wrong — it means the incoming
      transaction distribution has shifted significantly from training conditions.
      Possible causes: seasonality, new fraud patterns, upstream data pipeline change,
      or a model threshold that is systematically miscalibrated in production.

    Action: investigate the audit log, compare feature distributions, consider retraining.

    Returns NOT_ENOUGH_DATA if fewer than 50 predictions have been logged.
    """
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT is_fraud FROM predictions ORDER BY timestamp DESC LIMIT 1000"
        ).fetchall()

    if len(rows) < 50:
        return {
            "status":          "NOT_ENOUGH_DATA",
            "message":         f"Only {len(rows)} predictions logged. Need at least 50 for a meaningful drift estimate.",
            "records_checked": len(rows),
            "live_fraud_rate":  None,
            "baseline_fraud_rate": _metrics.get("fraud_pct"),
        }

    n_total     = len(rows)
    n_fraud     = sum(1 for r in rows if r["is_fraud"] == 1)
    live_rate   = n_fraud / n_total                              # fraction (0–1)
    # baseline_fraud_pct is stored as a percentage (e.g. 0.0535), convert to fraction
    baseline_pct   = _metrics.get("fraud_pct", 0)              # e.g. 0.0535
    baseline_rate  = baseline_pct / 100.0                       # e.g. 0.000535

    # Avoid division by zero if no fraud in training data
    if baseline_rate == 0:
        ratio  = float("inf") if live_rate > 0 else 1.0
    else:
        ratio  = live_rate / baseline_rate

    drifted = ratio > 2.0 or ratio < 0.5

    return {
        "status":              "DRIFT_DETECTED" if drifted else "OK",
        "records_checked":     n_total,
        "live_fraud_flagged":  n_fraud,
        "live_fraud_rate":     round(live_rate, 6),
        "baseline_fraud_rate": round(baseline_rate, 6),
        "ratio":               round(ratio, 4),
        "threshold_applied":   "flag if ratio > 2.0 or ratio < 0.5",
        "message": (
            f"Live fraud rate ({live_rate:.4%}) is {ratio:.2f}× the training baseline "
            f"({baseline_rate:.4%}). {'Investigate distribution shift.' if drifted else 'Within expected range.'}"
        ),
    }

@app.post("/predict", response_model=PredictionOutput, dependencies=[Depends(require_api_key)])
@limiter.limit("60/minute")
async def predict(request: Request, tx: TransactionInput, background_tasks: BackgroundTasks):
    """
    Async inference endpoint.

    model.predict_proba() and explainer.shap_values() are CPU-bound and blocking.
    Running them directly in an async def would hold the event loop and prevent
    other requests from being handled concurrently. run_in_threadpool() dispatches
    the work to Starlette's thread pool executor so the event loop stays free.

    The DB write is also synchronous (sqlite3). It runs in a BackgroundTask which
    itself fires in the thread pool after the response is returned.
    """
    try:
        result = await run_in_threadpool(
            _predict_one, tx.features, tx.threshold, tx.velocity_warning
        )
        tx_type = _tx_type_from_features(tx.features)
        background_tasks.add_task(
            _write_prediction, tx.features[0], result.fraud_probability,
            result.is_fraud, tx_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/predict/batch", response_model=BatchOutput, dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
async def predict_batch(request: Request, batch: BatchInput, background_tasks: BackgroundTasks):
    """
    Async batch inference endpoint — semaphore-limited concurrency.

    Each prediction runs in a thread pool via run_in_threadpool(). Without a
    semaphore, 100 transactions would fire 100 concurrent threads, consuming
    Starlette's entire 40-worker pool and starving all other endpoints.

    _BATCH_SEMAPHORE(10) caps concurrent batch threads at 10. The remaining
    transactions queue inside asyncio — they don't block the event loop —
    and drain as earlier predictions complete. Single /predict requests always
    have thread-pool headroom available regardless of batch load.

    DB writes are queued as BackgroundTasks after all predictions complete,
    keeping them off the response critical path.
    """
    async def _predict_with_semaphore(tx: TransactionInput) -> PredictionOutput:
        async with _BATCH_SEMAPHORE:
            return await run_in_threadpool(
                _predict_one, tx.features, tx.threshold, tx.velocity_warning
            )

    try:
        results: list[PredictionOutput] = list(
            await asyncio.gather(*[_predict_with_semaphore(tx) for tx in batch.transactions])
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    for tx, result in zip(batch.transactions, results):
        tx_type = _tx_type_from_features(tx.features)
        background_tasks.add_task(
            _write_prediction, tx.features[0], result.fraud_probability,
            result.is_fraud, tx_type
        )

    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchOutput(results=results, total=len(results), fraud_count=fraud_count)

@app.get("/audit", dependencies=[Depends(require_api_key)])
def get_audit(
    fraud_only: bool = Query(False, description="If true, return only flagged transactions"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Paginated audit log with optional fraud-only filter."""
    where = "WHERE is_fraud = 1" if fraud_only else ""
    with _get_db() as conn:
        rows = conn.execute(
            f"SELECT id, amount, probability, is_fraud, timestamp, tx_type "
            f"FROM predictions {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return {"total_returned": len(rows), "offset": offset, "records": [dict(r) for r in rows]}

@app.get("/random_sample", dependencies=[Depends(require_api_key)])
def get_random_sample():
    """Return a random transaction from the sample dataset with pre-engineered features."""
    if _df_sample is None:
        raise HTTPException(status_code=500, detail="paysim_sample.csv not found on server.")
    row     = _df_sample.sample(1)
    is_fraud = bool(row["isFraud"].values[0])
    tx_type  = str(row["type"].values[0])
    amount   = float(row["amount"].values[0])
    step     = int(row["step"].values[0])

    # Engineer features for this row
    X_row    = engineer_features(row)
    X_scaled = apply_scaler(X_row, scaler)
    features = X_row[FEATURE_NAMES].values.flatten().tolist()

    return {
        "features": features,
        "feature_names": FEATURE_NAMES,
        "is_fraud": is_fraud,
        "tx_type": tx_type,
        "amount": amount,
        "step": step,
    }

# ── Static frontend ────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
