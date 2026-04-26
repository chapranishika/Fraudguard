# FraudGuard — Financial Fraud Detection Pipeline

[![CI](https://github.com/<your-username>/fraudguard/actions/workflows/test.yml/badge.svg)](https://github.com/<your-username>/fraudguard/actions/workflows/test.yml) [![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen)](tests/) [![Python](https://img.shields.io/badge/python-3.11%2B-blue)](requirements.txt) [![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](src/api.py)

End-to-end financial fraud detection on the **PaySim** synthetic transactions dataset (1M rows, 1:1868 class imbalance). Covers domain-driven feature engineering, Bayesian hyperparameter optimisation (Optuna), probability calibration, async FastAPI inference with SHAP explainability, model versioning with hash-linked registry, drift detection, velocity feature schema intelligence, and 101 tests enforced by GitHub Actions CI on Python 3.11 + 3.12.

---

## Why This Isn't a Toy Project

Most fraud detection portfolios use the Kaggle creditcard dataset, where V1–V28 are pre-PCA'd by the original authors. There is no feature engineering to do — you hand anonymised vectors directly to a model. This project uses **PaySim**, which provides raw financial transaction fields, requiring deliberate domain-driven decisions:

**1. The class imbalance is severe and handled correctly.**
535 fraud cases in 1,000,000 transactions — a **1:1868 ratio**. A naive classifier achieves 99.95% accuracy by predicting everything as normal. This project addresses it in two independent layers:

- **Stratified split first.** `train_test_split(..., stratify=y)` ensures both train and test partitions preserve the 0.05% fraud rate. Without this, small datasets can produce test folds with zero fraud cases.
- **ADASYN on the train split only.** Oversampling is applied strictly after the split. Applying it before would allow synthetic minority samples to bleed into the test set, inflating every metric. ADASYN adaptively weights boundary examples — minority samples surrounded by majority neighbours get more synthetic neighbours generated near them — producing a harder and more realistic training distribution than uniform SMOTE.

**2. PR-AUC is used for model selection, not ROC-AUC.**
On a 99.95% majority class, ROC-AUC is dominated by the true-negative pool. A model can achieve 0.99 ROC-AUC while missing most fraud. PR-AUC (Average Precision) ignores true negatives entirely and measures retrieval quality on the minority class — it directly penalises false negatives and false positives where they cost money.

**3. Probabilities are calibrated.**
Tree ensembles produce scores, not probabilities. A raw XGBoost score of 0.73 does not mean 73% of such transactions are fraud. `CalibratedClassifierCV` (isotonic regression, cv=5) maps the raw score distribution to empirical frequencies, fitting calibration on held-out folds so no training data is reused.

**4. SHAP is wired to the right object.**
`CalibratedClassifierCV` wraps estimators in `.calibrated_classifiers_[i].estimator`. Passing the wrapper to `TreeExplainer` silently produces wrong SHAP values or crashes. This project accesses `calibrated_classifiers_[0].estimator` explicitly to give SHAP the raw tree structure.

**5. The test suite uses real picklable sklearn objects.**
No `MagicMock` for model artifacts — the previous architecture broke because `api.py` calls `_load("model.pkl")` at module import time, meaning patches applied after import are always too late. `tests/conftest.py` writes actual `DummyClassifier`, `StandardScaler`, and a module-level `_StubExplainer` to `models/` before any import fires. The stub class must be at module scope because Python's pickle protocol cannot serialise classes defined inside functions.

---

## Technical Architecture

```
paysim_sample.csv  (1M rows, 0.05% fraud)
        │
        ▼
 src/preprocess.py
 ┌─────────────────────────────────────────────────────────┐
 │  engineer_features()                                     │
 │    balance_delta_orig = newbalanceOrig + amount          │
 │                       - oldbalanceOrg   (acctg error)   │
 │    balance_delta_dest = oldbalanceDest + amount          │
 │                       - newbalanceDest                   │
 │    balance_zero_orig  = 1 if newbalanceOrig == 0         │
 │    balance_zero_dest  = 1 if oldbalanceDest == 0         │
 │    type_*             = one-hot (5 transaction types)    │
 │    → 15 features total                                   │
 └─────────────────────────────────────────────────────────┘
        │
        ▼
 StandardScaler.fit()  ← fitted on train split ONLY
        │
        ├── train (80%)  → ADASYN oversampling
        │                → RF / XGBoost / LightGBM training
        │                → compare by PR-AUC
        │                → CalibratedClassifierCV (isotonic, cv=5)
        │
        └── test  (20%)  → evaluation only, never seen by scaler or ADASYN
                         → precision_recall_curve → threshold_curve.json
                         → SHAP on calibrated_classifiers_[0].estimator
                         → metrics.json + trained_at + data_hash (MD5)

        ▼
 models/
 ├── model.pkl             CalibratedClassifierCV wrapping XGBoost
 ├── scaler.pkl            StandardScaler (continuous cols only)
 ├── explainer.pkl         SHAP TreeExplainer on raw estimator
 ├── metrics.json          All metrics + provenance
 └── threshold_curve.json  Full PR curve (312 points)

        ▼
 src/api.py  (FastAPI)
 ┌─────────────────────────────────────────────────────────┐
 │  POST /predict          → probability + SHAP + warnings  │
 │  POST /predict/batch    → bulk inference                 │
 │  GET  /threshold_analysis → PR curve for ops team        │
 │  GET  /audit            → paginated SQLite audit log     │
 │  GET  /metrics          → full metrics.json              │
 │  GET  /health           → liveness + model provenance    │
 │  GET  /random_sample    → real PaySim row for demo       │
 │                                                          │
 │  Auth: X-API-Key header on all data endpoints            │
 │  Audit: BackgroundTasks → SQLite, zero request latency   │
 │  Validation: input_warnings for OOD features (>5σ)       │
 └─────────────────────────────────────────────────────────┘
        │
        ├── frontend/     Vanilla HTML/JS operations terminal
        └── app.py        Streamlit dashboard (loads models directly)
```

---

## Feature Engineering

Fraud in PaySim follows a specific pattern: accounts are drained entirely (sender balance → 0) while the balance update at the destination is inconsistently recorded. This produces large non-zero `balance_delta` values:

| Feature | Engineering | Fraud signal |
|---|---|---|
| `balance_delta_orig` | `newbalanceOrig + amount - oldbalanceOrg` | Near-zero in honest tx; large in fraud |
| `balance_delta_dest` | `oldbalanceDest + amount - newbalanceDest` | Destination balance not updated properly |
| `balance_zero_orig` | `1 if newbalanceOrig == 0` | Sender wiped to zero |
| `balance_zero_dest` | `1 if oldbalanceDest == 0` | Recipient had nothing before transaction |
| `type_TRANSFER` | one-hot | Fraud exclusively on TRANSFER and CASH_OUT |
| `type_CASH_OUT` | one-hot | |
| `step` | raw | Velocity proxy — hour of simulation |

---

## How to Run

### Prerequisites
- Python 3.11+
- PaySim CSV from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### 1. Set up

```bash
git clone https://github.com/<your-username>/fraudguard.git
cd fraudguard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare dataset (one-time)

```python
# Run this once to create paysim_sample.csv in the project root
import pandas as pd
df = pd.read_csv("Synthetic_Financial_datasets_log.csv", nrows=1_000_000)
df.to_csv("paysim_sample.csv", index=False)
```

### 3. Train

```bash
python train.py
# ~3–5 min. Writes models/ artifacts and threshold_curve.json
```

### 4. Run API

```bash
uvicorn src.api:app --reload --port 8000
# Swagger UI → http://127.0.0.1:8000/docs
```

Test auth enforcement:
```bash
curl http://127.0.0.1:8000/health                                     # 200 no key
curl http://127.0.0.1:8000/metrics                                    # 401
curl http://127.0.0.1:8000/metrics -H "X-API-Key: demo-key-123"      # 200
```

Predict a classic fraud pattern (TRANSFER draining sender to zero):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-123" \
  -d '{
    "features": [50000,50000,0,0,0,0,50000,1,1,1,0,0,0,0,1],
    "threshold": 0.5
  }'
```

Threshold analysis — pick your operating point based on ops capacity:
```bash
curl http://127.0.0.1:8000/threshold_analysis -H "X-API-Key: demo-key-123"
```

Paginated audit log, fraud only:
```bash
curl "http://127.0.0.1:8000/audit?fraud_only=true&limit=20&offset=0" \
  -H "X-API-Key: demo-key-123"
```

### 5. Frontend options

```bash
# HTML/JS terminal (requires API running)
cd frontend && python -m http.server 5500
# Open http://localhost:5500

# Streamlit (loads models directly, no API needed)
streamlit run app.py
```

### 6. Tests

```bash
pytest tests/ -v
# 61 tests in ~8s — no pkl files, no dataset required
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | ✗ | Liveness only. Returns `{"status":"ok"}` — no internal data exposed |
| `GET` | `/info` | ✓ | Model provenance: `best_model`, `trained_at`, `data_hash`, `pr_auc_calibrated`, `feature_count` |
| `GET` | `/metrics` | ✓ | Full `metrics.json`: calibrated + raw metrics, model comparison, Optuna results, CV variance |
| `GET` | `/threshold_analysis` | ✓ | Full PR curve (threshold → precision/recall/F1) for ops threshold selection |
| `GET` | `/drift_report` | ✓ | Live fraud flag rate vs training baseline. `DRIFT_DETECTED` if ratio >2× or <0.5× |
| `POST` | `/predict` | ✓ | Single transaction → `fraud_probability`, `is_fraud`, `top_shap_features`, `input_warnings`, `velocity_warning` |
| `POST` | `/predict/batch` | ✓ | Up to 100 transactions → results array + `fraud_count`. Semaphore-limited to 10 concurrent threads |
| `GET` | `/audit` | ✓ | Paginated SQLite log. Query params: `fraud_only`, `limit` (max 500), `offset` |
| `GET` | `/random_sample` | ✓ | Random PaySim row with pre-engineered 17-feature vector for demo/testing |

**Feature vector order (17 floats):**
`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `balance_delta_orig`, `balance_delta_dest`, `balance_zero_orig`, `balance_zero_dest`, `orig_tx_count_so_far`, `orig_amount_sum_so_far`, `step`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`

---

## Docker

```bash
docker-compose up --build
# API → http://localhost:8000
# Streamlit → http://localhost:8501
```

> `paysim_sample.csv` and `models/` are volume-mounted. Both must exist locally before starting Docker.

---

## Security Notes

- `X-API-Key` header enforced on all data endpoints via FastAPI `APIKeyHeader` + `Depends`
- `/health` intentionally open — required for Docker/k8s liveness probes
- CORS restricted to `localhost:3000`, `127.0.0.1:8000`, `localhost:8501`, `localhost:5500`
- `paysim_sample.csv`, `*.pkl`, `*.db`, `*.sqlite` excluded via `.gitignore`
- `demo-key-123` is a public demo key. Replace with `os.environ["API_KEY"]` before deployment

---

## Demo Limitations

The HTML manual entry form computes `balance_delta_orig/dest` and `balance_zero` flags directly from the input fields using the same arithmetic as `engineer_features()` in Python — this is not a heuristic proxy. The only gap is that `step` is entered manually rather than derived from a real event stream.

For ground-truth behaviour: use the Research Lab "Pull Random Transaction" button, which fetches a real PaySim row, runs `engineer_features()` server-side, and sends the result to `/predict`.

---

## ML & Engineering Design Decisions

This section documents the technical reasoning behind every non-obvious choice in the pipeline. These are the questions you will be asked in a technical screen at Microsoft, Razorpay, or any serious ML engineering role — and the honest answers based on exactly what this codebase does.

---

### 1. Why PR-AUC, not ROC-AUC, for Model Selection

The dataset contains 535 fraud cases in 1,000,000 transactions — a **1:1868 class ratio** (0.0535% fraud). At this imbalance level, ROC-AUC is a misleading primary metric.

ROC-AUC plots True Positive Rate against False Positive Rate across all thresholds. False Positive Rate is `FP / (FP + TN)`. With 999,465 normal transactions, the true-negative pool is enormous — even a model that flags very little fraud can achieve a low FPR and therefore a high ROC-AUC score. A model predicting everything as normal gets ROC-AUC close to 0.5, but a weak fraud model that catches only 30% of fraud while being confident about normal transactions can reach 0.95+ ROC-AUC. That number is not meaningful for a fraud operations team.

PR-AUC (Average Precision) plots Precision against Recall across thresholds. It **never involves true negatives** — the denominator in both Precision (`TP / (TP + FP)`) and Recall (`TP / (TP + FN)`) only counts fraud-related predictions. Every missed fraud case and every false alarm directly reduces PR-AUC. This makes it the correct optimisation target when the minority class is what you actually care about catching.

This is why `train.py` selects the best model with:
```python
best_name = max(results, key=lambda m: results[m]["pr_auc"])
```
Not `roc_auc`. The difference in model selected can be significant when two models are close on ROC-AUC but diverge on their fraud retrieval quality.

---

### 2. CalibratedClassifierCV — Correcting Tree Ensemble Overconfidence

XGBoost, LightGBM, and RandomForest output **scores**, not **probabilities**. A raw XGBoost output of 0.73 does not mean 73% of such transactions are actually fraudulent. Tree ensembles are systematically overconfident — they push scores toward 0 and 1 more aggressively than the true empirical frequencies warrant.

This matters practically: if a downstream system uses the fraud score as a continuous risk signal (e.g., "apply step-up authentication at score > 0.4, block at score > 0.8"), it needs the scores to be calibrated — meaning a score of 0.4 should correspond to roughly 40% of such transactions being fraud in the real world.

This project wraps the best model with:
```python
best_model = CalibratedClassifierCV(
    estimator=best_model_raw,
    method="isotonic",
    cv=5
)
```

**Why isotonic over Platt scaling (sigmoid)?** Platt scaling fits a logistic sigmoid — it assumes the calibration curve is monotonic and S-shaped, which is often violated for tree ensembles. Isotonic regression fits a non-parametric monotone function, making no shape assumptions. On datasets large enough to fit 5-fold CV (which 1M rows easily is), isotonic is more accurate.

**Why cv=5?** The calibration mapping is fitted on held-out fold predictions — the model never sees its own predictions during calibration training. This prevents the calibration layer from overfitting to the training distribution.

**Critical implementation note:** `CalibratedClassifierCV` stores the 5 fitted `(base_estimator, calibrator)` pairs in `.calibrated_classifiers_`. When building the SHAP explainer, `TreeExplainer` requires a raw tree object — passing the wrapper raises an error or produces incorrect values. The correct access pattern, used in `train.py`, is:
```python
_raw_for_shap = best_model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(_raw_for_shap)
```

---

### 3. Data Leakage Prevention — Split Before Scaling

Preprocessing leakage occurs when information from the test set is used during any training step. With `StandardScaler`, the risk is specific: if you fit the scaler on the full dataset before splitting, the scaler's `mean_` and `scale_` are computed from a distribution that includes test-set rows. When you later evaluate on the test set, those rows are being scaled using statistics partially derived from themselves.

The original `load_and_preprocess()` function in this codebase did exactly this — it fitted the scaler before returning, and the train/test split happened afterwards in `train.py`. This has been fixed.

The correct sequence, enforced in the current `train.py`:

```python
# 1. Load raw CSV and engineer features — no scaling
X_raw, y = load_raw(CSV_PATH)

# 2. Split FIRST — test set is now invisible to everything that follows
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Fit scaler on train partition ONLY
scaler  = fit_scaler(X_train_raw)

# 4. Apply to both partitions using train-derived statistics
X_train = apply_scaler(X_train_raw, scaler)
X_test  = apply_scaler(X_test_raw, scaler)   # test rows scaled with train mean/std
```

The `load_raw()` function is intentionally designed to make the correct pattern hard to violate — it returns unscaled features and provides no scaler, forcing the caller to handle the split-then-scale sequence explicitly.

On a 1M-row dataset with stable feature distributions, the numerical difference from leakage is small. The conceptual error is not — and any senior ML interviewer will check the order of operations as the first question in a code review.

---

### 4. Sequential Velocity Features — Why Fraud is a Sequence Problem

The balance-error features (`balance_delta_orig`, `balance_zero_orig`) detect individual fraudulent transactions in isolation. But real fraud is often sequential: a fraudster drains multiple accounts in rapid succession, or repeatedly transfers to the same destination across short time windows.

Two velocity features are computed in `engineer_features()` on the full dataset sorted by `step`:

```python
out["orig_tx_count_so_far"] = (
    df.groupby("nameOrig")["amount"]
    .transform(lambda x: x.expanding().count())
)
out["orig_amount_sum_so_far"] = (
    df.groupby("nameOrig")["amount"]
    .transform(lambda x: x.expanding().sum())
)
```

`expanding()` produces a cumulative window from the first observed transaction for each `nameOrig` up to and including the current row. Because the dataframe is sorted by `step` before this computation, each row's velocity values reflect only its own chronological past — transactions that have not yet occurred at that simulation step are not included. This means there is **no future leakage within the feature itself**.

These features directly encode: *how many times has this sender transacted before this moment, and how much have they moved in total?* A fraudster who has already completed 14 large transfers in the same step period will have a very high `orig_tx_count_so_far` — a signal invisible to models that score each row in isolation.

**Inference note:** For single-transaction API requests, these values cannot be computed from the transaction alone. The current `/predict` endpoint accepts them as input features — the calling system is responsible for looking up the sender's current cumulative counts from a feature store or audit log. For demo purposes, setting both to 0 approximates a first-time sender.

---

### 5. Production Engineering Choices

**Rate limiting with SlowAPI (60 req/min on `/predict`).**
Without rate limiting, the `/predict` endpoint can be trivially exhausted via a scripted loop — either for denial of service or for probing the model's decision boundary. SlowAPI integrates with FastAPI's dependency injection system:

```python
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("60/minute")
def predict(request: Request, ...):
```

Three components are all required together. The `request: Request` parameter must be the first argument — SlowAPI extracts the client IP from it. Missing the `add_exception_handler` registration causes unhandled 500s instead of clean 429 responses.

**SQLite WAL mode for BackgroundTask concurrency.**
FastAPI's `BackgroundTasks` dispatches work to a thread pool after the HTTP response is returned. SQLite's default journal mode (`DELETE`) uses exclusive write locks — a background thread writing an audit row can block any concurrent reader and raises `OperationalError: database is locked` under parallel requests.

WAL (Write-Ahead Logging) mode separates read and write paths: readers access the stable main database file while a writer appends to a separate WAL file. Concurrent reads and one background write proceed without blocking each other. `check_same_thread=False` is additionally required because the connection is created in the request thread and used in the background worker thread:

```python
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
```

This handles single-process multi-threaded concurrency. For a multi-process deployment (gunicorn `--workers 4`), the correct solution is PostgreSQL with asyncpg — multiple processes cannot safely share a single SQLite WAL file.

**Environment variable secrets.**
The API key is read from the environment:
```python
_API_KEY = os.environ.get("API_KEY", "demo-key-123")
```

The fallback `"demo-key-123"` exists only for local development without a `.env` file. In any real deployment, `API_KEY` is injected via the host environment, a secrets manager (AWS Secrets Manager, GCP Secret Manager, Vault), or a CI/CD pipeline secret. The `.env` file is excluded from version control via `.gitignore`. A `.env.example` template is committed to document the required variables without exposing values.

Hardcoding secrets in source code is a security anti-pattern regardless of repository visibility — secrets committed to git persist in history even after deletion and can be extracted from any clone.

---

### 6. Known Limitation — Velocity Feature Inference Gap

`orig_tx_count_so_far` and `orig_amount_sum_so_far` are computed in `engineer_features()` using an expanding-window `groupby` on a dataframe sorted by `step`. This is well-defined for batch training: every row has access to its full transaction history up to that simulation step, the sort order is deterministic, and no future data bleeds backward. At training time this produces accurate, leakage-free cumulative signals.

At inference time the contract breaks. When a single transaction arrives at `/predict`, there is no surrounding dataframe to sort or group — the expanding window has nothing to expand over. The current API sidesteps this by accepting `orig_tx_count_so_far` and `orig_amount_sum_so_far` as raw float inputs in the 17-feature vector, delegating responsibility for supplying correct values to the calling system. For demo purposes the documentation suggests setting both to zero, which is equivalent to treating every request as a first-time sender and silently discards the sequential fraud signal the model was trained to exploit.

In a production payment system this gap would be closed with a **low-latency feature store**: a Redis cluster (or equivalent) maintaining a running hash per `nameOrig` with two fields — `tx_count` and `amount_sum` — updated atomically on every processed transaction. The inference path would read these counters in a single `HGET` call (sub-millisecond at co-location), inject them into the feature vector, and then call the model. The feature store update would happen asynchronously after the prediction is returned, keeping it off the critical latency path. Without this infrastructure, the velocity features contribute accurate signal during offline evaluation but degrade to a constant (zero) at runtime — a systematic discrepancy between training-time and inference-time feature distributions that inflates reported metrics relative to true deployed performance.

---

### 7. Known Limitation — SHAP Approximation on Calibrated Ensembles

`CalibratedClassifierCV` with `cv=5` does not produce a single model — it produces five `(base_estimator, calibrator)` pairs, one per fold, stored in `.calibrated_classifiers_[0]` through `[4]`. At prediction time, `predict_proba()` averages the calibrated outputs of all five pairs. The deployed probability for any given transaction is therefore a function of all five fold-0-through-4 base models filtered through their respective isotonic regression mappings.

`shap.TreeExplainer` requires a native tree object and cannot accept the `CalibratedClassifierCV` wrapper directly — passing the wrapper raises a `TypeError`. This project targets `calibrated_classifiers_[0].estimator`: the raw XGBoost tree from fold 0 only. The SHAP values returned by `/predict` are therefore attributions with respect to the fold-0 base model's raw score, not with respect to the final calibrated ensemble's output probability. In practice this is a reasonable approximation — all five folds were trained on the same resampled dataset with the same hyperparameters, so their tree structures and feature importance rankings are similar. The directional signal is preserved: features that drive the fold-0 score up or down drive the ensemble output in the same direction. However, the magnitude of each attribution is not precisely aligned with the calibrated probability the API returns, and there is no formal guarantee of accuracy at the individual-prediction level.

The correct production solution is `shap.KernelExplainer` or `shap.PermutationExplainer` applied to the full `CalibratedClassifierCV` object's `predict_proba` method directly. These model-agnostic explainers treat the calibrated ensemble as a black box and compute attributions with respect to its actual output — at the cost of significantly higher compute time per explanation (seconds per prediction versus milliseconds for `TreeExplainer`). For a latency-sensitive payment API, a pragmatic middle ground is to pre-compute KernelExplainer attributions on a large representative sample at training time for global importance reporting, while serving the faster `TreeExplainer` approximation at inference time with an explicit disclaimer that magnitudes are indicative rather than exact.
