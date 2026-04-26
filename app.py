"""
app.py  —  Streamlit UI for the Fraud Detection system.
Loads artifacts from models/ folder (run train.py first).
"""

import json
import pickle
import warnings
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import FEATURE_NAMES, N_FEATURES, apply_scaler, make_dataframe, engineer_features

matplotlib.use("Agg")
plt.style.use("dark_background")
matplotlib.rcParams['axes.facecolor'] = 'none'
matplotlib.rcParams['figure.facecolor'] = 'none'
warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")

st.set_page_config(page_title="Fraud Detection AI", page_icon="💳", layout="wide")

if "audit_log" not in st.session_state:
    st.session_state.audit_log = deque(maxlen=500)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load model artifacts using registry.json if present (versioned),
    otherwise fall back to unversioned model.pkl / scaler.pkl / explainer.pkl.
    Mirrors the resolution logic in src/api.py _resolve_artifacts().
    """
    registry_path = MODELS_DIR / "registry.json"
    try:
        if registry_path.exists():
            with open(registry_path) as f:
                reg = json.load(f)
            model_f    = reg["active_model"]
            scaler_f   = reg["active_scaler"]
            explainer_f = reg["active_explainer"]
        else:
            model_f, scaler_f, explainer_f = "model.pkl", "scaler.pkl", "explainer.pkl"

        model     = pickle.load(open(MODELS_DIR / model_f,    "rb"))
        scaler    = pickle.load(open(MODELS_DIR / scaler_f,   "rb"))
        explainer = pickle.load(open(MODELS_DIR / explainer_f,"rb"))
        return model, scaler, explainer
    except FileNotFoundError as e:
        st.error(f"⚠️ Missing model file: {e}. Run `python train.py` first.")
        st.stop()

@st.cache_data
def load_metrics():
    try:
        with open(MODELS_DIR / "metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("paysim_sample.csv")
    except Exception:
        return None

model, scaler, explainer = load_artifacts()
metrics = load_metrics()
data    = load_dataset()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
threshold = st.sidebar.slider("Fraud sensitivity", 0.0, 1.0, 0.5, 0.01,
    help="Lower = catch more fraud (more false alarms). Higher = only flag very suspicious transactions.")
mode = st.sidebar.radio("Input mode", ["Enter a transaction", "Random real sample", "Batch upload"])

if metrics:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Best model:** {metrics.get('best_model','—')}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 💳 Elite FraudGuard AI Terminal")
st.caption("PaySim financial fraud detection · PR-AUC optimised · CalibratedClassifierCV · SHAP explainability")

if metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC (calibrated)",  f"{metrics.get('pr_auc_calibrated',  metrics.get('pr_auc', 0)):.4f}")
    m2.metric("Precision (calibrated)",f"{metrics.get('precision_calibrated',metrics.get('precision',0)):.4f}")
    m3.metric("Recall (calibrated)",   f"{metrics.get('recall_calibrated',  metrics.get('recall', 0)):.4f}")
    m4.metric("F1 (calibrated)",       f"{metrics.get('f1_calibrated',      metrics.get('f1', 0)):.4f}")
    st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Batch upload":
    st.subheader("📂 Batch prediction")
    st.info(
        "Upload a PaySim-format CSV with columns: `step`, `type`, `amount`, "
        "`nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, "
        "`oldbalanceDest`, `newbalanceDest`. "
        "An `isFraud` column is used for ground-truth comparison if present."
    )
    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        X_batch  = engineer_features(batch_df)
        X_scaled = apply_scaler(X_batch, scaler)

        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        result_df = pd.DataFrame({
            "amount":           batch_df["amount"].values if "amount" in batch_df.columns else X_batch["amount"].values,
            "type":             batch_df["type"].values   if "type"   in batch_df.columns else "—",
            "fraud_probability": probs.round(4),
            "prediction":        preds,
            "label":             pd.Series(preds).map({0: "✅ Normal", 1: "⚠️ FRAUD"}).values,
        })

        if "isFraud" in batch_df.columns:
            result_df["ground_truth"] = batch_df["isFraud"].map({0: "Normal", 1: "FRAUD"}).values

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", len(preds))
        c2.metric("Flagged as fraud", int(preds.sum()))
        c3.metric("Normal", int(len(preds) - preds.sum()))

        st.dataframe(result_df, use_container_width=True)
        st.download_button("⬇️ Download results", result_df.to_csv(index=False), "predictions.csv")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# SINGLE TRANSACTION
# ═════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧾 Transaction details")

    # ── MANUAL ENTRY — PaySim schema ─────────────────────────────────────────
    if mode == "Enter a transaction":
        st.caption(
            "Enter real PaySim financial transaction fields. "
            "Balance delta and zero-flag features are computed automatically from these inputs."
        )

        customer_id = st.text_input(
            "Sender account ID (nameOrig)",
            placeholder="e.g. C1234567890",
            help="PII is obfuscated before logging"
        )

        c1f, c2f = st.columns(2)
        tx_type = c1f.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            index=0,
            help="Fraud only occurs on TRANSFER and CASH_OUT in PaySim"
        )
        amount = c2f.number_input(
            "Amount (₹)", min_value=0.0, max_value=10_000_000.0,
            value=50_000.0, step=1_000.0
        )

        c3f, c4f = st.columns(2)
        old_balance_orig = c3f.number_input(
            "Sender opening balance (oldbalanceOrg)",
            min_value=0.0, max_value=100_000_000.0,
            value=50_000.0, step=1_000.0,
            help="Sender's balance before this transaction"
        )
        new_balance_orig = c4f.number_input(
            "Sender closing balance (newbalanceOrig)",
            min_value=0.0, max_value=100_000_000.0,
            value=0.0, step=1_000.0,
            help="Sender's balance after this transaction. Set to 0 to simulate account drain."
        )

        c5f, c6f = st.columns(2)
        old_balance_dest = c5f.number_input(
            "Recipient opening balance (oldbalanceDest)",
            min_value=0.0, max_value=100_000_000.0,
            value=0.0, step=1_000.0
        )
        new_balance_dest = c6f.number_input(
            "Recipient closing balance (newbalanceDest)",
            min_value=0.0, max_value=100_000_000.0,
            value=0.0, step=1_000.0
        )

        c7f, c8f = st.columns(2)
        step = c7f.number_input(
            "Step (simulation hour)", min_value=1, max_value=744, value=1,
            help="Hour of the simulation — velocity proxy"
        )
        orig_tx_count = c8f.number_input(
            "Sender prior tx count (orig_tx_count_so_far)",
            min_value=0.0, value=0.0, step=1.0,
            help="Cumulative transactions sent by this account. Leave 0 if unknown (velocity warning will fire)."
        )
        orig_amount_sum = st.number_input(
            "Sender cumulative amount (orig_amount_sum_so_far)",
            min_value=0.0, value=0.0, step=1000.0,
            help="Total amount sent by this account to date. Leave 0 if unknown."
        )

        # Compute engineered features from raw inputs (mirrors engineer_features())
        balance_delta_orig = new_balance_orig + amount - old_balance_orig
        balance_delta_dest = old_balance_dest + amount - new_balance_dest
        balance_zero_orig  = 1.0 if new_balance_orig == 0.0 else 0.0
        balance_zero_dest  = 1.0 if old_balance_dest == 0.0 else 0.0

        # One-hot encode transaction type
        type_vec = {t: 0.0 for t in ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]}
        type_vec[tx_type] = 1.0

        # Build 17-feature vector matching FEATURE_NAMES exactly:
        # amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
        # balance_delta_orig, balance_delta_dest, balance_zero_orig, balance_zero_dest,
        # orig_tx_count_so_far, orig_amount_sum_so_far, step,
        # type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
        input_data = [
            amount, old_balance_orig, new_balance_orig,
            old_balance_dest, new_balance_dest,
            balance_delta_orig, balance_delta_dest,
            balance_zero_orig, balance_zero_dest,
            orig_tx_count, orig_amount_sum,
            float(step),
            type_vec["CASH_IN"], type_vec["CASH_OUT"], type_vec["DEBIT"],
            type_vec["PAYMENT"], type_vec["TRANSFER"],
        ]

        # Risk signal summary
        signals = []
        if tx_type in ["TRANSFER","CASH_OUT"]:   signals.append("high-risk tx type")
        if balance_zero_orig:                     signals.append("sender balance wiped to 0")
        if balance_zero_dest:                     signals.append("recipient had zero balance")
        if abs(balance_delta_dest) > 1000:        signals.append(f"recipient balance error ₹{balance_delta_dest:,.0f}")
        risk_str = " · ".join(signals) if signals else "no obvious fraud signals"
        st.info(f"**Fraud signals detected:** {risk_str}  |  Amount: ₹{amount:,.0f}  |  Type: {tx_type}")

    # ── SAMPLE FROM DATASET ──────────────────────────────────────────────────
    else:
        if data is not None:
            sample = data.sample(1, random_state=None)
            from src.preprocess import engineer_features
            X_samp     = engineer_features(sample)
            input_data = X_samp[FEATURE_NAMES].values.flatten().tolist()
            gt = int(sample["isFraud"].values[0])

            amount_val  = float(sample["amount"].values[0])
            tx_type_val = str(sample["type"].values[0])
            customer_id = f"dataset_user_{sample.index[0]}"

            st.success("✅ Real transaction loaded from dataset")

            info_cols = st.columns(4)
            info_cols[0].metric("Amount", f"₹{amount_val:,.2f}")
            info_cols[1].metric("Type", tx_type_val)
            info_cols[2].metric("Ground truth", "⚠️ FRAUD" if gt else "✅ Normal")
            info_cols[3].metric("Transaction ID", f"#{sample.index[0]}")

            with st.expander("View raw transaction fields"):
                display_cols = ["step","type","amount","oldbalanceOrg","newbalanceOrig",
                                "oldbalanceDest","newbalanceDest","isFraud"]
                st.dataframe(sample[[c for c in display_cols if c in sample.columns]],
                             use_container_width=True)
                st.caption(
                    "Features shown are raw PaySim fields. The model uses 17 engineered "
                    "features derived from these: balance deltas, zero-balance flags, "
                    "type one-hot encoding, and velocity counters."
                )
        else:
            st.error("paysim_sample.csv not found.")
            input_data = [0.0] * N_FEATURES

# ── PREDICTION ────────────────────────────────────────────────────────────────
with col2:
    st.subheader("📊 Result")

    if st.button("🚀 Run fraud check", use_container_width=True):
        # PII Obfuscation
        obfuscated_id = "N/A"
        if mode == "Enter a transaction" and customer_id:
            if "@" in customer_id:
                parts = customer_id.split("@")
                obfuscated_id = f"{parts[0][:2]}***@{parts[1]}" if len(parts[0]) > 2 else f"***@{parts[1]}"
            elif len(customer_id) >= 4:
                obfuscated_id = f"****-****-****-{customer_id[-4:]}"
            else:
                obfuscated_id = "****"
            st.info(f"🔒 Processing Zero-Trust ID: {obfuscated_id}")
        elif mode == "Random real sample":
            obfuscated_id = f"****-{(sample.index[0] % 9999):04d}"

        df_input  = make_dataframe(input_data)
        df_scaled = apply_scaler(df_input, scaler)
        prob      = float(model.predict_proba(df_scaled)[0][1])
        
        # 3. Audit Logging
        st.session_state.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": obfuscated_id,
            "amount": amount if mode == "Enter a transaction" else amount_val,
            "prob": round(prob, 4),
            "verdict": "FRAUD" if prob >= threshold else "NORMAL",
            "mode": mode
        })

        st.metric("Fraud probability", f"{prob:.1%}")
        st.progress(prob)

        if prob >= threshold:
            st.error("⚠️ FRAUD DETECTED\nThis transaction has been flagged. Review before processing.")
        else:
            st.success("✅ Transaction looks legitimate")

        # Risk gauge bar
        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.barh(["Normal", "Fraud"], [1 - prob, prob],
                color=["#2ecc71", "#e74c3c"], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.2,
                   label=f"Threshold ({threshold:.2f})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── SHAP explanation ─────────────────────────────────────────────────
        st.subheader("🔍 Why this decision?")

        shap_vals = explainer.shap_values(df_scaled)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[1]).flatten()[:N_FEATURES]
        else:
            sv = np.array(shap_vals).flatten()[:N_FEATURES]

        # Human-readable labels for PaySim SHAP features
        def _feat_label(name):
            labels = {
                "amount":                "Transaction amount",
                "oldbalanceOrg":         "Sender opening balance",
                "newbalanceOrig":        "Sender closing balance",
                "oldbalanceDest":        "Recipient opening balance",
                "newbalanceDest":        "Recipient closing balance",
                "balance_delta_orig":    "Sender balance accounting error",
                "balance_delta_dest":    "Recipient balance accounting error",
                "balance_zero_orig":     "Sender balance wiped to zero",
                "balance_zero_dest":     "Recipient had zero balance",
                "orig_tx_count_so_far":  "Sender prior transaction count",
                "orig_amount_sum_so_far":"Sender cumulative amount sent",
                "step":                  "Simulation step (time proxy)",
                "type_CASH_IN":          "Type: CASH_IN",
                "type_CASH_OUT":         "Type: CASH_OUT",
                "type_DEBIT":            "Type: DEBIT",
                "type_PAYMENT":          "Type: PAYMENT",
                "type_TRANSFER":         "Type: TRANSFER",
            }
            return labels.get(name, name)

        shap_df = pd.DataFrame({
            "Feature": [_feat_label(f) for f in FEATURE_NAMES],
            "SHAP value": sv,
            "Abs": np.abs(sv),
        }).sort_values("Abs", ascending=False).head(8)

        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP value"]]
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.barh(shap_df["Feature"][::-1], shap_df["SHAP value"][::-1], color=colors[::-1])
        ax2.axvline(0, color="grey", linewidth=0.8)
        ax2.set_xlabel("Impact on fraud score")
        ax2.set_title("Top factors in this decision")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.caption("🔴 Red bars increase fraud probability · 🟢 Green bars reduce it")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Model insights")

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions trained on", f"{metrics['total_transactions']:,}")
    c2.metric("Fraud cases in dataset",         f"{metrics['fraud_count']:,}")
    c3.metric("Fraud rate",                     f"{metrics['fraud_pct']:.3f}%")
    c4.metric("Best model",                     metrics["best_model"])

    st.subheader("🏆 Model comparison")
    comparison = metrics.get("comparison", {})
    if comparison:
        rows = []
        for name, m in comparison.items():
            rows.append({
                "Model":     name,
                "PR-AUC":    m.get("pr_auc", "—"),
                "ROC-AUC":   m.get("roc_auc", "—"),
                "Precision": m.get("precision", "—"),
                "Recall":    m.get("recall", "—"),
                "F1 Score":  m.get("f1", "—"),
                "Selected":  "✅ Best" if name == metrics["best_model"] else "",
            })
        comp_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(comp_df.style.highlight_max(
            subset=["PR-AUC","ROC-AUC","Precision","Recall","F1 Score"], color="#d4edda"
        ), use_container_width=True)

    cv = metrics.get("cross_validation")
    if cv:
        st.subheader("📐 Cross-validation variance (5-fold)")
        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("CV PR-AUC mean", f"{cv['mean_pr_auc']:.4f}")
        cv2.metric("CV PR-AUC std",  f"± {cv['std_pr_auc']:.4f}")
        cv3.metric("Folds", cv["n_folds"])
        st.caption(
            "ADASYN applied inside each fold — validation folds contain only real transactions. "
            f"Fold scores: {cv['fold_scores']}"
        )

    opt = metrics.get("optuna_tuning")
    if opt and opt.get("best_params"):
        with st.expander("⚙️ Optuna tuning results"):
            st.json(opt)

    st.subheader("Confusion matrix (test set)")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Normal", "Actual: Fraud"],
            columns=["Predicted: Normal", "Predicted: Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.subheader("🔬 Which signals matter most globally?")
    global_shap = metrics.get("global_shap", {})
    if global_shap:
        def _to_float(v):
            if isinstance(v, (list, tuple)):
                return float(np.mean(np.abs(v)))
            return float(v)

        def _global_label(name):
            labels = {
                "amount":                "Transaction amount",
                "oldbalanceOrg":         "Sender opening balance",
                "newbalanceOrig":        "Sender closing balance",
                "oldbalanceDest":        "Recipient opening balance",
                "newbalanceDest":        "Recipient closing balance",
                "balance_delta_orig":    "Sender balance accounting error",
                "balance_delta_dest":    "Recipient balance accounting error",
                "balance_zero_orig":     "Sender balance wiped to zero",
                "balance_zero_dest":     "Recipient had zero balance",
                "orig_tx_count_so_far":  "Sender prior tx count",
                "orig_amount_sum_so_far":"Sender cumulative amount",
                "step":                  "Simulation step (time)",
                "type_CASH_IN":          "Type: CASH_IN",
                "type_CASH_OUT":         "Type: CASH_OUT",
                "type_DEBIT":            "Type: DEBIT",
                "type_PAYMENT":          "Type: PAYMENT",
                "type_TRANSFER":         "Type: TRANSFER",
            }
            return labels.get(name, name)

        shap_series = pd.Series(
            {_global_label(k): _to_float(v) for k, v in global_shap.items()}
        ).sort_values(ascending=False).head(12)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        shap_series[::-1].plot(kind="barh", ax=ax3, color="#3498db")
        ax3.set_xlabel("Mean influence on fraud score (mean |SHAP|)")
        ax3.set_title("Top fraud signals across all transactions")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.caption("Higher bar = this signal has more influence on whether a transaction is flagged as fraud.")

elif data is not None:
    fraud_ratio = data["isFraud"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", f"{len(data):,}")
    c2.metric("Fraud %", f"{fraud_ratio*100:.4f}%")
    c3.metric("Normal %", f"{(1-fraud_ratio)*100:.4f}%")

st.markdown("---")
st.subheader("🔒 Forensic Audit Log (Zero-Trust)")
st.caption("Secure circular buffer (max 500 events). PII is irreversibly masked.")
if len(st.session_state.audit_log) > 0:
    st.dataframe(pd.DataFrame(st.session_state.audit_log).iloc[::-1], use_container_width=True)
else:
    st.info("No events logged yet.")

st.caption("FraudGuard v2 · PaySim dataset · PR-AUC selection · CalibratedClassifierCV · SHAP explainability")