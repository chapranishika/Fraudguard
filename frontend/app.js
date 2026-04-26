/**
 * FraudGuard — Operations Terminal
 * Targets PaySim feature schema (15 features)
 */

const API_BASE = 'http://127.0.0.1:8000';
const API_KEY  = 'demo-key-123';
const HEADERS  = { 'Content-Type': 'application/json', 'X-API-Key': API_KEY };

const XAI_MAP = {
    "amount":             "Transaction amount",
    "oldbalanceOrg":      "Sender opening balance",
    "newbalanceOrig":     "Sender closing balance",
    "oldbalanceDest":     "Recipient opening balance",
    "newbalanceDest":     "Recipient closing balance",
    "balance_delta_orig": "Sender balance accounting error",
    "balance_delta_dest": "Recipient balance accounting error",
    "balance_zero_orig":  "Sender balance wiped to zero",
    "balance_zero_dest":  "Recipient had zero balance",
    "step":               "Time step (velocity proxy)",
    "type_CASH_OUT":      "Cash-out transaction",
    "type_TRANSFER":      "Transfer transaction",
    "type_PAYMENT":       "Payment transaction",
    "type_CASH_IN":       "Cash-in transaction",
    "type_DEBIT":         "Debit transaction",
};

let terminalQueue = [];
let manualIdCounter = 1;

// ── Feature builder from manual form ────────────────────────────────────────
function buildFeaturesFromForm() {
    const amount    = parseFloat(document.getElementById('t-amount')?.value) || 0;
    const oldOrig   = parseFloat(document.getElementById('t-old-orig')?.value) || 0;
    const newOrig   = parseFloat(document.getElementById('t-new-orig')?.value) || Math.max(0, oldOrig - amount);
    const oldDest   = parseFloat(document.getElementById('t-old-dest')?.value) || 0;
    const newDest   = parseFloat(document.getElementById('t-new-dest')?.value) || oldDest + amount;
    const step      = parseInt(document.getElementById('t-step')?.value) || 1;
    const txType    = document.getElementById('t-type')?.value || 'CASH_OUT';

    const balDeltaOrig = newOrig + amount - oldOrig;
    const balDeltaDest = oldDest + amount - newDest;
    const balZeroOrig  = newOrig === 0 ? 1 : 0;
    const balZeroDest  = oldDest === 0 ? 1 : 0;

    const types = ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'];
    const typeVec = types.map(t => t === txType ? 1 : 0);

    return [
        amount, oldOrig, newOrig, oldDest, newDest,
        balDeltaOrig, balDeltaDest, balZeroOrig, balZeroDest,
        step,
        ...typeVec
    ];
}

// ── Render SHAP bar chart ───────────────────────────────────────────────────
function renderShapBars(containerId, topShap) {
    const container = document.getElementById(containerId);
    if (!container || !topShap) return;
    const entries = Object.entries(topShap).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0,8);
    const maxAbs  = Math.max(...entries.map(e => Math.abs(e[1])), 0.001);

    container.innerHTML = entries.map(([feat, val]) => {
        const label    = XAI_MAP[feat] || feat;
        const pct      = Math.abs(val) / maxAbs * 100;
        const color    = val > 0 ? 'var(--danger)' : 'var(--success)';
        const dir      = val > 0 ? '▲ increases risk' : '▼ reduces risk';
        return `
        <div style="margin-bottom:0.6rem">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
            <span>${label}</span>
            <span style="color:${color};font-weight:600">${val > 0 ? '+' : ''}${val.toFixed(4)} ${dir}</span>
          </div>
          <div style="background:rgba(255,255,255,0.08);border-radius:3px;height:8px">
            <div style="width:${pct}%;background:${color};height:8px;border-radius:3px;transition:width 0.4s"></div>
          </div>
        </div>`;
    }).join('');
}

// ── Add row to audit table ──────────────────────────────────────────────────
function appendAuditRow(id, amount, prob, isF, ts, txType) {
    const tbody = document.getElementById('audit-body');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${ts.replace('T',' ').split('.')[0]} UTC</td>
      <td>${id.slice(0,8)}…</td>
      <td>₹${parseFloat(amount).toFixed(2)}</td>
      <td><span class="badge ${isF ? 'danger' : 'success'}">${isF ? '⚠ FRAUD' : '✓ NORMAL'}</span></td>
      <td>${(prob*100).toFixed(2)}%</td>`;
    tbody.prepend(tr);
    // Cap at 100 rows
    while (tbody.rows.length > 100) tbody.deleteRow(tbody.rows.length-1);
}

document.addEventListener('DOMContentLoaded', async () => {

    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', () => {
        const r = document.documentElement;
        r.setAttribute('data-theme', r.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
    });

    // View switch
    const vTerminal  = document.getElementById('view-terminal');
    const vResearch  = document.getElementById('view-research');
    document.getElementById('mode-terminal')?.addEventListener('click', () => {
        vTerminal.style.display='grid'; vResearch.style.display='none';
        document.body.classList.remove('theme-classic-white');
    });
    document.getElementById('mode-research')?.addEventListener('click', () => {
        vTerminal.style.display='none'; vResearch.style.display='grid';
        document.body.classList.add('theme-classic-white');
    });

    // ── Load metrics on boot ──────────────────────────────────────────────────
    try {
        const res = await fetch(`${API_BASE}/metrics`, { headers: HEADERS });
        if (res.ok) {
            const m = await res.json();
            const auc = (m.pr_auc_calibrated ?? m.pr_auc ?? 0).toFixed(4);
            const f1  = (m.f1_calibrated ?? m.f1 ?? 0).toFixed(4);
            document.getElementById('kpi-auc').innerText = `PR-AUC: ${auc}`;
            document.getElementById('kpi-f1').innerText  = `F1: ${f1}`;
            if (m.confusion_matrix) {
                document.getElementById('cm-tn').innerText = m.confusion_matrix[0][0];
                document.getElementById('cm-fp').innerText = m.confusion_matrix[0][1];
                document.getElementById('cm-fn').innerText = m.confusion_matrix[1][0];
                document.getElementById('cm-tp').innerText = m.confusion_matrix[1][1];
            }
            if (m.best_model) document.getElementById('kpi-model').innerText = m.best_model;
            if (m.pr_auc_calibrated) document.getElementById('kpi-precision').innerText = m.precision_calibrated?.toFixed(4) || '--';
        }
    } catch(e) { console.warn('Metrics load failed', e); }

    // ── Add to queue ──────────────────────────────────────────────────────────
    document.getElementById('btn-add-manual')?.addEventListener('click', () => {
        if (terminalQueue.length >= 5) return alert('Queue full (max 5). Run fraud check first.');
        const features = buildFeaturesFromForm();
        const amount   = features[0];
        const txType   = document.getElementById('t-type')?.value || 'CASH_OUT';
        const id = `tx_m***_${String(manualIdCounter++).padStart(2,'0')}`;
        terminalQueue.push({ id, features, amount, txType });
        document.getElementById('queue-count').innerText = `Queue: ${terminalQueue.length}/5`;
        const row = document.createElement('div');
        row.className = 'feed-row'; row.id = `row-${id}`;
        row.innerHTML = `<div>${id}</div><div>₹${amount.toFixed(2)}</div><div>${txType}</div><div><span class="badge">PENDING</span></div>`;
        document.getElementById('live-feed').prepend(row);
    });

    // ── Run fraud check ────────────────────────────────────────────────────────
    document.getElementById('btn-simulate')?.addEventListener('click', async () => {
        if (terminalQueue.length === 0) return alert('Queue is empty.');
        const btn = document.getElementById('btn-simulate');
        btn.classList.add('loading');
        try {
            const payload = { transactions: terminalQueue.map(item => ({ features: item.features, threshold: 0.5 })) };
            const res = await fetch(`${API_BASE}/predict/batch`, { method:'POST', headers: HEADERS, body: JSON.stringify(payload) });
            if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
            const data = await res.json();

            data.results.forEach((r, i) => {
                const tx  = terminalQueue[i];
                const row = document.getElementById(`row-${tx.id}`);
                if (row) row.lastElementChild.innerHTML = r.is_fraud
                    ? `<span class="badge danger">FRAUD</span>`
                    : `<span class="badge success">NORMAL</span>`;
                appendAuditRow(tx.id, tx.amount, r.fraud_probability, r.is_fraud,
                    new Date().toISOString(), tx.txType);
                // Update live KPIs
                const kpiTotal   = document.getElementById('kpi-total');
                const kpiBlocked = document.getElementById('kpi-blocked');
                if (kpiTotal)   kpiTotal.innerText   = parseInt(kpiTotal.innerText||0) + 1;
                if (kpiBlocked && r.is_fraud) kpiBlocked.innerText = parseInt(kpiBlocked.innerText||0) + 1;
            });

            // Show SHAP for highest-risk transaction
            const maxRisk = data.results.reduce((a,b) => a.fraud_probability > b.fraud_probability ? a : b);
            renderShapBars('shap-bar-chart', maxRisk.top_shap_features);
            const aiBox = document.getElementById('terminal-ai-text');
            if (aiBox) {
                aiBox.style.display = 'block';
                const topKey  = Object.keys(maxRisk.top_shap_features)[0];
                const topName = XAI_MAP[topKey] || topKey;
                aiBox.innerHTML = maxRisk.is_fraud
                    ? `<strong>Batch complete:</strong> Highest-risk transaction flagged at <strong>${(maxRisk.fraud_probability*100).toFixed(1)}%</strong>. Primary signal: <em>${topName}</em>.`
                    : `<strong>Batch complete:</strong> All transactions cleared. No significant anomalies.`;
            }

            terminalQueue = [];
            document.getElementById('queue-count').innerText = 'Queue: 0/5';
        } catch(err) {
            alert('API Error: ' + err.message);
        } finally {
            btn.classList.remove('loading');
        }
    });

    // ── Batch CSV upload ───────────────────────────────────────────────────────
    document.getElementById('btn-batch')?.addEventListener('click', () => {
        const fi = document.createElement('input');
        fi.type = 'file'; fi.accept = '.csv';
        fi.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const text  = await file.text();
            const lines = text.split('\n').filter(l => l.trim());
            let start   = 0;
            if (isNaN(parseFloat(lines[0].split(',')[0]))) start = 1;
            const transactions = [];
            for (let i = start; i < lines.length; i++) {
                const cols     = lines[i].split(',');
                const features = Array.from({length: 15}, (_,j) => parseFloat(cols[j]) || 0);
                transactions.push({ features, threshold: 0.5 });
            }
            if (!transactions.length) return alert('CSV is empty or invalid.');
            document.getElementById('batch-modal').classList.add('active');
            try {
                const res  = await fetch(`${API_BASE}/predict/batch`, {
                    method:'POST', headers: HEADERS, body: JSON.stringify({ transactions })
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                document.getElementById('batch-progress').style.width = '100%';
                document.getElementById('batch-total').innerText  = data.total;
                document.getElementById('batch-fraud').innerText  = data.fraud_count;
                document.getElementById('batch-clean').innerText  = data.total - data.fraud_count;
                const closeBtn = document.getElementById('btn-close-batch');
                if (closeBtn) { closeBtn.disabled=false; closeBtn.innerText='Close Report'; }
            } catch(err) { alert('Batch failed: ' + err.message); }
        };
        fi.click();
    });

    document.getElementById('btn-close-batch')?.addEventListener('click', () => {
        document.getElementById('batch-modal').classList.remove('active');
        document.getElementById('batch-progress').style.width = '0%';
    });

    // ── Research Lab: random sample ──────────────────────────────────────────
    const btnRandom = document.getElementById('btn-random-row');
    if (btnRandom) {
        btnRandom.disabled = false;
        btnRandom.addEventListener('click', async () => {
            btnRandom.innerText = 'Fetching…'; btnRandom.disabled = true;
            try {
                const sRes  = await fetch(`${API_BASE}/random_sample`, { headers: HEADERS });
                if (!sRes.ok) throw new Error('Failed to fetch sample.');
                const s     = await sRes.json();

                document.getElementById('sim-details').style.display = 'block';
                document.getElementById('sim-amount').innerText = `₹${s.amount.toFixed(2)}`;
                document.getElementById('sim-time').innerText   = `Step ${s.step} | ${s.tx_type}`;
                const tSpan = document.getElementById('sim-truth');
                tSpan.innerText   = s.is_fraud ? 'FRAUD' : 'NORMAL';
                tSpan.style.color = s.is_fraud ? '#ef4444' : '#10b981';

                const pRes = await fetch(`${API_BASE}/predict`, {
                    method:'POST', headers: HEADERS,
                    body: JSON.stringify({ features: s.features, threshold: 0.5 })
                });
                const pred = await pRes.json();
                document.getElementById('research-score').innerText = `Probability: ${(pred.fraud_probability*100).toFixed(2)}%`;

                renderShapBars('research-shap-chart', pred.top_shap_features);

                const aiBox = document.getElementById('ai-explanation-text');
                if (aiBox) {
                    aiBox.style.display = 'block';
                    const topKey  = Object.keys(pred.top_shap_features)[0];
                    const topName = XAI_MAP[topKey] || topKey;
                    const match   = s.is_fraud === pred.is_fraud;
                    aiBox.innerHTML = `<strong>Model report:</strong> Classified as <strong>${pred.is_fraud ? 'HIGH RISK' : 'SAFE'}</strong> (${(pred.fraud_probability*100).toFixed(2)}%). Primary factor: <em>${topName}</em>. Ground truth match: <strong>${match ? '✓ YES' : '✗ NO'}</strong>.`;
                    if (pred.input_warnings?.length) {
                        aiBox.innerHTML += `<br><small style="color:var(--danger)">⚠ Input warnings: ${pred.input_warnings.join('; ')}</small>`;
                    }
                }

                // Research tab: load threshold curve into table
                loadThresholdTable();
            } catch(err) {
                alert('Error: ' + err.message);
            } finally {
                btnRandom.innerText = 'Pull Random Transaction';
                btnRandom.disabled = false;
            }
        });
    }

    // ── Research Lab: comparison table from metrics API ──────────────────────
    try {
        const res = await fetch(`${API_BASE}/metrics`, { headers: HEADERS });
        if (res.ok) {
            const m = await res.json();
            const cmp = m.comparison || {};
            if (cmp.XGBoost) {
                document.getElementById('comp-xgb-auc')?.setAttribute('id','comp-xgb-auc');
                const setCell = (id, val) => { const el=document.getElementById(id); if(el) el.innerText=val??'--'; };
                setCell('comp-xgb-auc',   cmp.XGBoost?.pr_auc);
                setCell('comp-xgb-f1',    cmp.XGBoost?.f1);
                setCell('comp-xgb-prec',  cmp.XGBoost?.precision);
                setCell('comp-rf-auc',    cmp.RandomForest?.pr_auc);
                setCell('comp-rf-f1',     cmp.RandomForest?.f1);
                setCell('comp-rf-prec',   cmp.RandomForest?.precision);
                setCell('comp-lgbm-auc',  cmp.LightGBM?.pr_auc);
                setCell('comp-lgbm-f1',   cmp.LightGBM?.f1);
                setCell('comp-lgbm-prec', cmp.LightGBM?.precision);
            }
        }
    } catch(e) {}
});

// ── Threshold analysis table ─────────────────────────────────────────────────
async function loadThresholdTable() {
    try {
        const res = await fetch(`${API_BASE}/threshold_analysis`, { headers: HEADERS });
        if (!res.ok) return;
        const data = await res.json();
        const tableEl = document.getElementById('threshold-table-body');
        if (!tableEl) return;
        // Sample ~20 evenly spaced points from the curve
        const curve = data.curve || [];
        const step  = Math.max(1, Math.floor(curve.length / 20));
        const rows  = curve.filter((_,i) => i % step === 0).slice(0,20);
        tableEl.innerHTML = rows.map(r =>
            `<tr>
               <td>${r.threshold}</td>
               <td>${r.precision}</td>
               <td>${r.recall}</td>
               <td>${r.f1}</td>
             </tr>`
        ).join('');
    } catch(e) {}
}
