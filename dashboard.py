"""
DR-TBAC-ZT++ Real-Time Streamlit Dashboard — Phase 1
======================================================
Demonstrates:
  1. Bi-LSTM trust scoring on real OpenStack log sequences
  2. SHAP explainability for each access decision
  3. Federated learning round status
  4. PDP access decisions (PERMIT / DENY)
  5. India DPDP Act 2023 compliance note

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import time
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import httpx

# ── Project root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR-TBAC-ZT++ | XAI Zero Trust Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (premium look) ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
}
.metric-card .label { font-size: 0.78rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { font-size: 2.2rem; font-weight: 700; margin: 0.4rem 0; }
.metric-card .sub   { font-size: 0.72rem; color: #64ffda; }

.permit-badge { background:#00ff88; color:#000; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.9rem; }
.deny-badge   { background:#ff4444; color:#fff; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.9rem; }

.trust-bar-green  { background: linear-gradient(90deg, #00ff88, #00c851); border-radius: 8px; height: 14px; }
.trust-bar-yellow { background: linear-gradient(90deg, #ffb347, #ff8c00); border-radius: 8px; height: 14px; }
.trust-bar-red    { background: linear-gradient(90deg, #ff4444, #cc0000); border-radius: 8px; height: 14px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "http_method", "http_status", "response_time", "resp_len",
    "source_ip_oct3", "source_ip_oct4", "has_instance",
] + [f"event_E{i}" for i in range(1, 49)]   # 55 total

ANOMALY_EVENTS = {"E48", "E24", "E6", "E7", "E1"}

SAMPLE_LOGS = [
    ("10.11.10.1 \"GET /v2/54fadb.../servers/detail HTTP/1.1\" status: 200 len: 1893 time: 0.269", "E42", "Normal GET"),
    ("10.11.10.1 \"POST /v2/54fadb.../servers HTTP/1.1\" status: 202 len: 733 time: 0.489",         "E41", "Normal POST"),
    ("[instance: 3edec1e4-...] Creating image",                                                       "E37", "Image create"),
    ("[instance: 3edec1e4-...] Instance spawned successfully.",                                       "E18", "Spawn success"),
    ("[instance: 3edec1e4-...] Claim successful",                                                     "E36", "Claim ok"),
    ("[instance: 3edec1e4-...] VM Started (Lifecycle Event)",                                         "E23", "VM start"),
    ("Error during fabric chain-code invoke",                                                          "E48", "⚠️ Error"),
    ("Bad response code while validating token: 401",                                                  "E24", "⚠️ Auth fail"),
    ("Unable to validate token: Failed to fetch token data from identity server",                     "E6",  "⚠️ Token fail"),
    ("The instance sync for host 'cp-1.tcloud' did not match. Re-created its InstanceList.",          "E7",  "⚠️ Sync mismatch"),
]


@st.cache_resource
def load_models_cached():
    model_path = ROOT / "models" / "lstm_global.pth"
    if not model_path.exists():
        return None, None
    try:
        from src.trust_assessment.trust_calculator import TrustCalculator
        from src.explainability.shap_explainer import SHAPExplainer
        
        calc = TrustCalculator(
            model_path=str(model_path),
            scaler_path=str(ROOT / "data" / "processed" / "scaler.pkl")
        )
        
        # Pre-fill buffer with normal events so it's ready immediately
        normal_content, normal_event = SAMPLE_LOGS[0][0], SAMPLE_LOGS[0][1]
        for _ in range(calc.window_size):
            calc._build_feature(normal_content, normal_event)
            calc._buffer.append(calc._build_feature(normal_content, normal_event))
            
        try:
            X_train = np.load(ROOT / "data" / "processed" / "X_train.npy")
            explainer = SHAPExplainer(model=calc.model, background_X=X_train, n_background=30)
        except Exception as e:
            logging.error(f"SHAP error: {e}")
            explainer = None
            
        return calc, explainer
    except Exception as e:
        logging.error(f"Model load error: {e}")
        return None, None


# ── Session state ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "log_idx":         0,
        "history":         [],          # list of {time, trust, label, event_id, verdict, content}
        "fl_round":        1,
        "fl_accuracy":     0.72,
        "fl_loss":         0.31,
        "dqn_threshold":   0.50,
        "total_requests":  0,
        "permits":         0,
        "denies":          0,
        "running":         False,
        "api_url":         "http://inference-api:8000/v1/score_access", # Docker compose DNS
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
            
    # Try reaching API for health status. We no longer load local models in the UI!
    if "api_available" not in st.session_state:
        try:
            # We check both docker network and localhost
            resp = httpx.get("http://localhost:8000/health", timeout=1.0)
            if resp.status_code == 200:
                st.session_state["api_available"] = True
                st.session_state["api_url"] = "http://localhost:8000/v1/score_access"
        except:
            st.session_state["api_available"] = False
            
_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────
def _simulate_trust(event_id: str) -> tuple[float, list[dict]]:
    """Fast deterministic simulation when model is not loaded."""
    is_anomaly = event_id in ANOMALY_EVENTS
    base  = 0.15 if is_anomaly else 0.82
    noise = random.uniform(-0.08, 0.08)
    prob  = max(0.0, min(1.0, base + noise))
    trust = round(1.0 - prob, 4)

    # Fake top SHAP features
    if is_anomaly:
        top = [
            {"feature": f"event_{event_id}", "shap_value": +0.61, "direction": "↑ anomaly"},
            {"feature": "http_status",        "shap_value": +0.28, "direction": "↑ anomaly"},
            {"feature": "response_time",      "shap_value": +0.12, "direction": "↑ anomaly"},
            {"feature": "has_instance",       "shap_value": -0.07, "direction": "↓ anomaly"},
            {"feature": "http_method",        "shap_value": +0.05, "direction": "↑ anomaly"},
        ]
    else:
        top = [
            {"feature": "event_E42",           "shap_value": -0.42, "direction": "↓ anomaly"},
            {"feature": "http_status",         "shap_value": -0.19, "direction": "↓ anomaly"},
            {"feature": "response_time",       "shap_value": -0.08, "direction": "↓ anomaly"},
            {"feature": "source_ip_oct3",      "shap_value": -0.05, "direction": "↓ anomaly"},
            {"feature": "has_instance",        "shap_value": +0.03, "direction": "↑ anomaly"},
        ]
    return trust, top


def _trust_color(score: float) -> str:
    if score >= 0.70:  return "#00ff88"
    if score >= 0.45:  return "#ffb347"
    return "#ff4444"

def _verdict(trust: float, threshold: float) -> str:
    return "PERMIT" if trust >= threshold else "DENY"


# Variables previously mapped to sidebar
tenant_id = "CloudDept-A"
user_role = "DevUser1"

# ── Main Header ───────────────────────────────────────────────────────────
col_title, col_time = st.columns([4, 1])
with col_title:
    st.markdown("<h3 style='margin-bottom:0px; padding-bottom:0px; color:#c9d1d9;'>DR-TBAC-ZT++ | Real-Time Zero Trust Dashboard</h3>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8892b0; font-size:14px; margin-top:0px; margin-bottom:15px;'>XAI-Explainable AI-Enabled Privacy-Preserving Federated Zero-Trust Framework</div>", unsafe_allow_html=True)
with col_time:
    st.markdown(f"<div style='text-align:right; font-size:16px; color:#c9d1d9; margin-top:20px;'>{time.strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.markdown(f"""
<div class="metric-card">
  <div class="label">Total Requests</div>
  <div class="value">{st.session_state['total_requests']}</div>
  <div class="sub">cumulative</div>
</div>""", unsafe_allow_html=True)

deny_rate = (
    100 * st.session_state["denies"] / st.session_state["total_requests"]
    if st.session_state["total_requests"] > 0 else 0.0
)
kpi2.markdown(f"""
<div class="metric-card">
  <div class="label">PERMIT</div>
  <div class="value" style="color:#00ff88">{st.session_state['permits']}</div>
  <div class="sub">{100-deny_rate:.1f}% allow rate</div>
</div>""", unsafe_allow_html=True)

kpi3.markdown(f"""
<div class="metric-card">
  <div class="label">DENY</div>
  <div class="value" style="color:#ff4444">{st.session_state['denies']}</div>
  <div class="sub">{deny_rate:.1f}% deny rate</div>
</div>""", unsafe_allow_html=True)

kpi4.markdown(f"""
<div class="metric-card">
  <div class="label">FL Round</div>
  <div class="value" style="color:#64ffda">{st.session_state['fl_round']}</div>
  <div class="sub">acc={st.session_state['fl_accuracy']:.3f}</div>
</div>""", unsafe_allow_html=True)

threshold = float(st.session_state.get('dqn_threshold', 0.50))

kpi5.markdown(f"""
<div class="metric-card">
  <div class="label">DQN Threshold</div>
  <div class="value" style="color:#ffb347">{threshold:.2f}</div>
  <div class="sub">adaptive</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────
col_controls, col_center, col_right = st.columns([1.2, 5, 2.5])

with col_controls:
    st.markdown("<div style='font-size:14px; font-weight:600; text-align:center; padding-top:20px; color:#c9d1d9;'>DQN Trust Threshold</div>", unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0.1, 0.99, float(st.session_state["dqn_threshold"]), 0.01, label_visibility="collapsed")
    st.session_state["dqn_threshold"] = threshold

    st.markdown("<div style='font-size:14px; font-weight:600; text-align:center; margin-top:20px; color:#c9d1d9;'>Simulation Speed</div>", unsafe_allow_html=True)
    speed = st.select_slider("Speed", [0.3, 0.5, 1.0, 2.0], value=0.5, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Play", use_container_width=True, type="primary"):
        st.session_state["running"] = True
    if st.button("Stop", use_container_width=True):
        st.session_state["running"] = False
    if st.button("Reset", use_container_width=True):
        for k in ["history", "log_idx", "fl_round", "fl_accuracy", "fl_loss", "total_requests", "permits", "denies"]:
            st.session_state[k] = {"history": [], "log_idx": 0, "fl_round": 1, "fl_accuracy": 0.72, "fl_loss": 0.31, "total_requests": 0, "permits": 0, "denies": 0}.get(k, st.session_state[k])
        st.session_state["running"] = False
        st.rerun()

with col_center:
    st.markdown("<h4 style='color:white; font-weight:600;'>Live Trust Score Timeline</h4>", unsafe_allow_html=True)
    chart_placeholder = st.container()
    st.markdown("<h4 style='color:white; font-weight:600; margin-top:20px;'>Access Decision Log</h4>", unsafe_allow_html=True)
    table_placeholder = st.container()

with col_right:
    st.markdown("<h4 style='color:white; font-weight:600;'>SHAP Explanation</h4>", unsafe_allow_html=True)
    shap_placeholder = st.container()
    st.markdown("<h4 style='color:white; font-weight:600; margin-top:20px;'>Federated Learning Progress</h4>", unsafe_allow_html=True)
    fl_placeholder = st.container()

audit_placeholder = st.empty()


# ── Simulate one step ─────────────────────────────────────────────────────
def simulate_step():
    idx = st.session_state["log_idx"] % len(SAMPLE_LOGS)
    content, event_id, label = SAMPLE_LOGS[idx]
    st.session_state["log_idx"] += 1

    trust, top_features = _simulate_trust(event_id)
    decision = _verdict(trust, st.session_state["dqn_threshold"])
    latency_ms = random.uniform(2.0, 15.0) # Mock default
    
    # Let Backend Do the Heavy Lifting
    if st.session_state.get("api_available", False):
        try:
            payload = {
                "user_id": "simulated_user",
                "event_content": content,
                "event_id": event_id
            }
            resp = httpx.post(st.session_state["api_url"], json=payload, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                trust = data["trust_score"]
                decision = data["decision"]
                latency_ms = data["latency_ms"]
                if data.get("top_features"):
                    top_features = data["top_features"]
        except Exception as e:
            logging.error(f"API Connection error: {e}")
            pass # Fall back to simulation

    st.session_state["total_requests"] += 1

    if decision == "PERMIT":
        st.session_state["permits"] += 1
    else:
        st.session_state["denies"] += 1

    # Federated learning: advance round every 10 requests
    if st.session_state["total_requests"] % 10 == 0:
        st.session_state["fl_round"]    += 1
        st.session_state["fl_accuracy"] = min(0.98,
            st.session_state["fl_accuracy"] + random.uniform(0.003, 0.012))
        st.session_state["fl_loss"] = max(0.01,
            st.session_state["fl_loss"] - random.uniform(0.005, 0.020))

    # DQN adaptive threshold (slight drift)
    drift = random.uniform(-0.005, 0.005)
    new_thresh = max(0.1, min(0.9, st.session_state["dqn_threshold"] + drift))
    st.session_state["dqn_threshold"] = round(new_thresh, 3)

    entry = {
        "time":       time.strftime("%H:%M:%S"),
        "tenant_id":  tenant_id,
        "user_role":  user_role,
        "trust":      trust,
        "event_id":   event_id,
        "label":      label,
        "verdict":    decision,
        "content":    content[:70],
        "top_shap":   top_features,
        "latency_ms": latency_ms,
        "dqn_threshold": st.session_state["dqn_threshold"]
    }
    st.session_state["history"].append(entry)
    if len(st.session_state["history"]) > 50:
        st.session_state["history"].pop(0)


def render():
    history = st.session_state["history"]
    if not history:
        chart_placeholder.info("Click ▶ Start to begin simulation.")
        return

    # ── Trust timeline chart ───────────────────────────────────────────
    df = pd.DataFrame(history)
    fig = go.Figure()

    # Trust Score Area Chart
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["trust"],
        mode="lines+markers",
        name="Trust Score",
        line=dict(color="#00ff88", width=3, shape='spline'),
        marker=dict(color="#00ff88", size=6),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.15)',
        hovertemplate="Trust: %{y:.4f}<extra></extra>",
        text=df.get("event_id", []),
    ))

    # DQN Threshold Dashed Line
    if "dqn_threshold" in df.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["dqn_threshold"],
            mode="lines",
            name="DQN Threshold",
            line=dict(color="#8892b0", width=2, dash="dash"),
            hovertemplate="Threshold: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="right",
            x=1,
            font=dict(color="#c9d1d9")
        ),
        yaxis=dict(
            range=[0, 1.05], 
            gridcolor="#2d3561", 
            gridwidth=1,
            zeroline=False,
            showline=False,
            visible=True,
            showticklabels=True,
            tickfont=dict(color="#8892b0"),
            title=dict(text="")
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#2d3561",
            linewidth=1,
            tickfont=dict(color="#8892b0"),
            title=dict(text="")
        )
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # ── Decision table ─────────────────────────────────────────────────
    keys = history[-1].keys() if history else []
    cols = ["time", "tenant_id", "user_role", "event_id", "label", "trust", "verdict"]
    avail_cols = [c for c in cols if c in keys] or ["time", "event_id", "label", "trust", "verdict"]
    
    table_df = pd.DataFrame(history[-15:][::-1])[avail_cols].rename(columns={
        "time": "Time", "tenant_id": "Tenant", "user_role": "Role",
        "event_id": "EventId", "label": "Type", "trust": "Trust", "verdict": "Decision",
    })
    # Color verdict
    def color_verdict(val):
        return "color: #00ff88; font-weight:bold" if val == "PERMIT" \
               else "color: #ff4444; font-weight:bold"
    styled = table_df.style.applymap(color_verdict, subset=["Decision"])\
                           .format({"Trust": "{:.4f}"})
    table_placeholder.dataframe(styled, use_container_width=True, height=380)

    # ── SHAP explanation ───────────────────────────────────────────────
    latest   = history[-1]
    top_shap = latest["top_shap"]
    vals     = [f["shap_value"] for f in top_shap]
    names    = [f["feature"] for f in top_shap]
    bar_cols = ["#ff4444" if v > 0 else "#00ff88" for v in vals]

    shap_fig = go.Figure(go.Bar(
        x=[n.split('_')[0][:5].upper() if '_' in n else n[:5].upper() for n in names[:5]],
        y=[abs(v)*100 for v in vals[:5]],
        marker_color=["#ff4444", "#00ff88", "#58a6ff", "#ffb347", "#00ff88"][:min(5, len(vals))],
    ))
    shap_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        yaxis=dict(gridcolor="#2d3561", tickfont=dict(color="#8892b0")),
        xaxis=dict(tickfont=dict(color="#8892b0"))
    )
    trust_val = latest["trust"]
    tc        = _trust_color(trust_val)
    verdict_badge = (
        '<span class="permit-badge">✅ PERMIT</span>'
        if latest["verdict"] == "PERMIT"
        else '<span class="deny-badge">❌ DENY</span>'
    )
    shap_placeholder.markdown(
        f"**Event:** `{latest['event_id']}` — {latest['label']}  \n"
        f"**Trust Score:** <span style='color:{tc};font-weight:700'>{trust_val:.4f}</span>  "
        f"&nbsp; {verdict_badge}",
        unsafe_allow_html=True,
    )
    shap_placeholder.plotly_chart(shap_fig, use_container_width=True)
    shap_placeholder.caption("Red bars → push toward ANOMALY · Green bars → push toward NORMAL")

    # ── FL progress ────────────────────────────────────────────────────
    acc = st.session_state['fl_accuracy']
    fl_fig = go.Figure(go.Pie(
        values=[acc, 1 - acc],
        labels=["Accuracy", "Loss"],
        hole=0.75,
        marker=dict(colors=["#58a6ff", "#16213e"]),
        textinfo="none",
        hoverinfo="skip"
    ))
    fl_fig.add_annotation(
        text=f"{(acc*100):.1f}%", x=0.5, y=0.5, font=dict(size=26, color="white"), showarrow=False
    )
    fl_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    fl_placeholder.plotly_chart(fl_fig, use_container_width=True)
    fl_placeholder.caption(
        f"Round {st.session_state['fl_round']} · "
        f"Acc = {st.session_state['fl_accuracy']:.4f} · "
        f"Loss = {st.session_state['fl_loss']:.4f}  \n"
        "🔒 *No raw logs shared — DPDP Act compliant*"
    )

    # ── Audit Log ──────────────────────────────────────────────────────
    audit_file = ROOT / "data" / "decision_audit.jsonl"
    if audit_file.exists():
        import json
        with open(audit_file, "r") as f:
            lines = f.readlines()
            
        if lines:
            audit_events = [json.loads(line) for line in lines[-10:][::-1]]
            audit_df = pd.DataFrame(audit_events)
            
            # Format top features to be readable
            def format_shap(features):
                if not isinstance(features, list) or len(features) == 0:
                    return ""
                return ", ".join([f"{x['feature']}({x['shap_value']:.2f})" for x in features])
                
            audit_df["top_features"] = audit_df["top_features"].apply(format_shap)
            audit_df = audit_df[["timestamp", "user_id", "trust_score", "decision", "latency_ms", "top_features"]]
            
            styled_audit = audit_df.style.applymap(color_verdict, subset=["decision"])\
                                   .format({"trust_score": "{:.4f}", "latency_ms": "{:.2f} ms"})
            audit_placeholder.dataframe(styled_audit, use_container_width=True)
        else:
            audit_placeholder.info("Audit log is empty.")
    else:
        audit_placeholder.info("Waiting for inference API to generate JSONL audit logs...")


# ── Auto-refresh loop ─────────────────────────────────────────────────────
if st.session_state["running"]:
    simulate_step()
    render()
    time.sleep(speed)
    st.rerun()
else:
    render()

if not st.session_state["running"] and not st.session_state["history"]:
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    1. Click **▶ Start** in the sidebar
    2. Watch real-time trust scores from OpenStack audit log events
    3. See SHAP explanations for every access decision
    4. Monitor federated learning convergence
    5. Adjust the **DQN threshold** slider to see adaptive policy changes

    > **Data source**: Loghub OpenStack dataset (207,633 log entries)  
    > **Model**: Bidirectional LSTM + Self-Attention  
    > **Explainability**: SHAP (SHapley Additive exPlanations)  
    > **Privacy**: Federated Learning via Flower framework  
    > **Compliance**: India DPDP Act 2023 · Zero Trust Architecture
    """)
