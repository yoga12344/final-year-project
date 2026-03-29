import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# Set Page Config
st.set_page_config(page_title="Zero-Trust Dashboard", page_icon="🔒", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme (identical to dashboard.py)
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

.permit-badge { background:#00ff88; color:#000; padding:8px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.deny-badge   { background:#ff4444; color:#fff; padding:8px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# INITIALIZE SESSION STATE
# -------------------------------------------------------------------------
if "threshold_history" not in st.session_state:
    st.session_state.threshold_history = [0.65] * 15

if "processed_events" not in st.session_state:
    st.session_state.processed_events = np.random.randint(1000, 2000)

if "events_sec" not in st.session_state:
    st.session_state.events_sec = np.random.randint(120, 150)

if "inference_latency" not in st.session_state:
    st.session_state.inference_latency = np.random.randint(10, 20)

if "throughput" not in st.session_state:
    st.session_state.throughput = np.random.randint(90, 100)

if "alerts" not in st.session_state:
    st.session_state.alerts = [
        {"time": "10:05:12 AM", "user": "SysAdmin", "event": "Successful login from known IP", "severity": "Low", "action": "PERMIT"},
        {"time": "10:12:45 AM", "user": "GuestUser", "event": "Failed database access attempts", "severity": "Medium", "action": "DENIED"},
        {"time": "10:20:00 AM", "user": "DevUser1", "event": "Rapid API calls detected", "severity": "High", "action": "MFA CHALLENGE"}
    ]

# -------------------------------------------------------------------------
# MAIN DASHBOARD HEADER
# -------------------------------------------------------------------------
st.title("🔒 Zero-Trust Access Control Dashboard")
st.caption("Phase 1 · Dynamic Trust Evaluation · Multi-Tenant Context · Adaptive DQN Thresholds")

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.markdown("## 👤 User Context & Behavior")
login_hour = st.sidebar.slider("Login Hour", 0, 23, 14)
geo_risk = st.sidebar.slider("Geo Risk", 0.0, 1.0, 0.2)
failed_attempts = st.sidebar.slider("Failed Attempts", 0, 10, 1)
api_calls = st.sidebar.slider("API Calls/min", 0, 30, 8)
seq_anomaly = st.sidebar.slider("Seq Anomaly Score", 0.0, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🏢 Tenant Context")
tenant_id = st.sidebar.selectbox("Tenant ID", ["CloudDept-A", "CloudDept-B", "Govt-Tenant"])
user_role = st.sidebar.selectbox("User Role", ["DevUser1", "AdminUser", "GuestUser"])
resource_domain = st.sidebar.selectbox("Resource Domain", ["VM cluster", "Database", "Object Storage"])

check_access = st.sidebar.button("▶ Check Access", use_container_width=True, type="primary")

# Execute Calculation on Check Access
if check_access:
    st.session_state.processed_events += 1
    st.session_state.events_sec = np.random.randint(120, 150)
    st.session_state.inference_latency = np.random.uniform(10.0, 20.0)
    st.session_state.throughput = np.random.randint(90, 100)

    # Calculate Trust Score
    trust = 0.85 - (login_hour < 6 or login_hour > 22) * 0.25 - geo_risk * 0.35 - failed_attempts * 0.08 - seq_anomaly * 0.30
    trust = max(0.0, min(1.0, trust))
    
    # Simulate DQN adjusting threshold
    current_threshold = st.session_state.threshold_history[-1]
    adjustment = np.random.uniform(-0.03, 0.03)
    new_threshold = max(0.4, min(0.9, current_threshold + adjustment))
    
    st.session_state.threshold_history.append(new_threshold)
    if len(st.session_state.threshold_history) > 15:
        st.session_state.threshold_history.pop(0)

    # Alert Trigger
    if trust < 0.5:
        now_str = datetime.now().strftime("%I:%M:%S %p")
        st.session_state.alerts.append({
            "time": now_str, 
            "user": user_role, 
            "event": "Malicious behavior pattern detected", 
            "severity": "High", 
            "action": "DENIED"
        })
        if len(st.session_state.alerts) > 5:
            st.session_state.alerts.pop(0)


# -------------------------------------------------------------------------
# METRIC CARDS ROW (Matching dashboard.py structure)
# -------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'''
<div class="metric-card">
  <div class="label">Processed Events</div>
  <div class="value">{st.session_state.processed_events:,}</div>
  <div class="sub">cumulative</div>
</div>''', unsafe_allow_html=True)
m2.markdown(f'''
<div class="metric-card">
  <div class="label">Events / sec</div>
  <div class="value">{st.session_state.events_sec}</div>
  <div class="sub">real-time ingest</div>
</div>''', unsafe_allow_html=True)
m3.markdown(f'''
<div class="metric-card">
  <div class="label">Inference Latency</div>
  <div class="value">{st.session_state.inference_latency:.1f} <span style="font-size:1.2rem;">ms</span></div>
  <div class="sub">avg response</div>
</div>''', unsafe_allow_html=True)
m4.markdown(f'''
<div class="metric-card">
  <div class="label">Throughput</div>
  <div class="value">{st.session_state.throughput} <span style="font-size:1.2rem;">req/s</span></div>
  <div class="sub">network capability</div>
</div>''', unsafe_allow_html=True)

st.divider()

if check_access:
    # -------------------------------------------------------------------------
    # DECISION ROW
    # -------------------------------------------------------------------------
    left, right = st.columns([3, 2])
    
    with left:
        st.markdown("### 📊 Dynamic Trust Evaluation")
        decision = "PERMIT" if trust >= new_threshold else "DENY"
        badge = f'<span class="permit-badge">✅ {decision}</span>' if decision == "PERMIT" else f'<span class="deny-badge">❌ {decision}</span>'
        
        if trust >= 0.75:
            tc = "#00ff88"
            r_label = "Low"
        elif trust >= 0.50:
            tc = "#ffb347"
            r_label = "Medium"
        else:
            tc = "#ff4444"
            r_label = "High"
            
        st.markdown(f"**Calculated Score:** <span style='color:{tc};font-weight:700;font-size:1.5rem'>{trust:.4f}</span> &nbsp;&nbsp; {badge}", unsafe_allow_html=True)
        st.markdown(f"**Behavioral Risk:** <span style='color:{tc}'>{r_label}</span>")
        
        # Simple progress bar
        st.progress(trust, text="Confidence Level")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏢 Current Tenant Context")
        tenant_df = pd.DataFrame([{"Tenant ID": tenant_id, "User Role": user_role, "Resource Domain": resource_domain}])
        st.table(tenant_df)

    with right:
        st.markdown("### 📈 Threshold Adjustment History")
        # Plotly chart similar to the ones in `dashboard.py`
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 16)),
            y=st.session_state.threshold_history,
            mode="lines+markers",
            line=dict(color="#64ffda", width=3),
            marker=dict(size=8),
            name="DQN Threshold"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=260,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Adjustment Step",
            yaxis=dict(range=[0.3, 1.0], title="Adaptive Threshold"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("DQN agent dynamically adjusts required threshold to balance risk / usability.")

else:
    st.info("👈 Adjust the context sliders in the sidebar and click **▶ Check Access** to generate Trust evaluations.")

# -------------------------------------------------------------------------
# CONTINUOUS MONITORING ROW
# -------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔔 Security Alerts & Continuous Monitoring")

def styled_alert(alert):
    if alert["severity"] == "High":
        color = "#ff4444"
    elif alert["severity"] == "Medium":
        color = "#ffb347"
    else:
        color = "#00ff88"
    return f"""
    <div style="background-color: #16213e; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 6px solid {color}; border-top: 1px solid #2d3561; border-right: 1px solid #2d3561; border-bottom: 1px solid #2d3561;">
        <span style="color: #8892b0; font-size: 0.85rem;">[{alert['time']}]</span> &nbsp;
        <b style="color:white;font-size:1.05rem;">{alert['user']}</b> &mdash; 
        <span style="color:#d1d5db;">{alert['event']}</span>
        <div style="float: right; font-size: 0.95rem;">
            Severity: <span style="color: {color}; font-weight: bold;">{alert['severity']}</span> &nbsp;|&nbsp; 
            Action: <b style="color:white;">{alert['action']}</b>
        </div>
    </div>
    """

for alert in reversed(st.session_state.alerts):
    st.markdown(styled_alert(alert), unsafe_allow_html=True)
