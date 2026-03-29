import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
import random

def init_session_state():
    """Initialize session state variables for the simulation."""
    defaults = {
        "running": False,
        "total_requests": 0,
        "permits": 0,
        "denies": 0,
        "fl_round": 1,
        "fl_accuracy": 0.72,
        "fl_loss": 0.31,
        "dqn_threshold": 0.65,
        "threshold_history": [0.65] * 15,
        "trust_history": [],
        "log_entries": [],
        "alerts": [
            {"time": "10:05:12 AM", "user": "AdminUser", "event": "Successful login from known IP", "severity": "Low", "action": "PERMIT"},
            {"time": "10:12:45 AM", "user": "GuestUser", "event": "Failed database access attempts", "severity": "Medium", "action": "DENIED"}
        ],
        "events_sec": 0,
        "inference_latency": 0.0,
        "throughput": 0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def generate_fake_request(threshold, current_tenant, current_role, current_domain):
    """Generate a simulated access request with random trust and adjust metrics."""
    # Simulate trust score around 0.2 to 0.95
    trust = np.random.beta(a=3, b=2)
    
    # Adjust DQN threshold slightly
    adjustment = np.random.uniform(-0.03, 0.03)
    new_threshold = max(0.18, min(0.99, threshold + adjustment))
    
    # Decision
    decision = "PERMIT" if trust >= new_threshold else "DENY"
    
    # Update global metrics
    st.session_state.total_requests += 1
    if decision == "PERMIT":
        st.session_state.permits += 1
    else:
        st.session_state.denies += 1
        
    st.session_state.events_sec = np.random.randint(120, 150)
    st.session_state.inference_latency = np.random.uniform(10.0, 25.0)
    st.session_state.throughput = np.random.randint(90, 110)
    
    # Threshold history
    st.session_state.threshold_history.append(new_threshold)
    if len(st.session_state.threshold_history) > 15:
        st.session_state.threshold_history.pop(0)
        
    # Trust history
    req_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.trust_history.append({
        "time": req_time,
        "trust": trust,
        "threshold": new_threshold,
        "decision": decision
    })
    if len(st.session_state.trust_history) > 50:
        st.session_state.trust_history.pop(0)
        
    # Log entry
    event_ids = ["E42", "E41", "E37", "E18", "E24", "E6", "E7"]
    event_types = ["Normal GET", "Normal POST", "Image create", "Spawn success", "⚠️ Auth fail", "⚠️ Token fail", "⚠️ Sync mismatch"]
    idx = np.random.randint(0, len(event_ids))
    
    st.session_state.log_entries.append({
        "Time": req_time,
        "EventId": event_ids[idx],
        "Type": event_types[idx],
        "Trust": f"{trust:.4f}",
        "Decision": decision,
        "Tenant ID": current_tenant,
        "User Role": current_role,
        "Resource Domain": current_domain
    })
    if len(st.session_state.log_entries) > 15:
        st.session_state.log_entries.pop(0)
        
    # Generate alert if trust < 0.3 (or 0.5 as requested previously)
    if trust < 0.3:
        st.session_state.alerts.append({
            "time": req_time,
            "user": current_role,
            "event": "Malicious behavior pattern detected - New User Detected (Low Trust)",
            "severity": "High",
            "action": "DENIED"
        })
        if len(st.session_state.alerts) > 6:
            st.session_state.alerts.pop(0)

    # occasionally bump FL round
    if st.session_state.total_requests % 15 == 0:
        st.session_state.fl_round += 1
        st.session_state.fl_accuracy = min(0.99, st.session_state.fl_accuracy + random.uniform(0.005, 0.015))
        st.session_state.fl_loss = max(0.01, st.session_state.fl_loss - random.uniform(0.01, 0.03))

    return trust, new_threshold, decision

def render_timeline_chart():
    """Render the Live Trust Score Timeline using Plotly."""
    df = pd.DataFrame(st.session_state.trust_history)
    fig = go.Figure()

    if not df.empty:
        # Trust Score (Green)
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["trust"],
            mode="lines+markers",
            name="Trust Score",
            line=dict(color="#00ff88", width=2),
            marker=dict(
                color=["#00ff88" if d == "PERMIT" else "#ff4444" for d in df["decision"]],
                size=6
            ),
            hovertemplate="Trust: %{y:.4f}<extra></extra>"
        ))
        
        # Threshold (Orange Dashed)
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["threshold"],
            mode="lines",
            name="DQN Threshold",
            line=dict(color="#ffb347", width=2, dash="dash"),
            hovertemplate="Threshold: %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=300,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(title="Request #", showgrid=False, linecolor="#30363d"),
        yaxis=dict(title="Trust Score", range=[0, 1.05], showgrid=True, gridcolor="#30363d", linecolor="#30363d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_shap_plot():
    """Render a dummy SHAP Explanation plot using Plotly horizontal bars."""
    features = ["http_status", "response_time", "has_instance", "source_ip", "event_E42"]
    
    # Randomize shap values slightly for animation feel
    vals = [
        random.uniform(0.2, 0.4),    # push to anomaly
        random.uniform(0.05, 0.15),  # push to anomaly
        random.uniform(-0.1, -0.01), # push to normal
        random.uniform(-0.2, -0.05), # push to normal
        random.uniform(-0.5, -0.3)   # push to normal
    ]
    
    colors = ["#ff4444" if v > 0 else "#00ff88" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=vals[::-1],
        y=features[::-1],
        orientation="h",
        marker_color=colors[::-1]
    ))
    
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=250,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis_title="SHAP value \u2192 anomaly (Red) / normal (Green)",
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(
        page_title="DR-TBAC-ZT++ Dashboard",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # -------------------------------------------------------------------------
    # Custom CSS Styling
    # -------------------------------------------------------------------------
    st.markdown("""
        <style>
        /* Dark Navy/Black Theme */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }
        h1, h2, h3, p, span, div {
            color: #c9d1d9;
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        [data-testid="stMetricLabel"] {
            color: #8b949e !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
        }
        div[data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] { color: #00ff88 !important; } /* PERMIT */
        div[data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] { color: #ff4444 !important; } /* DENY */
        div[data-testid="stMetric"]:nth-child(4) [data-testid="stMetricValue"] { color: #58a6ff !important; } /* FL ROUND */
        div[data-testid="stMetric"]:nth-child(5) [data-testid="stMetricValue"] { color: #ffb347 !important; } /* THRESHOLD */
        
        /* Badges & Tables */
        .badge-permit { background-color: #00ff88; color: #000; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
        .badge-deny { background-color: #ff4444; color: #fff; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
        .badge-fl { background-color: #1f6feb; color: #fff; padding: 6px 12px; border-radius: 6px; font-weight: 500; display: inline-block; margin-top: 10px; }
        
        hr { border-color: #30363d; }
        </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------
    st.markdown("<h1>🔐 DR-TBAC-ZT++ | Real-Time Zero Trust Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:1.1rem;'>XAI-Enabled Privacy-Preserving Federated Zero-Trust Framework</p>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # RESEARCH GAP 3: Scalability Challenge (Top Metrics & Sub-metrics)
    # -------------------------------------------------------------------------
    st.markdown("<div style='text-align:right; font-size:0.8rem; color:#8b949e;'>Scalability Metrics: "
                f"<span style='color:#58a6ff; font-weight:bold;'>Events/sec: {st.session_state.events_sec}</span> | "
                f"<span style='color:#58a6ff; font-weight:bold;'>Latency: {st.session_state.inference_latency:.1f} ms</span> | "
                f"<span style='color:#58a6ff; font-weight:bold;'>Throughput: {st.session_state.throughput} req/s</span></div>", 
                unsafe_allow_html=True)

    # Top Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Requests (Cum.)", st.session_state.total_requests)
    m2.metric("PERMIT", st.session_state.permits)
    m3.metric("DENY", st.session_state.denies)
    m4.metric("FL Round", st.session_state.fl_round)
    m5.metric("DQN Threshold", f"{st.session_state.dqn_threshold:.2f}")

    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    st.sidebar.title("⚙️ Controls")
    
    # RESEARCH GAP 2 (Control): AI-Driven Adaptive Access Control
    st.sidebar.markdown("### Settings")
    manual_threshold = st.sidebar.slider("DQN Trust Threshold", 0.18, 0.99, st.session_state.dqn_threshold, 0.01)
    sim_delay = st.sidebar.select_slider("Simulation Speed (Delay)", options=[0.3, 0.5, 1.0, 2.0], value=0.5)
    
    # Update state if slider is used manually while paused
    if not st.session_state.running and manual_threshold != st.session_state.dqn_threshold:
        st.session_state.dqn_threshold = manual_threshold

    # Buttons
    col_play, col_stop, col_reset = st.sidebar.columns(3)
    if col_play.button("▶ Play", use_container_width=True):
        st.session_state.running = True
    if col_stop.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False
    if col_reset.button("🔄 Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()

    # RESEARCH GAP 4: Multi-Tenant Authorization
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 Multi-Tenant Context")
    tenant_id = st.sidebar.selectbox("Tenant ID", ["CloudDept-A", "CloudDept-B", "Govt-Tenant"])
    user_role = st.sidebar.selectbox("User Role", ["DevUser1", "AdminUser", "GuestUser"])
    resource_domain = st.sidebar.selectbox("Resource Domain", ["VM cluster", "Database", "Object Storage"])

    # -------------------------------------------------------------------------
    # MAIN CONTENT GRID
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # RESEARCH GAP 1 & 2: Dynamic Trust Evaluation & Threshold History
        st.markdown("### 📡 Live Trust Score Timeline")
        chart_placeholder = st.empty()
        
        # RESEARCH GAP 1: Current Trust Status Row
        status_placeholder = st.empty()
        
        st.markdown("### 📋 Access Decision Log")
        table_placeholder = st.empty()

    with col_right:
        st.markdown("### 🔍 SHAP Explanation")
        shap_placeholder = st.empty()
        st.markdown("<p style='font-size:0.8rem; color:#8b949e;'>Red bars \u2192 push toward ANOMALY<br>Green bars \u2192 push toward NORMAL</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🌐 Federated Learning Progress")
        fl_placeholder = st.empty()


    # -------------------------------------------------------------------------
    # BOTTOM EXPANDERS
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    # RESEARCH GAP 5: Continuous Monitoring
    with st.expander("🔔 Security Alerts & Continuous Monitoring", expanded=True):
        alerts_placeholder = st.empty()

    with st.expander("📖 Real-Time Decision Audit Log", expanded=False):
        audit_placeholder = st.empty()

    # Footer
    st.markdown("""
        <div style='text-align:center; padding: 20px 0; color: #8b949e;'>
            <b>📚 Reference:</b> Wang et al. (2025). DR-TBAC: Dynamic Role & Trust-Based Access Control for Multi-Tenant Clouds.<br>
            <span class='badge-fl'>🇮🇳 DPDP Act 2023 Compliant</span>
        </div>
    """, unsafe_allow_html=True)


    # -------------------------------------------------------------------------
    # SIMULATION & RENDER LOGIC
    # -------------------------------------------------------------------------
    if st.session_state.running:
        with st.spinner("Simulating traffic..."):
            trust, new_thresh, decision = generate_fake_request(
                st.session_state.dqn_threshold, tenant_id, user_role, resource_domain
            )
            st.session_state.dqn_threshold = new_thresh

    # Draw Chart
    if st.session_state.trust_history:
        chart_placeholder.plotly_chart(render_timeline_chart(), use_container_width=True)
        
        latest = st.session_state.trust_history[-1]
        t_val = latest["trust"]
        if t_val >= 0.75: r_label, r_col = "Low", "#00ff88"
        elif t_val >= 0.5: r_label, r_col = "Medium", "#ffb347"
        else: r_label, r_col = "High", "#ff4444"
        
        warning_msg = "<span style='color:#ff4444; font-weight:bold; margin-left: 15px;'>⚠️ New User Detected – Low Trust</span>" if t_val < 0.3 else ""
        
        status_placeholder.markdown(f"""
            <div style='background-color:#161b22; padding: 10px; border-radius: 8px; border-left: 4px solid {r_col}; margin-bottom: 20px;'>
                <b>Current Trust Score:</b> <span style='color:{r_col}; font-size: 1.2rem;'>{t_val:.4f}</span> &nbsp;|&nbsp; 
                <b>User Behavior Risk:</b> <span style='color:{r_col};'>{r_label}</span>
                {warning_msg}
            </div>
        """, unsafe_allow_html=True)
    else:
        chart_placeholder.info("Click ▶ Play to begin simulation.")

    # Draw Table
    if st.session_state.log_entries:
        df_logs = pd.DataFrame(st.session_state.log_entries[::-1])
        # Subset for display
        df_display = df_logs[["Time", "EventId", "Type", "Trust", "Decision"]]
        
        def highlight_decision(val):
            if val == "PERMIT": return 'color: #000; background-color: #00ff88; font-weight: bold; border-radius:10px;'
            elif val == "DENY": return 'color: #fff; background-color: #ff4444; font-weight: bold; border-radius:10px;'
            return ''
        
        styled_df = df_display.style.applymap(highlight_decision, subset=["Decision"])
        table_placeholder.dataframe(styled_df, use_container_width=True, height=250, hide_index=True)
        audit_placeholder.dataframe(df_logs, use_container_width=True)
    
    # Draw SHAP
    shap_placeholder.plotly_chart(render_shap_plot(), use_container_width=True)
    
    # Draw FL
    fl_placeholder.markdown(f"""
        <div style='text-align:center;'>
            <h3>Round {st.session_state.fl_round}</h3>
            <p>Accuracy: <span style='color:#00ff88;'>{st.session_state.fl_accuracy:.4f}</span> | Loss: <span style='color:#ff4444;'>{st.session_state.fl_loss:.4f}</span></p>
            <div class='badge-fl'>🔒 No raw logs shared — DPDP Act compliant</div>
        </div>
    """, unsafe_allow_html=True)

    # Draw Alerts
    if st.session_state.alerts:
        alerts_html = ""
        for alert in reversed(st.session_state.alerts):
            s_col = "#ff4444" if alert["severity"]=="High" else "#ffb347" if alert["severity"]=="Medium" else "#00ff88"
            alerts_html += f"""
                <div style='padding: 8px; border-bottom: 1px solid #30363d;'>
                    <span style='color:#8b949e; font-size:0.85em;'>[{alert["time"]}]</span> 
                    <span style='color:#58a6ff; font-weight:bold;'>{alert["user"]}</span>: {alert["event"]} <br>
                    <span style='font-size:0.85em;'>Severity: <span style='color:{s_col}; font-weight:bold;'>{alert["severity"]}</span> | Action: <b>{alert["action"]}</b></span>
                </div>
            """
        alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # RE-RUN LOOP
    # -------------------------------------------------------------------------
    if st.session_state.running:
        time.sleep(sim_delay)
        st.rerun()

if __name__ == "__main__":
    main()
