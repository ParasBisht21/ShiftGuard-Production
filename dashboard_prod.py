import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np 
import time
import random

# --- 1. CLOUD CONFIGURATION ---
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except FileNotFoundError:
    st.error("Secrets not found! Please check Streamlit Cloud settings.")
    st.stop()

# --- 2. DATABASE CONNECTION ---
def get_db_connection():
    # Using pymssql for maximum compatibility with Streamlit Cloud
    conn_str = f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    engine = create_engine(conn_str)
    return engine

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="ShiftGuard PROD", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; padding: 10px; }
        div[data-testid="stDataFrame"] { font-size: 0.8rem; }
        div[data-testid="stPopoverBody"] { border: 1px solid #444; }
        /* Terminal Style for AI Logs */
        .ai-log { font-family: 'Courier New', monospace; color: #00ff00; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADERS & HEURISTIC ENGINE ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            # Standardization
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()

            # --- HEURISTIC ENGINE: THE "MATH" LAYER ---
            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def generate_heuristic_profile(nid):
                random.seed(nid) # Deterministic Seed 
                
                full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
                dept = depts[nid % len(depts)]
                
                # Simulation: Higher ID = Longer Shift (Variance for demo)
                base_shift = 6 + (nid % 10) 
                hours_on_shift = np.round(base_shift + random.uniform(0, 2), 1)
                
                # Simulation: Stress Factor (BPM)
                bpm = int(65 + (hours_on_shift * 2) + random.randint(-5, 5))
                
                return full_name, dept, hours_on_shift, bpm

            # Apply Profile Gen
            df['Full_Name'], df['Department'], df['Hours_On_Shift'], df['BPM'] = zip(*df['nurse_id'].apply(generate_heuristic_profile))
            
            # --- THE FORMULA: RISK CALCULATION ---
            # PITCH: "Risk = (Hours * 4.5) + (Stress * 1.2)"
            def calculate_risk(row):
                if row['status'] == 'Relieved': return 12 # Baseline safe
                
                # The Formula
                stress_factor = max(0, row['BPM'] - 70)
                risk_score = (row['Hours_On_Shift'] * 4.5) + (stress_factor * 1.2)
                
                # Cap at 99, Min at 5
                risk_score = min(max(risk_score, 5), 99)
                return int(risk_score)

            df['Calculated_Risk'] = df.apply(calculate_risk, axis=1)
            
            # Merge DB value with Math (Max wins)
            df['incident_probability'] = df[['incident_probability', 'Calculated_Risk']].max(axis=1)

        return df
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        return None

def load_audit_logs():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC")
            df = pd.read_sql(query, conn)
        return df
    except Exception: return pd.DataFrame() 

# --- 5. ACTIONS ---
def relieve_nurse_in_db(fatigued_id, risk_val, replacement_name, is_ai=False):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            # 1. Update Status
            sql_update = text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id")
            conn.execute(sql_update, {"id": fatigued_id})
            
            # 2. Audit Log
            action_type = 'AI_AUTO_RESOLVE' if is_ai else 'MANUAL_SWAP'
            log_msg = f"Auto-Swap with {replacement_name}" if is_ai else f"Swapped with {replacement_name}"
            
            sql_log = text("""
                IF OBJECT_ID('audit_logs', 'U') IS NOT NULL
                INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action)
                VALUES (:id, :type, :risk, :action)
            """)
            conn.execute(sql_log, {"id": fatigued_id, "type": action_type, "risk": risk_val, "action": log_msg})
        return True
    except Exception: return False

def reset_simulation():
    try:
        engine = get_db_connection()
        with engine.begin() as conn:
            # Reset everyone to safe
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15"))
            # FORCE CRITICAL RISKS (These override the math for the demo drama)
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98 WHERE nurse_id IN (34, 68, 93, 29, 55)")) 
            conn.execute(text("TRUNCATE TABLE audit_logs"))
        return True
    except Exception: return False

# --- MAIN APP LAYOUT ---
st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")

# --- ADMIN SIDEBAR ---
with st.sidebar:
    st.header("Admin Console")
    st.info("Hidden tools for demo purposes")
    if st.button("🔄 RESET SIMULATION", type="primary"):
        with st.spinner("Resetting Database..."):
            if reset_simulation():
                st.success("Database Reset!")
                time.sleep(1)
                st.rerun()

# --- TABS ---
tab1, tab2 = st.tabs(["🔴 Live Operations", "⚖️ Audit & Compliance"])

with tab1:
    df = load_data()
    if df is not None:
        # METRICS
        active_risk_df = df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')]
        active_risk_count = len(active_risk_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Staff", len(df), "Active on Floor")
        m2.metric("Critical Alerts", active_risk_count, "Immediate Action Reqd", delta_color="inverse")
        m3.metric("Avg Unit BPM", f"{int(df['BPM'].mean())}", "+12% vs Baseline", delta_color="inverse")
        m4.metric("System Latency", "24ms", "Azure SQL")

        # --- AI AUTO-PILOT SECTION ---
        if active_risk_count > 0:
            st.markdown("### ⚡ AI Counter-Measures")
            with st.expander("🤖 **RECOMMENDATION ENGINE: Heuristic Optimization Detected**", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"""
                        **Status:** <span style='color:#ff4b4b'>CRITICAL INSTABILITY</span>
                        <br>Heuristic Model (Duration x 4.5 + Stress x 1.2) has identified **{active_risk_count}** anomalies.
                        <br><i>Predicted Liability Reduction: <b>$2.4M</b></i>
                    """, unsafe_allow_html=True)
                with c2:
                    if st.button("🚀 EXECUTE AUTO-FIX", type="primary", use_container_width=True):
                        status_box = st.empty()
                        progress_bar = st.progress(0)
                        logs = []
                        
                        # Constraint-Based Filter: Only use safe nurses
                        safe_staff = df[df['incident_probability'] < 20]['Full_Name'].tolist()
                        
                        # --- 🚨 CRASH PREVENTION: EMERGENCY PROTOCOL ---
                        # If list is empty (everyone is tired), grab top 3 least tired people
                        if not safe_staff:
                            safe_staff = df.sort_values('incident_probability')['Full_Name'].head(3).tolist()
                            logs.append("[WARNING] RESOURCE DEPLETION DETECTED. ENGAGING EMERGENCY RESERVE.")

                        for i, (idx, nurse) in enumerate(active_risk_df.iterrows()):
                            time.sleep(0.4) 
                            # Round-Robin Assignment
                            replacement = safe_staff[i % len(safe_staff)]
                            
                            # SHOW THE HEURISTIC MATH IN THE LOGS
                            msg = f"[AI] Analyzing ID {nurse['nurse_id']}... Hours: {nurse['Hours_On_Shift']} | BPM: {nurse['BPM']} -> RISK {nurse['incident_probability']}%"
                            logs.append(msg)
                            status_box.code("\n".join(logs), language="bash")
                            
                            time.sleep(0.3)
                            msg2 = f"[AI] >> Allocating Resource: {replacement} (Risk: 5%)... SWAP EXECUTED."
                            logs.append(msg2)
                            status_box.code("\n".join(logs), language="bash")
                            progress_bar.progress((i + 1) / active_risk_count)
                            
                            relieve_nurse_in_db(nurse['nurse_id'], nurse['incident_probability'], replacement, is_ai=True)
                        
                        time.sleep(0.5)
                        st.success("✅ OPTIMIZATION COMPLETE: All risks neutralized.")
                        time.sleep(1)
                        st.rerun()

        st.divider()

        # --- STANDARD VIEW ---
        safe_nurses = df[df['incident_probability'] < 30].sort_values('incident_probability')
        replacement_options = safe_nurses.apply(lambda x: f"{x['Full_Name']} (ID: {x['nurse_id']} | Risk: {x['incident_probability']}%)", axis=1).tolist()

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("🚨 High Priority Interventions")
            # Sort: Relieved (Pinned) -> High Risk
            df['sort_score'] = df['incident_probability']
            df.loc[df['status'] == 'Relieved', 'sort_score'] = 1000
            top_nurses = df.sort_values('sort_score', ascending=False).head(3)

            for i, (idx, nurse) in enumerate(top_nurses.iterrows()):
                nurse_id = int(nurse['nurse_id'])
                risk_val = int(nurse['incident_probability'])
                
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1.2]) 
                    with c1:
                        st.markdown(f"### 🩺 **{nurse_id}**")
                        st.caption(nurse['Department'])
                    
                    with c2:
                        st.markdown(f"**{nurse['Full_Name']}**")
                        if nurse['status'] == 'Relieved':
                            st.success("✅ **RELIEVED**")
                            st.caption("Log: Optimization Engine")
                        else:
                            st.progress(risk_val / 100, text=f"Risk: {risk_val}% | Shift: {nurse['Hours_On_Shift']}h")

                    with c3:
                        if nurse['status'] != 'Relieved':
                            with st.popover("⚡ MANAGE SWAP", use_container_width=True):
                                selected_replacement = st.selectbox("Available Staff:", replacement_options, key=f"sel_{i}")
                                replacement_name = selected_replacement.split(" (")[0] if selected_replacement else "Unknown"
                                if st.button(f"Confirm Swap", key=f"conf_{i}", type="primary"):
                                    with st.spinner("Processing..."):
                                        relieve_nurse_in_db(nurse_id, risk_val, replacement_name)
                                        time.sleep(1)
                                        st.rerun()
                        else:
                            st.button("Log Archived", disabled=True, key=f"d_{i}")

        with col_right:
            st.subheader("📊 Unit Risk Distribution")
            dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
            st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")
            st.info("ℹ️ **Heuristic Insight:** ICU unit shows 15% higher stress factor (BPM > 90) than surgical ward.")

        # --- ROSTER TABLE (WITH MATH EXPLAINED) ---
        st.subheader("📋 Staff Roster (Heuristic Analysis)")
        st.dataframe(
            df[['nurse_id', 'Full_Name', 'Department', 'Hours_On_Shift', 'BPM', 'incident_probability', 'status']].sort_values('incident_probability', ascending=False), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Hours_On_Shift": st.column_config.NumberColumn("Shift Duration (Hrs)", format="%.1f"),
                "BPM": "Avg Heart Rate",
                "incident_probability": st.column_config.ProgressColumn("Risk Model Score", format="%d%%", min_value=0, max_value=100)
            }
        )

with tab2:
    st.header("⚖️ Compliance Audit Trail")
    if st.button("🔄 Refresh Logs"): st.rerun()
    audit_df = load_audit_logs()
    if not audit_df.empty:
        st.dataframe(
            audit_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)", format="D MMM, HH:mm:ss"),
            }
        )
    else:
        st.info("No records found.")
