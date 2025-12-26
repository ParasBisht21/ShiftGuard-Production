import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np 
import time

# --- 1. CLOUD CONFIGURATION ---
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except FileNotFoundError:
    st.error("Secrets not found! Please check Streamlit Cloud settings.")
    st.stop()

# --- 2. DATABASE CONNECTION (Linux/Cloud Compatible) ---
def get_db_connection():
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
        /* Tab Styling */
        button[data-baseweb="tab"] {
            font-size: 1.2rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADERS ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            # Rename for app logic
            if 'fatigue_risk' in df.columns:
                df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns:
                df['status'] = df['status'].str.strip()

            # --- DENSITY ENRICHMENT (Fake Data for Demo) ---
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            shifts = ['Day (07:00)', 'Night (19:00)', 'Swing (12:00)']
            
            df['Department'] = df['nurse_id'].apply(lambda x: depts[int(x) % len(depts)])
            df['Shift_Start'] = df['nurse_id'].apply(lambda x: shifts[int(x) % len(shifts)])
            df['Hours_On_Shift'] = df.apply(lambda row: 
                np.round(8 + (row['incident_probability']/20) + (int(row['nurse_id'])%3), 1), axis=1)
            df['BPM'] = df.apply(lambda row: 
                int(70 + (row['incident_probability']/2) + (int(row['nurse_id'])%10)), axis=1)

        return df
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        return None

def load_audit_logs():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            # Fetch last 50 logs, newest first
            query = text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC")
            df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame() # Return empty if table doesn't exist yet

# --- 5. ACTIONS (WRITE BACK + LOGGING) ---
def relieve_nurse_in_db(nurse_id, current_risk):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            # 1. Update the Nurse Status
            sql_update = text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id")
            conn.execute(sql_update, {"id": nurse_id})
            
            # 2. Write to Audit Log (The "Black Box")
            sql_log = text("""
                INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action)
                VALUES (:id, 'SHIFT_SWAP', :risk, 'Authorized via Dashboard')
            """)
            conn.execute(sql_log, {"id": nurse_id, "risk": current_risk})
            
        return True
    except Exception as e:
        st.error(f"Write Error: {e}")
        return False

# --- MAIN APP ---
st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")

# TABS FOR MULTI-PAGE FEEL
tab1, tab2 = st.tabs(["🔴 Live Operations", "⚖️ Audit & Compliance"])

# --- TAB 1: THE COMMAND CENTER ---
with tab1:
    df = load_data()
    if df is not None:
        st.caption(f"LIVE CONNECTION | {len(df)} Staff Members Monitored | Location: Central India Node")
        
        # METRICS
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: st.metric("Total Staff", len(df), "Active on Floor")
        with m2: 
            critical = len(df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')])
            st.metric("Critical Alerts", critical, "Immediate Action Reqd", delta_color="inverse")
        with m3: 
            avg_bpm = int(df['BPM'].mean())
            st.metric("Avg Unit Stress", f"{avg_bpm} BPM", "+12% vs Baseline", delta_color="inverse")
        with m4:
            long_shifts = len(df[df['Hours_On_Shift'] > 12])
            st.metric("Overtime Violations", long_shifts, "Compliance Risk")
        with m5: st.metric("System Latency", "24ms", "Azure SQL")

        st.divider()

        # SPLIT VIEW
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("🚨 High Priority Interventions")
            
            # Sort: Relieved first (pinned), then High Risk
            df['sort_score'] = df['incident_probability']
            df.loc[df['status'] == 'Relieved', 'sort_score'] = 1000
            top_nurses = df.sort_values('sort_score', ascending=False).head(3)

            for i, (idx, nurse) in enumerate(top_nurses.iterrows()):
                nurse_id = int(nurse['nurse_id'])
                risk_val = int(nurse['incident_probability'])
                
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        st.markdown(f"### 🩺 **ID: {nurse_id}**")
                        st.caption(f"Dept: {nurse['Department']}")
                    
                    with c2:
                        if nurse['status'] == 'Relieved':
                            st.success("✅ **SHIFT RELIEVED**")
                            st.caption("Audit Log ID: Generated")
                        else:
                            st.markdown(f"**Fatigue Risk: {risk_val}%**")
                            st.progress(risk_val / 100)
                            st.markdown(f"⏱️ **Shift:** {nurse['Hours_On_Shift']}h | ❤️ **BPM:** {nurse['BPM']}")

                    with c3:
                        if nurse['status'] != 'Relieved':
                            if st.button(f"⚡ SWAP SHIFT", key=f"btn_{i}", type="primary"):
                                with st.spinner("Authorizing & Logging..."):
                                    success = relieve_nurse_in_db(nurse_id, risk_val)
                                    time.sleep(1)
                                if success:
                                    st.toast(f"✅ Backup Called. Incident Logged for ID {nurse_id}")
                                    time.sleep(1)
                                    st.rerun()
                        else:
                            st.button("Archived", disabled=True, key=f"d_{i}")

        with col_right:
            st.subheader("📊 Unit Status")
            dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
            st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")
            st.info("ℹ️ **Insight:** ICU unit is showing 15% higher fatigue levels than surgical ward.")

        # ROSTER TABLE
        st.subheader("📋 Live Staff Roster (All Units)")
        display_cols = ['nurse_id', 'Department', 'Shift_Start', 'Hours_On_Shift', 'BPM', 'incident_probability', 'status']
        st.dataframe(
            df[display_cols].sort_values('incident_probability', ascending=False),
            column_config={
                "incident_probability": st.column_config.ProgressColumn("Risk Score", format="%d%%", min_value=0, max_value=100),
                "nurse_id": st.column_config.TextColumn("Staff ID"),
            },
            use_container_width=True,
            hide_index=True,
            height=300
        )

# --- TAB 2: AUDIT LOGS (THE LEGAL LAYER) ---
with tab2:
    st.header("⚖️ Compliance Audit Trail")
    st.markdown("Immutable record of all high-risk interventions. Data is stored in Azure SQL 'audit_logs' table.")
    
    if st.button("🔄 Refresh Logs"):
        st.rerun()
        
    audit_df = load_audit_logs()
    
    if not audit_df.empty:
        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)", format="D MMM, HH:mm:ss"),
                "log_id": "Log ID",
                "nurse_id": "Staff ID",
                "risk_level_at_time": "Risk %",
                "manager_action": "Action Taken"
            }
        )
    else:
        st.info("No interventions recorded yet. Go to 'Live Operations' and Swap a Shift to generate a log.")
