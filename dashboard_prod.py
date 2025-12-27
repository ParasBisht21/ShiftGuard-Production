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
        /* Make popover buttons look sharper */
        div[data-testid="stPopoverBody"] { border: 1px solid #444; }
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADERS ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            # Standardize Columns
            if 'fatigue_risk' in df.columns:
                df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns:
                df['status'] = df['status'].str.strip()

            # --- "THE MATRIX" DATA GENERATOR ---
            # We generate names/depts based on ID so they stay consistent on refresh
            
            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James", "Maria", "John"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def get_fake_profile(nid):
                # Use ID as seed so Nurse 93 is ALWAYS "Sarah Chen"
                random.seed(nid) 
                fname = random.choice(first_names)
                lname = random.choice(last_names)
                dept = depts[nid % len(depts)]
                return f"{fname} {lname}", dept

            df['Full_Name'], df['Department'] = zip(*df['nurse_id'].apply(get_fake_profile))
            
            # Generate Biometrics
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
            query = text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC")
            df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame() 

# --- 5. ACTIONS (NOW WITH REPLACEMENT LOGIC) ---
def relieve_nurse_in_db(fatigued_id, risk_val, replacement_name):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            # 1. Update the Fatigued Nurse
            sql_update = text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id")
            conn.execute(sql_update, {"id": fatigued_id})
            
            # 2. Log WHO replaced them (The new feature)
            log_msg = f"Swapped with {replacement_name}"
            sql_log = text("""
                INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action)
                VALUES (:id, 'MANUAL_SWAP', :risk, :action)
            """)
            conn.execute(sql_log, {"id": fatigued_id, "risk": risk_val, "action": log_msg})
            
        return True
    except Exception as e:
        st.error(f"Write Error: {e}")
        return False

# --- MAIN APP ---
st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")

tab1, tab2 = st.tabs(["🔴 Live Operations", "⚖️ Audit & Compliance"])

with tab1:
    df = load_data()
    if df is not None:
        st.caption(f"LIVE CONNECTION | {len(df)} Staff Members Monitored | Location: Central India Node")
        
        # CALCULATE METRICS
        active_risk = len(df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Staff", len(df), "Active on Floor")
        m2.metric("Critical Alerts", active_risk, "Immediate Action Reqd", delta_color="inverse")
        m3.metric("Avg Unit BPM", f"{int(df['BPM'].mean())}", "+12% vs Baseline", delta_color="inverse")
        m4.metric("System Latency", "24ms", "Azure SQL")

        st.divider()

        # --- GET LIST OF SAFE REPLACEMENTS (Low Risk Nurses) ---
        safe_nurses = df[df['incident_probability'] < 30].sort_values('incident_probability')
        # Create a dropdown list like: "Sarah Chen (Risk: 12%)"
        replacement_options = safe_nurses.apply(
            lambda x: f"{x['Full_Name']} (ID: {x['nurse_id']} | Risk: {x['incident_probability']}%)", axis=1
        ).tolist()

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
                    c1, c2, c3 = st.columns([1, 2, 1.2]) # Wider column for the button
                    with c1:
                        st.markdown(f"### 🩺 **{nurse_id}**")
                        st.caption(nurse['Department'])
                    
                    with c2:
                        st.markdown(f"**{nurse['Full_Name']}**")
                        if nurse['status'] == 'Relieved':
                            st.success("✅ **RELIEVED**")
                        else:
                            st.progress(risk_val / 100, text=f"Risk: {risk_val}% | Shift: {nurse['Hours_On_Shift']}h")

                    with c3:
                        if nurse['status'] != 'Relieved':
                            # --- THE NEW FEATURE: POPOVER MENU ---
                            with st.popover("⚡ MANAGE SWAP", use_container_width=True):
                                st.markdown("### Select Replacement")
                                st.caption("Choose an available low-risk nurse to take over.")
                                
                                # Dropdown to pick the new nurse
                                selected_replacement = st.selectbox(
                                    "Available Staff:", 
                                    replacement_options,
                                    key=f"sel_{i}"
                                )
                                
                                # Extract just the name for the log
                                replacement_name = selected_replacement.split(" (")[0]
                                
                                if st.button(f"Confirm Swap", key=f"conf_{i}", type="primary"):
                                    with st.spinner("Updating Roster..."):
                                        success = relieve_nurse_in_db(nurse_id, risk_val, replacement_name)
                                        time.sleep(1)
                                    if success:
                                        st.toast(f"✅ Swap Confirmed: {replacement_name} is en route.")
                                        time.sleep(1)
                                        st.rerun()
                        else:
                            st.button("Log Archived", disabled=True, key=f"d_{i}")

        with col_right:
            st.subheader("📊 Live Unit Status")
            dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
            st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")

        st.subheader("📋 Staff Roster")
        st.dataframe(
            df[['nurse_id', 'Full_Name', 'Department', 'incident_probability', 'status']].sort_values('incident_probability', ascending=False),
            use_container_width=True,
            hide_index=True
        )

with tab2:
    st.header("⚖️ Compliance Audit Trail")
    if st.button("🔄 Refresh Logs"): st.rerun()
    audit_df = load_audit_logs()
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No swaps recorded yet.")
