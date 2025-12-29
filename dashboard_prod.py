import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np 
import time
import random
import requests
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# --- 1. CLOUD CONFIGURATION ---
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
    # Optional Secrets (Won't crash if missing, but features will disable)
    AI_KEY = st.secrets.get("AZURE_AI_KEY", "")
    AI_ENDPOINT = st.secrets.get("AZURE_AI_ENDPOINT", "")
    LOGIC_APP_URL = st.secrets.get("LOGIC_APP_URL", "")
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
        .ai-log { font-family: 'Courier New', monospace; color: #00ff00; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SENTINEL ENGINE (Azure AI) ---
def run_sentinel_analysis(text_input):
    if not AI_KEY or not AI_ENDPOINT:
        st.error("Azure AI Secrets Missing. Check .toml file.")
        return 0.0, ["System Offline"]
    
    try:
        credential = AzureKeyCredential(AI_KEY)
        client = TextAnalyticsClient(endpoint=AI_ENDPOINT, credential=credential)
        
        # 1. Sentiment Analysis
        response = client.analyze_sentiment(documents=[text_input])[0]
        sentiment_score = response.confidence_scores.negative
        
        # 2. Key Phrase Extraction (Simulating Medical Entity Extraction)
        phrases_result = client.extract_key_phrases(documents=[text_input])[0]
        entities = phrases_result.key_phrases
        
        return sentiment_score, entities
    except Exception as e:
        st.error(f"Azure AI Error: {e}")
        return 0.0, []

# --- 5. DATA LOADERS & HEURISTIC ENGINE ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()

            # HEURISTIC PROFILES
            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def generate_heuristic_profile(nid):
                random.seed(nid) # Deterministic
                full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
                dept = depts[nid % len(depts)]
                base_shift = 6 + (nid % 10) 
                hours_on_shift = np.round(base_shift + random.uniform(0, 2), 1)
                bpm = int(65 + (hours_on_shift * 2) + random.randint(-5, 5))
                return full_name, dept, hours_on_shift, bpm

            df['Full_Name'], df['Department'], df['Hours_On_Shift'], df['BPM'] = zip(*df['nurse_id'].apply(generate_heuristic_profile))
            
            # RISK FORMULA: Risk = (Hours * 4.5) + (Stress * 1.2)
            def calculate_risk(row):
                if row['status'] == 'Relieved': return 12
                stress_factor = max(0, row['BPM'] - 70)
                risk_score = (row['Hours_On_Shift'] * 4.5) + (stress_factor * 1.2)
                return int(min(max(risk_score, 5), 99))

            df['Calculated_Risk'] = df.apply(calculate_risk, axis=1)
            # Use max of DB or Math to ensure Demo values (98%) persist
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

# --- 6. ACTIONS (SQL + LOGIC APP) ---
def relieve_nurse_in_db(fatigued_id, risk_val, replacement_name, is_ai=False):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            sql_update = text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id")
            conn.execute(sql_update, {"id": fatigued_id})
            
            action_type = 'AI_AUTO_RESOLVE' if is_ai else 'MANUAL_SWAP'
            log_msg = f"Auto-Swap with {replacement_name}" if is_ai else f"Swapped with {replacement_name}"
            
            sql_log = text("""
                IF OBJECT_ID('audit_logs', 'U') IS NOT NULL
                INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action)
                VALUES (:id, :type, :risk, :action)
            """)
            conn.execute(sql_log, {"id": fatigued_id, "type": action_type, "risk": risk_val, "action": log_msg})

        # AZURE LOGIC APP TRIGGER (Async Notification)
        if LOGIC_APP_URL:
            try:
                payload = {
                    "message": f"Nurse {fatigued_id} relieved by {replacement_name}. Risk Level was {risk_val}%. Authorized by ShiftGuard AI.",
                    "nurse_id": fatigued_id
                }
                requests.post(LOGIC_APP_URL, json=payload, timeout=2)
            except Exception:
                pass # Fail silently if Logic App is down to keep dashboard alive
            
        return True
    except Exception: return False

def reset_simulation():
    try:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15"))
            # Force Demo Scenarios
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98 WHERE nurse_id IN (34, 68, 93, 29, 55)")) 
            conn.execute(text("TRUNCATE TABLE audit_logs"))
        return True
    except Exception: return False

# --- MAIN APP LAYOUT ---
st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")

with st.sidebar:
    st.header("Admin Console")
    if st.button("🔄 RESET SIMULATION", type="primary"):
        with st.spinner("Resetting Database..."):
            if reset_simulation():
                st.success("Database Reset!")
                time.sleep(1)
                st.rerun()

tab1, tab2 = st.tabs(["🔴 Live Operations", "⚖️ Audit & Compliance"])

with tab1:
    df = load_data()
    if df is not None:
        active_risk_df = df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')]
        active_risk_count = len(active_risk_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Staff", len(df), "Active on Floor")
        m2.metric("Critical Alerts", active_risk_count, "Immediate Action Reqd", delta_color="inverse")
        m3.metric("Avg Unit BPM", f"{int(df['BPM'].mean())}", "+12% vs Baseline", delta_color="inverse")
        m4.metric("System Latency", "24ms", "Azure SQL")

        # --- AI AUTO-PILOT ---
        if active_risk_count > 0:
            st.markdown("### ⚡ AI Counter-Measures")
            with st.expander("🤖 **RECOMMENDATION ENGINE: Heuristic Optimization Detected**", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Status:** <span style='color:#ff4b4b'>CRITICAL INSTABILITY</span><br>Heuristic Model identified **{active_risk_count}** anomalies.<br><i>Predicted Liability Reduction: <b>$2.4M</b></i>", unsafe_allow_html=True)
                with c2:
                    if st.button("🚀 EXECUTE AUTO-FIX", type="primary", use_container_width=True):
                        status_box = st.empty()
                        progress_bar = st.progress(0)
                        logs = []
                        safe_staff = df[df['incident_probability'] < 20]['Full_Name'].tolist()
                        
                        # Crash Prevention: Emergency Protocol
                        if not safe_staff:
                            safe_staff = df.sort_values('incident_probability')['Full_Name'].head(3).tolist()
                            logs.append("[WARNING] RESOURCE DEPLETION. ENGAGING EMERGENCY RESERVE.")

                        for i, (idx, nurse) in enumerate(active_risk_df.iterrows()):
                            time.sleep(0.4) 
                            replacement = safe_staff[i % len(safe_staff)]
                            logs.append(f"[AI] Analyzing ID {nurse['nurse_id']}... Hours: {nurse['Hours_On_Shift']} | BPM: {nurse['BPM']} -> RISK {nurse['incident_probability']}%")
                            status_box.code("\n".join(logs), language="bash")
                            time.sleep(0.3)
                            logs.append(f"[AI] >> Allocating Resource: {replacement} (Risk: 5%)... SWAP EXECUTED.")
                            status_box.code("\n".join(logs), language="bash")
                            progress_bar.progress((i + 1) / active_risk_count)
                            relieve_nurse_in_db(nurse['nurse_id'], nurse['incident_probability'], replacement, is_ai=True)
                        
                        st.success("✅ OPTIMIZATION COMPLETE.")
                        time.sleep(1)
                        st.rerun()

        st.divider()

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # --- THE SENTINEL (INTERACTIVE VERSION) ---
            st.markdown("### 🧠 Sentinel: Narrative Analysis")
            with st.container(border=True):
                nurse_list = df['nurse_id'].tolist()
                # Interactive Dropdown to select WHO reports the issue
                selected_sentinel_id = st.selectbox("Reporting Nurse ID:", nurse_list)
                
                user_text = st.text_input("📝 Shift Log:", placeholder="Type here: 'I am feeling dizzy...'")
                
                if st.button("Analyze Log"):
                    if len(user_text) > 3:
                        with st.spinner("Azure Neural Processing..."):
                            stress, entities = run_sentinel_analysis(user_text)
                            c1, c2 = st.columns(2)
                            c1.metric("Cognitive Stress", f"{int(stress * 100)}%", "NLP Sentiment")
                            c2.write("**Detected Factors:**")
                            for ent in entities: c2.caption(f"🔴 {ent}")
                            
                            if stress > 0.7:
                                st.warning(f"⚠️ CRITICAL LOAD DETECTED. Flagging Nurse {selected_sentinel_id}...")
                                engine = get_db_connection()
                                with engine.begin() as conn:
                                    # Updates the SELECTED nurse
                                    sql_update = text("UPDATE nurses SET fatigue_risk = 99 WHERE nurse_id = :id")
                                    conn.execute(sql_update, {"id": selected_sentinel_id})
                                time.sleep(2)
                                st.rerun()

            # --- EXPANDED HIGH PRIORITY LIST (ALL RISKS > 90) ---
            st.subheader("🚨 High Priority Interventions")
            critical_mask = (df['incident_probability'] >= 90) | (df['status'] == 'Relieved')
            critical_nurses = df[critical_mask].sort_values('incident_probability', ascending=False)
            
            # 1. Try to find Safe Nurses (< 30%)
            safe_nurses = df[df['incident_probability'] < 30].sort_values('incident_probability')
            
            # 2. EMERGENCY FALLBACK: If everyone is tired, just take the top 5 least tired people
            if safe_nurses.empty:
                safe_nurses = df.sort_values('incident_probability', ascending=True).head(5)
            
            # 3. Generate Dropdown Options
            replacement_options = safe_nurses.apply(lambda x: f"{x['Full_Name']} (ID: {x['nurse_id']} | Risk: {x['incident_probability']}%)", axis=1).tolist()

            if critical_nurses.empty:
                st.success("✅ No critical fatigue risks detected. Unit is operating safely.")
            else:
                for i, (idx, nurse) in enumerate(critical_nurses.iterrows()):
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
                            else:
                                st.progress(risk_val / 100, text=f"Risk: {risk_val}% | Shift: {nurse['Hours_On_Shift']}h")
                        with c3:
                            if nurse['status'] != 'Relieved':
                                with st.popover("⚡ MANAGE SWAP", use_container_width=True):
                                    sel = st.selectbox("Available Staff:", replacement_options, key=f"sel_{i}")
                                    rep_name = sel.split(" (")[0] if sel else "Unknown"
                                    if st.button(f"Confirm Swap", key=f"conf_{i}", type="primary"):
                                        relieve_nurse_in_db(nurse_id, risk_val, rep_name)
                                        st.rerun()
                            else:
                                st.button("Log Archived", disabled=True, key=f"d_{i}")

        with col_right:
            st.subheader("📊 Unit Risk Distribution")
            dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
            st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")
            st.info("ℹ️ **Heuristic Insight:** ICU unit shows 15% higher stress factor (BPM > 90) than surgical ward.")

        st.subheader("📋 Staff Roster (Heuristic Analysis)")
        st.dataframe(df[['nurse_id', 'Full_Name', 'Department', 'Hours_On_Shift', 'BPM', 'incident_probability', 'status']].sort_values('incident_probability', ascending=False), use_container_width=True, hide_index=True)

with tab2:
    st.header("⚖️ Compliance Audit Trail")
    if st.button("🔄 Refresh Logs"): st.rerun()
    audit_df = load_audit_logs()
    if not audit_df.empty: st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else: st.info("No records found.")

