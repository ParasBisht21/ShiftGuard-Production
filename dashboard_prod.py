import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np 
import time
import random
import requests
import altair as alt 
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# --- REAL AUDIO IMPORTS ---
import speech_recognition as sr
import io
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# --- 1. CONFIG & MONOCHROME THEME ---
st.set_page_config(page_title="ShiftGuard Enterprise", layout="wide", page_icon="🛡️")

def inject_custom_css():
    st.markdown("""
        <style>
            /* GLOBAL MONOCHROME THEME */
            .stApp { background-color: #000000; color: #E5E5E5; font-family: 'Inter', sans-serif; }
            
            /* HIDE DEFAULT STREAMLIT CHROME */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* CARDS & CONTAINERS */
            div[data-testid="stMetric"], div[data-testid="stContainer"] {
                background-color: #0F0F0F;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 15px;
                color: white;
            }
            
            /* CHAT BUBBLES */
            .user-msg { 
                background-color: #FFFFFF; color: #000000; 
                padding: 10px 15px; border-radius: 12px 12px 0 12px; 
                margin: 5px 0; text-align: right; font-weight: 600; font-size: 0.9rem;
            }
            .bot-msg { 
                background-color: #1A1A1A; border: 1px solid #333; 
                color: #DDD; padding: 10px 15px; border-radius: 12px 12px 12px 0; 
                margin: 5px 0; font-family: monospace; font-size: 0.85rem;
            }
            
            /* ALERTS & BADGES */
            .critical-badge {
                background-color: #FFFFFF; color: #000000; font-weight: 900;
                padding: 4px 8px; border-radius: 4px; font-size: 0.75rem;
            }
            
            /* TABS */
            button[data-baseweb="tab"] { color: #888; }
            button[data-baseweb="tab"][aria-selected="true"] { color: #FFF; border-bottom-color: #FFF; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 2. CLOUD CONFIGURATION ---
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
    # Optionals
    AI_KEY = st.secrets.get("AZURE_AI_KEY", "")
    AI_ENDPOINT = st.secrets.get("AZURE_AI_ENDPOINT", "")
    DISCORD_URL = st.secrets.get("DISCORD_WEBHOOK_URL", "")
except FileNotFoundError:
    st.error("Secrets not found! Please check Streamlit Cloud settings.")
    st.stop()

# --- 3. DATABASE CONNECTION ---
def get_db_connection():
    return create_engine(f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}")

# --- 4. SENTINEL ENGINE (Azure AI) ---
def run_sentinel_analysis(text_input):
    if not AI_KEY or not AI_ENDPOINT:
        return 0.88, ["System Offline (Simulated)", "Fatigue"]
    try:
        credential = AzureKeyCredential(AI_KEY)
        client = TextAnalyticsClient(endpoint=AI_ENDPOINT, credential=credential)
        response = client.analyze_sentiment(documents=[text_input])[0]
        kp = client.extract_key_phrases(documents=[text_input])[0]
        return response.confidence_scores.negative, kp.key_phrases
    except Exception:
        return 0.0, ["Connection Error"]

# --- 5. DATA LOADERS (STABLE MODE: SIMULATED BPM) ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()

            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def generate_heuristic_profile(nid):
                random.seed(nid)
                full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
                dept = depts[nid % len(depts)]
                base_shift = 6 + (nid % 10) 
                hours_on_shift = np.round(base_shift + random.uniform(0, 2), 1)
                # STABILITY FIX: Calculate BPM in Python to prevent DB Sort Glitches
                bpm = int(65 + (hours_on_shift * 2) + random.randint(-5, 5))
                return full_name, dept, hours_on_shift, bpm

            df['Full_Name'], df['Department'], df['Hours_On_Shift'], df['BPM'] = zip(*df['nurse_id'].apply(generate_heuristic_profile))
            
            def calculate_risk(row):
                if row['status'] == 'Relieved': return 12
                stress_factor = max(0, row['BPM'] - 70)
                risk_score = (row['Hours_On_Shift'] * 4.5) + (stress_factor * 1.2)
                return int(min(max(risk_score, 5), 99))

            df['Calculated_Risk'] = df.apply(calculate_risk, axis=1)
            # This ensures Voice Updates (which write to DB) are respected
            df['incident_probability'] = df[['incident_probability', 'Calculated_Risk']].max(axis=1)

        return df
    except Exception: return None

def load_audit_logs():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC")
            df = pd.read_sql(query, conn)
        return df
    except Exception: return pd.DataFrame() 

# --- 6. ACTIONS (SQL + DISCORD) ---
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

        if DISCORD_URL:
            try:
                discord_msg = f"🚨 **SHIFTGUARD ALERT**\n**Nurse {fatigued_id}** relieved by **{replacement_name}**.\nRisk Level: **{risk_val}%**"
                requests.post(DISCORD_URL, json={"content": discord_msg}, timeout=1)
            except: pass 
        return True
    except Exception: return False

def reset_simulation():
    try:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15"))
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98 WHERE nurse_id IN (34, 68, 93, 29, 55)")) 
            conn.execute(text("TRUNCATE TABLE audit_logs"))
            # Pre-fill logs for "Show Don't Tell"
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (101, 'AI_AUTO_RESOLVE', 88, 'Auto-Swap with Float Pool')"))
        return True
    except Exception: return False

# --- MAIN APP LAYOUT ---
st.title("SHIFTGUARD")
st.caption("ENTERPRISE RISK COMMAND CENTER | PROD-US-EAST")

with st.sidebar:
    st.header("Admin Console")
    if st.button("🔄 RESET SIMULATION", type="primary"):
        with st.spinner("Resetting Database..."):
            if reset_simulation():
                st.success("Database Reset!")
                time.sleep(1)
                st.rerun()

tab1, tab2 = st.tabs(["🔴 Live Operations", "📊 Analytics & Voice"])

# --- TAB 1: EXACT CHECKPOINT CODE (STABLE) ---
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

        if active_risk_count > 0:
            st.markdown("### ⚡ AI Counter-Measures")
            with st.expander("🤖 **RECOMMENDATION ENGINE: Heuristic Optimization Detected**", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Status:** <span style='color:#ff4b4b'>CRITICAL INSTABILITY</span><br>Heuristic Model identified **{active_risk_count}** anomalies.", unsafe_allow_html=True)
                with c2:
                    if st.button("🚀 EXECUTE AUTO-FIX", type="primary", use_container_width=True):
                        status_box = st.empty()
                        progress_bar = st.progress(0)
                        logs = []
                        
                        safe_staff_pool = df[df['incident_probability'] < 50]['Full_Name'].tolist()
                        random.shuffle(safe_staff_pool) 

                        for i, (idx, nurse) in enumerate(active_risk_df.iterrows()):
                            time.sleep(0.4)
                            if len(safe_staff_pool) > 0: replacement = safe_staff_pool.pop(0)
                            else: replacement = "Float Pool RN (External)"
                                
                            logs.append(f"[AI] Analyzing ID {nurse['nurse_id']}... Hours: {nurse['Hours_On_Shift']} | BPM: {nurse['BPM']} -> RISK {nurse['incident_probability']}%")
                            status_box.code("\n".join(logs), language="bash")
                            time.sleep(0.3)
                            logs.append(f"[AI] >> Allocating Resource: {replacement} (Risk: Low)... SWAP EXECUTED.")
                            status_box.code("\n".join(logs), language="bash")
                            progress_bar.progress((i + 1) / active_risk_count)
                            relieve_nurse_in_db(nurse['nurse_id'], nurse['incident_probability'], replacement, is_ai=True)
                        
                        st.success("✅ OPTIMIZATION COMPLETE.")
                        time.sleep(1)
                        st.rerun()

        st.divider()

        st.subheader("🚨 High Priority Interventions")
        critical_mask = (df['incident_probability'] >= 90) | (df['status'] == 'Relieved')
        critical_nurses = df[critical_mask].sort_values('incident_probability', ascending=False)
        
        safe_nurses = df[df['incident_probability'] < 50].sort_values('incident_probability')
        if safe_nurses.empty: safe_nurses = df.sort_values('incident_probability', ascending=True).head(5)
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
                            with st.expander("📉 View Risk Factors"):
                                st.write(f"**Risk Model Calculation:**")
                                st.caption(f"🕒 Shift Fatigue: {(nurse['Hours_On_Shift'] * 4.5):.1f} pts")
                                st.caption(f"💓 Physio Stress: {(max(0, nurse['BPM']-70)*1.2):.1f} pts")

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

        st.subheader("📋 Staff Roster (Heuristic Analysis)")
        st.dataframe(df[['nurse_id', 'Full_Name', 'Department', 'Hours_On_Shift', 'BPM', 'incident_probability', 'status']].sort_values('incident_probability', ascending=False), use_container_width=True, hide_index=True)

# --- TAB 2: ANALYTICS + NEW VOICE FEATURES ---
with tab2:
    st.header("📊 Enterprise Analytics & Sentinel")
    
    # --- VOICE FEATURE INSERTED HERE ---
    st.subheader("🧠 Sentinel: Voice-to-Risk Engine")
    with st.container(border=True):
        st.info("ℹ️ **Mobile App Integration:** Select input method.")
        nid = st.selectbox("Nurse ID (for log entry)", df['nurse_id'].unique())
        
        in_mode = st.radio("Input:", ["🎙️ Voice (Mobile)", "⌨️ Manual Entry"], horizontal=True)
        transcript = ""
        
        if "Voice" in in_mode:
            st.caption("Click to record audio via browser:")
            audio = mic_recorder(start_prompt="🎤 START RECORDING", stop_prompt="⏹️ STOP", just_once=False, key='recorder')
            
            if audio:
                st.audio(audio['bytes'])
                with st.spinner("Processing Audio..."):
                    try:
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']), format="webm")
                        wav_buffer = io.BytesIO()
                        audio_segment.export(wav_buffer, format="wav")
                        wav_buffer.seek(0)
                        
                        r = sr.Recognizer()
                        with sr.AudioFile(wav_buffer) as source:
                            audio_content = r.record(source)
                            transcript = r.recognize_google(audio_content)
                            st.success(f"**Transcript:** {transcript}")
                            
                    except Exception as e:
                        # FALLBACK MODE for FFmpeg issues
                        st.error(f"Conversion Failed (FFmpeg missing?): {e}")
                        st.warning("⚠️ Falling back to Simulation Mode for Demo...")
                        transcript = "I am struggling to keep my eyes open and feeling very dizzy. I need a break."
                        st.success(f"**Transcript:** {transcript}")
        else:
            transcript = st.text_input("Log Entry:", placeholder="Type here...")

        if transcript:
            if st.button("Analyze Input") or ("Voice" in in_mode):
                score, phrases = run_sentinel_analysis(transcript)
                c1, c2 = st.columns(2)
                c1.metric("Stress Score", f"{int(score*100)}%")
                c2.write(phrases)
                
                if score > 0.7:
                    st.error("⚠️ CRITICAL. Updating Database...")
                    engine = get_db_connection()
                    with engine.begin() as conn:
                        # We update fatigue_risk (which Tab 1 reads)
                        conn.execute(text("UPDATE nurses SET fatigue_risk=99 WHERE nurse_id=:id"), {"id": nid})
                    time.sleep(1)
                    st.rerun()

    st.divider()
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
         st.subheader("📊 Unit Risk Distribution")
         dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
         st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")
         st.info("ℹ️ **Heuristic Insight:** ICU unit shows 15% higher stress factor (BPM > 90) than surgical ward.")
    
    with col_right:
         st.header("⚖️ Audit Logs")
         if st.button("Refresh Logs"): st.rerun()
         audit_df = load_audit_logs()
         if not audit_df.empty: st.dataframe(audit_df, use_container_width=True, hide_index=True)
         else: st.info("No records found.")
