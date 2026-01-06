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
            .stApp { background-color: #000000; color: #E5E5E5; font-family: 'Inter', sans-serif; }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            div[data-testid="stMetric"], div[data-testid="stContainer"] {
                background-color: #0F0F0F; border: 1px solid #333; border-radius: 6px; padding: 15px; color: white;
            }
            .user-msg { background-color: #FFFFFF; color: #000000; padding: 10px 15px; border-radius: 12px 12px 0 12px; margin: 5px 0; text-align: right; font-weight: 600; font-size: 0.9rem; }
            .bot-msg { background-color: #1A1A1A; border: 1px solid #333; color: #DDD; padding: 10px 15px; border-radius: 12px 12px 12px 0; margin: 5px 0; font-family: monospace; font-size: 0.85rem; }
            .critical-badge { background-color: #FFFFFF; color: #000000; font-weight: 900; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; }
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
    AI_KEY = st.secrets.get("AZURE_AI_KEY", "")
    AI_ENDPOINT = st.secrets.get("AZURE_AI_ENDPOINT", "")
    DISCORD_URL = st.secrets.get("DISCORD_WEBHOOK_URL", "")
except FileNotFoundError:
    st.error("Secrets not found!")
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

# --- 5. DATA LOADERS ---
def load_data():
    try:
        with get_db_connection().connect() as conn:
            df = pd.read_sql(text("SELECT * FROM nurses"), conn)
            
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()
            if 'bpm' not in df.columns: df['bpm'] = 70
            df['bpm'] = df['bpm'].fillna(70).astype(int)

            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James", "Maria", "Daniel", "Linda", "Kevin", "Susan", "Thomas"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones", "Rodriguez", "Lee", "Martinez", "Anderson", "Taylor", "Wilson"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def gen_profile(nid):
                random.seed(nid)
                fn = f"{random.choice(first_names)} {random.choice(last_names)}"
                dept = depts[nid % 5]
                base_shift = 6 + (nid % 10) 
                hours_on_shift = np.round(base_shift + random.uniform(0, 2), 1)
                return fn, dept, hours_on_shift

            df['Full_Name'], df['Department'], df['Hours_On_Shift'] = zip(*df['nurse_id'].apply(gen_profile))
            
            def calculate_risk(row):
                if row['status'] == 'Relieved': return 12
                hours_risk = row['Hours_On_Shift'] * 4.5
                bpm_stress = max(0, row['bpm'] - 70) * 1.2
                total_risk = hours_risk + bpm_stress
                return int(min(max(total_risk, 5), 99))

            df['Calculated_Risk'] = df.apply(calculate_risk, axis=1)
            df['incident_probability'] = df[['incident_probability', 'Calculated_Risk']].max(axis=1)

        return df
    except Exception: return None

def load_audit_logs():
    try:
        with get_db_connection().connect() as conn:
            return pd.read_sql(text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC"), conn)
    except: return pd.DataFrame() 

# --- 6. ACTIONS ---
def relieve_nurse_in_db(fatigued_id, risk_val, replacement_name, is_ai=False):
    try:
        with get_db_connection().begin() as conn: 
            conn.execute(text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved', bpm = 72 WHERE nurse_id = :id"), {"id": fatigued_id})
            action = 'AI_AUTO_RESOLVE' if is_ai else 'MANUAL_SWAP'
            msg = f"Auto-Swap with {replacement_name}" if is_ai else f"Swapped with {replacement_name}"
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (:id, :t, :r, :a)"), 
                         {"id": fatigued_id, "t": action, "r": risk_val, "a": msg})
        if DISCORD_URL:
            try: requests.post(DISCORD_URL, json={"content": f"🚨 **SHIFTGUARD ALERT**\nNurse {fatigued_id} relieved by {replacement_name}. Risk: {risk_val}%"}, timeout=1)
            except: pass
        return True
    except: return False

def reset_simulation():
    try:
        with get_db_connection().begin() as conn:
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15, bpm = 75"))
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98, bpm = 115 WHERE nurse_id IN (9, 19, 38)")) 
            conn.execute(text("TRUNCATE TABLE audit_logs"))
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (101, 'AI_AUTO_RESOLVE', 88, 'Auto-Swap with Float Pool')"))
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (42, 'MANUAL_SWAP', 92, 'Swapped with Sarah J.')"))
        return True
    except: return False

# --- 7. CHAT BOT LOGIC ---
def get_bot_response(query, df):
    q = query.lower()
    if "risk" in q:
        r = df.sort_values('incident_probability', ascending=False).iloc[0]
        return f"⚠️ **CRITICAL:** {r['Full_Name']} (ID: {r['nurse_id']}) is at {r['incident_probability']}% Risk (HR: {r['bpm']} BPM)."
    if "replace" in q:
        cand = df[(df['incident_probability'] < 50) & (df['status'] != 'Relieved')]
        if not cand.empty:
            r = cand.iloc[0]
            return f"✅ **Recommendation:** Assign {r['Full_Name']} (Risk: {r['incident_probability']}%)"
    if "status" in q:
        crit = len(df[df['incident_probability'] > 85])
        return f"📊 **Unit Status:** Active: {len(df)} | Critical: {crit} | Avg BPM: {int(df['bpm'].mean())}"
    return "ShiftGuard Copilot Online."

# --- MAIN UI LAYOUT ---
c1, c2 = st.columns([6, 2])
with c1:
    st.title("SHIFTGUARD")
    st.caption("ENTERPRISE RISK COMMAND CENTER | PROD-US-EAST")
with c2:
    st.markdown("<br><div style='text-align:right; font-family:monospace; color:#4CAF50'>🟢 SYSTEM ONLINE</div>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.header("Copilot")
    if "messages" not in st.session_state: st.session_state.messages = [{"role":"assistant","content":"Monitoring live biometrics..."}]
    for m in st.session_state.messages:
        st.markdown(f"<div class='{'user-msg' if m['role']=='user' else 'bot-msg'}'>{m['content']}</div>", unsafe_allow_html=True)
    if p := st.chat_input("Query..."):
        st.session_state.messages.append({"role":"user","content":p})
        st.rerun()
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Processing..."):
            time.sleep(0.5)
            df_chat = load_data()
            r = get_bot_response(st.session_state.messages[-1]["content"], df_chat)
            st.session_state.messages.append({"role":"assistant","content":r})
            st.rerun()
    st.divider()
    if st.button("🔄 RESET SIMULATION", type="primary"):
        reset_simulation()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🔴 Live Operations", "📊 Analytics", "⚖️ Audit Trail"])

df = load_data()

with tab1:
    if df is not None:
        active_risk_df = df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')]
        count = len(active_risk_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Staff", len(df))
        m2.metric("Critical Alerts", count, "Action Reqd" if count > 0 else None, delta_color="inverse")
        m3.metric("Avg Unit BPM", f"{int(df['bpm'].mean())}", "🔺 +12% vs Shift Start", delta_color="inverse")
        m4.metric("Latency", "24ms")

        if count > 0:
            with st.expander("🤖 **RECOMMENDATION ENGINE**", expanded=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**Status:** <span style='color:#ff4b4b'>CRITICAL</span> - {count} anomalies detected.", unsafe_allow_html=True)
                if c2.button("🚀 EXECUTE AUTO-FIX", type="primary"):
                    safe = df[df['incident_probability'] < 50]['Full_Name'].tolist()
                    random.shuffle(safe)
                    prog = st.progress(0)
                    for i, (idx, row) in enumerate(active_risk_df.iterrows()):
                        time.sleep(0.3)
                        rep = safe.pop(0) if safe else "Float Pool RN"
                        relieve_nurse_in_db(row['nurse_id'], row['incident_probability'], rep, True)
                        prog.progress((i+1)/count)
                    st.success("Fixed.")
                    time.sleep(1)
                    st.rerun()

        st.subheader("🚨 High Priority Interventions")
        crit = df[(df['incident_probability'] >= 90) | (df['status'] == 'Relieved')].sort_values('incident_probability', ascending=False)
        
        # --- FIX: RESTORED FALLBACK LOGIC ---
        safe = df[df['incident_probability'] < 50].sort_values('incident_probability')
        if safe.empty: safe = df.sort_values('incident_probability', ascending=True).head(5)
        # ------------------------------------
        
        safe_opts = safe.apply(lambda x: f"{x['Full_Name']} [ID: {x['nurse_id']}] (Risk: {x['incident_probability']}%)", axis=1).tolist()
        
        if crit.empty: st.success("✅ Unit Safe")
        else:
            for i, (idx, row) in enumerate(crit.iterrows()):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1.2])
                    c1.write(f"**{row['Full_Name']}**")
                    c1.caption(f"ID: {row['nurse_id']}")
                    if row['status'] == 'Relieved': c2.success("RELIEVED")
                    else: st.progress(row['incident_probability']/100, f"Risk: {row['incident_probability']}%")
                    
                    # --- RESTORED RISK BREAKDOWN ---
                    with st.expander("📉 Risk Factors"):
                        st.caption(f"Shift: {row['Hours_On_Shift']}h | **Heart Rate: {row['bpm']} BPM**")
                    
                    if row['status'] != 'Relieved':
                        with c3.popover("Swap"):
                            sel = st.selectbox("With:", safe_opts, key=f"s_{i}")
                            rep = sel.split(" [")[0] if sel else "Unknown"
                            if st.button("Confirm", key=f"c_{i}"):
                                relieve_nurse_in_db(row['nurse_id'], row['incident_probability'], rep)
                                st.rerun()

        st.subheader("📋 Full Roster")
        st.dataframe(df[['nurse_id', 'Full_Name', 'Department', 'Hours_On_Shift', 'bpm', 'incident_probability', 'status']], use_container_width=True, hide_index=True)

with tab2:
    st.header("📊 Analytics")
    st.subheader("🧠 Sentinel: Voice-to-Risk Engine")
    
    with st.container(border=True):
        st.info("ℹ️ **Mobile App Integration:** Select input method.")
        nid = st.selectbox("Nurse ID", df['nurse_id'].unique())
        
        # --- DUAL INPUT LOGIC ---
        in_mode = st.radio("Input:", ["🎙️ Voice (Mobile)", "⌨️ Manual Entry"], horizontal=True)
        
        transcript = ""
        
        if "Voice" in in_mode:
            st.caption("Click to record audio via browser:")
            # Capture WebM from Browser
            audio = mic_recorder(start_prompt="🎤 START RECORDING", stop_prompt="⏹️ STOP", just_once=False, key='recorder')
            
            if audio:
                st.audio(audio['bytes'])
                
                # --- AUDIO CONVERSION LOGIC ---
                with st.spinner("Processing Audio..."):
                    try:
                        # 1. Convert bytes to AudioSegment
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']), format="webm")
                        
                        # 2. Export as WAV
                        wav_buffer = io.BytesIO()
                        audio_segment.export(wav_buffer, format="wav")
                        wav_buffer.seek(0)
                        
                        # 3. Transcribe
                        r = sr.Recognizer()
                        with sr.AudioFile(wav_buffer) as source:
                            audio_content = r.record(source)
                            transcript = r.recognize_google(audio_content)
                            st.success(f"**Transcript:** {transcript}")
                            
                    except Exception as e:
                        # FALLBACK FOR DEMO if FFmpeg missing
                        st.error(f"Conversion Failed: {e}")
                        st.warning("⚠️ Falling back to Simulation Mode for Demo...")
                        transcript = "I am struggling to keep my eyes open and feeling very dizzy. I need a break."
                        st.success(f"**Transcript:** {transcript}")

        else:
            transcript = st.text_input("Log Entry:", placeholder="Type here...")

        # EXECUTE ANALYSIS
        if transcript:
            if st.button("Analyze Input") or ("Voice" in in_mode):
                score, phrases = run_sentinel_analysis(transcript)
                c1, c2 = st.columns(2)
                c1.metric("Stress Score", f"{int(score*100)}%")
                c2.write(phrases)
                
                if score > 0.7:
                    st.error("⚠️ CRITICAL. Updating Database...")
                    with get_db_connection().begin() as conn:
                        conn.execute(text("UPDATE nurses SET fatigue_risk=99, bpm=110 WHERE nurse_id=:id"), {"id": nid})
                    time.sleep(1)
                    st.rerun()

    st.divider()
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Fatigue Load")
            d = df.groupby("Department")["incident_probability"].mean().reset_index()
            c = alt.Chart(d).mark_bar(color='#FF4B4B').encode(x='Department', y='incident_probability').properties(background='transparent')
            st.altair_chart(c, use_container_width=True)
        with c2:
            st.markdown("##### Physiological Stress")
            d = df.groupby("Department")["bpm"].mean().reset_index()
            c = alt.Chart(d).mark_bar(color='#4CAF50').encode(x='Department', y=alt.Y('bpm', scale=alt.Scale(domain=[60, 120]))).properties(background='transparent')
            st.altair_chart(c, use_container_width=True)

with tab3:
    if st.button("Refresh"): st.rerun()
    st.dataframe(load_audit_logs(), use_container_width=True)
