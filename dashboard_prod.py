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

# --- 5. DATA LOADERS (NOW READS BPM FROM DB) ---
def load_data():
    try:
        with get_db_connection().connect() as conn:
            # PULL EVERYTHING FROM DB (Including new 'bpm' column)
            df = pd.read_sql(text("SELECT * FROM nurses"), conn)
            
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()

            # Handle Null BPMs (Default to 70 Resting HR)
            if 'bpm' not in df.columns: df['bpm'] = 70
            df['bpm'] = df['bpm'].fillna(70).astype(int)

            # Heuristic Engine: Profile Generation Only
            first_names = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James"]
            last_names = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def gen_profile(nid):
                random.seed(nid) # Deterministic
                fn = f"{first_names[nid % 10]} {last_names[nid % 10]}"
                dept = depts[nid % 5]
                base_shift = 6 + (nid % 10) 
                hours_on_shift = np.round(base_shift + random.uniform(0, 2), 1)
                return fn, dept, hours_on_shift

            df['Full_Name'], df['Department'], df['Hours_On_Shift'] = zip(*df['nurse_id'].apply(gen_profile))
            
            # --- REAL-TIME RISK CALCULATION WITH BPM ---
            def calculate_risk(row):
                if row['status'] == 'Relieved': return 12
                
                # Factor 1: Shift Length
                hours_risk = row['Hours_On_Shift'] * 4.5
                
                # Factor 2: Bio-Metric Stress (BPM > 70)
                # Every heartbeat above 70 adds 1.2% risk
                bpm_stress = max(0, row['bpm'] - 70) * 1.2
                
                total_risk = hours_risk + bpm_stress
                
                # Clamp between 5 and 99
                return int(min(max(total_risk, 5), 99))

            df['Calculated_Risk'] = df.apply(calculate_risk, axis=1)
            # Use max of DB value or Calculated value to allow overrides
            df['incident_probability'] = df[['incident_probability', 'Calculated_Risk']].max(axis=1)

        return df
    except Exception: return None

def load_audit_logs():
    try:
        with get_db_connection().connect() as conn:
            return pd.read_sql(text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC"), conn)
    except: return pd.DataFrame() 

# --- 6. ACTIONS (SQL + DISCORD) ---
def relieve_nurse_in_db(fatigued_id, risk_val, replacement_name, is_ai=False):
    try:
        with get_db_connection().begin() as conn: 
            # Relieving a nurse resets their BPM to 72 (Resting)
            conn.execute(text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved', bpm = 72 WHERE nurse_id = :id"), {"id": fatigued_id})
            
            action = 'AI_AUTO_RESOLVE' if is_ai else 'MANUAL_SWAP'
            msg = f"Auto-Swap with {replacement_name}" if is_ai else f"Swapped with {replacement_name}"
            
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (:id, :t, :r, :a)"), 
                         {"id": fatigued_id, "t": action, "r": risk_val, "a": msg})

        if DISCORD_URL:
            try:
                requests.post(DISCORD_URL, json={"content": f"🚨 **SHIFTGUARD ALERT**\nNurse {fatigued_id} relieved by {replacement_name}. Risk: {risk_val}%"}, timeout=1)
            except: pass
        return True
    except: return False

def reset_simulation():
    try:
        with get_db_connection().begin() as conn:
            # Reset everyone to Safe Status and Normal BPM
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15, bpm = 75"))
            
            # Create Specific Demo Scenarios (High Risk / High BPM)
            # IDs 9, 19, 38 set to High Stress
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98, bpm = 115 WHERE nurse_id IN (9, 19, 38)")) 
            
            conn.execute(text("TRUNCATE TABLE audit_logs"))
        return True
    except: return False

# --- 7. CHAT BOT LOGIC ---
def get_bot_response(query, df):
    q = query.lower()
    
    # INTENT: High Risk
    if "risk" in q or "critical" in q or "danger" in q:
        riskiest = df.sort_values('incident_probability', ascending=False).iloc[0]
        return f"⚠️ **CRITICAL ALERT:**\n\n**{riskiest['Full_Name']}** (ID: {riskiest['nurse_id']}) is at **{riskiest['incident_probability']}% Risk**.\n\nHeart Rate: **{riskiest['bpm']} BPM** (High Stress).\nImmediate relief recommended."

    # INTENT: Replacement
    if "replace" in q or "swap" in q:
        target_name = ""
        for name in df['Full_Name'].unique():
            if name.split()[0].lower() in q: target_name = name; break
        
        # Recommendation Engine
        candidates = df[(df['incident_probability'] < 50) & (df['status'] != 'Relieved')]
        if target_name:
            target_dept = df[df['Full_Name'] == target_name].iloc[0]['Department']
            dept_cands = candidates[candidates['Department'] == target_dept]
            if not dept_cands.empty: candidates = dept_cands
            
        if not candidates.empty:
            rec = candidates.sort_values('incident_probability').iloc[0]
            return f"✅ **Recommendation:**\n\nAssign **{rec['Full_Name']}**.\n- Dept: {rec['Department']}\n- Risk: {rec['incident_probability']}%\n- HR: {rec['bpm']} BPM (Normal)"
        return "⚠️ No low-risk staff available. Activate Float Pool."

    # INTENT: Status
    if "status" in q or "report" in q:
        crit = len(df[df['incident_probability'] > 85])
        avg_bpm = int(df['bpm'].mean())
        return f"📊 **Unit Status Report:**\n\n- **Active Staff:** {len(df)}\n- **Critical Alerts:** {crit}\n- **Avg Unit BPM:** {avg_bpm} (Live Data)"

    return "ShiftGuard Copilot Online. I can help with:\n- Identifying high risk staff\n- Finding replacements\n- Real-time bio-metric status"

# --- MAIN UI LAYOUT ---

# 1. Header
c1, c2 = st.columns([6, 2])
with c1:
    st.title("SHIFTGUARD")
    st.caption("ENTERPRISE RISK COMMAND CENTER | PROD-US-EAST")
with c2:
    st.markdown("<br><div style='text-align:right; font-family:monospace; color:#4CAF50'>🟢 SYSTEM ONLINE</div>", unsafe_allow_html=True)

st.divider()

# 2. Sidebar (Chat & Admin)
with st.sidebar:
    st.header("Copilot")
    if "messages" not in st.session_state: 
        st.session_state.messages = [{"role":"assistant","content":"Connected to Azure SQL. Monitoring live biometrics."}]
    
    # Render Chat
    for m in st.session_state.messages:
        role_class = "user-msg" if m['role'] == "user" else "bot-msg"
        st.markdown(f"<div class='{role_class}'>{m['content']}</div>", unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask about risks or staffing..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.rerun()
        
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Analyzing roster..."):
            time.sleep(0.5) 
            df_chat = load_data()
            response = get_bot_response(st.session_state.messages[-1]["content"], df_chat)
            st.session_state.messages.append({"role":"assistant","content":response})
            st.rerun()
            
    st.markdown("---")
    st.header("Admin Console")
    if st.button("🔄 RESET SIMULATION", type="primary"):
        reset_simulation()
        st.session_state.messages = [{"role":"assistant","content":"♻️ Database Reset. Starting new shift simulation."}]
        st.rerun()

# 3. Tabs
tab1, tab2, tab3 = st.tabs(["🔴 Live Operations", "📊 Analytics", "⚖️ Audit Trail"])

df = load_data()

with tab1:
    if df is not None:
        # Metrics
        active_risk_df = df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')]
        active_risk_count = len(active_risk_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Staff", len(df))
        m2.metric("Critical Alerts", active_risk_count, delta="Action Reqd" if active_risk_count > 0 else None, delta_color="inverse")
        
        # BPM Metric (Live from DB)
        m3.metric("Avg Unit BPM", f"{int(df['bpm'].mean())}", "Bio-Data Live")
        m4.metric("Latency", "24ms")

        # AI Auto-Fix Section
        if active_risk_count > 0:
            with st.expander("🤖 **RECOMMENDATION ENGINE: Heuristic Optimization Detected**", expanded=True):
                c1, c2 = st.columns([3, 1])
                with c1: st.markdown(f"**Status:** <span style='color:#ff4b4b'>CRITICAL INSTABILITY</span><br>Risk Model identified **{active_risk_count}** anomalies.", unsafe_allow_html=True)
                with c2:
                    if st.button("🚀 EXECUTE AUTO-FIX", type="primary", use_container_width=True):
                        
                        # POOL LOGIC
                        safe_staff_pool = df[df['incident_probability'] < 50]['Full_Name'].tolist()
                        random.shuffle(safe_staff_pool)
                        
                        prog = st.progress(0)
                        for i, (idx, nurse) in enumerate(active_risk_df.iterrows()):
                            time.sleep(0.3)
                            
                            if len(safe_staff_pool) > 0:
                                rep = safe_staff_pool.pop(0)
                            else:
                                rep = "Float Pool RN (External)"
                                
                            relieve_nurse_in_db(nurse['nurse_id'], nurse['incident_probability'], rep, is_ai=True)
                            prog.progress((i+1)/active_risk_count)
                        st.success("Optimization Complete.")
                        time.sleep(1)
                        st.rerun()

        # Detailed Roster
        st.subheader("🚨 High Priority Interventions")
        critical_nurses = df[(df['incident_probability'] >= 90) | (df['status'] == 'Relieved')].sort_values('incident_probability', ascending=False)
        
        # Safe pool for manual swap dropdowns
        safe_nurses = df[df['incident_probability'] < 50].sort_values('incident_probability')
        if safe_nurses.empty: safe_nurses = df.sort_values('incident_probability', ascending=True).head(5)
        safe_nurses_list = safe_nurses.apply(lambda x: f"{x['Full_Name']} (Risk: {x['incident_probability']}%)", axis=1).tolist()
        
        if critical_nurses.empty:
            st.success("✅ Unit Operating Safely.")
        else:
            for i, (idx, nurse) in enumerate(critical_nurses.iterrows()):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 1.2]) 
                    with c1: 
                        st.markdown(f"### 🩺 **{nurse['nurse_id']}**")
                        st.caption(nurse['Department'])
                    with c2:
                        st.markdown(f"**{nurse['Full_Name']}**")
                        if nurse['status'] == 'Relieved': st.markdown("<span class='critical-badge'>RELIEVED</span>", unsafe_allow_html=True)
                        else: st.progress(nurse['incident_probability'] / 100, text=f"Risk: {nurse['incident_probability']}%")
                        
                        # Risk Breakdown Expander
                        with st.expander("📉 Risk Factors"):
                            st.caption(f"Shift: {nurse['Hours_On_Shift']}h | **Heart Rate: {nurse['bpm']} BPM**")
                    with c3:
                        if nurse['status'] != 'Relieved':
                            with st.popover("⚡ MANAGE SWAP", use_container_width=True):
                                sel = st.selectbox("Staff:", safe_nurses_list, key=f"s_{i}")
                                rep_name = sel.split(" (")[0] if sel else "Unknown"
                                if st.button(f"Confirm", key=f"c_{i}", type="primary"):
                                    relieve_nurse_in_db(nurse['nurse_id'], nurse['incident_probability'], rep_name)
                                    st.rerun()

        st.subheader("📋 Full Roster")
        st.dataframe(df[['nurse_id', 'Full_Name', 'Department', 'Hours_On_Shift', 'bpm', 'incident_probability', 'status']], use_container_width=True, hide_index=True)

with tab2:
    st.header("📊 Enterprise Analytics")
    
    # Sentinel Analysis
    st.subheader("🧠 Sentinel NLP Engine")
    with st.container(border=True):
        nid = st.selectbox("Nurse ID", df['nurse_id'].unique())
        txt = st.text_input("Shift Log", placeholder="I am feeling dizzy...")
        if st.button("Analyze Log"):
            score, phrases = run_sentinel_analysis(txt)
            c1, c2 = st.columns(2)
            c1.metric("Stress Score", f"{int(score*100)}%")
            c2.write(phrases)
            if score > 0.7:
                st.error("⚠️ CRITICAL. Updating Database...")
                with get_db_connection().begin() as conn:
                    # Update Risk AND set BPM to indicate stress
                    conn.execute(text("UPDATE nurses SET fatigue_risk=99, bpm=110 WHERE nurse_id=:id"), {"id": nid})
                time.sleep(1)
                st.rerun()
    
    st.divider()
    
    # GRAPHS with White Text (Altair)
    if df is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Fatigue Load by Department")
            chart_data = df.groupby("Department")["incident_probability"].mean().reset_index()
            c = alt.Chart(chart_data).mark_bar(color='#FF4B4B').encode(
                x=alt.X('Department', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('incident_probability', title='Avg Risk %', axis=alt.Axis(labelColor='white', titleColor='white'))
            ).configure_view(stroke=None).properties(background='transparent', height=300)
            st.altair_chart(c, use_container_width=True)

        with c2:
            st.markdown("##### Bio-Metric Stress (Avg BPM)")
            bpm_data = df.groupby("Department")["bpm"].mean().reset_index()
            c = alt.Chart(bpm_data).mark_bar(color='#4CAF50').encode(
                x=alt.X('Department', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('bpm', title='Heart Rate (BPM)', scale=alt.Scale(domain=[60, 120]), axis=alt.Axis(labelColor='white', titleColor='white'))
            ).configure_view(stroke=None).properties(background='transparent', height=300)
            st.altair_chart(c, use_container_width=True)

with tab3:
    st.header("⚖️ Audit Logs")
    if st.button("Refresh"): st.rerun()
    st.dataframe(load_audit_logs(), use_container_width=True, hide_index=True)
