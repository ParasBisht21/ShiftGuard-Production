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

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="ShiftGuard Enterprise", layout="wide", page_icon="🛡️")

def inject_custom_css():
    st.markdown("""
        <style>
            .stApp { background-color: #000000; color: #E5E5E5; font-family: 'Inter', sans-serif; }
            #MainMenu, footer, header {visibility: hidden;}
            div[data-testid="stMetric"], div[data-testid="stContainer"] {
                background-color: #0F0F0F; border: 1px solid #333; border-radius: 6px; padding: 15px; color: white;
            }
            .user-msg { background-color: #FFF; color: #000; padding: 10px; border-radius: 12px 12px 0 12px; margin: 5px 0; text-align: right; }
            .bot-msg { background-color: #222; border: 1px solid #444; color: #DDD; padding: 10px; border-radius: 12px 12px 12px 0; margin: 5px 0; }
            .critical-badge { background-color: #FFF; color: #000; font-weight: 900; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; }
            /* Force Tab Text Color */
            button[data-baseweb="tab"] { color: #888; }
            button[data-baseweb="tab"][aria-selected="true"] { color: #FFF; border-bottom-color: #FFF; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- 2. CLOUD CONNECTION ---
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

def get_db_connection():
    return create_engine(f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}")

# --- 3. LOGIC ENGINES ---
def run_sentinel_analysis(text_input):
    if not AI_KEY or not AI_ENDPOINT: return 0.88, ["System Offline", "Fatigue"]
    try:
        credential = AzureKeyCredential(AI_KEY)
        client = TextAnalyticsClient(endpoint=AI_ENDPOINT, credential=credential)
        response = client.analyze_sentiment(documents=[text_input])[0]
        kp = client.extract_key_phrases(documents=[text_input])[0]
        return response.confidence_scores.negative, kp.key_phrases
    except: return 0.0, ["Connection Error"]

def load_data():
    try:
        with get_db_connection().connect() as conn:
            df = pd.read_sql(text("SELECT * FROM nurses"), conn)
            
            if 'fatigue_risk' in df.columns: df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns: df['status'] = df['status'].str.strip()

            first = ["Sarah", "Mike", "Jessica", "David", "Emily", "Robert", "Jennifer", "William", "Lisa", "James"]
            last = ["Chen", "Smith", "Patel", "Johnson", "Kim", "Garcia", "Singh", "Miller", "Wong", "Jones"]
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            
            def gen_profile(nid):
                random.seed(nid)
                fn = f"{first[nid % 10]} {last[nid % 10]}"
                dept = depts[nid % 5]
                base = 6 + (nid % 10) 
                hours = np.round(base + random.uniform(0, 2), 1)
                bpm = int(65 + (hours * 2) + random.randint(-5, 5))
                return fn, dept, hours, bpm

            df['Full_Name'], df['Department'], df['Hours_On_Shift'], df['BPM'] = zip(*df['nurse_id'].apply(gen_profile))
            
            def calc_risk(row):
                if row['status'] == 'Relieved': return 12
                stress = max(0, row['BPM'] - 70)
                risk = (row['Hours_On_Shift'] * 4.5) + (stress * 1.2)
                return int(min(max(risk, 5), 99))

            df['Calculated_Risk'] = df.apply(calc_risk, axis=1)
            df['incident_probability'] = df[['incident_probability', 'Calculated_Risk']].max(axis=1)
        return df
    except Exception as e:
        return None

def load_audit_logs():
    try:
        with get_db_connection().connect() as conn:
            return pd.read_sql(text("SELECT TOP 50 * FROM audit_logs ORDER BY timestamp DESC"), conn)
    except: return pd.DataFrame() 

def relieve_nurse_in_db(nid, risk, rep, is_ai=False):
    try:
        with get_db_connection().begin() as conn: 
            conn.execute(text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id"), {"id": nid})
            action = 'AI_COPILOT_AUTO' if is_ai else 'MANUAL_SWAP'
            msg = f"Auto-Swap with {rep}" if is_ai else f"Swapped with {rep}"
            conn.execute(text("INSERT INTO audit_logs (nurse_id, action_type, risk_level_at_time, manager_action) VALUES (:id, :t, :r, :a)"), 
                         {"id": nid, "t": action, "r": risk, "a": msg})
        if DISCORD_URL: requests.post(DISCORD_URL, json={"content": f"🚨 **ALERT** Nurse {nid} relieved by {rep}. Risk: {risk}%"}, timeout=1)
        return True
    except: return False

def reset_simulation():
    try:
        with get_db_connection().begin() as conn:
            conn.execute(text("UPDATE nurses SET status = 'Active', fatigue_risk = 15"))
            conn.execute(text("UPDATE nurses SET fatigue_risk = 98 WHERE nurse_id IN (34, 68, 93, 29, 55)")) 
            conn.execute(text("TRUNCATE TABLE audit_logs"))
        return True
    except: return False

def get_bot_response(q, df):
    q = q.lower()
    if "risk" in q:
        r = df.sort_values('incident_probability', ascending=False).iloc[0]
        return f"⚠️ **CRITICAL:** {r['Full_Name']} ({r['Department']}) is at {r['incident_probability']}% Risk."
    if "status" in q:
        c = len(df[df['incident_probability']>85])
        return f"📊 **Status:** Active Staff: {len(df)} | Critical: {c} | Avg Risk: {int(df['incident_probability'].mean())}%"
    return "I can help with Risk Analysis and Staffing."

# --- 4. MAIN LAYOUT ---
c1, c2 = st.columns([6, 2])
with c1:
    st.title("SHIFTGUARD")
    st.caption("ENTERPRISE RISK COMMAND CENTER | PROD-US-EAST")
with c2:
    st.markdown("<br><div style='text-align:right; font-family:monospace; color:#4CAF50'>🟢 SYSTEM ONLINE</div>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.header("Copilot")
    if "messages" not in st.session_state: st.session_state.messages = [{"role":"assistant","content":"System Active."}]
    for m in st.session_state.messages:
        st.markdown(f"<div class='{'user-msg' if m['role']=='user' else 'bot-msg'}'>{m['content']}</div>", unsafe_allow_html=True)
    if p := st.chat_input("Query..."):
        st.session_state.messages.append({"role":"user","content":p})
        st.rerun()
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            df_chat = load_data()
            if df_chat is not None:
                r = get_bot_response(st.session_state.messages[-1]["content"], df_chat)
                st.session_state.messages.append({"role":"assistant","content":r})
            st.rerun()
    st.divider()
    if st.button("🔄 Reset"):
        reset_simulation()
        st.rerun()

# 3 TABS NOW
tab1, tab2, tab3 = st.tabs(["🔴 Live Operations", "📊 Analytics", "⚖️ Audit Trail"])

with tab1:
    df = load_data()
    if df is None:
        st.error("🚨 Database Connection Failed. Please check secrets.")
    else:
        # A. METRICS
        active_risk_df = df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')]
        count = len(active_risk_df)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Staff", len(df))
        m2.metric("Critical Risks", count, "Action Reqd" if count > 0 else None, delta_color="inverse")
        m3.metric("Avg Unit BPM", f"{int(df['BPM'].mean())}")
        m4.metric("Latency", "24ms")

        st.divider()

        # C. AI AUTO-FIX
        if count > 0:
            with st.expander("🤖 **RECOMMENDATION ENGINE**", expanded=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**Status:** <span style='color:#ff4b4b'>CRITICAL</span> - {count} anomalies detected.", unsafe_allow_html=True)
                if c2.button("🚀 EXECUTE AUTO-FIX", type="primary"):
                    safe = df[df['incident_probability'] < 20]['Full_Name'].tolist()
                    prog = st.progress(0)
                    for i, (idx, row) in enumerate(active_risk_df.iterrows()):
                        time.sleep(0.3)
                        rep = safe[i % len(safe)] if safe else "Float"
                        relieve_nurse_in_db(row['nurse_id'], row['incident_probability'], rep, True)
                        prog.progress((i+1)/count)
                    st.success("Optimization Complete.")
                    time.sleep(1)
                    st.rerun()

        # D. ROSTER & SENTINEL
        c_left, c_right = st.columns([2, 1])
        
        with c_left:
            st.subheader("🚨 High Priority Interventions")
            crit_nurses = df[(df['incident_probability'] >= 90) | (df['status'] == 'Relieved')].sort_values('incident_probability', ascending=False)
            safe_opts = df[df['incident_probability'] < 30]['Full_Name'].tolist()

            if crit_nurses.empty: st.success("✅ Unit Safe")
            else:
                for i, (idx, row) in enumerate(crit_nurses.iterrows()):
                    with st.container():
                        c1, c2, c3 = st.columns([1, 2, 1])
                        c1.write(f"**{row['Full_Name']}**")
                        c1.caption(f"ID: {row['nurse_id']}")
                        if row['status'] == 'Relieved': c2.success("RELIEVED")
                        else: c2.progress(row['incident_probability']/100, f"Risk: {row['incident_probability']}%")
                        
                        if row['status'] != 'Relieved':
                            with c3.popover("Swap"):
                                sel = st.selectbox("With:", safe_opts, key=f"s_{i}")
                                if st.button("Confirm", key=f"c_{i}"):
                                    relieve_nurse_in_db(row['nurse_id'], row['incident_probability'], sel)
                                    st.rerun()
        
        with c_right:
             st.subheader("🧠 Sentinel NLP")
             with st.container(border=True):
                 nid = st.selectbox("ID", df['nurse_id'].unique())
                 txt = st.text_input("Log", placeholder="Dizzy...")
                 if st.button("Analyze"):
                     s, p = run_sentinel_analysis(txt)
                     st.metric("Stress", f"{int(s*100)}%")
                     st.write(p)
                     if s > 0.7:
                         with get_db_connection().begin() as conn:
                             conn.execute(text("UPDATE nurses SET fatigue_risk=99 WHERE nurse_id=:id"), {"id": nid})
                         st.error("Updated Risk Model")
                         time.sleep(1)
                         st.rerun()

with tab2:
    st.header("📊 Enterprise Analytics")
    if df is not None:
        c1, c2 = st.columns(2)
        
        # We use Altair to ensure WHITE text for axes
        with c1:
            st.markdown("##### Fatigue Load by Department")
            chart_data = df.groupby("Department")["incident_probability"].mean().reset_index()
            c = alt.Chart(chart_data).mark_bar(color='#FF4B4B').encode(
                x=alt.X('Department', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('incident_probability', title='Avg Risk %', axis=alt.Axis(labelColor='white', titleColor='white'))
            ).configure_view(stroke=None).properties(background='transparent', height=300)
            st.altair_chart(c, use_container_width=True)

        with c2:
            st.markdown("##### Staffing Distribution")
            count_data = df["Department"].value_counts().reset_index()
            count_data.columns = ["Department", "Count"]
            c = alt.Chart(count_data).mark_bar(color='#333333').encode(
                x=alt.X('Department', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('Count', title='Staff Count', axis=alt.Axis(labelColor='white', titleColor='white'))
            ).configure_view(stroke=None).properties(background='transparent', height=300)
            st.altair_chart(c, use_container_width=True)

with tab3:
    if st.button("Refresh"): st.rerun()
    st.dataframe(load_audit_logs(), use_container_width=True)
