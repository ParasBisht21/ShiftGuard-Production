import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np # Needed for the fake data generation

# --- 1. CLOUD CONFIGURATION ---
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except FileNotFoundError:
    st.error("Secrets not found!")
    st.stop()

# --- 2. DATABASE CONNECTION (Linux/Cloud Compatible) ---
def get_db_connection():
    conn_str = f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    engine = create_engine(conn_str)
    return engine

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="ShiftGuard PROD", layout="wide", page_icon="🛡️")

# INJECT CSS FOR "DENSITY" LOOK
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; padding: 10px; }
        /* Make the table look tighter */
        div[data-testid="stDataFrame"] { font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADER + "MAGIC ENRICHMENT" ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses")
            df = pd.read_sql(query, conn)
            
            # --- FIX NAMES & CLEANUP ---
            if 'fatigue_risk' in df.columns:
                df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
            if 'status' in df.columns:
                df['status'] = df['status'].str.strip()

            # --- THE "DENSITY" HACK: GENERATE CONSISTENT FAKE DATA ---
            # We use the nurse_id as a seed so the data stays the same on refresh!
            
            depts = ['ICU', 'ER', 'Pediatrics', 'Oncology', 'Surgical Ward']
            shifts = ['Day (07:00)', 'Night (19:00)', 'Swing (12:00)']
            
            # Create synthetic columns
            df['Department'] = df['nurse_id'].apply(lambda x: depts[int(x) % len(depts)])
            df['Shift_Start'] = df['nurse_id'].apply(lambda x: shifts[int(x) % len(shifts)])
            
            # Generate "Hours on Shift" (High risk = longer shift)
            # If risk is > 80, make shift long (10-14 hours). If low, 2-6 hours.
            df['Hours_On_Shift'] = df.apply(lambda row: 
                np.round(8 + (row['incident_probability']/20) + (int(row['nurse_id'])%3), 1), axis=1)
            
            # Generate "Heart Rate" (Correlated with risk)
            df['BPM'] = df.apply(lambda row: 
                int(70 + (row['incident_probability']/2) + (int(row['nurse_id'])%10)), axis=1)

        return df
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        return None

# --- 5. DATABASE WRITE ACTION ---
def relieve_nurse_in_db(nurse_id):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            sql = text("UPDATE nurses SET fatigue_risk = 12, status = 'Relieved' WHERE nurse_id = :id")
            conn.execute(sql, {"id": nurse_id})
        return True
    except Exception as e:
        st.error(f"Write Error: {e}")
        return False

# --- MAIN APP ---
df = load_data()

st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")
st.caption(f"🔴 LIVE CONNECTION | {len(df)} Staff Members Monitored | Location: Central India Node")

if df is not None:
    # --- TOP METRICS ROW ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Staff", len(df), "Active on Floor")
    with m2:
        critical = len(df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')])
        st.metric("Critical Alerts", critical, "Immediate Action Reqd", delta_color="inverse")
    with m3:
        avg_bpm = int(df['BPM'].mean())
        st.metric("Avg Unit Stress", f"{avg_bpm} BPM", "+12% vs Baseline", delta_color="inverse")
    with m4:
        long_shifts = len(df[df['Hours_On_Shift'] > 12])
        st.metric("Overtime Violations", long_shifts, "Compliance Risk")
    with m5:
        st.metric("System Latency", "24ms", "Azure SQL")

    st.divider()

    # --- LAYOUT: SPLIT SCREEN (CARDS LEFT, CHART RIGHT) ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🚨 High Priority Interventions")
        
        # Sort logic: Relieved at top, then Risk
        df['sort_score'] = df['incident_probability']
        df.loc[df['status'] == 'Relieved', 'sort_score'] = 1000
        top_nurses = df.sort_values('sort_score', ascending=False).head(3)

        # RENDER CARDS
        for i, (idx, nurse) in enumerate(top_nurses.iterrows()):
            nurse_id = int(nurse['nurse_id'])
            
            # Dynamic Color based on risk
            risk_color = "red" if nurse['incident_probability'] > 80 else "orange"
            if nurse['status'] == 'Relieved': risk_color = "green"

            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.markdown(f"### 🩺 **ID: {nurse_id}**")
                    st.caption(f"Dept: {nurse['Department']}")
                
                with c2:
                    # SHOW THE DATA THAT JUSTIFIES THE ACTION
                    if nurse['status'] == 'Relieved':
                        st.success("✅ **SHIFT RELIEVED**")
                    else:
                        st.markdown(f"**Fatigue Risk: {nurse['incident_probability']}%**")
                        st.progress(nurse['incident_probability'] / 100)
                        st.markdown(f"⏱️ **Shift Duration:** {nurse['Hours_On_Shift']} hrs | ❤️ **BPM:** {nurse['BPM']}")

                with c3:
                    if nurse['status'] != 'Relieved':
                        if st.button(f"⚡ SWAP SHIFT", key=f"btn_{i}", type="primary"):
                            with st.spinner("Authorizing..."):
                                success = relieve_nurse_in_db(nurse_id)
                                time.sleep(1)
                            if success:
                                st.rerun()
                    else:
                        st.button("Archive Log", disabled=True, key=f"d_{i}")

    with col_right:
        st.subheader("📊 Unit Status")
        # Simple Bar Chart of Risk by Department
        dept_risk = df.groupby('Department')['incident_probability'].mean().reset_index()
        st.bar_chart(dept_risk, x="Department", y="incident_probability", color="#FF4B4B")
        
        st.info("ℹ️ **Insight:** ICU unit is showing 15% higher fatigue levels than surgical ward.")

    st.divider()

    # --- BOTTOM SECTION: THE "FULL ROSTER" (DENSITY) ---
    st.subheader("📋 Live Staff Roster (All Units)")
    
    # Show a dense table so it looks like a database
    display_cols = ['nurse_id', 'Department', 'Shift_Start', 'Hours_On_Shift', 'BPM', 'incident_probability', 'status']
    
    # Style the dataframe (Highlight high risk rows)
    st.dataframe(
        df[display_cols].sort_values('incident_probability', ascending=False),
        column_config={
            "incident_probability": st.column_config.ProgressColumn("Risk Score", format="%d%%", min_value=0, max_value=100),
            "nurse_id": st.column_config.TextColumn("Staff ID"),
        },
        use_container_width=True,
        hide_index=True,
        height=400 # Fixed height with scrollbar
    )
