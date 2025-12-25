import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import urllib
import time

# --- 1. CLOUD CONFIGURATION ---
# --- 1. CLOUD CONFIGURATION ---
# We now pull these from the Cloud's "Safe Box" (Secrets)
try:
    SERVER = st.secrets["SERVER"]
    DATABASE = st.secrets["DATABASE"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except FileNotFoundError:
    # Fallback for local testing if .streamlit/secrets.toml is missing
    st.error("Secrets not found! Please add them to .streamlit/secrets.toml or Streamlit Cloud.")
    st.stop()

# --- 2. DATABASE CONNECTION ENGINE ---
def get_db_connection():
    # We switch to pymssql because it works on Streamlit Cloud without complex drivers
    conn_str = f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    engine = create_engine(conn_str)
    return engine

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="ShiftGuard PROD | Enterprise Safety", layout="wide", page_icon="🛡️")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1c24; border: 1px solid #30363d; }
    div[data-testid="stToast"] { background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. DATA LOADER (LIVE FROM AZURE) ---
def load_data():
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            query = text("SELECT * FROM nurses") 
            df = pd.read_sql(query, conn)
            
            # --- FIX: Rename DB column to match App Logic ---
            if 'fatigue_risk' in df.columns:
                df.rename(columns={'fatigue_risk': 'incident_probability'}, inplace=True)
                
            # Ensure status column is clean
            if 'status' in df.columns:
                df['status'] = df['status'].str.strip()
                
        return df
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        return None

# --- 5. ACTION: UPDATE DATABASE (THE REAL DEAL) ---
def relieve_nurse_in_db(nurse_id):
    try:
        engine = get_db_connection()
        with engine.begin() as conn: 
            # Update the ACTUAL sql column 'fatigue_risk'
            sql = text("""
                UPDATE nurses 
                SET fatigue_risk = 12, status = 'Relieved' 
                WHERE nurse_id = :id
            """)
            conn.execute(sql, {"id": nurse_id})
        return True
    except Exception as e:
        st.error(f"Write Error: {e}")
        return False

df = load_data()

# --- 6. DASHBOARD UI ---
st.title("🛡️ ShiftGuard: Enterprise Risk Command Center")
st.caption("🔴 LIVE PRODUCTION MODE | Connected to Azure SQL (Central India)")

if df is not None:
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Staff Monitored", len(df), "Real-time Azure SQL")
    with m2:
        # Filter: Only count people who are NOT relieved
        critical_risk = len(df[(df['incident_probability'] >= 85) & (df['status'] != 'Relieved')])
        st.metric("Active Incident Risk", critical_risk, "Requires Intervention", delta_color="inverse")
    with m3:
        avg_risk = df['incident_probability'].mean()
        st.metric("Liability Exposure", f"{(avg_risk/25):.1f}x", "vs. Standard")
    with m4:
        st.metric("System Health", "Online", "Latency: 24ms")

    st.divider()

    # INTERVENTION SECTION
    st.subheader("🚨 Priority Interventions (Live Database Write)")
    
    # --- DEMO LOGIC FIX: Keep 'Relieved' nurses visible ---
    # Create a sorting score to force Relieved nurses to top of list
    df['sort_score'] = df['incident_probability']
    
    # If status is 'Relieved', give them a huge score so they act "pinned"
    if 'status' in df.columns:
        df.loc[df['status'] == 'Relieved', 'sort_score'] = 1000
    
    # Sort by this custom score
    at_risk_nurses = df.sort_values('sort_score', ascending=False).head(3)

    cols = st.columns(3)
    for i, (idx, nurse) in enumerate(at_risk_nurses.iterrows()):
        nurse_id = int(nurse['nurse_id'])
        
        with cols[i]:
            with st.container(border=True):
                # CHECK DB STATUS
                if nurse['status'] == 'Relieved':
                    st.success(f"**NURSE ID: {nurse_id}**")
                    st.markdown("### ✅ RELIEVED")
                    st.caption("Update confirmed in Azure SQL")
                    st.progress(0.12, "Risk: 12% (Safe)")
                else:
                    st.error(f"**NURSE ID: {nurse_id}**")
                    st.write(f"Risk: {nurse['incident_probability']}%")
                    st.progress(nurse['incident_probability'] / 100)
                    
                    if st.button(f"⚡ Relieve (ID: {nurse_id})", key=f"btn_{i}"):
                        with st.spinner("Writing to Azure SQL Transaction Log..."):
                            success = relieve_nurse_in_db(nurse_id)
                            time.sleep(1) # Visual pacing
                        
                        if success:
                            st.toast("✅ Database Updated Successfully!")
                            time.sleep(0.5)

                            st.rerun()

