import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# ============================================================================ 
# PATH SETUP
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
ENGINE_PATH = ROOT / "src" / "simulator" / "strategy_engine.py"

# Ensure src is importable
sys.path.append(str(ROOT / "src"))
from simulator.strategy_engine import RealWorldStrategyEngine

# ============================================================================ 
# STREAMLIT CONFIG
# ============================================================================

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide", initial_sidebar_state="expanded")
# 🏎️ PROFESSIONAL & CLASSIC F1 CSS 🏎️
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Roboto+Mono:wght@400;500;700&display=swap');
    
    /* Smooth fade in */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Slide from left */
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Slide from right */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Scale up animation */
    @keyframes scaleUp {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Subtle shine effect */
    @keyframes shine {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    /* Gentle pulse */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Border draw animation */
    @keyframes borderDraw {
        0% { width: 0; }
        100% { width: 100%; }
    }
    
    /* Dash animation for racing lines */
    @keyframes dash {
        to { stroke-dashoffset: 0; }
    }
    
    /* Bounce subtle */
    @keyframes bounceSubtle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Main app background - clean white/light gray */
    .stApp {
        background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Title styling - professional F1 look */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        color: #e10600 !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-align: center;
        padding: 30px 0 10px 0;
        font-size: 2.8rem !important;
        animation: scaleUp 0.8s ease-out;
        position: relative;
        margin-bottom: 10px !important;
    }
    
    h1::before {
        content: '🏎️';
        position: absolute;
        left: 20px;
        animation: slideInLeft 1s ease-out;
    }
    
    h1::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 4px;
        background: linear-gradient(90deg, transparent, #e10600, transparent);
        animation: borderDraw 1.2s ease-out;
    }
    
    /* Subheader styling */
    h2 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        font-size: 1.8rem !important;
        margin-top: 30px !important;
        padding-bottom: 12px !important;
        border-bottom: 3px solid #e10600;
        animation: slideInLeft 0.7s ease-out;
    }
    
    h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #2c2c2c !important;
        font-size: 1.4rem !important;
        margin-top: 25px !important;
        padding-left: 15px !important;
        border-left: 4px solid #e10600;
        animation: slideInLeft 0.6s ease-out;
    }
    
    /* Sidebar styling - clean and professional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%) !important;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #e10600 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        padding: 15px 0;
        border-bottom: 2px solid #e10600;
    }
    
    [data-testid="stSidebar"] label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #2c2c2c !important;
        font-size: 0.95rem !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stSlider {
        animation: fadeIn 0.5s ease-out;
    }
    
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #d0d0d0 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #e10600 !important;
        box-shadow: 0 2px 8px rgba(225, 6, 0, 0.15);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #e10600 !important;
        box-shadow: 0 0 0 3px rgba(225, 6, 0, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #e10600 !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: #ffffff !important;
        border: 3px solid #e10600 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    
    /* Checkbox */
    .stCheckbox {
        animation: fadeIn 0.6s ease-out;
    }
    
    .stCheckbox label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #2c2c2c !important;
    }
    
    /* Button styling - professional F1 red */
    .stButton > button {
        background: linear-gradient(135deg, #e10600 0%, #c40500 100%) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 12px 40px !important;
        border: none !important;
        border-radius: 8px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(225, 6, 0, 0.3);
        cursor: pointer;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(225, 6, 0, 0.4);
        background: linear-gradient(135deg, #ff0700 0%, #e10600 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(225, 6, 0, 0.3);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        animation: fadeIn 0.5s ease-out;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Success message */
    [data-testid="stNotificationContentSuccess"] {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-left: 5px solid #28a745 !important;
        color: #155724 !important;
        font-weight: 600 !important;
    }
    
    /* Error message */
    [data-testid="stNotificationContentError"] {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border-left: 5px solid #dc3545 !important;
        color: #721c24 !important;
        font-weight: 600 !important;
    }
    
    /* DataFrames - clean professional look */
    .stDataFrame {
        animation: scaleUp 0.8s ease-out;
    }
    
    .stDataFrame > div {
        border: 2px solid #e10600 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.15);
        background: #ffffff !important;
    }
    
    /* Table headers */
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #e10600 0%, #c40500 100%) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 14px 12px !important;
        font-size: 0.9rem !important;
    }
    
    /* Table rows */
    .stDataFrame tbody tr {
        background: #ffffff !important;
        transition: all 0.3s ease;
        border-bottom: 1px solid #f0f0f0 !important;
    }
    
    .stDataFrame tbody tr:hover {
        background: linear-gradient(90deg, #fff5f5 0%, #ffffff 100%) !important;
        transform: translateX(5px);
        box-shadow: -3px 0 0 0 #e10600;
    }
    
    .stDataFrame tbody tr td {
        font-family: 'Roboto Mono', monospace !important;
        color: #2c2c2c !important;
        padding: 12px !important;
    }
    
    /* Metric boxes - professional cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        animation: bounceSubtle 2s ease-in-out infinite;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(225, 6, 0, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    [data-testid="stMetric"]:hover::before {
        left: 100%;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 8px 25px rgba(225, 6, 0, 0.2);
        border-color: #e10600;
        animation: none;
    }
    
    [data-testid="stMetric"] label {
        color: #666666 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e10600 !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #e10600 !important;
    }
    
    /* Plots */
    .stPlotlyChart, .element-container:has(img) {
        animation: scaleUp 1s ease-out;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.12);
        transition: all 0.4s ease;
        border: 2px solid #f0f0f0;
    }
    
    .stPlotlyChart:hover, .element-container:has(img):hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(225, 6, 0, 0.25);
        border-color: #e10600;
    }
    
    /* Markdown content */
    .stMarkdown {
        animation: fadeIn 0.5s ease-out;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Paragraph text */
    p {
        color: #2c2c2c !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }
    
    /* Strong/bold text */
    strong {
        color: #e10600 !important;
        font-weight: 700 !important;
    }
    
    /* Code blocks */
    code {
        background: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        color: #e10600 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'Roboto Mono', monospace !important;
        font-weight: 500 !important;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 40px 0;
    }
    
    /* Caption text */
    .stCaption {
        color: #666666 !important;
        font-family: 'Inter', sans-serif !important;
        font-style: italic;
        text-align: center;
    }
    
    /* Info icon styling */
    .stInfo {
        background: linear-gradient(90deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border-left: 5px solid #17a2b8 !important;
        border-radius: 8px !important;
    }
    
    /* Column containers */
    [data-testid="column"] {
        animation: slideInLeft 0.6s ease-out;
    }
    
    [data-testid="column"]:nth-child(2) {
        animation: fadeIn 0.7s ease-out;
    }
    
    [data-testid="column"]:nth-child(3) {
        animation: slideInRight 0.6s ease-out;
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.2s ease;
    }
    
    /* Remove any default padding that might look cramped */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("F1 Race Strategy Simulator")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1rem; margin-top: -10px;'>"
    "Professional Monte Carlo Strategy Analysis</p>", 
    unsafe_allow_html=True
)

# ============================================================================ 
# UNZIP MODELS (if zipped)
# ============================================================================

def unzip_models():
    """Extract zipped models if they exist."""
    for zip_file in MODELS_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(MODELS_DIR)

unzip_models()  # Run this at startup

# ============================================================================ 
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_all_models():
    """Load all models (.pkl) from models directory."""
    models = {}
    for model_file in MODELS_DIR.glob("*.pkl"):
        try:
            models[model_file.stem] = joblib.load(model_file)
        except Exception as e:
            st.error(f"Failed to load model {model_file.name}: {e}")
            st.stop()
    return models

models = load_all_models()

# ============================================================================ 
# LOAD MASTER DATA
# ============================================================================

master_zip = PROCESSED / "master_lap_by_lap.zip"
master_csv = PROCESSED / "master_lap_by_lap.csv"

if master_zip.exists():
    with zipfile.ZipFile(master_zip, 'r') as z:
        with z.open("master_lap_by_lap.csv") as f:
            master = pd.read_csv(f)
elif master_csv.exists():
    master = pd.read_csv(master_csv)
else:
    st.error("Processed master_lap_by_lap data not found.")
    st.stop()

# ============================================================================ 
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")
seasons = sorted(master['year'].unique().tolist())
season = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)

races = master[master['year'] == season][['raceId','name']].drop_duplicates().sort_values('name')
race_map = dict(zip(races['name'], races['raceId']))
race_name = st.sidebar.selectbox("Race", races['name'].tolist())
race_id = int(race_map[race_name])

drivers_in_race = master[master['raceId'] == race_id][['driverId','forename','surname']].drop_duplicates()
drivers_in_race['driver_name'] = drivers_in_race['forename'] + " " + drivers_in_race['surname']
driver_map = dict(zip(drivers_in_race['driver_name'], drivers_in_race['driverId']))
driver_name = st.sidebar.selectbox("Driver", drivers_in_race['driver_name'].tolist())
driver_id = int(driver_map[driver_name])

st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy Parameters**")
current_lap = int(master[(master['raceId']==race_id) & (master['driverId']==driver_id)]['lap'].iloc[0])
total_laps = int(master[master['raceId']==race_id]['lap'].max())
max_stops = st.sidebar.slider("Maximum pit stops", 1, 3, 2)
consider_sc = st.sidebar.checkbox("Include safety car analysis", value=True)
run_button = st.sidebar.button("🚀 Run Simulation")

# ============================================================================ 
# INITIALIZE ENGINE
# ============================================================================

@st.cache_resource
def get_engine():
    return RealWorldStrategyEngine()

engine = get_engine()

# ============================================================================ 
# RACE OVERVIEW
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])
with col1: st.markdown(f"**📍 Race:** {race_name}")
with col2: st.markdown(f"**👤 Driver:** {driver_name}")
with col3: st.markdown(f"**🆔** {race_id}")

col1, col2 = st.columns(2)
with col1: st.markdown(f"**Starting Lap:** {current_lap}")
with col2: st.markdown(f"**Total Laps:** {total_laps}")

# ============================================================================ 
# RUN SIMULATION
# ============================================================================

if run_button:
    with st.spinner("⏳ Running Monte Carlo strategy simulation..."):
        try:
            res = engine.simulate_strategy(
                race_id=race_id,
                driver_id=driver_id,
                current_lap=current_lap,
                total_laps=total_laps,
                max_stop=max_stops,
                consider_sc=consider_sc
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    st.success("✅ Simulation Complete")

    # --- Optimal Strategy
    st.markdown("---")
    st.markdown("## 🏆 Optimal Strategy")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Pit Laps", str(res['best_strategy']))
    with col2: st.metric("Number of Stops", res['stops'])
    with col3: st.metric("Expected Time", f"{res['expected_total_time_s']:.2f}s")

    # --- Strategy Comparison
    df_cand = pd.DataFrame([{
        "Strategy": str(c['pit_strategy']),
        "Stops": c['stops'],
        "Expected Time (s)": round(c['expected_total_time_s'], 2)
    } for c in res['all_candidates']])
    st.markdown("---")
    st.markdown("## 📊 Strategy Comparison")
    st.dataframe(df_cand.sort_values('Expected Time (s)').reset_index(drop=True), width="stretch", height=300)

# ============================================================================ 
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Built with Streamlit • Powered by Machine Learning • Formula 1 Strategy Optimization")
