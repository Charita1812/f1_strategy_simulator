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
# HELPER FUNCTIONS FOR LOADING ZIPPED FILES
# ============================================================================

def load_pickle_from_zip(zip_path, filename):
    """
    Load a pickle file from a ZIP archive.
    Uses pickle.loads() to avoid file object compatibility issues.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(filename) as f:
                # Read all bytes first, then unpickle
                data = f.read()
                return pickle.loads(data)
    except Exception as e:
        st.error(f"Failed to load {filename} from {zip_path}: {e}")
        raise


def load_csv_from_zip(zip_path, filename):
    """
    Load a CSV file from a ZIP archive.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(filename) as f:
                return pd.read_csv(f)
    except Exception as e:
        st.error(f"Failed to load {filename} from {zip_path}: {e}")
        raise


# ============================================================================
# PATH SETUP AND FILE LOADING
# ============================================================================

# Paths - adjust if you changed structure
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Add src to sys.path for imports
sys.path.append(str(ROOT / "src"))

# Load models from ZIP files
@st.cache_resource
def load_all_models():
    """Load all ML models. Cached to avoid reloading on every interaction."""
    models = {}
    
    # Load zipped models
    try:
        models['safety_car'] = load_pickle_from_zip(
            MODELS_DIR / "safety_car_model.zip", 
            "safety_car_model.pkl"
        )
        models['undercut'] = load_pickle_from_zip(
            MODELS_DIR / "undercut_model.zip", 
            "undercut_model.pkl"
        )
    except Exception as e:
        st.error(f"Error loading zipped models: {e}")
        st.stop()
    
    # Load regular pickle files (if they exist as non-zipped)
    try:
        lap_time_path = MODELS_DIR / "lap_time_predictor.pkl"
        tyre_wear_path = MODELS_DIR / "tyre_wear_predictor.pkl"
        pit_window_path = MODELS_DIR / "pit_window_model.pkl"
        
        if lap_time_path.exists():
            with open(lap_time_path, "rb") as f:
                models['lap_time'] = pickle.load(f)
        
        if tyre_wear_path.exists():
            with open(tyre_wear_path, "rb") as f:
                models['tyre_wear'] = pickle.load(f)
        
        if pit_window_path.exists():
            with open(pit_window_path, "rb") as f:
                models['pit_window'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading regular pickle models: {e}")
        st.stop()
    
    return models


@st.cache_data
def load_master_data():
    """Load master lap-by-lap data. Cached to avoid reloading."""
    try:
        # Try loading from ZIP first
        zip_path = PROCESSED / "master_lap_by_lap.zip"
        if zip_path.exists():
            return load_csv_from_zip(zip_path, "master_lap_by_lap.csv")
        
        # Fallback to regular CSV
        csv_path = PROCESSED / "master_lap_by_lap.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        st.error(f"Data file not found at {PROCESSED}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading master data: {e}")
        st.stop()


# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="F1 Strategy Simulator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL F1 CSS STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Roboto+Mono:wght@400;500;700&display=swap');
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes scaleUp {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes borderDraw {
        0% { width: 0; }
        100% { width: 100%; }
    }
    @keyframes bounceSubtle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(to bottom, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Title styling */
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
    
    /* Subheaders */
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
    
    /* Sidebar */
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
    
    /* Select boxes */
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
    
    /* Button styling */
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
        width: 100%;
        margin-top: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(225, 6, 0, 0.4);
        background: linear-gradient(135deg, #ff0700 0%, #e10600 100%) !important;
    }
    
    /* DataFrames */
    .stDataFrame > div {
        border: 2px solid #e10600 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.15);
    }
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #e10600 0%, #c40500 100%) !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stDataFrame tbody tr:hover {
        background: linear-gradient(90deg, #fff5f5 0%, #ffffff 100%) !important;
        transform: translateX(5px);
        box-shadow: -3px 0 0 0 #e10600;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        animation: bounceSubtle 2s ease-in-out infinite;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 8px 25px rgba(225, 6, 0, 0.2);
        border-color: #e10600;
        animation: none;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e10600 !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Text */
    p {
        color: #2c2c2c !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }
    strong {
        color: #e10600 !important;
        font-weight: 700 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

st.title("F1 Race Strategy Simulator")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1rem; margin-top: -10px;'>"
    "Professional Monte Carlo Strategy Analysis</p>", 
    unsafe_allow_html=True
)

# Load data and models with error handling
with st.spinner("Loading data and models..."):
    master = load_master_data()
    models = load_all_models()

# Import strategy engine
try:
    from simulator.strategy_engine import RealWorldStrategyEngine
except ImportError as e:
    st.error(f"Failed to import strategy engine: {e}")
    st.stop()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# Season selection
seasons = sorted(master['year'].unique().tolist())
season = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)

# Race selection
races = master[master['year'] == season][['raceId','name']].drop_duplicates().sort_values('name')
race_map = dict(zip(races['name'].tolist(), races['raceId'].tolist()))
race_name = st.sidebar.selectbox("Race", races['name'].tolist())
race_id = int(race_map[race_name])

# Driver selection
drivers_in_race = master[master['raceId'] == race_id][['driverId','forename','surname']].drop_duplicates()
drivers_in_race['driver_name'] = drivers_in_race['forename'] + " " + drivers_in_race['surname']
driver_map = dict(zip(drivers_in_race['driver_name'], drivers_in_race['driverId']))
driver_name = st.sidebar.selectbox("Driver", drivers_in_race['driver_name'].tolist())
driver_id = int(driver_map[driver_name])

# Strategy parameters
st.sidebar.markdown("---")
st.sidebar.markdown("**Strategy Parameters**")
current_lap = int(master[(master['raceId']==race_id) & (master['driverId']==driver_id)]['lap'].iloc[0])
total_laps = int(master[master['raceId']==race_id]['lap'].max())
max_stops = st.sidebar.slider("Maximum pit stops", 1, 3, 2)
consider_sc = st.sidebar.checkbox("Include safety car analysis", value=True)
run_button = st.sidebar.button("🚀 Run Simulation")

# ============================================================================
# INSTANTIATE ENGINE (CACHED)
# ============================================================================

@st.cache_resource
def get_engine():
    """Initialize strategy engine. Cached to avoid reloading models."""
    return RealWorldStrategyEngine()

engine = get_engine()

# ============================================================================
# RACE OVERVIEW SECTION
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown(f"**📍 Race:** {race_name}")
with col2:
    st.markdown(f"**👤 Driver:** {driver_name}")
with col3:
    st.markdown(f"**🆔** {race_id}")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Starting Lap:** {current_lap}")
with col2:
    st.markdown(f"**Total Laps:** {total_laps}")

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
    
    # ========================================================================
    # OPTIMAL STRATEGY DISPLAY
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🏆 Optimal Strategy")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pit Laps", str(res['best_strategy']))
    with col2:
        st.metric("Number of Stops", res['stops'])
    with col3:
        st.metric("Expected Time", f"{res['expected_total_time_s']:.2f}s")

    # ========================================================================
    # STRATEGY COMPARISON TABLE
    # ========================================================================
    
    df_cand = pd.DataFrame([{
        "Strategy": str(c['pit_strategy']),
        "Stops": c['stops'],
        "Expected Time (s)": round(c['expected_total_time_s'], 2)
    } for c in res['all_candidates']])
    
    st.markdown("---")
    st.markdown("## 📊 Strategy Comparison")
    st.dataframe(
        df_cand.sort_values('Expected Time (s)').reset_index(drop=True),
        use_container_width=True,
        height=300
    )

    # ========================================================================
    # SINGLE-STOP STRATEGY ANALYSIS
    # ========================================================================
    
    one_stop = [c for c in res['all_candidates'] if c['stops']==1]
    if one_stop:
        st.markdown("---")
        st.markdown("## 📈 Single-Stop Strategy Analysis")
        
        pit_laps = [c['pit_strategy'][0] for c in one_stop]
        times = [c['expected_total_time_s'] for c in one_stop]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot line segments
        for i in range(len(pit_laps)-1):
            ax.plot(pit_laps[i:i+2], times[i:i+2], linewidth=3.5,
                    color='#e10600', alpha=0.8, zorder=2)
        
        # Add markers
        ax.scatter(pit_laps, times, s=120, color='#ffffff', edgecolor='#e10600', 
                   linewidth=3, zorder=3, alpha=0.95)
        ax.scatter(pit_laps, times, s=200, color='#e10600', alpha=0.2, zorder=1)
        
        # Highlight optimal strategy
        min_idx = times.index(min(times))
        ax.scatter([pit_laps[min_idx]], [times[min_idx]], s=250, color='#FFD700', 
                   edgecolor='#e10600', linewidth=4, zorder=4, marker='*', label='Optimal')
        
        # Add annotations
        for i, (lap, time) in enumerate(zip(pit_laps, times)):
            ax.annotate(f'{time:.1f}s', 
                       xy=(lap, time), 
                       xytext=(0, 10 if i % 2 == 0 else -15),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       fontweight='600',
                       color='#e10600',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='#e10600', linewidth=1.5, alpha=0.9))
        
        ax.set_xlabel("Pit Stop Lap", fontsize=13, fontweight='600', color='#2c2c2c')
        ax.set_ylabel("Expected Race Time (seconds)", fontsize=13, fontweight='600', color='#2c2c2c')
        ax.set_title("Single-Stop Strategy Performance", fontsize=15, fontweight='700', 
                     color='#1a1a1a', pad=20)
        ax.grid(True, alpha=0.3, color='#cccccc', linestyle='-', linewidth=0.7)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#e10600')
            spine.set_linewidth(2)
        
        st.pyplot(fig)

    # ========================================================================
    # HISTORICAL LAP TIME ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.markdown("## 🏁 Historical Lap Time Analysis")
    df_driver = master[(master['raceId']==race_id) & (master['driverId']==driver_id)].sort_values('lap')
    
    if not df_driver.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        
        # Area fill
        ax2.fill_between(df_driver['lap'], df_driver['lap_time_s'], alpha=0.3, color='#ff6b6b')
        
        # Main line
        ax2.plot(df_driver['lap'], df_driver['lap_time_s'], linewidth=3, 
                 color='#e10600', alpha=0.9, zorder=2)
        
        # Markers
        ax2.scatter(df_driver['lap'], df_driver['lap_time_s'], s=60, 
                    color='#ffffff', edgecolor='#e10600', linewidth=2, zorder=3, alpha=0.8)
        
        # Highlight fastest lap
        fastest_idx = df_driver['lap_time_s'].idxmin()
        fastest_lap = df_driver.loc[fastest_idx, 'lap']
        fastest_time = df_driver.loc[fastest_idx, 'lap_time_s']
        ax2.scatter([fastest_lap], [fastest_time], s=300, color='#FFD700', 
                    edgecolor='#e10600', linewidth=3, zorder=4, marker='*', label='Fastest Lap')
        
        # Annotation
        ax2.annotate(f'Fastest: {fastest_time:.2f}s\nLap {fastest_lap}', 
                    xy=(fastest_lap, fastest_time), 
                    xytext=(15, -30),
                    textcoords='offset points',
                    ha='left',
                    fontsize=10,
                    fontweight='700',
                    color='#e10600',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFD700', 
                             edgecolor='#e10600', linewidth=2, alpha=0.95),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                   color='#e10600', lw=2))
        
        ax2.set_xlabel("Lap Number", fontsize=13, fontweight='600', color='#2c2c2c')
        ax2.set_ylabel("Lap Time (seconds)", fontsize=13, fontweight='600', color='#2c2c2c')
        ax2.set_title(f"{driver_name} - Lap Time Evolution", fontsize=15, fontweight='700', 
                     color='#1a1a1a', pad=20)
        ax2.grid(True, alpha=0.3, color='#cccccc', linestyle='-', linewidth=0.7)
        ax2.set_facecolor('#f8f9fa')
        fig2.patch.set_facecolor('white')
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#e10600')
            spine.set_linewidth(2)
        
        st.pyplot(fig2)
    else:
        st.info("ℹ️ No historical lap data available for this driver and race selection.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Built with Streamlit • Powered by Machine Learning • Formula 1 Strategy Optimization")

