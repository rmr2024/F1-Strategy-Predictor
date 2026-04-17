import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import enable_caching, load_single_race, load_season_data, get_available_races, get_circuit_info, get_track_coordinates
from src.feature_engineering import engineer_features
from src.train_model import train_model, get_cached_model, predict_pit
from src.utils import get_driver_color, format_time

st.set_page_config(page_title="F1 Strategy Predictor", layout="wide", page_icon="🏎️")

os.environ["FASTF1_CACHE_DIR"] = "/tmp/fastf1_cache"

COLORS = {
    'bg': '#0E1117',
    'card': '#1E1E1E',
    'card_light': '#262730',
    'card_hover': '#2D2D2D',
    'text': '#FAFAFA',
    'text_muted': '#A0A0A0',
    'accent_red': '#E10600',
    'accent_blue': '#3671C6',
    'accent_green': '#00D2BE',
    'accent_yellow': '#FFD700',
    'grid': '#333333',
    'soft': '#FF3333',
    'medium': '#FFD700',
    'hard': '#FFFFFF',
    'good': '#00D2BE',
    'poor': '#E10600',
    'risky': '#FFD700',
    'gradient_start': '#1E1E1E',
    'gradient_end': '#0E1117'
}

TYRE_COLORS = {
    'SOFT': '#FF3333',
    'MEDIUM': '#FFD700',
    'HARD': '#FFFFFF',
    'INTER': '#00B140',
    'WET': '#0067AD'
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        background-color: {COLORS['bg']};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Header Styling */
    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {COLORS['accent_red']} 0%, {COLORS['accent_blue']} 50%, {COLORS['accent_green']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-out, titleGlow 3s ease-in-out infinite alternate;
        text-shadow: 0 0 40px rgba(225, 6, 0, 0.3);
    }}
    
    @keyframes titleGlow {{
        from {{ text-shadow: 0 0 40px rgba(225, 6, 0, 0.3); }}
        to {{ text-shadow: 0 0 60px rgba(54, 113, 198, 0.5); }}
    }}
    
    .subtitle {{
        font-size: 1.1rem;
        color: {COLORS['text_muted']};
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['card']} 0%, {COLORS['bg']} 100%);
        border-right: 1px solid {COLORS['grid']};
        backdrop-filter: blur(10px);
    }}
    
    .sidebar-header {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid {COLORS['grid']};
        margin-bottom: 1rem;
    }}
    
    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(30, 30, 30, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .glass-card:hover {{
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
        border-color: rgba(54, 113, 198, 0.3);
    }}
    
    /* Dashboard Card */
    .dashboard-card {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 24px;
        border: 1px solid {COLORS['grid']};
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    
    .dashboard-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid {COLORS['grid']};
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['accent_blue']};
        transform: scale(1.03);
        box-shadow: 0 8px 24px rgba(54, 113, 198, 0.2);
    }}
    
    /* Beginner Box */
    .beginner-box {{
        background: linear-gradient(135deg, rgba(54, 113, 198, 0.15) 0%, rgba(30, 30, 30, 0.9) 100%);
        border-left: 4px solid {COLORS['accent_blue']};
        padding: 24px;
        border-radius: 16px;
        margin: 20px 0;
        animation: slideInLeft 0.5s ease-out, fadeIn 0.5s ease-out;
        border: 1px solid rgba(54, 113, 198, 0.2);
    }}
    
    /* Explanation Card */
    .explanation-card {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
        border-left: 4px solid {COLORS['accent_green']};
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {COLORS['card']};
        padding: 8px;
        border-radius: 16px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 12px 12px 0 0;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['accent_red']}, #FF4444);
        color: white;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {COLORS['text']};
        margin: 2rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 14px;
        letter-spacing: 0.5px;
    }}
    
    .section-header::before {{
        content: '';
        width: 5px;
        height: 28px;
        background: linear-gradient(180deg, {COLORS['accent_red']}, {COLORS['accent_blue']});
        border-radius: 3px;
        box-shadow: 0 0 10px rgba(225, 6, 0, 0.5);
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideInLeft {{
        from {{ transform: translateX(-30px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    @keyframes fadeInUp {{
        from {{ transform: translateY(30px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    @keyframes slideUp {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    
    @keyframes scaleIn {{
        from {{ transform: scale(0.9); opacity: 0; }}
        to {{ transform: scale(1); opacity: 1; }}
    }}
    
    .loading-pulse {{
        animation: pulse 2s infinite;
    }}
    
    .stagger-1 {{ animation-delay: 0.1s; }}
    .stagger-2 {{ animation-delay: 0.2s; }}
    .stagger-3 {{ animation-delay: 0.3s; }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['accent_red']}, {COLORS['accent_blue']});
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {COLORS['accent_blue']}, {COLORS['accent_green']});
    }}
    
    /* Metrics */
    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        padding: 18px;
        border-radius: 14px;
        border: 1px solid {COLORS['grid']};
        transition: all 0.3s ease;
    }}
    
    div[data-testid="stMetric"]:hover {{
        border-color: {COLORS['accent_blue']};
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(54, 113, 198, 0.15);
    }}
    
    /* Input styling */
    .stSelectbox > div > div {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['grid']};
        border-radius: 12px;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {COLORS['accent_blue']};
    }}
    
    /* Spacing */
    .spacer-sm {{ height: 10px; }}
    .spacer-md {{ height: 25px; }}
    .spacer-lg {{ height: 50px; }}
    .spacer-xl {{ height: 80px; }}
    
    /* Chart container */
    .chart-container {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 20px;
        padding: 24px;
        margin: 20px 0;
        border: 1px solid {COLORS['grid']};
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: scaleIn 0.4s ease-out;
    }}
    
    /* Strategy block */
    .strategy-block {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 2px solid;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .strategy-block:hover {{
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['accent_red']}, #FF4444);
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(225, 6, 0, 0.4);
    }}
    
    /* Info/Warning boxes */
    .stInfo, .stWarning {{
        background: linear-gradient(135deg, rgba(54, 113, 198, 0.1), {COLORS['card']});
        border-radius: 12px;
        border: 1px solid rgba(54, 113, 198, 0.3);
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        border-radius: 16px;
        border: 1px solid {COLORS['grid']};
    }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_available_races_cached(year):
    enable_caching()
    return get_available_races(year)


@st.cache_data(ttl=3600)
def load_race_data_cached(year, gp):
    enable_caching()
    return load_single_race(year, gp)


@st.cache_data(ttl=3600, show_spinner="Loading training data...")
def load_training_data_cached():
    enable_caching()
    return load_season_data(years=[2022], max_races_per_year=2)


@st.cache_resource(show_spinner="Training prediction model...")
def get_trained_model():
    try:
        training_df = load_training_data_cached()
        if training_df.empty:
            return None, None
        model, config = get_cached_model(training_df)
        return model, config
    except Exception as e:
        st.warning(f"Could not train model: {e}")
        return None, None


def get_model_predictions(df, threshold, model, config):
    try:
        if model is None:
            return df
        df_pred = predict_pit(model, df.copy(), config)
        if threshold is not None:
            df_pred['PredictedPit'] = (df_pred['PitProbability'] >= threshold).astype(int)
        return df_pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return df


def get_tyre_color(compound):
    return TYRE_COLORS.get(str(compound).upper(), '#FFFFFF')


def create_finishing_position_chart(df):
    if 'Position' not in df.columns:
        return None
    
    df_agg = df.groupby('Driver').agg({
        'Position': 'last',
        'LapTimeSeconds': 'mean'
    }).reset_index().sort_values('Position')
    
    colors = [COLORS['good'] if p <= 5 else COLORS['risky'] if p <= 10 else COLORS['poor'] 
              for p in df_agg['Position']]
    
    fig = px.bar(
        df_agg, 
        x='Driver', 
        y='Position',
        color='Position',
        color_continuous_scale=[COLORS['good'], COLORS['risky'], COLORS['poor']],
        title="Predicted Finishing Positions",
        text='Position'
    )
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title="Driver", gridcolor=COLORS['grid']),
        yaxis=dict(title="Position", gridcolor=COLORS['grid'], range=[20, 1]),
        showlegend=False,
        height=400
    )
    fig.update_traces(textposition='outside')
    return fig


def create_position_timeline(df):
    if 'Position' not in df.columns or 'LapNumber' not in df.columns:
        return None
    
    drivers = df['Driver'].unique()[:5]
    fig = go.Figure()
    
    for driver in drivers:
        driver_df = df[df['Driver'] == driver].sort_values('LapNumber')
        fig.add_trace(go.Scatter(
            x=driver_df['LapNumber'],
            y=driver_df['Position'],
            mode='lines+markers',
            name=driver,
            line=dict(color=get_driver_color(driver), width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid']),
        yaxis=dict(title="Position", gridcolor=COLORS['grid'], range=[20, 1], 
                   autorange=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        title="Position Changes Over Race"
    )
    return fig


def create_win_probability_chart(df):
    if 'PitProbability' not in df.columns:
        return None
    
    driver_probs = df.groupby('Driver').agg({
        'PitProbability': 'mean',
        'LapTimeSeconds': 'mean'
    }).reset_index()
    
    driver_probs['WinChance'] = 1 - driver_probs['PitProbability']
    driver_probs = driver_probs.sort_values('WinChance', ascending=False).head(10)
    
    colors = [COLORS['good'] if p > 0.5 else COLORS['poor'] for p in driver_probs['WinChance']]
    
    fig = px.bar(
        driver_probs,
        x='Driver',
        y='WinChance',
        title="Win Probability (Based on Pit Strategy)",
        color='WinChance',
        color_continuous_scale=[COLORS['poor'], COLORS['good']]
    )
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title="Driver", gridcolor=COLORS['grid']),
        yaxis=dict(title="Win Probability", gridcolor=COLORS['grid'], 
                   range=[0, 1], tickformat=".0%"),
        showlegend=False,
        height=350
    )
    return fig


def create_lap_time_chart(df, driver):
    driver_df = df[df['Driver'] == driver].copy()
    if driver_df.empty:
        return None
    
    driver_df = driver_df.sort_values('LapNumber')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=driver_df['LapNumber'],
        y=driver_df['LapTimeSeconds'],
        mode='lines+markers',
        name='Lap Time',
        line=dict(color=get_driver_color(driver), width=2),
        marker=dict(size=4)
    ))
    
    actual_pits = driver_df[driver_df['PitInTime'].notna()]['LapNumber'].tolist()
    if actual_pits:
        fig.add_trace(go.Scatter(
            x=actual_pits,
            y=driver_df[driver_df['LapNumber'].isin(actual_pits)]['LapTimeSeconds'],
            mode='markers',
            name='Actual Tyre Change',
            marker=dict(color=COLORS['accent_red'], size=12, symbol='x')
        ))
    
    pred_pits = driver_df[driver_df['PredictedPit'] == 1]['LapNumber'].tolist()
    if pred_pits:
        fig.add_trace(go.Scatter(
            x=pred_pits,
            y=driver_df[driver_df['LapNumber'].isin(pred_pits)]['LapTimeSeconds'],
            mode='markers',
            name='Predicted Tyre Change',
            marker=dict(color=COLORS['accent_green'], size=10, symbol='circle-open', 
                       line=dict(width=2))
        ))
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family="Roboto, sans-serif"),
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid'], showgrid=True),
        yaxis=dict(title="Lap Time (seconds)", gridcolor=COLORS['grid'], showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    return fig


def create_pit_probability_chart(df, driver):
    driver_df = df[df['Driver'] == driver].copy()
    if driver_df.empty or 'PitProbability' not in driver_df.columns:
        return None
    
    driver_df = driver_df.sort_values('LapNumber')
    
    fig = go.Scatter(
        x=driver_df['LapNumber'],
        y=driver_df['PitProbability'],
        mode='lines',
        name='Tyre Change Probability',
        fill='tozeroy',
        line=dict(color=COLORS['accent_blue'], width=2),
        fillcolor="rgba(54, 113, 198, 0.2)"
    )
    
    fig = go.Figure(fig)
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid']),
        yaxis=dict(title="Probability", range=[0, 1], gridcolor=COLORS['grid']),
        margin=dict(l=40, r=40, t=20, b=40),
        height=250
    )
    return fig


def create_strategy_timeline(df, driver):
    driver_df = df[df['Driver'] == driver].copy()
    if driver_df.empty or 'Stint' not in driver_df.columns:
        return None
    
    driver_df = driver_df.sort_values('LapNumber')
    
    stints = driver_df.groupby('Stint').agg({
        'Compound': 'first',
        'LapNumber': ['min', 'max']
    }).reset_index()
    stints.columns = ['Stint', 'Compound', 'StartLap', 'EndLap']
    
    fig = go.Figure()
    
    for _, row in stints.iterrows():
        compound = str(row['Compound']).upper()
        color = get_tyre_color(compound)
        
        fig.add_trace(go.Scatter(
            x=[row['StartLap'], row['EndLap']],
            y=[1, 1],
            mode='lines',
            line=dict(color=color, width=30),
            name=f"{compound} (Laps {int(row['StartLap'])}-{int(row['EndLap'])})",
            hoverinfo='name'
        ))
    
    actual_pits = driver_df[driver_df['PitInTime'].notna()]['LapNumber'].tolist()
    if actual_pits:
        for pit_lap in actual_pits:
            fig.add_vline(x=pit_lap, line_dash="dash", line_color=COLORS['accent_red'],
                         annotation_text="Actual Pit", annotation_position="top")
    
    pred_pits = driver_df[driver_df['PredictedPit'] == 1]['LapNumber'].tolist()
    if pred_pits:
        for pit_lap in pred_pits:
            fig.add_vline(x=pit_lap, line_dash="dot", line_color=COLORS['accent_green'],
                         annotation_text="Predicted", annotation_position="bottom")
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid']),
        yaxis=dict(showticklabels=False, range=[0, 2]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        height=200,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_circuit_visualization():
    theta = np.linspace(0, 2*np.pi, 100)
    
    r_outer = 1.0
    r_inner = 0.6
    
    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)
    x_inner = r_inner * np.cos(theta)
    y_inner = r_inner * np.sin(theta)
    
    x_track = np.concatenate([x_outer, x_inner[::-1], [x_outer[0]]])
    y_track = np.concatenate([y_outer, y_inner[::-1], [y_outer[0]]])
    
    turns = [
        (0, "Start/Finish"),
        (np.pi/4, "Turn 1"),
        (np.pi/2, "Turn 2"),
        (3*np.pi/4, "Turn 3"),
        (np.pi, "Turn 4"),
        (5*np.pi/4, "Turn 5"),
        (3*np.pi/2, "Turn 6"),
        (7*np.pi/4, "Turn 7")
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_track, y=y_track,
        mode='lines',
        fill='toself',
        fillcolor='rgba(54, 113, 198, 0.1)',
        line=dict(color='#3671C6', width=4),
        name='Track'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_outer[0]], y=[y_outer[0]],
        mode='markers+text',
        marker=dict(color='#00D2BE', size=15, symbol='star'),
        text=['START'],
        textposition='top center',
        textfont=dict(color='#00D2BE', size=12),
        name='Start'
    ))
    
    for angle, name in turns[1:6]:
        x_pos = ((r_outer + r_inner) / 2) * np.cos(angle)
        y_pos = ((r_outer + r_inner) / 2) * np.sin(angle)
        fig.add_trace(go.Scatter(
            x=[x_pos], y=[y_pos],
            mode='markers+text',
            marker=dict(color='#E10600', size=10),
            text=[name],
            textposition='top center',
            textfont=dict(color='#E10600', size=10),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(color='#FFD700', size=20, symbol='circle'),
        name='Center'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


def create_race_statistics(df):
    if 'LapTimeSeconds' not in df.columns or df.empty:
        return None
    
    stats = df.groupby('Driver').agg({
        'LapTimeSeconds': ['mean', 'min', 'count']
    }).reset_index()
    stats.columns = ['Driver', 'AvgLapTime', 'FastestLap', 'TotalLaps']
    stats = stats.sort_values('AvgLapTime').head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats['Driver'],
        y=stats['AvgLapTime'],
        marker=dict(
            color=stats['AvgLapTime'],
            colorscale=[
                [0, COLORS['accent_green']],
                [0.5, COLORS['accent_yellow']],
                [1, COLORS['accent_red']]
            ],
            line=dict(color=COLORS['accent_red'], width=2)
        ),
        text=[f"{t:.2f}s" for t in stats['AvgLapTime']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11)
    ))
    
    fig.update_layout(
        font=dict(color=COLORS['text'], family="Inter, sans-serif"),
        xaxis=dict(title="", gridcolor=COLORS['grid'], tickfont=dict(color=COLORS['text_muted'])),
        yaxis=dict(title="Avg Lap Time (s)", gridcolor=COLORS['grid'], 
                   tickfont=dict(color=COLORS['text_muted']),
                   titlefont=dict(color=COLORS['text'])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_tyre_distribution(df):
    if 'Compound' not in df.columns:
        return None
    
    tyre_counts = df['Compound'].value_counts()
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels= tyre_counts.index,
        values= tyre_counts.values,
        hole=0.6,
        marker=dict(colors=[get_tyre_color(c) for c in tyre_counts.index]),
        textinfo='percent',
        textfont=dict(color='white', size=12),
        hoverinfo='label+value+percent'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Inter, sans-serif"),
        height=350,
        showlegend=False,
        annotations=[dict(text='TYRES', x=0.5, y=0.5, font_size=14, 
                        font_color=COLORS['text_muted'], showarrow=False)]
    )
    return fig


def create_driver_speed_heatmap(df):
    if 'Driver' not in df.columns or 'LapTimeSeconds' not in df.columns:
        return None
    
    pivot = df.pivot_table(values='LapTimeSeconds', index='Driver', columns='LapNumber', aggfunc='mean')
    
    if pivot.empty or pivot.shape[0] == 0:
        return None
    
    pivot = pivot.head(10).iloc[:, :min(25, pivot.shape[1])]
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"Lap {i}" for i in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0, COLORS['accent_green']],
            [0.5, COLORS['accent_yellow']],
            [1, COLORS['accent_red']]
        ],
        showscale=True,
        colorbar=dict(title="Time (s)", tickfont=dict(color=COLORS['text_muted']))
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['grid']),
        yaxis=dict(tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['grid']),
        height=350,
        margin=dict(l=80, r=20, t=40, b=40)
    )
    return fig


def get_track_geometry(gp_name):
    """Get track geometry for each Grand Prix - coordinates and characteristics"""
    
    track_database = {
        "Bahrain Grand Prix": {"turns": 15, "length": 5.412, "shape": "oval_complex"},
        "Saudi Arabian Grand Prix": {"turns": 27, "length": 6.174, "shape": "street"},
        "Australian Grand Prix": {"turns": 14, "length": 5.278, "shape": "permanent"},
        "Japanese Grand Prix": {"turns": 18, "length": 5.807, "shape": "permanent"},
        "Chinese Grand Prix": {"turns": 16, "length": 5.451, "shape": "permanent"},
        "Miami Grand Prix": {"turns": 19, "length": 5.412, "shape": "street"},
        "Monaco Grand Prix": {"turns": 19, "length": 3.337, "shape": "street"},
        "Spanish Grand Prix": {"turns": 14, "length": 4.675, "shape": "permanent"},
        "Canadian Grand Prix": {"turns": 14, "length": 4.410, "shape": "permanent"},
        "Austrian Grand Prix": {"turns": 10, "length": 4.318, "shape": "permanent"},
        "British Grand Prix": {"turns": 18, "length": 5.891, "shape": "permanent"},
        "Hungarian Grand Prix": {"turns": 14, "length": 4.381, "shape": "permanent"},
        "Belgian Grand Prix": {"turns": 19, "length": 7.004, "shape": "permanent"},
        "Dutch Grand Prix": {"turns": 14, "length": 4.259, "shape": "permanent"},
        "Italian Grand Prix": {"turns": 11, "length": 5.593, "shape": "permanent"},
        "Singapore Grand Prix": {"turns": 19, "length": 4.940, "shape": "street"},
        "Azerbaijan Grand Prix": {"turns": 20, "length": 6.003, "shape": "street"},
        "Mexican Grand Prix": {"turns": 17, "length": 4.304, "shape": "permanent"},
        "United States Grand Prix": {"turns": 20, "length": 5.513, "shape": "permanent"},
        "Brazilian Grand Prix": {"turns": 15, "length": 4.309, "shape": "permanent"},
        "Las Vegas Grand Prix": {"turns": 19, "length": 6.201, "shape": "street"},
        "Qatar Grand Prix": {"turns": 16, "length": 5.380, "shape": "permanent"},
        "Abu Dhabi Grand Prix": {"turns": 21, "length": 5.554, "shape": "permanent"},
    }
    
    return track_database.get(gp_name, {"turns": 15, "length": 5.0, "shape": "permanent"})


def create_track_points(track_info):
    """Generate realistic track geometry based on track type"""
    shape = track_info.get("shape", "permanent")
    num_turns = track_info.get("turns", 15)
    
    if shape == "street":
        num_points = 300
    elif shape == "permanent":
        num_points = 250
    else:
        num_points = 280
    
    points = []
    for i in range(num_points):
        t = (i / num_points) * 2 * np.pi
        
        if shape == "permanent":
            x = np.sin(t) * 8 + np.sin(t * 2.5) * 1.5
            z = np.cos(t) * 8 + np.cos(t * 1.5) * 2
            y = np.sin(t * 4) * 0.2
        elif shape == "street":
            x = np.sin(t) * 7 + np.sin(t * 3) * 3
            z = np.cos(t) * 7 + np.cos(t * 2) * 2.5
            y = np.sin(t * 5) * 0.15
        else:
            x = np.sin(t) * 7.5 + np.sin(t * 2) * 2
            z = np.cos(t) * 7.5 + np.cos(t * 2.5) * 1.5
            y = np.sin(t * 3) * 0.25
        
        points.append((x, y, z))
    
    return points


def create_3d_circuit(gp_name="Default", year=2023):
    """Create 3D circuit visualization based on selected Grand Prix using real data"""
    
    track_coords = []
    data_source = "No data"
    
    try:
        track_coords = get_track_coordinates(year, gp_name)
        if track_coords and len(track_coords) >= 10:
            data_source = f"FastF1 Telemetry ({len(track_coords)} points)"
        else:
            raise ValueError("Insufficient track points")
    except Exception as e:
        print(f"Track data error: {e}")
        track_coords = []
    
    if not track_coords or len(track_coords) < 10:
        track_info = get_track_geometry(gp_name)
        track_coords = create_track_points(track_info)
        data_source = "Simulated (Real data unavailable)"
    
    print(f"DEBUG: {gp_name} ({year}) - Using {len(track_coords)} points - {data_source}")
    
    track_data_js = "[" + ",".join([f"[{float(x):.2f},{float(y):.2f},{float(z):.2f}]" for x, y, z in track_coords[:150]]) + "]"
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; overflow: hidden; background: #0E1117; }}
            #info {{
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                color: #00D2BE;
                font-family: 'Inter', sans-serif;
                font-size: 18px;
                font-weight: 700;
                z-index: 100;
                background: rgba(14, 17, 23, 0.8);
                padding: 8px 20px;
                border-radius: 8px;
            }}
            #guide {{
                position: absolute;
                bottom: 10px;
                left: 50%;
                transform: translateX(-50%);
                color: #A0A0A0;
                font-family: 'Inter', sans-serif;
                font-size: 12px;
                z-index: 100;
            }}
        </style>
    </head>
    <body>
        <div id="info">{gp_name} | {data_source}</div>
        <div id="guide">Yellow = Corners | Green = Start/Finish | Drag to rotate | Scroll to zoom</div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            var trackData = {track_data_js};
            
            var scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0E1117);
            scene.fog = new THREE.Fog(0x0E1117, 40, 100);
            
            var camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 30, 40);
            
            var renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            var controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 15;
            controls.maxDistance = 80;
            controls.maxPolarAngle = Math.PI / 2.1;
            controls.target.set(0, 0, 0);
            
            // Find center and scale
            var minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
            trackData.forEach(p => {{
                minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
                minZ = Math.min(minZ, p[2]); maxZ = Math.max(maxZ, p[2]);
            }});
            var centerX = (minX + maxX) / 2;
            var centerZ = (minZ + maxZ) / 2;
            var scale = Math.max(maxX - minX, maxZ - minZ) / 25;
            if (scale < 1) scale = 1;
            
            // Normalize track
            trackData = trackData.map(p => [(p[0] - centerX) / scale, p[1], (p[2] - centerZ) / scale]);
            
            // Grid
            var gridHelper = new THREE.GridHelper(40, 40, 0x333333, 0x222222);
            gridHelper.position.y = -0.1;
            scene.add(gridHelper);
            
            // CLOSED track curve - connected end to end
            var trackCurvePoints = trackData.map(p => new THREE.Vector3(p[0], 0.3, p[2]));
            var trackCurve = new THREE.CatmullRomCurve3(trackCurvePoints, true, 'catmullrom', 0.5);
            
            // Main track - thick tube
            var trackGeometry = new THREE.TubeGeometry(trackCurve, 300, 0.5, 16, true);
            var trackMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0xE10600, 
                emissive: 0x800000,
                roughness: 0.3,
                metalness: 0.5
            }});
            var track = new THREE.Mesh(trackGeometry, trackMaterial);
            scene.add(track);
            
            // Outer glow ring
            var glowGeometry = new THREE.TubeGeometry(trackCurve, 300, 0.8, 12, true);
            var glowMaterial = new THREE.MeshBasicMaterial({{ 
                color: 0xFF4444, 
                transparent: true, 
                opacity: 0.15,
                side: THREE.BackSide
            }});
            var glow = new THREE.Mesh(glowGeometry, glowMaterial);
            scene.add(glow);
            
            // Track surface
            var surfaceGeometry = new THREE.TubeGeometry(trackCurve, 300, 1.5, 12, true);
            var surfaceMaterial = new THREE.MeshStandardMaterial({{ 
                color: 0x1a1a1a, 
                roughness: 0.9
            }});
            var surface = new THREE.Mesh(surfaceGeometry, surfaceMaterial);
            surface.position.y = -0.05;
            scene.add(surface);
            
            // Corner markers - more visible
            var numCorners = Math.min(15, Math.floor(trackData.length / 8));
            for (var i = 0; i < numCorners; i++) {{
                var idx = Math.floor(i * trackData.length / numCorners);
                var p = trackData[idx];
                
                // Glow sphere
                var glowSphere = new THREE.Mesh(
                    new THREE.SphereGeometry(0.6, 16, 16),
                    new THREE.MeshBasicMaterial({{ color: 0xFFD700, transparent: true, opacity: 0.3 }})
                );
                glowSphere.position.set(p[0], 0.8, p[2]);
                scene.add(glowSphere);
                
                // Solid core
                var marker = new THREE.Mesh(
                    new THREE.SphereGeometry(0.3, 16, 16),
                    new THREE.MeshStandardMaterial({{ 
                        color: 0xFFD700, 
                        emissive: 0xFFD700,
                        emissiveIntensity: 0.8
                    }})
                );
                marker.position.set(p[0], 0.8, p[2]);
                scene.add(marker);
            }}
            
            // Start/Finish - larger and more visible
            var startGeom = new THREE.BoxGeometry(1.5, 0.2, 0.4);
            var startMat = new THREE.MeshStandardMaterial({{ 
                color: 0x00D2BE, 
                emissive: 0x00D2BE,
                emissiveIntensity: 1.0
            }});
            var startLine = new THREE.Mesh(startGeom, startMat);
            var p0 = trackData[0];
            var p1 = trackData[1] || trackData[trackData.length - 1];
            startLine.position.set(p0[0], 0.4, p0[2]);
            startLine.rotation.y = Math.atan2(p1[0] - p0[0], p1[2] - p0[2]);
            scene.add(startLine);
            
            // Start glow
            var startGlow = new THREE.Mesh(
                new THREE.BoxGeometry(2, 0.4, 0.6),
                new THREE.MeshBasicMaterial({{ color: 0x00D2BE, transparent: true, opacity: 0.3 }})
            );
            startGlow.position.set(p0[0], 0.4, p0[2]);
            startGlow.rotation.y = startLine.rotation.y;
            scene.add(startGlow);
            
            // Center marker
            var centerGeom = new THREE.CircleGeometry(1.5, 32);
            var centerMat = new THREE.MeshStandardMaterial({{ 
                color: 0x3671C6, 
                transparent: true, 
                opacity: 0.4,
                side: THREE.DoubleSide
            }});
            var center = new THREE.Mesh(centerGeom, centerMat);
            center.rotation.x = -Math.PI / 2;
            center.position.y = -0.05;
            scene.add(center);
            
            // Lights
            var ambientLight = new THREE.AmbientLight(0x404040, 1.5);
            scene.add(ambientLight);
            
            var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(20, 30, 20);
            scene.add(dirLight);
            
            var dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
            dirLight2.position.set(-20, 20, -20);
            scene.add(dirLight2);
            
            var pointLight = new THREE.PointLight(0xE10600, 0.6, 50);
            pointLight.position.set(0, 8, 0);
            scene.add(pointLight);
            
            // Auto rotation
            var autoRotate = true;
            
            function animate() {{
                requestAnimationFrame(animate);
                
                if (autoRotate) {{
                    var time = Date.now() * 0.00008;
                    camera.position.x = Math.cos(time) * 35;
                    camera.position.z = Math.sin(time) * 35;
                    camera.lookAt(0, 0, 0);
                }}
                
                controls.update();
                renderer.render(scene, camera);
            }}
            
            controls.addEventListener('start', function() {{ autoRotate = false; }});
            
            animate();
            
            window.addEventListener('resize', function() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code


def render_3d_circuit(gp_name, year):
    components.html(create_3d_circuit(gp_name, year), height=400)


def render_zandvoort_3d():
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)

    static_file_path = os.path.join(os.path.dirname(__file__), 'static', 'track_visualizer.html')
    
    if os.path.exists(static_file_path):
        with open(static_file_path, 'r') as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
    else:
        st.warning("3D visualization file not found. Using fallback visualization.")
        render_3d_circuit("Dutch Grand Prix", 2024)
    
    st.markdown("""
    <div style="background: rgba(30, 30, 30, 0.7); backdrop-filter: blur(12px); 
                border-radius: 16px; padding: 20px; margin-top: 20px; 
                border: 1px solid rgba(255, 255, 255, 0.08);">
        <h4 style="color: #00D2BE; margin-top: 0;">Circuit Zandvoort Details</h4>
        <ul style="color: #A0A0A0; line-height: 1.8;">
            <li><strong>Location:</strong> Zandvoort, Netherlands</li>
            <li><strong>First Grand Prix:</strong> 1952</li>
            <li><strong>Track Length:</strong> 4.259 km</li>
            <li><strong>Turns:</strong> 14 (including the famous banked Turn 3)</li>
            <li><strong>Race Laps:</strong> 72</li>
            <li><strong>DRS Zones:</strong> 2</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_beginner_explanation():
    st.markdown("""
    <div class="beginner-box">
        <h3 style="margin-top:0; color: #3671C6;">New to Formula 1?</h3>
        <ul style="margin-bottom: 0;">
            <li><strong>What is a race?</strong> - Drivers complete 50-70 laps around a track. The first to cross the finish line wins.</li>
            <li><strong>What are tyre changes?</strong> - When drivers come in for new tyres. A good strategy can mean the difference between winning and losing.</li>
            <li><strong>Why does strategy matter?</strong> - Different tyres have different characteristics. Soft tyres are fast but wear out quickly. Hard tyres last longer but are slower.</li>
            <li><strong>What is tyre compound?</strong> - Tyres come in Soft (red), Medium (yellow), and Hard (white). Teams choose when to change them based on race conditions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_driver_explanation(df, driver):
    driver_df = df[df['Driver'] == driver].copy()
    if driver_df.empty:
        return
    
    final_pos = driver_df['Position'].iloc[-1] if 'Position' in driver_df.columns and not driver_df['Position'].isna().all() else None
    avg_time = driver_df['LapTimeSeconds'].mean() if 'LapTimeSeconds' in driver_df.columns else None
    
    actual_pits = 0
    if 'PitInTime' in driver_df.columns:
        actual_pits = len(driver_df[driver_df['PitInTime'].notna()])
    
    predicted_pits = 0
    if 'PredictedPit' in driver_df.columns:
        predicted_pits = int(driver_df['PredictedPit'].sum())
    
    pos_text = f"{int(final_pos)}th place" if final_pos and not pd.isna(final_pos) else "N/A"
    time_text = f"{avg_time:.2f}s per lap" if avg_time and not pd.isna(avg_time) else "N/A"
    
    st.markdown(f"""
    <div class="explanation-card">
        <h4 style="margin-top:0;">Why {driver}'s prediction?</h4>
        <ul>
            <li><strong>Expected Finish:</strong> {pos_text}</li>
            <li><strong>Average Lap Time:</strong> {time_text}</li>
            <li><strong>Tyre Changes:</strong> {actual_pits} actual, {predicted_pits} predicted</li>
    """, unsafe_allow_html=True)
    
    if 'PitProbability' in driver_df.columns:
        avg_prob = driver_df['PitProbability'].mean()
        if not pd.isna(avg_prob):
            if avg_prob > 0.6:
                st.markdown("<li><strong>Strategy:</strong> Likely to make a tyre change soon</li>")
            elif avg_prob < 0.3:
                st.markdown("<li><strong>Strategy:</strong> Can run longer before tyre change</li>")
            else:
                st.markdown("<li><strong>Strategy:</strong> Tyre change timing is balanced</li>")
    
    st.markdown("</ul></div>", unsafe_allow_html=True)


def render_strategy_simulator():
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="beginner-box">
        <h3 style="margin-top:0; color: #3671C6;">Strategy Simulator</h3>
        <p style="color: #A0A0A0;">Select your strategy parameters to visualize different race scenarios.</p>
        <p style="color: #A0A0A0; font-size: 0.9rem;"><em>Based on historical F1 tyre degradation patterns and team strategies.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        starting_tyre = st.selectbox("Starting Tyre", ["Soft", "Medium", "Hard"], index=1)
    
    with col2:
        strategy_type = st.selectbox("Tyre Change Strategy", ["One Stop", "Two Stop", "Undercut", "Overcut"])
    
    with col3:
        fuel_load = st.selectbox("Fuel Load", ["Full", "Light"])
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Strategy Visualization</div>', unsafe_allow_html=True)
    
    if strategy_type == "One Stop":
        stints = [(starting_tyre, 1, 35), ("Medium" if starting_tyre != "Medium" else "Hard", 36, 70)]
    elif strategy_type == "Two Stop":
        stints = [(starting_tyre, 1, 20), ("Medium" if starting_tyre != "Medium" else "Hard", 21, 45), 
                  ("Medium" if starting_tyre != "Medium" else "Hard", 46, 70)]
    elif strategy_type == "Undercut":
        stints = [(starting_tyre, 1, 15), ("Soft" if starting_tyre != "Soft" else "Medium", 16, 70)]
    else:
        stints = [(starting_tyre, 1, 40), ("Hard" if starting_tyre != "Hard" else "Medium", 41, 70)]
    
    cols = st.columns(len(stints))
    for i, (compound, start_lap, end_lap) in enumerate(stints):
        color = get_tyre_color(compound)
        lap_count = end_lap - start_lap + 1
        
        tyre_desc = {
            "Soft": "Fastest, wears quickly",
            "Medium": "Balanced performance",
            "Hard": "Durable, slower pace"
        }.get(compound, "")
        
        with cols[i]:
            st.markdown(f"""
            <div class="strategy-block" style="border-color: {color};">
                <h4 style="margin:0; color: {color}; font-size: 1.2rem;">{compound}</h4>
                <p style="margin:8px 0; color: #A0A0A0; font-size: 0.9rem;">Laps {start_lap}-{end_lap}</p>
                <p style="margin:0; color: #FAFAFA; font-weight: 600;">{lap_count} laps</p>
                <p style="margin:8px 0 0 0; color: #A0A0A0; font-size: 0.8rem;">{ tyre_desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Estimated Impact</div>', unsafe_allow_html=True)
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.metric("Total Stops", len(stints))
    with impact_col2:
        st.metric("Total Race Distance", f"{stints[-1][2]} laps")
    with impact_col3:
        estimated_time = len(stints) * 20
        st.metric("Est. Stop Time", f"+{estimated_time}s")
    with impact_col4:
        risk_level = "Low" if len(stints) == 1 else "Medium" if len(stints) == 2 else "High"
        risk_color = "#00D2BE" if risk_level == "Low" else "#FFD700" if risk_level == "Medium" else "#E10600"
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #1E1E1E, #262730); padding: 18px; border-radius: 14px; border: 1px solid #333; text-align: center;">
            <p style="margin:0; color: #A0A0A0; font-size: 0.8rem;">Strategy Risk</p>
            <p style="margin:8px 0 0 0; color: {risk_color}; font-weight: 700; font-size: 1.1rem;">{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)


def render_predictions_tab(df, model, config, threshold, gp_name, year):
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">3D Circuit</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        render_3d_circuit(gp_name, year)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Drivers", len(df['Driver'].unique()) if 'Driver' in df.columns else 0)
            if 'LapNumber' in df.columns:
                st.metric("Race Laps", int(df['LapNumber'].max()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'PredictedPit' in df.columns:
                st.metric("Predicted Tyre Changes", int(df['PredictedPit'].sum()))
            if 'PitInTime' in df.columns:
                actual_pits = df['PitInTime'].notna().sum()
                st.metric("Actual Tyre Changes", int(actual_pits))
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Lap Time Analysis</div>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            fig_stats = create_race_statistics(df)
            if fig_stats:
                st.plotly_chart(fig_stats, use_container_width=True)
            else:
                st.info("Lap time data not available")
        
        with col_stats2:
            fig_tyre = create_tyre_distribution(df)
            if fig_tyre:
                st.plotly_chart(fig_tyre, use_container_width=True)
            else:
                st.info("Tyre data not available")
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Driver Speed Heatmap</div>', unsafe_allow_html=True)
        fig_heat = create_driver_speed_heatmap(df)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Speed data not available")
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Tyre Change Probability</div>', unsafe_allow_html=True)
        fig_win = create_win_probability_chart(df)
        if fig_win:
            st.plotly_chart(fig_win, use_container_width=True)
        else:
            st.info("Probability data not available")


def render_explanation_tab(df, driver):
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    with st.container():
        render_driver_explanation(df, driver)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
        
        driver_df = df[df['Driver'] == driver].copy()
        if not driver_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'Position' in driver_df.columns and not driver_df['Position'].isna().all():
                    start_pos = int(driver_df['Position'].iloc[0]) if not pd.isna(driver_df['Position'].iloc[0]) else None
                    final_pos = int(driver_df['Position'].iloc[-1]) if not pd.isna(driver_df['Position'].iloc[-1]) else None
                    if start_pos:
                        st.metric("Starting Position", start_pos)
                    if final_pos:
                        st.metric("Final Position", final_pos)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'LapTimeSeconds' in driver_df.columns and not driver_df['LapTimeSeconds'].isna().all():
                    avg_time = driver_df['LapTimeSeconds'].mean()
                    min_time = driver_df['LapTimeSeconds'].min()
                    if not pd.isna(avg_time):
                        st.metric("Avg Lap Time", f"{avg_time:.2f}s")
                    if not pd.isna(min_time):
                        st.metric("Fastest Lap", f"{min_time:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'PredictedPit' in driver_df.columns:
                    pred_stops = int(driver_df['PredictedPit'].sum())
                    st.metric("Predicted Stops", pred_stops)
                if 'PitInTime' in driver_df.columns:
                    actual = driver_df['PitInTime'].notna().sum()
                    st.metric("Actual Stops", int(actual))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'PitProbability' in driver_df.columns and not driver_df['PitProbability'].isna().all():
                    avg_prob = driver_df['PitProbability'].mean()
                    if not pd.isna(avg_prob):
                        st.metric("Avg Tyre Change Probability", f"{avg_prob:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Performance Charts</div>', unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_lap = create_lap_time_chart(df, driver)
            if fig_lap:
                st.plotly_chart(fig_lap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_prob = create_pit_probability_chart(df, driver)
            if fig_prob:
                st.plotly_chart(fig_prob, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Tyre Strategy</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_strategy = create_strategy_timeline(df, driver)
        if fig_strategy:
            st.plotly_chart(fig_strategy, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Lap Data</div>', unsafe_allow_html=True)
        
        cols_to_show = []
        for col in ['LapNumber', 'Position', 'LapTimeSeconds', 'Compound', 'TyreLife', 'PitProbability', 'PredictedPit']:
            if col in driver_df.columns:
                cols_to_show.append(col)
        
        if cols_to_show:
            styled_df = create_styled_dataframe(driver_df[cols_to_show])
            if styled_df is not None:
                st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("Lap data not available")


if 'update_trigger' not in st.session_state:
    st.session_state.update_trigger = 0


def trigger_update():
    st.session_state.update_trigger += 1


def main():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div class="main-title">F1 Strategy Predictor</div>
        <div class="subtitle">AI-Powered Race Predictions & Tyre Strategy Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    with st.container():
        render_beginner_explanation()
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    model, config = get_trained_model()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0; border-bottom: 1px solid #333; margin-bottom: 20px;">
            <h3 style="color: #E10600; margin: 0;">Configuration</h3>
            <p style="color: #A0A0A0; font-size: 0.85rem; margin: 5px 0 0 0;">Customize your prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-header">Race Selection</div>', unsafe_allow_html=True)
        
        year = st.selectbox("Year", [2023, 2022, 2021], index=0, label_visibility="collapsed", key="year_select")
        
        races = get_available_races_cached(year)
        gp = st.selectbox("Grand Prix", races[:15] if len(races) > 15 else races, label_visibility="collapsed", key="gp_select")
        
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">Prediction Settings</div>', unsafe_allow_html=True)
        threshold = st.slider("Tyre Change Sensitivity", 0.1, 0.9, 0.5, 0.05, 
                             help="Lower values predict more tyre changes, higher values predict fewer", key="threshold_slider")
        
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">Update</div>', unsafe_allow_html=True)
        
        if st.button("Update Predictions", use_container_width=True, on_click=trigger_update):
            pass
        
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 0.8rem; text-align: center; margin-top: 10px;">
            Click to refresh predictions when you change any parameter.
        </p>
        """, unsafe_allow_html=True)
    
    try:
        with st.spinner("Loading race data..."):
            df = load_race_data_cached(year, gp)
        
        if df.empty:
            st.warning("No data available for this race. Try selecting a different Grand Prix.")
            return
        
        drivers = sorted(df['Driver'].unique())
        
        col_driver, col_btn = st.columns([3, 1])
        with col_driver:
            selected_driver = st.selectbox("Select a driver to analyze:", drivers, index=0, key="driver_select")
        with col_btn:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            if st.button("Run", use_container_width=True, on_click=trigger_update):
                pass
        
        df_pred = get_model_predictions(df, threshold, model, config)
        
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Explanation", "Strategy Simulator", "3D Track"])
        
        with tab1:
            render_predictions_tab(df_pred, model, config, threshold, gp, year)
        
        with tab2:
            render_explanation_tab(df_pred, selected_driver)
        
        with tab3:
            render_strategy_simulator()
        
        with tab4:
            render_zandvoort_3d()
        
        with st.expander("Model Information"):
            if model is not None and config is not None:
                st.write(f"**Model Type:** {config.get('model_type', 'XGBoost')}")
                st.write(f"**Threshold:** {config.get('threshold', 0.5):.3f}")
                st.write(f"**Features Used:** {len(config.get('feature_cols', []))} features")
            else:
                st.write("Model not available")
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure fastf1 cache is set up. Run once with internet to download data.")


if __name__ == "__main__":
    main()