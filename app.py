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


def create_styled_dataframe(df):
    if df is None or df.empty:
        return None
    
    def style_cell(val):
        if isinstance(val, (int, float)):
            if not pd.isna(val):
                return f'color: {COLORS["text"]}; background-color: {COLORS["card"]}'
        return f'color: {COLORS["text"]}; background-color: {COLORS["card"]}'
    
    try:
        styled = df.style.applymap(style_cell)
        styled = styled.set_properties(**{
            'background-color': COLORS['card'],
            'color': COLORS['text'],
            'border-color': COLORS['grid'],
            'border-style': 'solid',
            'border-width': '1px',
            'padding': '8px',
            'text-align': 'center',
            'font-family': 'Inter, sans-serif',
        })
        styled = styled.set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', COLORS['card_light']),
                ('color', COLORS['text']),
                ('font-weight', '600'),
                ('padding', '10px'),
                ('border-bottom', f'2px solid {COLORS["accent_blue"]}'),
            ]},
            {'selector': 'td:hover', 'props': [
                ('background-color', COLORS['card_hover']),
            ]},
        ])
        return styled
    except Exception as e:
        return None


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
        xaxis=dict(title="Driver", gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_blue'], spikethickness=2, spikemode="across"),
        yaxis=dict(title="Position", gridcolor=COLORS['grid'], range=[20, 1], showspikes=True, spikecolor=COLORS['accent_blue'], spikethickness=2, spikemode="across"),
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
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_green'], spikethickness=1, spikemode="toaxis"),
        yaxis=dict(title="Position", gridcolor=COLORS['grid'], range=[20, 1], 
                   autorange=False, showspikes=True, spikecolor=COLORS['accent_green'], spikethickness=1, spikemode="toaxis"),
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
        xaxis=dict(title="Driver", gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_yellow'], spikethickness=1),
        yaxis=dict(title="Win Probability", gridcolor=COLORS['grid'], 
                   range=[0, 1], tickformat=".0%", showspikes=True, spikecolor=COLORS['accent_yellow'], spikethickness=1),
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
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid'], showgrid=True, showspikes=True, spikecolor=COLORS['accent_red'], spikethickness=1, spikemode="toaxis"),
        yaxis=dict(title="Lap Time (seconds)", gridcolor=COLORS['grid'], showgrid=True, showspikes=True, spikecolor=COLORS['accent_red'], spikethickness=1, spikemode="toaxis"),
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
        xaxis=dict(title="Lap Number", gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_blue'], spikethickness=1),
        yaxis=dict(title="Probability", range=[0, 1], gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_blue'], spikethickness=1),
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
        xaxis=dict(title=dict(text="", font=dict(color=COLORS['text'])), gridcolor=COLORS['grid'], tickfont=dict(color=COLORS['text_muted']), showspikes=True, spikecolor=COLORS['accent_green'], spikethickness=1),
        yaxis=dict(title=dict(text="Avg Lap Time (s)", font=dict(color=COLORS['text'])), gridcolor=COLORS['grid'], 
                   tickfont=dict(color=COLORS['text_muted']),
                   showspikes=True, spikecolor=COLORS['accent_green'], spikethickness=1),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg']
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
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
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
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_red'], spikethickness=1),
        yaxis=dict(tickfont=dict(color=COLORS['text_muted']), gridcolor=COLORS['grid'], showspikes=True, spikecolor=COLORS['accent_red'], spikethickness=1),
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


TRACK_DATA = {
    "Monaco Grand Prix": {
        "name": "Circuit de Monaco",
        "coords": [
            (0.85, 0.68), (0.85, 0.60), (0.82, 0.55), (0.75, 0.52), (0.65, 0.53),
            (0.58, 0.52), (0.52, 0.48), (0.48, 0.40), (0.47, 0.32), (0.45, 0.25),
            (0.40, 0.20), (0.32, 0.18), (0.25, 0.20), (0.20, 0.25), (0.18, 0.30),
            (0.15, 0.35), (0.12, 0.40), (0.10, 0.48), (0.12, 0.55), (0.18, 0.62),
            (0.25, 0.68), (0.35, 0.72), (0.45, 0.70), (0.55, 0.65), (0.62, 0.62),
            (0.68, 0.65), (0.72, 0.68), (0.78, 0.70), (0.85, 0.68)
        ]
    },
    "Silverstone Circuit": {
        "name": "Silverstone",
        "coords": [
            (0.50, 0.20), (0.55, 0.22), (0.60, 0.25), (0.65, 0.28), (0.70, 0.25),
            (0.75, 0.20), (0.78, 0.15), (0.80, 0.10), (0.78, 0.05), (0.72, 0.03),
            (0.65, 0.02), (0.58, 0.02), (0.50, 0.03), (0.42, 0.05), (0.35, 0.08),
            (0.28, 0.12), (0.22, 0.18), (0.18, 0.25), (0.15, 0.32), (0.12, 0.40),
            (0.10, 0.48), (0.12, 0.55), (0.18, 0.60), (0.25, 0.65), (0.32, 0.68),
            (0.40, 0.70), (0.48, 0.72), (0.50, 0.68), (0.50, 0.60), (0.50, 0.50),
            (0.50, 0.40), (0.50, 0.30), (0.50, 0.20)
        ]
    },
    "Spa-Francorchamps": {
        "name": "Spa",
        "coords": [
            (0.20, 0.75), (0.25, 0.72), (0.30, 0.68), (0.35, 0.65), (0.40, 0.62),
            (0.45, 0.60), (0.50, 0.58), (0.55, 0.55), (0.60, 0.50), (0.62, 0.45),
            (0.63, 0.40), (0.62, 0.35), (0.60, 0.30), (0.58, 0.25), (0.55, 0.22),
            (0.52, 0.20), (0.50, 0.18), (0.48, 0.15), (0.50, 0.12), (0.55, 0.10),
            (0.62, 0.08), (0.70, 0.06), (0.78, 0.05), (0.85, 0.08), (0.88, 0.12),
            (0.90, 0.18), (0.88, 0.25), (0.85, 0.30), (0.82, 0.35), (0.80, 0.40),
            (0.78, 0.45), (0.75, 0.50), (0.70, 0.55), (0.65, 0.58), (0.58, 0.62),
            (0.50, 0.65), (0.42, 0.68), (0.35, 0.72), (0.28, 0.75), (0.22, 0.78),
            (0.20, 0.75)
        ]
    },
    "default": {
        "name": "F1 Circuit",
        "coords": [
            (0.50, 0.20), (0.60, 0.22), (0.70, 0.25), (0.78, 0.30), (0.82, 0.38),
            (0.80, 0.45), (0.75, 0.50), (0.70, 0.55), (0.65, 0.60), (0.58, 0.65),
            (0.50, 0.68), (0.42, 0.65), (0.35, 0.60), (0.30, 0.55), (0.25, 0.50),
            (0.20, 0.45), (0.18, 0.38), (0.22, 0.30), (0.30, 0.25), (0.40, 0.22),
            (0.50, 0.20)
        ]
    }
}


def create_3d_circuit(gp_name="Default", year=2023, svg_track_path=None):
    """Create broadcast-quality 3D circuit visualization with real track data"""
    
    track_key = None
    for key in TRACK_DATA.keys():
        if key.lower() in gp_name.lower() or gp_name.lower() in key.lower():
            track_key = key
            break
    
    if track_key is None:
        track_key = "default"
    
    track_info = TRACK_DATA[track_key]
    data_source = f"Real Track Data - {track_info['name']}"
    
    track_data_js = "[" + ",".join([f"[{x:.4f},{0},{y:.4f}]" for x, y in track_info['coords']]) + "]"
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                margin: 0; 
                overflow: hidden; 
                background: linear-gradient(135deg, #0a0a12 0%, #0d1117 50%, #0a0a12 100%);
                font-family: 'Inter', -apple-system, sans-serif;
            }}
            #canvas-container {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            #overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 10;
            }}
            #title-bar {{
                position: absolute;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                text-align: center;
                pointer-events: none;
            }}
            #event-title {{
                font-size: 14px;
                font-weight: 500;
                color: #00D2BE;
                letter-spacing: 4px;
                text-transform: uppercase;
                margin-bottom: 4px;
                text-shadow: 0 0 20px rgba(0, 210, 190, 0.6);
            }}
            #gp-name {{
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: 2px;
                text-shadow: 0 0 30px rgba(0, 210, 190, 0.4), 0 2px 10px rgba(0,0,0,0.8);
            }}
            #data-source {{
                font-size: 10px;
                color: #6b7280;
                margin-top: 6px;
                letter-spacing: 1px;
            }}
            #controls-hint {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 24px;
                pointer-events: none;
            }}
            .hint-item {{
                font-size: 11px;
                color: #6b7280;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .hint-key {{
                background: rgba(255,255,255,0.1);
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid rgba(255,255,255,0.15);
                font-size: 10px;
                color: #9ca3af;
            }}
            #telemetry-panel {{
                position: absolute;
                bottom: 60px;
                left: 20px;
                background: rgba(10, 10, 18, 0.85);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(0, 210, 190, 0.2);
                border-radius: 12px;
                padding: 16px 20px;
                min-width: 200px;
                pointer-events: none;
            }}
            .telem-row {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                align-items: center;
            }}
            .telem-row:last-child {{ margin-bottom: 0; }}
            .telem-label {{
                font-size: 10px;
                color: #6b7280;
                letter-spacing: 1px;
                text-transform: uppercase;
            }}
            .telem-value {{
                font-size: 16px;
                font-weight: 600;
                color: #00D2BE;
                text-shadow: 0 0 10px rgba(0, 210, 190, 0.5);
            }}
            .telem-unit {{
                font-size: 10px;
                color: #6b7280;
                margin-left: 4px;
            }}
            #loading {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #00D2BE;
                font-size: 14px;
                letter-spacing: 3px;
                z-index: 100;
            }}
        </style>
    </head>
    <body>
        <div id="loading">INITIALIZING TELEMETRY...</div>
        <div id="canvas-container"></div>
        <div id="overlay">
            <div id="title-bar">
                <div id="event-title">Pre-Season Testing 2024</div>
                <div id="gp-name">{gp_name}</div>
                <div id="data-source">{data_source}</div>
            </div>
            <div id="telemetry-panel">
                <div class="telem-row">
                    <span class="telem-label">Speed</span>
                    <span><span class="telem-value" id="speed-val">287</span><span class="telem-unit">KM/H</span></span>
                </div>
                <div class="telem-row">
                    <span class="telem-label">Throttle</span>
                    <span><span class="telem-value" id="throttle-val">95</span><span class="telem-unit">%</span></span>
                </div>
                <div class="telem-row">
                    <span class="telem-label">Lap</span>
                    <span><span class="telem-value" id="lap-val">1</span><span class="telem-unit">/ 72</span></span>
                </div>
            </div>
            <div id="controls-hint">
                <div class="hint-item"><span class="hint-key">DRAG</span> Rotate</div>
                <div class="hint-item"><span class="hint-key">SCROLL</span> Zoom</div>
                <div class="hint-item"><span class="hint-key">RIGHT DRAG</span> Pan</div>
            </div>
        </div>
        
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>
        
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            import {{ SVGLoader }} from 'three/addons/loaders/SVGLoader.js';
            import {{ EffectComposer }} from 'three/addons/postprocessing/EffectComposer.js';
            import {{ RenderPass }} from 'three/addons/postprocessing/RenderPass.js';
            import {{ UnrealBloomPass }} from 'three/addons/postprocessing/UnrealBloomPass.js';
            import {{ OutputPass }} from 'three/addons/postprocessing/OutputPass.js';
            
            const trackData = {track_data_js};
            
            let scene, camera, renderer, controls, composer;
            let trackCurve, carMarker, carLight, trailParticles;
            let clock = new THREE.Clock();
            let autoRotate = true;
            let lastInteraction = 0;
            let carProgress = 0;
            
            function extractSvgTrackPaths(svgData) {{
                const points = [];
                const paths = svgData.paths;
                
                paths.forEach(path => {{
                    const shapes = path.toShapes(true);
                    shapes.forEach(shape => {{
                        const numPoints = 300;
                        const spacedPoints = shape.getSpacedPoints(numPoints);
                        spacedPoints.forEach(p => {{
                            points.push(new THREE.Vector3(p.x, 0, -p.y));
                        }});
                    }});
                }});
                
                return points;
            }}
            
            function normalizeTrackPoints(points) {{
                if (points.length === 0) return points;
                
                let minX = Infinity, maxX = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                
                points.forEach(p => {{
                    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
                    minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
                }});
                
                const centerX = (minX + maxX) / 2;
                const centerZ = (minZ + maxZ) / 2;
                const scale = Math.max(maxX - minX, maxZ - minZ) / 20;
                
                return points.map(p => new THREE.Vector3(
                    (p.x - centerX) / scale,
                    p.y / scale,
                    (p.z - centerZ) / scale
                ));
            }}
            
            async function loadSvgTrack(url) {{
                return new Promise((resolve, reject) => {{
                    const loader = new SVGLoader();
                    loader.load(url, (data) => {{
                        const rawPoints = extractSvgTrackPaths(data);
                        const normalizedPoints = normalizeTrackPoints(rawPoints);
                        resolve(normalizedPoints);
                    }}, undefined, (error) => {{
                        reject(error);
                    }});
                }});
            }}
            
            async function init() {{
                try {{
                    // Scene
                    scene = new THREE.Scene();
                    scene.fog = new THREE.FogExp2(0x0a0a12, 0.015);
                    
                    // Camera - cinematic 3/4 angle
                    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
                    camera.position.set(25, 20, 30);
                    
                    // Renderer
                    renderer = new THREE.WebGLRenderer({{ 
                        antialias: true,
                        powerPreference: "high-performance"
                    }});
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    renderer.toneMapping = THREE.ACESFilmicToneMapping;
                    renderer.toneMappingExposure = 1.2;
                    renderer.setClearColor(0x0a0a12);
                    document.getElementById('canvas-container').appendChild(renderer.domElement);
                    
                    // Post-processing
                    composer = new EffectComposer(renderer);
                    const renderPass = new RenderPass(scene, camera);
                    composer.addPass(renderPass);
                    
                    const bloomPass = new UnrealBloomPass(
                        new THREE.Vector2(window.innerWidth, window.innerHeight),
                        1.5,   // strength
                        0.4,   // radius
                        0.2    // threshold
                    );
                    composer.addPass(bloomPass);
                    
                    const outputPass = new OutputPass();
                    composer.addPass(outputPass);
                    
                    // Controls
                    controls = new OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.minDistance = 15;
                    controls.maxDistance = 80;
                    controls.maxPolarAngle = Math.PI / 2.2;
                    controls.target.set(0, 0, 0);
                    controls.autoRotate = false;
                    
                    controls.addEventListener('start', () => {{
                        autoRotate = false;
                        lastInteraction = Date.now();
                    }});
                    
                    // Normalize and create track curve
                    const curvePoints = getNormalizedTrackPoints(trackData);
                    trackCurve = new THREE.CatmullRomCurve3(curvePoints, true, 'catmullrom', 0.5);
                    
                    createEnvironment();
                    createTrack();
                    createLighting();
                    createCarMarker();
                    createParticles();
                    
                    document.getElementById('loading').style.display = 'none';
                    animate();
                    
                }} catch (error) {{
                    console.error('Initialization error:', error);
                    document.getElementById('loading').textContent = 'ERROR LOADING';
                    document.getElementById('loading').style.color = '#E10600';
                }}
            }}
            
            function getNormalizedTrackPoints(data) {{
                let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
                data.forEach(p => {{
                    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
                    minZ = Math.min(minZ, p[2]); maxZ = Math.max(maxZ, p[2]);
                }});
                const centerX = (minX + maxX) / 2;
                const centerZ = (minZ + maxZ) / 2;
                const scale = Math.max(maxX - minX, maxZ - minZ) / 20;
                
                return data.map(p => new THREE.Vector3(
                    (p[0] - centerX) / scale,
                    0.2,
                    (p[2] - centerZ) / scale
                ));
            }}
            
            function createEnvironment() {{
                // Subtle reflective floor
                const floorGeom = new THREE.PlaneGeometry(100, 100);
                const floorMat = new THREE.MeshStandardMaterial({{
                    color: 0x080810,
                    roughness: 0.8,
                    metalness: 0.3
                }});
                const floor = new THREE.Mesh(floorGeom, floorMat);
                floor.rotation.x = -Math.PI / 2;
                floor.position.y = -0.1;
                scene.add(floor);
                
                // Minimal grid
                const gridHelper = new THREE.GridHelper(60, 30, 0x1a1a2e, 0x0d0d15);
                gridHelper.material.opacity = 0.3;
                gridHelper.material.transparent = true;
                scene.add(gridHelper);
            }}
            
            function createTrack() {{
                // Create a single thick glowing track tube
                const trackRadius = 0.8;
                const tubularSegments = 500;
                
                // Main glowing track
                const trackGeom = new THREE.TubeGeometry(trackCurve, tubularSegments, trackRadius, 32, true);
                const trackMat = new THREE.MeshStandardMaterial({{
                    color: 0x2a2a2a,
                    emissive: 0x00D2BE,
                    emissiveIntensity: 1.2,
                    metalness: 0.9,
                    roughness: 0.1,
                    side: THREE.DoubleSide
                }});
                const track = new THREE.Mesh(trackGeom, trackMat);
                track.position.y = 0;
                scene.add(track);
                
                // Inner glow tube (brighter)
                const innerGlowGeom = new THREE.TubeGeometry(trackCurve, tubularSegments, trackRadius * 0.6, 24, true);
                const innerGlowMat = new THREE.MeshBasicMaterial({{
                    color: 0x00D2BE,
                    transparent: true,
                    opacity: 0.4
                }});
                const innerGlow = new THREE.Mesh(innerGlowGeom, innerGlowMat);
                innerGlow.position.y = 0.1;
                scene.add(innerGlow);
                
                // Outer glow effect (larger, more transparent)
                const outerGlowGeom = new THREE.TubeGeometry(trackCurve, tubularSegments, trackRadius * 1.5, 24, true);
                const outerGlowMat = new THREE.MeshBasicMaterial({{
                    color: 0x00D2BE,
                    transparent: true,
                    opacity: 0.15,
                    side: THREE.BackSide
                }});
                const outerGlow = new THREE.Mesh(outerGlowGeom, outerGlowMat);
                outerGlow.position.y = 0;
                scene.add(outerGlow);
                
                // Track surface base
                const baseGeom = new THREE.TubeGeometry(trackCurve, tubularSegments, trackRadius * 1.8, 16, true);
                const baseMat = new THREE.MeshStandardMaterial({{
                    color: 0x0a0a0f,
                    emissive: 0x050508,
                    emissiveIntensity: 0.5,
                    metalness: 0.3,
                    roughness: 0.8
                }});
                const base = new THREE.Mesh(baseGeom, baseMat);
                base.position.y = -0.3;
                scene.add(base);
                
                // Start/Finish line markers
                const startPos = trackCurve.getPointAt(0);
                const startTangent = trackCurve.getTangentAt(0);
                
                // Checkered start line
                for (let i = -2; i <= 2; i++) {{
                    const checkerGeom = new THREE.BoxGeometry(0.15, 0.02, 0.3);
                    const checkerMat = new THREE.MeshStandardMaterial({{
                        color: i % 2 === 0 ? 0xffffff : 0x000000,
                        emissive: i % 2 === 0 ? 0xffffff : 0x000000,
                        emissiveIntensity: 0.8
                    }});
                    const checker = new THREE.Mesh(checkerGeom, checkerMat);
                    checker.position.copy(startPos);
                    checker.position.y = 0.6;
                    checker.position.x += i * 0.15;
                    checker.lookAt(startPos.clone().add(startTangent));
                    scene.add(checker);
                }}
                
                // Start/finish glow arch
                const archGeom = new THREE.TorusGeometry(1.5, 0.05, 8, 32, Math.PI);
                const archMat = new THREE.MeshStandardMaterial({{
                    color: 0x00D2BE,
                    emissive: 0x00D2BE,
                    emissiveIntensity: 2,
                    metalness: 1,
                    roughness: 0
                }});
                const arch = new THREE.Mesh(archGeom, archMat);
                arch.position.copy(startPos);
                arch.position.y = 1;
                arch.rotation.x = Math.PI / 2;
                arch.lookAt(startPos.clone().add(startTangent));
                arch.rotation.x = Math.PI / 2;
                scene.add(arch);
            }}
            
            function createLighting() {{
                // Bright ambient for visibility
                const ambient = new THREE.AmbientLight(0x404060, 0.8);
                scene.add(ambient);
                
                // Strong directional light from above
                const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
                dirLight.position.set(10, 30, 10);
                scene.add(dirLight);
                
                // Cyan accent lights
                const cyanLight1 = new THREE.PointLight(0x00D2BE, 3, 40);
                cyanLight1.position.set(-10, 8, 10);
                scene.add(cyanLight1);
                
                const cyanLight2 = new THREE.PointLight(0x00D2BE, 2, 30);
                cyanLight2.position.set(10, 6, -10);
                scene.add(cyanLight2);
                
                // Purple accent
                const purpleLight = new THREE.PointLight(0x9333ea, 2, 35);
                purpleLight.position.set(0, 10, 15);
                scene.add(purpleLight);
                
                // Warm fill light
                const warmLight = new THREE.PointLight(0xff6600, 1, 25);
                warmLight.position.set(-8, 5, -8);
                scene.add(warmLight);
            }}
            
            function createCarMarker() {{
                // Car body - larger for visibility
                const carGeom = new THREE.SphereGeometry(0.6, 24, 24);
                const carMat = new THREE.MeshStandardMaterial({{
                    color: 0xE10600,
                    emissive: 0xE10600,
                    emissiveIntensity: 2.0,
                    metalness: 0.8,
                    roughness: 0.2
                }});
                carMarker = new THREE.Mesh(carGeom, carMat);
                scene.add(carMarker);
                
                // Car light (follows car)
                carLight = new THREE.PointLight(0xE10600, 5, 12);
                scene.add(carLight);
                
                // Glow sprite
                const spriteMat = new THREE.SpriteMaterial({{
                    map: createGlowTexture(),
                    color: 0xE10600,
                    transparent: true,
                    blending: THREE.AdditiveBlending
                }});
                const glowSprite = new THREE.Sprite(spriteMat);
                glowSprite.scale.set(4, 4, 1);
                carMarker.add(glowSprite);
            }}
            
            function createGlowTexture() {{
                const canvas = document.createElement('canvas');
                canvas.width = 128;
                canvas.height = 128;
                const ctx = canvas.getContext('2d');
                
                const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
                gradient.addColorStop(0, 'rgba(255,255,255,1)');
                gradient.addColorStop(0.3, 'rgba(255,255,255,0.8)');
                gradient.addColorStop(0.6, 'rgba(0,210,190,0.3)');
                gradient.addColorStop(1, 'rgba(0,210,190,0)');
                
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, 128, 128);
                
                const texture = new THREE.CanvasTexture(canvas);
                return texture;
            }}
            
            function createParticles() {{
                // Speed particles
                const particleCount = 500;
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(particleCount * 3);
                const colors = new Float32Array(particleCount * 3);
                
                for (let i = 0; i < particleCount; i++) {{
                    positions[i * 3] = (Math.random() - 0.5) * 60;
                    positions[i * 3 + 1] = Math.random() * 15;
                    positions[i * 3 + 2] = (Math.random() - 0.5) * 60;
                    
                    const color = new THREE.Color();
                    color.setHSL(0.5 + Math.random() * 0.2, 0.8, 0.6);
                    colors[i * 3] = color.r;
                    colors[i * 3 + 1] = color.g;
                    colors[i * 3 + 2] = color.b;
                }}
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                
                const material = new THREE.PointsMaterial({{
                    size: 0.08,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.6,
                    blending: THREE.AdditiveBlending
                }});
                
                trailParticles = new THREE.Points(geometry, material);
                scene.add(trailParticles);
            }}
            
            function updateTelemetry() {{
                const speed = Math.floor(250 + Math.random() * 80);
                const throttle = Math.floor(70 + Math.random() * 30);
                const lap = Math.floor(carProgress * 72) + 1;
                
                document.getElementById('speed-val').textContent = speed;
                document.getElementById('throttle-val').textContent = throttle;
                document.getElementById('lap-val').textContent = Math.min(lap, 72);
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                
                const delta = clock.getDelta();
                const time = clock.getElapsedTime();
                
                // Resume auto-rotation after 4 seconds of inactivity
                if (!autoRotate && Date.now() - lastInteraction > 4000) {{
                    autoRotate = true;
                }}
                
                // Auto-rotate camera
                if (autoRotate) {{
                    const rotSpeed = 0.02;
                    const radius = 25;
                    camera.position.x = Math.cos(time * rotSpeed) * radius * 0.8;
                    camera.position.z = Math.sin(time * rotSpeed) * radius;
                    camera.position.y = 15 + Math.sin(time * 0.015) * 3;
                    camera.lookAt(0, 0, 0);
                }}
                
                // Move car along track
                carProgress = (carProgress + delta * 0.015) % 1;
                const carPos = trackCurve.getPointAt(carProgress);
                const carTangent = trackCurve.getTangentAt(carProgress);
                
                carMarker.position.copy(carPos);
                carMarker.position.y += 1.0;
                carLight.position.copy(carMarker.position);
                
                // Pulse effect
                const pulse = 0.9 + Math.sin(time * 4) * 0.15;
                carMarker.scale.setScalar(pulse);
                
                // Update telemetry every 100ms
                if (Math.floor(time * 10) % 5 === 0) {{
                    updateTelemetry();
                }}
                
                // Animate particles
                if (trailParticles) {{
                    const positions = trailParticles.geometry.attributes.position.array;
                    for (let i = 0; i < positions.length; i += 3) {{
                        positions[i + 1] += 0.02;
                        if (positions[i + 1] > 15) positions[i + 1] = 0;
                    }}
                    trailParticles.geometry.attributes.position.needsUpdate = true;
                }}
                
                controls.update();
                composer.render();
            }}
            
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
                composer.setSize(window.innerWidth, window.innerHeight);
            }});
            
            init();
        </script>
    </body>
    </html>
    """
    return html_code


def render_3d_circuit(gp_name, year, svg_track_path=None):
    components.html(create_3d_circuit(gp_name, year, svg_track_path), height=500, scrolling=False)


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