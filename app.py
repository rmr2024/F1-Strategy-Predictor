import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import enable_caching, load_single_race, load_season_data, get_available_races
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
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, {COLORS['accent_red']}, {COLORS['accent_blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.5s ease-in;
    }}
    
    .subtitle {{
        font-size: 1rem;
        color: {COLORS['text_muted']};
        margin-bottom: 1.5rem;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['card']}, {COLORS['bg']});
        border-right: 1px solid {COLORS['grid']};
    }}
    
    .sidebar-header {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {COLORS['text']};
        padding: 1rem 0;
        border-bottom: 2px solid {COLORS['accent_red']};
        margin-bottom: 1rem;
    }}
    
    /* Card Styling */
    .dashboard-card {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid {COLORS['grid']};
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    
    .dashboard-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }}
    
    .metric-card {{
        background: {COLORS['card']};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid {COLORS['grid']};
        transition: all 0.2s ease;
    }}
    
    .metric-card:hover {{
        border-color: {COLORS['accent_blue']};
    }}
    
    /* Beginner Box */
    .beginner-box {{
        background: linear-gradient(135deg, {COLORS['card']}, {COLORS['card_light']});
        border-left: 4px solid {COLORS['accent_blue']};
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        animation: slideInLeft 0.4s ease-out;
    }}
    
    .beginner-title {{
        color: {COLORS['accent_blue']};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }}
    
    /* Explanation Card */
    .explanation-card {{
        background: linear-gradient(145deg, {COLORS['card']}, {COLORS['card_light']});
        border-radius: 16px;
        padding: 24px;
        margin: 15px 0;
        border-left: 4px solid {COLORS['accent_green']};
        animation: fadeInUp 0.4s ease-out;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['card']};
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['accent_red']};
        color: white;
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['text']};
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .section-header::before {{
        content: '';
        width: 4px;
        height: 24px;
        background: {COLORS['accent_red']};
        border-radius: 2px;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideInLeft {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    @keyframes fadeInUp {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    .loading-pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['grid']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['text_muted']};
    }}
    
    /* Metrics spacing */
    div[data-testid="stMetric"] {{
        background: {COLORS['card']};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {COLORS['grid']};
    }}
    
    /* Input styling */
    .stSelectbox > div > div {{
        background-color: {COLORS['card']};
        border: 1px solid {COLORS['grid']};
    }}
    
    /* Spacing utilities */
    .spacer-md {{ height: 20px; }}
    .spacer-lg {{ height: 40px; }}
    
    /* Chart container */
    .chart-container {{
        background: {COLORS['card']};
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
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
                   autorange="reversed"),
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


def create_styled_dataframe(df, max_rows=10):
    if df.empty:
        return None
    
    display_df = df.head(max_rows).copy()
    
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].round(3)
    
    return display_df


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
    
    final_pos = driver_df['Position'].iloc[-1] if 'Position' in driver_df.columns else None
    avg_time = driver_df['LapTimeSeconds'].mean() if 'LapTimeSeconds' in driver_df.columns else None
    actual_pits = len(driver_df[driver_df['PitInTime'].notna()])
    predicted_pits = driver_df['PredictedPit'].sum() if 'PredictedPit' in driver_df.columns else 0
    
    st.markdown(f"""
    <div class="explanation-card">
        <h4 style="margin-top:0;">Why {driver}'s prediction?</h4>
        <ul>
            <li><strong>Expected Finish:</strong> {int(final_pos) if final_pos else 'N/A'}th place</li>
            <li><strong>Average Lap Time:</strong> {avg_time:.2f}s per lap</li>
            <li><strong>Tyre Change Stops:</strong> {actual_pits} actual, {predicted_pits} predicted</li>
    """, unsafe_allow_html=True)
    
    if 'PitProbability' in driver_df.columns:
        avg_prob = driver_df['PitProbability'].mean()
        if avg_prob > 0.6:
            st.markdown("<li><strong>Strategy:</strong> Likely to make a tyre change soon</li>")
        elif avg_prob < 0.3:
            st.markdown("<li><strong>Strategy:</strong> Can run longer before tyre change</li>")
        else:
            st.markdown("<li><strong>Strategy:</strong> Tyre change timing is balanced</li>")
    
    st.markdown("</ul></div>", unsafe_allow_html=True)


def render_strategy_simulator():
    st.markdown("### Strategy Simulator")
    st.markdown("Adjust strategy parameters to see how they might affect the race:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox("Starting Tyre", ["Soft", "Medium", "Hard"], index=1)
    
    with col2:
        st.selectbox("Tyre Change Strategy", ["One Stop", "Two Stop", "Undercut", "Overcut"])
    
    with col3:
        st.selectbox("Fuel Load", ["Full", "Light"])
    
    st.info("Strategy simulation is coming soon! This feature will let you test different race strategies.")


def render_predictions_tab(df, model, config, threshold):
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Race Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            fig_pos = create_finishing_position_chart(df)
            if fig_pos:
                st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Drivers", len(df['Driver'].unique()) if 'Driver' in df.columns else 0)
            if 'Position' in df.columns:
                st.metric("Race Laps", int(df['LapNumber'].max()) if 'LapNumber' in df.columns else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'PredictedPit' in df.columns:
                st.metric("Predicted Tyre Changes", int(df['PredictedPit'].sum()))
            if 'PitInTime' in df.columns:
                actual_pits = df['PitInTime'].notna().sum()
                st.metric("Actual Tyre Changes", int(actual_pits))
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Position Timeline</div>', unsafe_allow_html=True)
        fig_timeline = create_position_timeline(df)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-header">Win Probability</div>', unsafe_allow_html=True)
        fig_win = create_win_probability_chart(df)
        if fig_win:
            st.plotly_chart(fig_win, use_container_width=True)


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
                if 'Position' in driver_df.columns:
                    st.metric("Starting Position", int(driver_df['Position'].iloc[0]))
                    st.metric("Final Position", int(driver_df['Position'].iloc[-1]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'LapTimeSeconds' in driver_df.columns:
                    st.metric("Avg Lap Time", f"{driver_df['LapTimeSeconds'].mean():.2f}s")
                    st.metric("Fastest Lap", f"{driver_df['LapTimeSeconds'].min():.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'PredictedPit' in driver_df.columns:
                    st.metric("Predicted Stops", int(driver_df['PredictedPit'].sum()))
                if 'PitInTime' in driver_df.columns:
                    actual = driver_df['PitInTime'].notna().sum()
                    st.metric("Actual Stops", int(actual))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'PitProbability' in driver_df.columns:
                    st.metric("Avg Pit Probability", f"{driver_df['PitProbability'].mean():.1%}")
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
        styled_df = create_styled_dataframe(driver_df[['LapNumber', 'Position', 'LapTimeSeconds', 
                                                        'Compound', 'TyreLife', 'PitProbability', 
                                                        'PredictedPit']])
        if styled_df is not None:
            st.dataframe(styled_df, use_container_width=True)


def main():
    st.markdown('<div class="main-title">F1 Strategy Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine Learning Race Predictions & Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
    
    with st.container():
        render_beginner_explanation()
    
    st.markdown('<div class="spacer-lg"></div>', unsafe_allow_html=True)
    
    model, config = get_trained_model()
    
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Race Selection</div>', unsafe_allow_html=True)
        
        year = st.selectbox("Year", [2023, 2022, 2021], index=0, label_visibility="collapsed")
        
        races = get_available_races_cached(year)
        gp = st.selectbox("Grand Prix", races[:15] if len(races) > 15 else races, label_visibility="collapsed")
        
        st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-header">Prediction Settings</div>', unsafe_allow_html=True)
        threshold = st.slider("Tyre Change Sensitivity", 0.1, 0.9, 0.5, 0.05, 
                             help="Lower values predict more tyre changes, higher values predict fewer")
        
        st.markdown('<div class="spacer-md"></div></div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("Loading race data..."):
            df = load_race_data_cached(year, gp)
        
        if df.empty:
            st.warning("No data available for this race. Try selecting a different Grand Prix.")
            return
        
        drivers = sorted(df['Driver'].unique())
        selected_driver = st.selectbox("Select a driver to analyze:", drivers, index=0)
        
        df_pred = get_model_predictions(df, threshold, model, config)
        
        tab1, tab2, tab3 = st.tabs(["Predictions", "Explanation", "Strategy Simulator"])
        
        with tab1:
            render_predictions_tab(df_pred, model, config, threshold)
        
        with tab2:
            render_explanation_tab(df_pred, selected_driver)
        
        with tab3:
            render_strategy_simulator()
        
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