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
    'risky': '#FFD700'
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
    .stApp {{ background-color: {COLORS['bg']}; }}
    .stSelectbox label, .stSlider label, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 
    {{ color: {COLORS['text']} !important; }}
    div[data-testid="stMetricValue"] {{ color: {COLORS['text']} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {COLORS['text_muted']} !important; }}
    .css-1d391kg {{ padding-top: 1rem; }}
    .feature-card {{
        background-color: {COLORS['card']};
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }}
    .strategy-block {{
        display: inline-block;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 5px;
        font-weight: bold;
        text-align: center;
    }}
    .beginner-box {{
        background-color: {COLORS['card_light']};
        border-left: 4px solid {COLORS['accent_blue']};
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    .explanation-card {{
        background-color: {COLORS['card']};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
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
            name='Actual Pit Stop',
            marker=dict(color=COLORS['accent_red'], size=12, symbol='x')
        ))
    
    pred_pits = driver_df[driver_df['PredictedPit'] == 1]['LapNumber'].tolist()
    if pred_pits:
        fig.add_trace(go.Scatter(
            x=pred_pits,
            y=driver_df[driver_df['LapNumber'].isin(pred_pits)]['LapTimeSeconds'],
            mode='markers',
            name='Predicted Pit Stop',
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
        name='Pit Stop Probability',
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
            <li><strong>What are pit stops?</strong> - When drivers come in for tyre changes. A good strategy can mean the difference between winning and losing.</li>
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
            st.markdown("<li><strong>Strategy:</strong> Likely to pit early due to tyre wear</li>")
        elif avg_prob < 0.3:
            st.markdown("<li><strong>Strategy:</strong> Can extend tyre stint, fewer pit stops needed</li>")
        else:
            st.markdown("<li><strong>Strategy:</strong> Balanced approach expected</li>")
    
    st.markdown("</ul></div>", unsafe_allow_html=True)


def render_strategy_simulator():
    st.markdown("### Strategy Simulator")
    st.markdown("Adjust strategy parameters to see how they might affect the race:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox("Starting Tyre", ["Soft", "Medium", "Hard"], index=1)
    
    with col2:
        st.selectbox("Pit Stop Strategy", ["One Stop", "Two Stop", "Undercut", "Overcut"])
    
    with col3:
        st.selectbox("Fuel Load", ["Full", "Light"])
    
    st.info("Strategy simulation is coming soon! This feature will let you test different race strategies.")


def render_predictions_tab(df, model, config, threshold):
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        fig_pos = create_finishing_position_chart(df)
        if fig_pos:
            st.plotly_chart(fig_pos, use_container_width=True)
    
    with col2:
        st.metric("Total Drivers", len(df['Driver'].unique()) if 'Driver' in df.columns else 0)
        if 'Position' in df.columns:
            st.metric("Race Laps", int(df['LapNumber'].max()) if 'LapNumber' in df.columns else 0)
    
    with col3:
        if 'PredictedPit' in df.columns:
            st.metric("Predicted Pit Stops", int(df['PredictedPit'].sum()))
        if 'PitInTime' in df.columns:
            actual_pits = df['PitInTime'].notna().sum()
            st.metric("Actual Pit Stops", int(actual_pits))
    
    st.markdown("---")
    
    fig_timeline = create_position_timeline(df)
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown("---")
    
    fig_win = create_win_probability_chart(df)
    if fig_win:
        st.plotly_chart(fig_win, use_container_width=True)


def render_explanation_tab(df, driver):
    render_driver_explanation(df, driver)
    
    st.markdown("---")
    st.markdown("### Driver Performance Details")
    
    driver_df = df[df['Driver'] == driver].copy()
    if not driver_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Position' in driver_df.columns:
                st.metric("Starting Position", int(driver_df['Position'].iloc[0]))
                st.metric("Final Position", int(driver_df['Position'].iloc[-1]))
        
        with col2:
            if 'LapTimeSeconds' in driver_df.columns:
                st.metric("Avg Lap Time", f"{driver_df['LapTimeSeconds'].mean():.2f}s")
                st.metric("Fastest Lap", f"{driver_df['LapTimeSeconds'].min():.2f}s")
        
        with col3:
            if 'PredictedPit' in driver_df.columns:
                st.metric("Predicted Stops", int(driver_df['PredictedPit'].sum()))
            if 'PitInTime' in driver_df.columns:
                actual = driver_df['PitInTime'].notna().sum()
                st.metric("Actual Stops", int(actual))
        
        with col4:
            if 'PitProbability' in driver_df.columns:
                st.metric("Avg Pit Probability", f"{driver_df['PitProbability'].mean():.1%}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_lap = create_lap_time_chart(df, driver)
            if fig_lap:
                st.plotly_chart(fig_lap, use_container_width=True)
        
        with col_chart2:
            fig_prob = create_pit_probability_chart(df, driver)
            if fig_prob:
                st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Tyre Strategy Timeline")
        fig_strategy = create_strategy_timeline(df, driver)
        if fig_strategy:
            st.plotly_chart(fig_strategy, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Lap-by-Lap Data")
        styled_df = create_styled_dataframe(driver_df[['LapNumber', 'Position', 'LapTimeSeconds', 
                                                        'Compound', 'TyreLife', 'PitProbability', 
                                                        'PredictedPit']])
        if styled_df is not None:
            st.dataframe(styled_df, use_container_width=True)


def main():
    st.title("F1 Strategy Predictor")
    st.markdown("### Machine Learning Race Predictions")
    
    st.markdown("---")
    render_beginner_explanation()
    st.markdown("---")
    
    model, config = get_trained_model()
    
    with st.sidebar:
        st.header("Race Selection")
        
        year = st.selectbox("Year", [2023, 2022, 2021], index=0)
        
        races = get_available_races_cached(year)
        gp = st.selectbox("Grand Prix", races[:15] if len(races) > 15 else races)
        
        st.header("Prediction Settings")
        threshold = st.slider("Pit Stop Sensitivity", 0.1, 0.9, 0.5, 0.05, 
                             help="Lower values predict more pit stops, higher values predict fewer")
        
        st.header("Driver Selection")
    
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