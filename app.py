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

st.set_page_config(page_title="F1-DRS | Race Strategy", layout="wide", page_icon="🏎️")

os.environ["FASTF1_CACHE_DIR"] = "/tmp/fastf1_cache"

COLORS = {
    'bg': '#0E1117',
    'card': '#1E1E1E',
    'text': '#FAFAFA',
    'text_muted': '#A0A0A0',
    'accent_red': '#E10600',
    'accent_blue': '#3671C6',
    'accent_green': '#00D2BE',
    'grid': '#333333'
}

st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLORS['bg']}; }}
    .stSelectbox label, .stSlider label, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 
    {{ color: {COLORS['text']} !important; }}
    .css-1d391kg {{ padding-top: 1rem; }}
    div[data-testid="stMetricValue"] {{ color: {COLORS['text']} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {COLORS['text_muted']} !important; }}
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
            name='Actual Pit',
            marker=dict(color=COLORS['accent_red'], size=12, symbol='x')
        ))
    
    pred_pits = driver_df[driver_df['PredictedPit'] == 1]['LapNumber'].tolist()
    if pred_pits:
        fig.add_trace(go.Scatter(
            x=pred_pits,
            y=driver_df[driver_df['LapNumber'].isin(pred_pits)]['LapTimeSeconds'],
            mode='markers',
            name='Predicted Pit',
            marker=dict(color=COLORS['accent_green'], size=10, symbol='circle-open', line=dict(width=2))
        ))
    
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family="Roboto, sans-serif"),
        xaxis=dict(
            title="Lap Number",
            gridcolor=COLORS['grid'],
            showgrid=True
        ),
        yaxis=dict(
            title="Lap Time (s)",
            gridcolor=COLORS['grid'],
            showgrid=True
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    return fig


def create_pit_probability_chart(df, driver):
    driver_df = df[df['Driver'] == driver].copy()
    if driver_df.empty or 'PitProbability' not in driver_df.columns:
        return None
    
    driver_df = driver_df.sort_values('LapNumber')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=driver_df['LapNumber'],
        y=driver_df['PitProbability'],
        mode='lines',
        name='Pit Probability',
        fill='tozeroy',
        line=dict(color=COLORS['accent_blue'], width=2),
        fillcolor=f"rgba(54, 113, 198, 0.2)"
    ))
    
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


def main():
    st.title("F1-DRS")
    st.markdown("### Race Strategy Predictor")
    
    model, config = get_trained_model()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        year = st.selectbox("Year", [2023, 2022, 2021], index=0)
    with col2:
        races = get_available_races_cached(year)
        gp = st.selectbox("Grand Prix", races[:10] if len(races) > 10 else races)
    with col3:
        threshold = st.slider("Pit Threshold", 0.1, 0.9, 0.5, 0.05)
    
    try:
        df = load_race_data_cached(year, gp)
        
        if df.empty:
            st.warning("No data available for this race.")
            return
        
        drivers = sorted(df['Driver'].unique())
        selected_driver = st.selectbox("Select Driver", drivers)
        
        df_pred = get_model_predictions(df, threshold, model, config)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            fig1 = create_lap_time_chart(df_pred, selected_driver)
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
        
        with col_chart2:
            st.markdown(f"**{selected_driver} Strategy**")
            driver_df = df_pred[df_pred['Driver'] == selected_driver].sort_values('LapNumber')
            
            stint_info = driver_df.groupby('Stint').agg({
                'Compound': 'first',
                'LapNumber': ['min', 'max'],
                'TyreLife': 'max'
            }).reset_index()
            stint_info.columns = ['Stint', 'Compound', 'StartLap', 'EndLap', 'MaxTyreLife']
            
            for _, row in stint_info.iterrows():
                st.markdown(f"**Stint {int(row['Stint'])}**: {row['Compound']} | Laps {int(row['StartLap'])}-{int(row['EndLap'])}")
            
            actual_pits = driver_df[driver_df['PitInTime'].notna()]['LapNumber'].tolist()
            if actual_pits:
                st.markdown(f"**Actual Pit Laps**: {actual_pits}")
            
            pred_pits = driver_df[driver_df['PredictedPit'] == 1]['LapNumber'].tolist()
            if pred_pits:
                st.markdown(f"**Predicted Pit Laps**: {pred_pits}")
        
        st.markdown("---")
        
        st.markdown("### Pit Probability")
        fig2 = create_pit_probability_chart(df_pred, selected_driver)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        with st.expander("Model Details"):
            if model is not None and config is not None:
                st.write(f"**Model Type**: {config.get('model_type', 'XGBoost')}")
                st.write(f"**Default Threshold**: {config.get('threshold', 0.5):.3f}")
                st.write(f"**Features**: {config.get('feature_cols', [])}")
            else:
                st.write("Model not available")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure fastf1 cache is set up. Run once with internet to download data.")


if __name__ == "__main__":
    main()