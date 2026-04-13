import pandas as pd
import numpy as np
import joblib
from src.feature_engineering import engineer_features, get_feature_columns

MODEL_DIR = "models"


def load_model():
    model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    config = joblib.load(f"{MODEL_DIR}/model_config.pkl")
    return model, config


def predict_pit_stops(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    model, config = load_model()
    if threshold is None:
        threshold = config['threshold']
    
    df = engineer_features(df)
    feature_cols = config['feature_cols']
    
    X = df[feature_cols].fillna(0)
    df['PitProbability'] = model.predict_proba(X)[:, 1]
    df['PredictedPit'] = (df['PitProbability'] >= threshold).astype(int)
    
    return df


def get_pit_windows(df: pd.DataFrame, threshold: float = 0.5) -> list:
    df = predict_pit_stops(df, threshold)
    pit_laps = df[df['PredictedPit'] == 1][['LapNumber', 'PitProbability', 'Driver']].to_dict('records')
    return pit_laps


def explain_prediction(df: pd.DataFrame, lap_idx: int) -> dict:
    model, config = load_model()
    feature_cols = config['feature_cols']
    
    X = df[feature_cols].fillna(0)
    row = X.iloc[[lap_idx]]
    
    exp = model.get_booster().trees_to_dataframe()
    tree = model.get_booster()
    
    importances = dict(zip(feature_cols, tree.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {'top_features': sorted_imp}