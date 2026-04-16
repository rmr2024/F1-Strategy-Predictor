import pandas as pd
import numpy as np
from typing import Optional
from src.feature_engineering import engineer_features, get_feature_columns


def predict_pit(model: object, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    if model is None:
        df = df.copy()
        df['PitProbability'] = 0.5
        df['PredictedPit'] = 0
        return df
    
    df = df.copy()
    
    try:
        df = engineer_features(df)
    except Exception:
        pass
    
    feature_cols = config.get('feature_cols', get_feature_columns()) if config else get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) == 0:
        df['PitProbability'] = 0.5
        df['PredictedPit'] = 0
        return df
    
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    
    try:
        df['PitProbability'] = model.predict_proba(X)[:, 1]
    except Exception:
        df['PitProbability'] = 0.5
    
    threshold = config.get('threshold', 0.5) if config else 0.5
    df['PredictedPit'] = (df['PitProbability'] >= threshold).astype(int)
    
    return df


def predict_pit_stops(df: pd.DataFrame, threshold: float = None, model=None, config=None) -> pd.DataFrame:
    if model is None or config is None:
        from src.train_model import get_cached_model, load_season_data
        try:
            training_df = load_season_data(years=[2022], max_races_per_year=2)
            if training_df.empty:
                df = df.copy()
                df['PitProbability'] = 0.5
                df['PredictedPit'] = 0
                return df
            model, config = get_cached_model(training_df)
        except Exception:
            df = df.copy()
            df['PitProbability'] = 0.5
            df['PredictedPit'] = 0
            return df
    
    if threshold is not None:
        config = config.copy()
        config['threshold'] = threshold
    
    return predict_pit(model, df, config)


def get_pit_windows(df: pd.DataFrame, threshold: float = 0.5, model=None, config=None) -> list:
    df = predict_pit_stops(df, threshold, model, config)
    pit_laps = df[df['PredictedPit'] == 1][['LapNumber', 'PitProbability', 'Driver']].to_dict('records')
    return pit_laps


def explain_prediction(df: pd.DataFrame, lap_idx: int, model=None, config=None) -> dict:
    if model is None or config is None:
        from src.train_model import get_cached_model, load_season_data
        try:
            training_df = load_season_data(years=[2022], max_races_per_year=2)
            if training_df.empty:
                return {'top_features': []}
            model, config = get_cached_model(training_df)
        except Exception:
            return {'top_features': []}
    
    feature_cols = config.get('feature_cols', get_feature_columns())
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) == 0:
        return {'top_features': []}
    
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    
    try:
        tree = model.get_booster()
        importances = dict(zip(available, tree.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        return {'top_features': sorted_imp}
    except Exception:
        return {'top_features': []}