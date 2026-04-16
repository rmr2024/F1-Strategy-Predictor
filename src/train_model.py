import pandas as pd
import numpy as np
import sys
import os
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import engineer_features, get_feature_columns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def train_model(df: pd.DataFrame) -> Tuple[object, dict]:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        raise ValueError("Training data is empty or None")
    
    df = engineer_features(df)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) == 0:
        raise ValueError("No features available for training")
    
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['PitNextLap'].fillna(0).astype(int)
    
    valid_idx = ~(X.isna().any(axis=1) | X.isnull().any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(y.unique()) < 2:
        raise ValueError("Training data must contain both classes (pit and no-pit)")
    
    n_pits = y.sum()
    n_non_pits = len(y) - n_pits
    scale_pos = n_non_pits / n_pits if n_pits > 0 else 1.0
    
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    elif HAS_SKLEARN:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    else:
        raise ImportError("Neither xgboost nor sklearn is available")
    
    model.fit(X, y)
    
    default_threshold = 0.5
    
    config = {
        'threshold': default_threshold,
        'feature_cols': available,
        'scale_pos_weight': scale_pos,
        'model_type': 'XGBoost' if HAS_XGB else 'Sklearn'
    }
    
    return model, config


def get_cached_model(_df_for_hash: pd.DataFrame) -> Tuple[object, dict]:
    if _df_for_hash is None or (_df_for_hash is not None and isinstance(_df_for_hash, pd.DataFrame) and _df_for_hash.empty):
        raise ValueError("Training data is empty")
    return train_model(_df_for_hash)


def predict_pit(model: object, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    if model is None:
        raise ValueError("Model is required for prediction")
    
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


if __name__ == "__main__":
    from src.data_loader import load_season_data
    
    print("Loading training data (limited to 2 races)...")
    df = load_season_data(years=[2022], max_races_per_year=2)
    
    if df.empty:
        print("No data loaded. Check your internet connection and FastF1 availability.")
        sys.exit(1)
    
    print(f"Loaded {len(df)} lap records")
    
    print("Training model...")
    model, config = train_model(df)
    
    print(f"Model type: {config['model_type']}")
    print(f"Features: {config['feature_cols']}")
    print("Training complete!")