import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from src.data_loader import load_season_data
from src.feature_engineering import prepare_training_data, get_feature_columns
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_training_data(years: list = [2021, 2022]) -> pd.DataFrame:
    cache_path = f"{MODEL_DIR}/training_data_cache.pkl"
    if os.path.exists(cache_path):
        print("Loading cached training data...")
        return pd.read_pickle(cache_path)
    print("Loading and engineering features...")
    df = load_season_data(years)
    df = prepare_training_data(df)
    df.to_pickle(cache_path)
    return df


def calculate_class_weight(df: pd.DataFrame) -> float:
    y = df['PitNextLap']
    n_pits = y.sum()
    n_non_pits = len(y) - n_pits
    return n_non_pits / n_pits if n_pits > 0 else 1.0


def tune_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def train_model(train_years: list = [2021, 2022], test_year: int = 2023):
    df = load_training_data(train_years)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    
    X = df[available].fillna(0)
    y = df['PitNextLap']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_pos = calculate_class_weight(df)
    print(f"Class imbalance ratio: {scale_pos:.2f}")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred = model.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, y_pred))
    
    threshold = tune_threshold(model, X_val, y_val)
    print(f"Optimal threshold: {threshold:.3f}")
    
    model_path = f"{MODEL_DIR}/xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    joblib.dump({'threshold': threshold, 'feature_cols': available}, f"{MODEL_DIR}/model_config.pkl")
    print(f"Config saved to {MODEL_DIR}/model_config.pkl")
    
    return model, threshold


if __name__ == "__main__":
    train_model()