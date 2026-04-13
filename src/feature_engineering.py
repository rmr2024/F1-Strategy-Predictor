import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_pace_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Driver', 'LapNumber']).copy()
    df['LapTimeMedian'] = df.groupby(['Driver', 'GrandPrix'])['LapTimeSeconds'].transform('median')
    df['PaceDelta'] = df['LapTimeSeconds'] - df['LapTimeMedian']
    return df


def create_degradation_slope(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(['Driver', 'GrandPrix', 'LapNumber']).copy()
    df['DegSlope'] = df.groupby(['Driver', 'GrandPrix'])['LapTimeSeconds'].transform(
        lambda x: x.rolling(window, min_periods=1).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=True
        )
    )
    return df


def create_fuel_load(df: pd.DataFrame, total_laps: int = 60) -> pd.DataFrame:
    df = df.sort_values(['Driver', 'LapNumber']).copy()
    df['FuelLoad'] = 1 - (df['LapNumber'] / total_laps)
    return df


def create_stint_progress(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Driver', 'GrandPrix', 'LapNumber']).copy()
    df['StintProgress'] = df.groupby(['Driver', 'GrandPrix', 'Stint'])['LapNumber'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1)
    )
    return df


def create_pit_next_lap_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Driver', 'LapNumber']).copy()
    df['PitNextLap'] = df.groupby(['Driver', 'GrandPrix'])['PitInTime'].transform(
        lambda x: x.notna().astype(int).shift(-1).fillna(0).astype(int)
    )
    return df


def encode_categorical(df: pd.DataFrame, encoders: dict = None) -> tuple:
    if encoders is None:
        encoders = {}
    categorical_cols = ['Compound', 'Team', 'Driver']
    for col in categorical_cols:
        if col in df.columns:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                df[col + '_encoded'] = encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col + '_encoded'] = encoders[col].transform(df[col].astype(str))
    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_pace_delta(df)
    df = create_degradation_slope(df)
    df = create_fuel_load(df)
    df = create_stint_progress(df)
    df = create_pit_next_lap_target(df)
    df, _ = encode_categorical(df)
    return df


def get_feature_columns() -> list:
    return [
        'LapNumber', 'TyreLife', 'Stint', 'TrackStatus',
        'LapTimeSeconds', 'PaceDelta', 'DegSlope', 'FuelLoad', 'StintProgress',
        'Compound_encoded', 'Team_encoded'
    ]


def prepare_training_data(df: pd.DataFrame) -> tuple:
    df = engineer_features(df)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df['PitNextLap']
    return X, y, available