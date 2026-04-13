import os
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_data(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_pickle(path)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def format_time(seconds: float) -> str:
    if pd.isna(seconds) or seconds <= 0:
        return "--:--.---"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{mins}:{secs:02d}.{ms:03d}"


def get_driver_color(driver: str) -> str:
    colors = {
        'VER': '#3671C6', 'NOR': '#FF8700', 'LEC': '#E8002D', 'HAM': '#00D2BE',
        'RUS': '#00D2BE', 'ALO': '#009E9E', 'BOT': '#9C0000', 'GAS': '#6CD3BF',
        'TSU': '#2B4562', 'ZHO': '#A50044', 'MAG': '#CAC6DA', 'ALB': '#FFFFFF',
        'STR': '#15A2F6', 'POB': '#FFC300', 'RAI': '#229090', 'KVY': '#7C0200'
    }
    return colors.get(driver, '#888888')


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b != 0 else default
    except (TypeError, ZeroDivisionError):
        return default