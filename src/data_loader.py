import os
import pandas as pd
import fastf1

DATA_DIR = "data"
YEARS = [2021, 2022, 2023]
RACES_PER_YEAR = 20

_cache_enabled = False


def enable_caching(cache_dir: str = None) -> None:
    """Ensure FastF1 caching is enabled (only once)"""
    global _cache_enabled
    if _cache_enabled:
        return
    
    if cache_dir is None:
        cache_dir = os.environ.get("FASTF1_CACHE_DIR", "/tmp")
    
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    _cache_enabled = True


def load_race_data(year: int, gp: str, session_type: str = "R") -> pd.DataFrame:
    """Load a single race session"""
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    laps = session.laps

    df = laps[[
        'Driver', 'LapTime', 'LapNumber', 'Compound', 'TyreLife',
        'Stint', 'TrackStatus', 'PitOutTime', 'PitInTime',
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Team'
    ]].copy()

    # Convert lap time to seconds
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    df['Year'] = year
    df['GrandPrix'] = gp

    return df


def load_season_data(years: list = YEARS) -> pd.DataFrame:
    """Load multiple seasons"""
    enable_caching()

    all_data = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)

            for _, event in schedule.iterrows():
                try:
                    gp = event['EventName']
                    df = load_race_data(year, gp)
                    all_data.append(df)
                    print(f"Loaded {year} {gp}")

                except Exception as e:
                    print(f"Skipped {year} {gp}: {e}")

        except Exception as e:
            print(f"Schedule error {year}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def get_available_races(year: int) -> list:
    """Return list of races for a given year"""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule['EventName'].tolist()
    except Exception:
        return []


def load_single_race(year: int, gp: str, session_type: str = "R") -> pd.DataFrame:
    """Load one race (used by Streamlit app)"""
    enable_caching()
    return load_race_data(year, gp, session_type)
