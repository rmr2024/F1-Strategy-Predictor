import os
import pandas as pd
import numpy as np
import fastf1
import tempfile

DATA_DIR = "data"
YEARS = [2021, 2022, 2023]
RACES_PER_YEAR = 20
MAX_TRAINING_RACES = 3

_cache_enabled = False
_temp_cache_dir = None


def get_temp_cache_dir() -> str:
    """Get or create a temporary cache directory"""
    global _temp_cache_dir
    if _temp_cache_dir is None:
        _temp_cache_dir = tempfile.mkdtemp(prefix="fastf1_cache_")
    return _temp_cache_dir


def enable_caching(cache_dir: str = None) -> None:
    """Ensure FastF1 caching is enabled (only once)"""
    global _cache_enabled
    if _cache_enabled:
        return
    
    if cache_dir is None:
        cache_dir = os.environ.get("FASTF1_CACHE_DIR", get_temp_cache_dir())
    
    os.makedirs(cache_dir, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(cache_dir)
        _cache_enabled = True
    except Exception as e:
        print(f"Warning: Could not enable cache: {e}")


def load_race_data(year: int, gp: str, session_type: str = "R", 
                   load_telemetry: bool = False, load_weather: bool = False) -> pd.DataFrame:
    """Load a single race session with minimal data loading"""
    session = fastf1.get_session(year, gp, session_type)
    load_params = {"laps": True, "telemetry": load_telemetry, "weather": load_weather}
    session.load(**load_params)

    laps = session.laps

    df = laps[[
        'Driver', 'LapTime', 'LapNumber', 'Compound', 'TyreLife',
        'Stint', 'TrackStatus', 'PitOutTime', 'PitInTime',
        'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Team'
    ]].copy()

    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    df['Year'] = year
    df['GrandPrix'] = gp

    return df


def get_circuit_info(year: int, gp: str) -> dict:
    """Get circuit information including corners and rotation"""
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load(telemetry=False, laps=False, weather=False)
        circuit = session.get_circuit_info()
        
        return {
            'corners': circuit.corners.to_dict('records') if circuit.corners is not None else [],
            'rotation': circuit.rotation if circuit.rotation else 0,
            'location': session.event['Location'],
            'name': session.event['Name']
        }
    except Exception as e:
        print(f"Error getting circuit info: {e}")
        return {'corners': [], 'rotation': 0, 'location': '', 'name': gp}


def get_track_coordinates(year: int, gp: str) -> list:
    """Get actual track coordinates from fastest lap telemetry"""
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load(telemetry=True, laps=True, weather=False)
        
        lap = session.laps.pick_fastest()
        if lap is None:
            return []
        
        pos = lap.get_pos_data()
        if pos.empty or 'X' not in pos.columns or 'Y' not in pos.columns:
            return []
        
        track = pos[['X', 'Y']].dropna().values
        
        circuit = session.get_circuit_info()
        track_angle = circuit.rotation / 180 * np.pi if circuit.rotation else 0
        
        if track_angle != 0:
            cos_a = np.cos(track_angle)
            sin_a = np.sin(track_angle)
            rotated = np.zeros_like(track)
            rotated[:, 0] = track[:, 0] * cos_a - track[:, 1] * sin_a
            rotated[:, 1] = track[:, 0] * sin_a + track[:, 1] * cos_a
            track = rotated
        
        track_list = track.tolist()
        
        return [[t[0], 0, t[1]] for t in track_list]
    except Exception as e:
        print(f"Error getting track coordinates: {e}")
        return []


def load_season_data(years: list = None, max_races_per_year: int = None) -> pd.DataFrame:
    """Load limited seasons data for faster training"""
    if years is None:
        years = [2022]
    
    if max_races_per_year is None:
        max_races_per_year = MAX_TRAINING_RACES
    
    enable_caching()

    all_data = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races_processed = 0

            for _, event in schedule.iterrows():
                if races_processed >= max_races_per_year:
                    break
                try:
                    gp = event['EventName']
                    event_format = event.get('EventFormat', 'conventional')
                    if event_format in ['conventional', 'sprint', 'sprint_shootout']:
                        df = load_race_data(year, gp, load_telemetry=False, load_weather=False)
                        all_data.append(df)
                        races_processed += 1
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
