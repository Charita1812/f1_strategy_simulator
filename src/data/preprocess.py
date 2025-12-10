# src/data/preprocess.py
import pandas as pd
import numpy as np
from .load_data import load_all

def time_to_ms(time_str):
    if pd.isna(time_str): return np.nan
    try:
        s = str(time_str).strip()
        if s.isdigit():
            return int(s)
        parts = s.split(':')
        if len(parts) == 1:
            return int(round(float(parts[0]) * 1000))
        elif len(parts) == 2:
            mins = int(parts[0]); secs = float(parts[1])
            return int(round((mins*60 + secs)*1000))
        else:
            hrs = int(parts[0]); mins = int(parts[1]); secs = float(parts[2])
            return int(round((hrs*3600 + mins*60 + secs)*1000))
    except:
        return np.nan

def preprocess_all():
    dfs = load_all()
    lap_times = dfs['lap_times']
    pit_stops = dfs['pit_stops']
    races = dfs['races']
    drivers = dfs['drivers']
    constructors = dfs['constructors']
    results = dfs['results']

    # convert lap time strings to ms
    if 'milliseconds' not in lap_times.columns:
        lap_times['milliseconds'] = lap_times['time'].apply(time_to_ms)
    lap_times = lap_times[lap_times['milliseconds'].notna()]
    lap_times['lap_time_ms'] = lap_times['milliseconds']

    # make basic merges - keep it minimal here
    # ... (for full logic use the data_pipeline.py implementation)
    return {
        "lap_times": lap_times,
        "pit_stops": pit_stops,
        "races": races,
        "drivers": drivers,
        "constructors": constructors,
        "results": results
    }
