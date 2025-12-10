# src/data/load_data.py
import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'raw')
RAW_DIR = os.path.abspath(RAW_DIR)

def load_csv(name):
    path = os.path.join(RAW_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put CSVs in data/raw/")
    return pd.read_csv(path, low_memory=False)

def load_all():
    files = {
        "lap_times": "lap_times.csv",
        "pit_stops": "pit_stops.csv",
        "races": "races.csv",
        "drivers": "drivers.csv",
        "constructors": "constructors.csv",
        "results": "results.csv"
    }
    dfs = {}
    for k,v in files.items():
        dfs[k] = load_csv(v)
    return dfs
