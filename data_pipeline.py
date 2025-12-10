# data_pipeline.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---- CONFIG ----
RAW_DIR = "data/raw"            # place your downloaded CSVs here
PROC_DIR = "data/processed"     # outputs saved here
os.makedirs(PROC_DIR, exist_ok=True)

# Filenames (update if your csv names differ)
FILES = {
    "lap_times": "lap_times.csv",
    "pit_stops": "pit_stops.csv",
    "races": "races.csv",
    "drivers": "drivers.csv",
    "constructors": "constructors.csv",
    "results": "results.csv"
}

# ---- UTILS ----
def safe_read_csv(path, **kwargs):
    print(f"Loading {path} ...")
    return pd.read_csv(path, low_memory=False, **kwargs)

def time_to_ms(time_str):
    """
    Convert a time string 'M:SS.mmm' or 'MM:SS.mmm' to milliseconds.
    If input is numeric, return as-is.
    """
    if pd.isna(time_str):
        return np.nan
    if isinstance(time_str, (int, float, np.integer, np.floating)):
        return int(time_str)
    s = str(time_str).strip()
    # If already milliseconds (only digits)
    if s.isdigit():
        return int(s)
    # Format examples: '1:23.456' or '83.456' or '1:02:345' (unlikely)
    parts = s.split(':')
    try:
        if len(parts) == 1:
            # seconds.millis
            secs = float(parts[0])
            return int(round(secs * 1000))
        elif len(parts) == 2:
            mins = int(parts[0])
            secs = float(parts[1])
            return int(round((mins * 60 + secs) * 1000))
        else:
            # hours:mins:secs
            hrs = int(parts[0]); mins = int(parts[1]); secs = float(parts[2])
            return int(round((hrs*3600 + mins*60 + secs)*1000))
    except Exception:
        return np.nan

# ---- LOAD RAW FILES ----
lap_times = safe_read_csv(os.path.join(RAW_DIR, FILES["lap_times"]))
pit_stops = safe_read_csv(os.path.join(RAW_DIR, FILES["pit_stops"]))
races = safe_read_csv(os.path.join(RAW_DIR, FILES["races"]))
drivers = safe_read_csv(os.path.join(RAW_DIR, FILES["drivers"]))
constructors = safe_read_csv(os.path.join(RAW_DIR, FILES["constructors"]))
results = safe_read_csv(os.path.join(RAW_DIR, FILES["results"]))

# ---- NORMALIZE lap_times ----
# Expected columns (common Kaggle/Egrast): raceId, driverId, lap, position, time, milliseconds
print("Preprocessing lap_times ...")
if 'milliseconds' not in lap_times.columns:
    # convert time strings to milliseconds
    lap_times['milliseconds'] = lap_times['time'].apply(time_to_ms)

lap_times['lap'] = lap_times['lap'].astype(int)
lap_times['raceId'] = lap_times['raceId'].astype(int)
lap_times['driverId'] = lap_times['driverId'].astype(int)
lap_times['lap_time_ms'] = lap_times['milliseconds']
lap_times['lap_time_s'] = lap_times['lap_time_ms'] / 1000.0

# drop obviously bad rows
lap_times = lap_times[lap_times['lap_time_ms'].notna()]

# ---- NORMALIZE pit_stops ----
# Expected: raceId, driverId, stop (stop number), lap, time, duration, milliseconds, tyre
print("Preprocessing pit_stops ...")
if 'milliseconds' not in pit_stops.columns:
    if 'duration' in pit_stops.columns:
        # duration might be 'xx.xxx' string seconds -> convert
        def dur_to_ms(x):
            if pd.isna(x): return np.nan
            try:
                return int(round(float(x) * 1000))
            except:
                return np.nan
        pit_stops['milliseconds'] = pit_stops['duration'].apply(dur_to_ms)
    else:
        pit_stops['milliseconds'] = np.nan

# Ensure types
pit_stops['raceId'] = pit_stops['raceId'].astype(int)
pit_stops['driverId'] = pit_stops['driverId'].astype(int)
pit_stops['lap'] = pit_stops['lap'].astype(int)

# normalize tyre column if exists (some datasets call it 'tyre' or 'tyres')
tyre_cols = [c for c in pit_stops.columns if 'tyr' in c.lower() or 'tyre' in c.lower() or 'tyres' in c.lower()]
if len(tyre_cols) == 0:
    pit_stops['compound'] = np.nan
else:
    pit_stops['compound'] = pit_stops[tyre_cols[0]].astype(str).str.upper()

pit_stops['pit_duration_ms'] = pit_stops['milliseconds']

# ---- JOIN lap_times with races/drivers/constructors ----
print("Merging metadata (races, drivers, constructors)...")
races_small = races[['raceId','year','round','name','date','circuitId']].copy()
drivers_small = drivers[['driverId','driverRef','code','forename','surname','nationality']].copy()
constructors_small = constructors[['constructorId','name','nationality']].copy()

lap = lap_times.merge(races_small, on='raceId', how='left')
lap = lap.merge(drivers_small, on='driverId', how='left')

# if results contains constructorId per driver, merge to know team in that race
if 'constructorId' in results.columns:
    results_small = results[['raceId','driverId','constructorId']].copy()
    lap = lap.merge(results_small, on=['raceId','driverId'], how='left')
    lap = lap.merge(constructors_small, on='constructorId', how='left', suffixes=('','_constructor'))
else:
    lap['constructorId'] = np.nan
    lap['name'] = lap.get('name_constructor', np.nan)

# ---- COMPUTE STINTS & TYRE AGE ----
# We'll determine stint_id by counting pit stops up to that lap (inclusive)
print("Computing stint_id and tyre_age per driver-race ...")
# Build a lookup of pit laps per (race,driver)
pit_lookup = pit_stops.groupby(['raceId','driverId'])['lap'].apply(list).to_dict()

def compute_stints_for_group(df):
    race = df['raceId'].iloc[0]
    driver = df['driverId'].iloc[0]
    pits = pit_lookup.get((race, driver), [])
    pits_sorted = sorted(pits)
    # We'll create an array of stint IDs: start at 1, increment when lap > pit_lap
    stint_ids = []
    tyre_ages = []
    current_stint = 1
    last_pit_lap = 0
    for lapnum in df['lap'].values:
        # if a pit happened on THIS lap, then that lap is the pit lap -> treat as end of previous stint
        # We'll assume tyre_age resets on next lap after the pit (common approach)
        if lapnum <= (pits_sorted[-1] if pits_sorted else 0):
            # find number of pits that have lap < current lap
            pits_before = [p for p in pits_sorted if p < lapnum]
            current_stint = len(pits_before) + 1
            last_pit = pits_before[-1] if pits_before else 0
            tyre_age = lapnum - last_pit
        else:
            # after last known pit
            current_stint = len(pits_sorted) + 1
            last_pit = pits_sorted[-1] if pits_sorted else 0
            tyre_age = lapnum - last_pit
        stint_ids.append(current_stint)
        tyre_ages.append(tyre_age)
    df = df.copy()
    df['stint_id'] = stint_ids
    df['tyre_age_laps'] = tyre_ages
    return df

lap = lap.sort_values(['raceId','driverId','lap'])
lap = lap.groupby(['raceId','driverId'], group_keys=False).apply(compute_stints_for_group)

# ---- ATTACH PIT INFO to laps (which laps had pit events, pit duration, compound switched to) ----
print("Attaching pit info to lap rows ...")
# Map pit stop info keyed by (raceId, driverId, lap)
pit_stops_indexed = pit_stops.set_index(['raceId','driverId','lap'])
def get_pit_info(row):
    key = (row['raceId'], row['driverId'], row['lap'])
    if key in pit_stops_indexed.index:
        r = pit_stops_indexed.loc[key]
        # if multiple stops same lap (rare), take first
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return pd.Series({
            'pit_happened': True,
            'pit_stop_number': r.get('stop', np.nan) if 'stop' in r.index else np.nan,
            'pit_duration_ms': r.get('pit_duration_ms', np.nan),
            'pit_compound': r.get('compound', np.nan)
        })
    else:
        return pd.Series({'pit_happened': False, 'pit_stop_number': np.nan, 'pit_duration_ms': np.nan, 'pit_compound': np.nan})

pit_info_df = lap.apply(get_pit_info, axis=1)
lap = pd.concat([lap.reset_index(drop=True), pit_info_df.reset_index(drop=True)], axis=1)

# ---- COMPUTE PIT DELTA (approx) ----
# pit_delta approximates the time lost due to a stop compared to the in/out lap baseline.
# Simple heuristic: pit delta = pit_duration_ms + (outsidelap_time - expected_lap_time)
# We'll compute median lap_time_ms for that driver on that stint before pit to estimate expected lap time.
print("Computing pit delta approximations ...")

# Prepare a helper to compute median of previous N laps for that driver-race before pit lap
def compute_pit_deltas(pit_df, lap_df, lookback=3):
    records = []
    grouped = pit_df.groupby(['raceId','driverId'])
    for (raceId, driverId), group in grouped:
        for _, pit in group.iterrows():
            lapnum = int(pit['lap'])
            # get preceding laps
            mask = (lap_df['raceId']==raceId) & (lap_df['driverId']==driverId) & (lap_df['lap'] < lapnum)
            prev_laps = lap_df.loc[mask].sort_values('lap', ascending=False).head(lookback)
            if len(prev_laps) == 0:
                continue
            expected_ms = prev_laps['lap_time_ms'].median()
            # out lap (lap before pit) and in lap (lap after pit)
            out_lap = prev_laps.iloc[0]['lap_time_ms']
            in_mask = (lap_df['raceId']==raceId) & (lap_df['driverId']==driverId) & (lap_df['lap'] == lapnum+1)
            in_lap_ms = lap_df.loc[in_mask, 'lap_time_ms'].squeeze() if not lap_df.loc[in_mask].empty else np.nan
            pit_duration_ms = pit.get('pit_duration_ms', np.nan)
            # observed pit delta rough estimate:
            # pit_delta = pit_duration + (in_lap_ms - expected_ms) + (out_lap - expected_ms)
            # simplified: pit_delta = pit_duration_ms + (in_lap_ms - out_lap)   (but both include traffic effects)
            pit_delta = np.nan
            if not np.isnan(pit_duration_ms):
                pit_delta = pit_duration_ms
            if not np.isnan(in_lap_ms):
                # add in vs out delta
                pit_delta = (pit_delta if not np.isnan(pit_delta) else 0) + (in_lap_ms - out_lap)
            records.append({
                'raceId': raceId,
                'driverId': driverId,
                'pit_lap': lapnum,
                'pit_duration_ms': pit_duration_ms,
                'estimated_pit_delta_ms': pit_delta
            })
    return pd.DataFrame.from_records(records)

pit_deltas_df = compute_pit_deltas(pit_stops, lap)
# merge pit_deltas back to pit_stops and also to lap rows (on pit lap)
pit_stops = pit_stops.merge(pit_deltas_df, left_on=['raceId','driverId','lap'], right_on=['raceId','driverId','pit_lap'], how='left')
# map to lap table
lap = lap.merge(pit_deltas_df[['raceId','driverId','pit_lap','estimated_pit_delta_ms']], left_on=['raceId','driverId','lap'], right_on=['raceId','driverId','pit_lap'], how='left')
lap = lap.drop(columns=['pit_lap'])

# ---- STINT-LEVEL AGGREGATES (per driver-race-stint) ----
print("Computing stint-level aggregates ...")
stint_agg = lap.groupby(['raceId','driverId','stint_id']).agg(
    stint_start_lap = ('lap','min'),
    stint_end_lap = ('lap','max'),
    stint_length = ('lap', lambda x: x.max() - x.min() + 1),
    mean_lap_time_ms = ('lap_time_ms','median'),
    median_lap_time_ms = ('lap_time_ms','median'),
    std_lap_time_ms = ('lap_time_ms','std'),
    max_tyre_age = ('tyre_age_laps','max')
).reset_index()

# Calculate a simple degradation slope (linear fit) per stint if there are >=3 laps
def degradation_slope(group):
    if group['lap'].nunique() < 3:
        return np.nan
    # simple linear regression slope of lap_time_ms ~ tyre_age_laps
    x = group['tyre_age_laps'].values.reshape(-1,1)
    y = group['lap_time_ms'].values
    from sklearn.linear_model import LinearRegression
    try:
        lr = LinearRegression().fit(x, y)
        return lr.coef_[0]  # ms per lap
    except:
        return np.nan

slopes = []
for _, row in stint_agg.iterrows():
    mask = (lap['raceId']==row['raceId']) & (lap['driverId']==row['driverId']) & (lap['stint_id']==row['stint_id'])
    grp = lap.loc[mask]
    slopes.append(degradation_slope(grp))
stint_agg['degradation_slope_ms_per_lap'] = slopes

# ---- SAVE PROCESSED FILES ----
print("Saving processed outputs ...")
lap.to_csv(os.path.join(PROC_DIR, "master_lap_by_lap.csv"), index=False)
pit_stops.to_csv(os.path.join(PROC_DIR, "pit_stops_enriched.csv"), index=False)
stint_agg.to_csv(os.path.join(PROC_DIR, "stint_aggregates.csv"), index=False)

print("Done. Outputs saved to", PROC_DIR)
