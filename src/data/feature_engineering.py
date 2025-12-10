# src/data/feature_engineering.py

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
)

FEATURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "features")
)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_master():
    """Load the output of the data pipeline."""
    path = os.path.join(PROCESSED_DIR, "master_lap_by_lap.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run data_pipeline.py first!")
    return pd.read_csv(path)


# ---------------------------------------------------------
# 1) DRIVER BASELINE (driver pace rating)
# ---------------------------------------------------------

def compute_driver_baseline(df):
    """
    Computes per-driver performance offset.
    Good for ML (drivers like Alonso/Hamilton consistently faster).
    """
    driver_avg = (
        df.groupby("driverId")["lap_time_ms"]
        .median()
        .rename("driver_baseline_ms")
    )
    return df.merge(driver_avg, on="driverId", how="left")


# ---------------------------------------------------------
# 2) CONSTRUCTOR BASELINE (team performance pace)
# ---------------------------------------------------------

def compute_constructor_baseline(df):
    constructor_avg = (
        df.groupby("constructorId")["lap_time_ms"]
        .median()
        .rename("constructor_baseline_ms")
    )
    return df.merge(constructor_avg, on="constructorId", how="left")


# ---------------------------------------------------------
# 3) FUEL LOAD PROXY (laps remaining)
# ---------------------------------------------------------

def compute_fuel_factor(df):
    """
    Races start heavy & fuel burns every lap.
    Approximation: more laps remaining → slower lap.
    """
    df["laps_remaining"] = df.groupby("raceId")["lap"].transform(
        lambda x: x.max() - x
    )
    return df


# ---------------------------------------------------------
# 4) TRACK EVOLUTION (grip increases as race continues)
# ---------------------------------------------------------

def compute_track_evolution(df):
    df["track_evolution_lap"] = df.groupby("raceId")["lap"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return df


# ---------------------------------------------------------
# 5) TRAFFIC FACTOR (position impact)
# ---------------------------------------------------------

def compute_traffic(df):
    """
    Simple proxy: cars in front affect pace.
    Position 1 = clean air (best)
    Position 20 = heavy traffic (worst)
    """
    df["traffic_index"] = df["position"].clip(upper=20)  # cap weird values
    return df


# ---------------------------------------------------------
# 6) TYRE AGE EFFECT (already computed but ensure clean)
# ---------------------------------------------------------

def clean_tyre_age(df):
    df["tyre_age_laps"] = df["tyre_age_laps"].fillna(0)
    return df


# ---------------------------------------------------------
# 7) TARGET VARIABLE (tyre degradation slope)
# ---------------------------------------------------------

def compute_target_variable(df):
    """
    Target = next lap time difference (degradation)
    lap_time(t+1) - lap_time(t)
    """
    df["next_lap_time_ms"] = df.groupby(["raceId", "driverId"])["lap_time_ms"].shift(-1)
    df["degradation_ms"] = df["next_lap_time_ms"] - df["lap_time_ms"]

    # remove invalid rows
    df = df[df["degradation_ms"].notna()]
    return df


# ---------------------------------------------------------
# FINAL FEATURE ENGINEERING PIPELINE
# ---------------------------------------------------------

def build_features():
    print("🔧 Loading processed master dataset...")
    df = load_master()

    print("🔧 Computing driver baseline...")
    df = compute_driver_baseline(df)

    print("🔧 Computing constructor baseline...")
    df = compute_constructor_baseline(df)

    print("🔧 Computing fuel factor...")
    df = compute_fuel_factor(df)

    print("🔧 Computing track evolution...")
    df = compute_track_evolution(df)

    print("🔧 Computing traffic index...")
    df = compute_traffic(df)

    print("🔧 Cleaning tyre age...")
    df = clean_tyre_age(df)

    print("🔧 Computing degradation target variable...")
    df = compute_target_variable(df)

    # SAVE
    ensure_dir(FEATURES_DIR)
    output_path = os.path.join(FEATURES_DIR, "features_master.csv")

    df.to_csv(output_path, index=False)

    print(f"🎉 Feature dataset saved to: {output_path}")
    print(f"Total rows: {len(df):,}")


if __name__ == "__main__":
    build_features()
