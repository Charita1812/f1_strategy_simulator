# src/models/undercut_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

CSV_PATH = "data/processed/master_lap_by_lap.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "undercut_model.pkl")


def load_data():
    print(f"Loading dataset from {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ ERROR: File not found -> {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Normalize column names (lowercase, no spaces)
    df.columns = df.columns.str.strip().str.lower()

    # Required columns
    required = [
        "driverid", "lap", "lap_time_s",
        "tyre_age_laps", "pit_happened",
        "pit_stop_number", "estimated_pit_delta_ms",
        "position"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"❌ Missing columns: {missing}")

    return df


def engineer_undercut_features(df: pd.DataFrame):
    print("Engineering undercut features...")

    df = df.sort_values(["driverid", "lap"]).reset_index(drop=True)

    # Previous lap time
    df["prev_lap_time"] = df.groupby("driverid")["lap_time_s"].shift(1)
    df["prev_lap_time"] = df["prev_lap_time"].fillna(df["lap_time_s"].mean())

    # Lap delta (pace fall-off)
    df["lap_time_delta"] = df["lap_time_s"] - df["prev_lap_time"]

    # Track position gains (if car ahead pits, you gain track)
    df["positions_gained"] = (
        df.groupby("driverid")["position"].shift(1) - df["position"]
    ).fillna(0)

    # Next lap time → used to compute undercut effect
    df["next_lap_time"] = df.groupby("driverid")["lap_time_s"].shift(-1)

    # Undercut effect = improvement in next lap
    df["undercut_effect"] = df["lap_time_s"] - df["next_lap_time"]
    df["undercut_effect"] = df["undercut_effect"].fillna(
        df["undercut_effect"].mean()
    )

    return df


def train_undercut_model():
    df = load_data()
    df = engineer_undercut_features(df)

    features = [
        "lap", "lap_time_s", "tyre_age_laps",
        "prev_lap_time", "lap_time_delta",
        "positions_gained", "pit_happened",
        "pit_stop_number", "estimated_pit_delta_ms"
    ]

    X = df[features]
    y = df["undercut_effect"]

    print(f"Training model on {len(df)} laps...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"✅ Undercut Model R² Score: {score:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"🎉 Model saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    train_undercut_model()
