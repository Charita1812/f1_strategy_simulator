# src/models/tyre_wear_predictor.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def train_tyre_wear_predictor():
    print("Loading processed master data...")
    df = pd.read_csv("data/processed/master_lap_by_lap.csv")

    # -------------------------
    # Feature engineering
    # -------------------------

    df['tyre_compound_code'] = df['pit_compound'].astype('category').cat.codes
    df['fuel_load'] = df.groupby('stint_id')['lap'].transform(lambda x: max(x) - x)
    df['track_evolution'] = df['lap'] / df['lap'].max()
    df['driver_code'] = df['driverId'].astype('category').cat.codes
    df['constructor_code'] = df['constructorId'].astype('category').cat.codes
    df['traffic'] = df['position'] / df['position'].max()
    df['last_lap_time'] = df.groupby(['raceId','driverId'])['lap_time_s'].shift(1)
    df['last_lap_time'] = df['last_lap_time'].fillna(df['lap_time_s'].mean())

    # Target: next lap minus current lap
    df['next_lap_time_s'] = df.groupby(['raceId','driverId'])['lap_time_s'].shift(-1)
    df['tyre_wear_s'] = df['next_lap_time_s'] - df['lap_time_s']
    df = df.dropna(subset=['tyre_wear_s'])

    # -------------------------
    # Feature columns
    # -------------------------
    feature_cols = [
        'tyre_age_laps',
        'tyre_compound_code',
        'fuel_load',
        'track_evolution',
        'driver_code',
        'constructor_code',
        'traffic',
        'last_lap_time'
    ]
    X = df[feature_cols]
    y = df['tyre_wear_s']

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Model training
    # -------------------------
    print("Training Tyre Wear Predictor...")
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Tyre Wear Predictor R² Score: {score:.3f}")

    # -------------------------
    # Save model
    # -------------------------
    joblib.dump(model, "models/tyre_wear_predictor.pkl")
    print("Tyre Wear Predictor saved to models/tyre_wear_predictor.pkl")


if __name__ == "__main__":
    train_tyre_wear_predictor()
