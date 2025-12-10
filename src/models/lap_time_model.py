# src/models/lap_time_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def train_lap_time_model():
    print("Loading processed master data...")
    df = pd.read_csv("data/processed/master_lap_by_lap.csv")

    # -------------------------
    # Feature engineering
    # -------------------------

    # Tyre compound categorical encoding
    df['tyre_compound_code'] = df['pit_compound'].astype('category').cat.codes

    # Fuel load proxy: laps remaining in stint
    df['fuel_load'] = df.groupby('stint_id')['lap'].transform(lambda x: max(x) - x)

    # Track evolution: normalized lap within race
    df['track_evolution'] = df['lap'] / df['lap'].max()

    # Driver and constructor encoding
    df['driver_code'] = df['driverId'].astype('category').cat.codes
    df['constructor_code'] = df['constructorId'].astype('category').cat.codes

    # Traffic effect: position normalized
    df['traffic'] = df['position'] / df['position'].max()

    # -------------------------
    # Add last lap time (previous lap)
    # -------------------------
    df['last_lap_time'] = df.groupby(['raceId','driverId'])['lap_time_s'].shift(1)
    df['last_lap_time'] = df['last_lap_time'].fillna(df['lap_time_s'].mean())

    # Target variable
    y = df['lap_time_s']

    # Feature columns
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

    # Drop rows with missing values
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X[mask]
    y = y[mask]

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Train model
    # -------------------------
    print("Training Lap Time Prediction Model...")
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Lap Time Model R² Score: {score:.3f}")

    # -------------------------
    # Save model
    # -------------------------
    joblib.dump(model, "models/lap_time_model.pkl")
    print("Lap Time Model saved to models/lap_time_model.pkl")


if __name__ == "__main__":
    train_lap_time_model()
