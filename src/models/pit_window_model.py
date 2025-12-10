import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def train_pit_window_model():
    print("Loading processed master data...")
    df = pd.read_csv("data/processed/master_lap_by_lap.csv")

    # -----------------------------------------
    # SAFETY CHECKS
    # -----------------------------------------
    required = ['lap', 'lap_time_s', 'tyre_age_laps', 'constructorId']
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise Exception(f"Your dataset is missing required columns: {missing}")

    # -----------------------------------------
    # Feature engineering
    # -----------------------------------------

    # Constructor encoding
    df['constructor_code'] = df['constructorId'].astype('category').cat.codes

    # Safety Car detection based on anomaly lap times
    mean_lt = df['lap_time_s'].mean()
    std_lt = df['lap_time_s'].std()
    df['is_safety_car'] = (df['lap_time_s'] > mean_lt + 2.5 * std_lt).astype(int)

    # Pace deterioration (pit trigger logic)
    df['rolling_best'] = df.groupby('driverId')['lap_time_s'].cummin()
    df['pace_delta'] = df['lap_time_s'] - df['rolling_best']

    # Remove NaNs
    df = df.dropna(subset=['pace_delta'])

    # -----------------------------------------
    # Features + labels
    # -----------------------------------------
    feature_cols = [
        'lap',
        'tyre_age_laps',
        'constructor_code',
        'is_safety_car'
    ]

    X = df[feature_cols]
    y = df['pace_delta']

    # -----------------------------------------
    # Train-test split
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------
    # Model training
    # -----------------------------------------
    print("Training Pit Window Model...")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Pit Window Model R² Score: {score:.3f}")

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    joblib.dump(model, "models/pit_window_model.pkl")
    print("Pit Window Model saved to models/pit_window_model.pkl")


if __name__ == "__main__":
    train_pit_window_model()
