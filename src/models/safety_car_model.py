import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MASTER_DATA_PATH = "data/processed/master_lap_by_lap.csv"
MODEL_SAVE_PATH = "models/safety_car_model.pkl"

def train_safety_car_model():

    print("Loading processed master data...")
    df = pd.read_csv(MASTER_DATA_PATH)

    # ---------------------------
    #  FIX: no 'status' column exists
    #  In real F1 data, safety car can be inferred from:
    #  - lap_time spikes
    #  - gaps collapsing
    #  - yellow flags columns (if exists)
    # ---------------------------

    # If no safety car info exists, generate approximate one:
    if "safety_car" not in df.columns:
        print("⚠ No 'safety_car' column found, generating an approximate version...")

        df["safety_car"] = 0

        # Rule: laps with +8 seconds slower = likely SC
        df["safety_car"] = np.where(
            df["lap_time_s"] >
            df["lap_time_s"].rolling(10, min_periods=1).mean() + 8,
            1,
            0
        )

    # ---------------------------
    # Features for predicting SC probability next lap
    # ---------------------------
    features = [
        "lap_number",
        "lap_time_s",
        "tyre_age_laps",
        "track_evolution",
        "traffic"
    ]

    # Keep only columns that actually exist
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["safety_car"]

    # Remove NaN rows
    df = df.dropna(subset=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Safety Car Prediction Model...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Safety Car Predictor Accuracy: {score:.3f}")

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Safety Car Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_safety_car_model()
