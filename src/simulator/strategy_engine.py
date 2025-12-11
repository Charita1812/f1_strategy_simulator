# src/simulator/strategy_engine.py
"""
Real-World F1 Strategy Engine
- Predicts lap times, tyre degradation, pit delta, undercut gains
- Supports multi-stint Monte Carlo simulation
- Robust to missing models or columns
"""

import os
import pandas as pd
import numpy as np
import joblib
import zipfile
from typing import Dict, Any

# -------------------------
# Paths
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")


class RealWorldStrategyEngine:
    def __init__(self):
        # --- Load master lap-by-lap CSV safely ---
        zip_path = os.path.join(PROCESSED, "master_lap_by_lap.zip")
        csv_folder = os.path.join(PROCESSED, "master_lap_by_lap")
        csv_path = os.path.join(csv_folder, "master_lap_by_lap.csv")

        if not os.path.exists(csv_path):
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Data missing: {zip_path}. Run data pipeline first.")
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(PROCESSED)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found after extracting ZIP: {csv_path}")

        self.master = pd.read_csv(csv_path)

        # --- Load models ---
        self.lap_time_model = self._load_model("lap_time_model.pkl")
        self.tyre_model = self._load_model("tyre_wear_predictor.pkl")
        self.pit_model = self._load_model("pit_window_model.pkl")

        # Load models inside ZIPs
        self.sc_model = self._load_model_from_zip("safety_car_model.zip", "safety_car_model.pkl")
        self.undercut_model = self._load_model_from_zip("undercut_model.zip", "undercut_model.pkl")

        # Global fallback pit delta
        self.global_pit_delta_s = 20.0

    # -------------------------
    # Model loading helpers
    # -------------------------
    def _load_model(self, filename: str):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            print(f"Loading model: {filename}")
            return joblib.load(path)
        print(f"⚠ Warning: Model {filename} not found, using fallback.")
        return None

    def _load_model_from_zip(self, zip_filename: str, pkl_filename: str):
        zip_path = os.path.join(MODELS_DIR, zip_filename)
        if not os.path.exists(zip_path):
            print(f"⚠ Warning: {zip_filename} missing, fallback for {pkl_filename}")
            return None
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(pkl_filename) as f:
                    print(f"Loading model from ZIP: {pkl_filename}")
                    return joblib.load(f)
        except Exception as e:
            print(f"⚠ Error loading {pkl_filename} from {zip_filename}: {e}")
            return None

    # -------------------------
    # Build lap features
    # -------------------------
    def _build_lap_features(self, row: Dict[str, Any], tyre_age_laps: int, last_lap_time: float, traffic_factor: float = 1.0):
        features = {
            'tyre_age_laps': tyre_age_laps,
            'tyre_compound_code': row.get('tyre_compound_code', 0),
            'fuel_load': row.get('laps_remaining', 0),
            'track_evolution': row.get('track_evolution', 0.5),
            'driver_code': row.get('driver_code', 0),
            'constructor_code': row.get('constructor_code', 0),
            'traffic': traffic_factor,
            'last_lap_time': last_lap_time
        }
        return features

    # -------------------------
    # Predictions
    # -------------------------
    def predict_lap_time(self, features: Dict[str, Any]) -> float:
        if self.lap_time_model is None:
            return features.get('last_lap_time', 90.0) + 0.05
        return float(self.lap_time_model.predict(pd.DataFrame([features]))[0])

    def predict_tyre_degradation(self, features: Dict[str, Any]) -> float:
        if self.tyre_model is None:
            return 0.05
        return float(self.tyre_model.predict(pd.DataFrame([features]))[0])

    def predict_pit_delta(self, features: Dict[str, Any], is_sc: bool = False) -> float:
        if is_sc or self.sc_model is None:
            return self.global_pit_delta_s
        X = pd.DataFrame([{
            'lap': features.get('lap', 1),
            'tyre_age_laps': features.get('tyre_age_laps', 0),
            'constructorId': features.get('constructorId', 0),
            'is_safety_car': int(is_sc)
        }])
        try:
            return float(self.pit_model.predict(X)[0]) if self.pit_model else self.global_pit_delta_s
        except Exception:
            return self.global_pit_delta_s

    def predict_undercut_gain(self, features: Dict[str, Any]) -> float:
        if self.undercut_model is None:
            return 1.5
        return float(self.undercut_model.predict(pd.DataFrame([features]))[0])

    # -------------------------
    # Strategy simulation
    # -------------------------
    def simulate_strategy(self,
                          race_id: int,
                          driver_id: int,
                          current_lap: int,
                          total_laps: int,
                          max_stop: int = 2,
                          consider_sc: bool = True) -> Dict[str, Any]:

        df_driver = self.master[(self.master['raceId'] == race_id) & (self.master['driverId'] == driver_id)]
        if df_driver.empty:
            raise ValueError("No data for driver in race")

        row = df_driver[df_driver['lap'] == current_lap].iloc[0].to_dict()
        last_lap_time = row.get('lap_time_s', 90.0)
        tyre_age = row.get('tyre_age_laps', 1)

        candidate_laps = list(range(current_lap+1, min(total_laps+1, current_lap+15)))
        results = []

        for pit_lap in candidate_laps:
            for stops in range(1, max_stop+1):
                pit_strategy = [pit_lap + i*(total_laps - pit_lap)//stops for i in range(stops)]
                sim_time = 0.0
                lap_time = last_lap_time
                tyre = tyre_age

                for lap in range(current_lap, total_laps):
                    if lap+1 in pit_strategy:
                        pit_features = {'lap': lap+1, 'tyre_age_laps': tyre, 'constructorId': row.get('constructorId', 0)}
                        delta = self.predict_pit_delta(pit_features, is_sc=consider_sc)
                        sim_time += delta
                        tyre = 0
                        lap_time = lap_time - 1.0

                    lap_features = self._build_lap_features(row, tyre, lap_time)
                    lap_time = self.predict_lap_time(lap_features)
                    lap_time += self.predict_tyre_degradation(lap_features)
                    sim_time += lap_time
                    tyre += 1

                results.append({
                    'pit_strategy': pit_strategy,
                    'stops': stops,
                    'expected_total_time_s': sim_time
                })

        best = min(results, key=lambda x: x['expected_total_time_s'])
        return {
            'best_strategy': best['pit_strategy'],
            'stops': best['stops'],
            'expected_total_time_s': best['expected_total_time_s'],
            'all_candidates': results
        }


# -------------------------
# CLI / Test
# -------------------------
if __name__ == "__main__":
    engine = RealWorldStrategyEngine()
    race_id = int(engine.master['raceId'].iloc[0])
    driver_id = int(engine.master['driverId'].iloc[0])
    current_lap = int(engine.master['lap'].iloc[0])
    total_laps = int(engine.master['lap'].max())

    sample = engine.simulate_strategy(race_id, driver_id, current_lap, total_laps)
    import pprint
    pprint.pprint(sample)
