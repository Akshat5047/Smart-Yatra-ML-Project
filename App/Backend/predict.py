import os
import sqlite3
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
PICKLES_DIR = os.path.join(BASE_DIR, "..", "Pickles")
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "smart_tourism.db")

VALID_BUDGET_ACCOMMODATION_TIERS = {"budget", "mid", "premium"}
ACCOMMODATION_TIER_ALIASES = {
    "standard": "mid",
    "mid-range": "mid",
    "midrange": "mid",
    "luxury": "premium",
}
VALID_BUDGET_TRANSPORT_MODES = {"auto", "bike", "bus", "car", "train"}

# ============================================================
# BUDGET
# ============================================================

budget_model = joblib.load(os.path.join(PICKLES_DIR, "Budget", "best_trip_cost_model.pkl"))

cost_cols = ['travel_cost_est', 'stay_cost_est', 'food_cost_est','entry_fees_est', 'tolls_and_parking_est']

def _build_budget_preprocessor():
    conn = sqlite3.connect(DB_PATH)
    raw_df = pd.read_sql("SELECT * FROM trip_budget_prediction;", conn)
    conn.close()
    df = raw_df.copy()
    df.columns = df.columns.str.lower()

    y = df[cost_cols]
    X = df.drop(columns=["travel_cost_est", "stay_cost_est", "food_cost_est", "entry_fees_est"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('ordinal', OrdinalEncoder(), ['accommodation_tier']),
        ('nominal', OneHotEncoder(handle_unknown='ignore'), ['transport_mode', 'season']),
        ('numeric', StandardScaler(), ['duration_days', 'num_travelers', 'route_distance_km'])
    ])
    preprocessor.fit(X_train)
    return preprocessor

budget_preprocessor = _build_budget_preprocessor()


def _normalize_budget_inputs(trip: dict) -> dict:
    normalized_trip = dict(trip)

    accommodation_tier = str(normalized_trip["accommodation_tier"]).strip().lower()
    accommodation_tier = ACCOMMODATION_TIER_ALIASES.get(accommodation_tier, accommodation_tier)
    if accommodation_tier not in VALID_BUDGET_ACCOMMODATION_TIERS:
        accommodation_tier = "mid"

    transport_mode = str(normalized_trip["transport_mode"]).strip().lower()
    if transport_mode not in VALID_BUDGET_TRANSPORT_MODES:
        transport_mode = "car"

    normalized_trip["accommodation_tier"] = accommodation_tier.capitalize() if accommodation_tier != "mid" else "Mid"
    normalized_trip["transport_mode"] = transport_mode
    return normalized_trip


def predict_budget(trip: dict) -> dict:
    trip = _normalize_budget_inputs(trip)

    new_trip = pd.DataFrame([{
        'duration_days': trip['duration_days'],
        'num_travelers': trip['num_travelers'],
        'route_distance_km': trip['route_distance_km'],
        'transport_mode': trip['transport_mode'],
        'accommodation_tier': trip['accommodation_tier'],
        'season': trip['season'],
    }])

    X_final = budget_preprocessor.transform(new_trip)
    predicted = budget_model.predict(X_final)

    result = dict(zip(cost_cols, predicted[0].tolist()))
    result['predicted_total_cost'] = sum(result.values())
    return result


# ============================================================
# CROWD — mean-encoding pipeline
# (crowd_model.pkl + preprocessor.pkl + 5 mean-map pkl)
# ============================================================

crowd_model = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "crowd_model.pkl"))
crowd_preprocessor = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "preprocessor.pkl"))
crowd_spot_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "spot_mean_map.pkl"))
crowd_district_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "district_mean_map.pkl"))
crowd_month_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "month_mean_map.pkl"))
crowd_season_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "season_mean_map.pkl"))
crowd_global_mean = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "global_visitor_mean.pkl"))


def predict_crowd(data: dict) -> dict:
    new_row = pd.DataFrame([data])

    new_row['spot_name_mean'] = new_row['spot_name'].map(crowd_spot_mean_map).fillna(crowd_global_mean)
    new_row['district_mean'] = new_row['district'].map(crowd_district_mean_map).fillna(crowd_global_mean)
    new_row['month_mean'] = new_row['month'].map(crowd_month_mean_map).fillna(crowd_global_mean)
    new_row['season_mean'] = new_row['season'].map(crowd_season_mean_map).fillna(crowd_global_mean)

    X_final = crowd_preprocessor.transform(new_row)
    predicted_visitors = crowd_model.predict(X_final)

    return {"predicted_total_visitors": float(predicted_visitors[0])}


# ============================================================
# CLIMATE
# ============================================================

class ClimateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=24, num_layers=1, output_size=None, dropout=0.2):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

climate_meta = joblib.load(os.path.join(PICKLES_DIR, "Climate", "best_climate_metadata.pkl"))
seq_len = climate_meta['seq_len']
last_known_date = climate_meta['last_known_date']
target_cols = climate_meta['target_cols']

climate_model = ClimateLSTM(input_size=len(target_cols), hidden_size=24, num_layers=1,output_size=len(target_cols), dropout=0.2)
state_dict = torch.load(
    os.path.join(PICKLES_DIR, "Climate", "best_climate_lstm_model.pt.zip"),
    map_location="cpu", weights_only=True
)
climate_model.load_state_dict(state_dict)
climate_model.eval()


def _load_climate_history():
    conn = sqlite3.connect(DB_PATH)
    raw = pd.read_sql("SELECT * FROM climate_dataset", conn)
    conn.close()
    raw['Date'] = pd.to_datetime(raw['Date'])

    raw['Is_Rain_Day'] = (raw['Rainfall_mm'] >= 1.0).astype(int)
    raw['Rainfall_Percent'] = (
        raw.groupby('District')['Is_Rain_Day']
           .transform(lambda s: s.rolling(7, min_periods=1).mean() * 100)
    )

    statewide = raw.groupby('Date')[target_cols].mean().asfreq('D')
    diffed = statewide.diff().dropna()

    district_daily = raw.groupby(['District', 'Date'])[target_cols].mean()
    district_baseline = district_daily.xs(last_known_date, level='Date')

    return diffed, district_baseline

climate_diffed, climate_district_baseline = _load_climate_history()


transport_feature_cols = [
    "distance_km",
    "budget_limit",
    "num_people",
    "rainfall_mm",
    "road_access_rating",
]
transport_model = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_model.pkl"))
transport_scaler = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_scaler.pkl"))
transport_label_encoder = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_label_encoder.pkl"))


def predict_climate(data: dict) -> dict:
    forecast_date = pd.Timestamp(data['forecast_date'])
    days_ahead = (forecast_date - last_known_date).days

    if days_ahead < 1:
        return {"error": f"forecast_date must be strictly after {last_known_date.date()}"}
    if data['district'] not in climate_district_baseline.index:
        return {"error": f"'{data['district']}' not found in climate data"}

    current_seq = torch.tensor(climate_diffed.values[-seq_len:], dtype=torch.float32).unsqueeze(0)
    baseline = climate_district_baseline.loc[data['district']]

    future_diffs = []
    daily_forecast = []
    with torch.no_grad():
        for i in range(days_ahead):
            next_diff = climate_model(current_seq)
            future_diffs.append(next_diff.squeeze(0).numpy())
            current_seq = torch.cat([current_seq[:, 1:, :], next_diff.unsqueeze(1)], dim=1)

            cumulative_so_far = np.sum(future_diffs, axis=0)
            day_values = baseline + cumulative_so_far
            day_date = last_known_date + pd.Timedelta(days=i + 1)
            daily_forecast.append({
                "forecast_date": day_date.strftime("%Y-%m-%d"),
                **dict(zip(target_cols, day_values.tolist()))
            })

    final_day = daily_forecast[-1]
    return {
        **{k: v for k, v in final_day.items() if k != "forecast_date"},
        "daily_forecast": daily_forecast,
    }


def predict_transport_mode(data: dict) -> dict:
    new_row = pd.DataFrame([{
        'distance_km': data['distance_km'],
        'budget_limit': data['budget_limit'],
        'num_people': data['num_people'],
        'rainfall_mm': data['rainfall_mm'],
        'road_access_rating': data['road_access_rating'],
    }])[transport_feature_cols]

    X_scaled = transport_scaler.transform(new_row)
    predicted_class = transport_model.predict(X_scaled)[0]
    predicted_mode = transport_label_encoder.inverse_transform([predicted_class])[0]

    probs = transport_model.predict_proba(X_scaled)[0]
    probability_by_mode = dict(zip(transport_label_encoder.classes_, probs.tolist()))

    return {
        "recommended_transport_mode": predicted_mode,
        "confidence": float(max(probs)),
        "probabilities": probability_by_mode
    }
