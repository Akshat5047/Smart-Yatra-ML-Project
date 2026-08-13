# ============================================================
# IMPORTS
# ============================================================
import altair as alt
import base64
import folium
import html
from branca.element import MacroElement, Template
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import json
import numpy as np
import os
import pandas as pd
import requests
import streamlit as st
from folium.plugins import AntPath
from PIL import Image, ImageDraw, ImageFont
from streamlit_folium import st_folium
import sqlite3
import joblib
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# CONFIG / CONSTANTS
# ============================================================
# API_URL removed — backend merged directly into this app, no more HTTP hop.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Data"))
# Models used to be loaded by a separate FastAPI backend (App/Backend) — now
# loaded directly in this app instead, since backend + frontend are combined.
PICKLES_DIR = os.path.join(BASE_DIR, "..", "Pickles")
DB_PATH = os.path.join(DATA_DIR, "smart_tourism.db")
CROWD_DATA_PATH = os.path.join(DATA_DIR, "crowd_data.csv")


# ============================================================
# SUPABASE CLIENT
# (previously lived only in the backend's config.py/database.py —
#  now needed directly here since this app talks to Supabase itself)
# ============================================================
load_dotenv(os.path.join(BASE_DIR, "..", "..", ".env"))


def _get_secret(key: str) -> str | None:
    """Reads a secret from Streamlit Cloud's st.secrets when deployed, falling
    back to a local .env file (via python-dotenv) when running locally."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


# ============================================================
# PREDICTION LOGIC
# (merged in from the former App/Backend/predict.py — models now
#  load once, in-process, alongside the rest of this app)
# ============================================================
VALID_BUDGET_ACCOMMODATION_TIERS = {"budget", "mid", "premium"}
ACCOMMODATION_TIER_ALIASES = {
    "standard": "mid",
    "mid-range": "mid",
    "midrange": "mid",
    "luxury": "premium",
}
VALID_BUDGET_TRANSPORT_MODES = {"auto", "bike", "bus", "car", "train"}

# ---- Budget ----
@st.cache_resource(show_spinner="Loading budget model...")
def _get_budget_model():
    return joblib.load(os.path.join(PICKLES_DIR, "Budget", "best_trip_cost_model.pkl"))


cost_cols = ['travel_cost_est', 'stay_cost_est', 'food_cost_est', 'entry_fees_est', 'tolls_and_parking_est']


@st.cache_resource(show_spinner="Loading budget model...")
def _get_budget_preprocessor():
    """Loaded lazily, only the first time predict_budget() is actually called —
    not at app startup — and cached after that via st.cache_resource."""
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
    budget_model = _get_budget_model()
    budget_preprocessor = _get_budget_preprocessor()

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


# ---- Crowd (mean-encoding pipeline) ----
@st.cache_resource(show_spinner="Loading crowd model...")
def _get_crowd_resources():
    """Loaded lazily, only the first time predict_crowd() is actually called."""
    model = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "crowd_model.pkl"))
    preprocessor = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "preprocessor.pkl"))
    spot_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "spot_mean_map.pkl"))
    district_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "district_mean_map.pkl"))
    month_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "month_mean_map.pkl"))
    season_mean_map = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "season_mean_map.pkl"))
    global_mean = joblib.load(os.path.join(PICKLES_DIR, "Crowd", "global_visitor_mean.pkl"))
    return model, preprocessor, spot_mean_map, district_mean_map, month_mean_map, season_mean_map, global_mean


def predict_crowd(data: dict) -> dict:
    (crowd_model, crowd_preprocessor, crowd_spot_mean_map, crowd_district_mean_map,
     crowd_month_mean_map, crowd_season_mean_map, crowd_global_mean) = _get_crowd_resources()

    new_row = pd.DataFrame([data])

    new_row['spot_name_mean'] = new_row['spot_name'].map(crowd_spot_mean_map).fillna(crowd_global_mean)
    new_row['district_mean'] = new_row['district'].map(crowd_district_mean_map).fillna(crowd_global_mean)
    new_row['month_mean'] = new_row['month'].map(crowd_month_mean_map).fillna(crowd_global_mean)
    new_row['season_mean'] = new_row['season'].map(crowd_season_mean_map).fillna(crowd_global_mean)

    X_final = crowd_preprocessor.transform(new_row)
    predicted_visitors = crowd_model.predict(X_final)

    return {"predicted_total_visitors": float(predicted_visitors[0])}


# ---- Climate (LSTM) ----
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


@st.cache_resource(show_spinner="Loading climate model...")
def _get_climate_resources():
    """Loaded lazily, only the first time predict_climate() is actually called —
    this is the heaviest module (torch LSTM + a 346k-row SQL read), so deferring
    it matters most here."""
    climate_meta = joblib.load(os.path.join(PICKLES_DIR, "Climate", "best_climate_metadata.pkl"))
    seq_len = climate_meta['seq_len']
    last_known_date = climate_meta['last_known_date']
    target_cols = climate_meta['target_cols']

    model = ClimateLSTM(input_size=len(target_cols), hidden_size=24, num_layers=1, output_size=len(target_cols), dropout=0.2)
    state_dict = torch.load(
        os.path.join(PICKLES_DIR, "Climate", "best_climate_lstm_model.pt.zip"),
        map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

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

    return model, seq_len, last_known_date, target_cols, diffed, district_baseline


def predict_climate(data: dict) -> dict:
    climate_model, seq_len, last_known_date, target_cols, climate_diffed, climate_district_baseline = _get_climate_resources()

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


# ---- Transport Mode ----
transport_feature_cols = [
    "distance_km",
    "budget_limit",
    "num_people",
    "rainfall_mm",
    "road_access_rating",
]


@st.cache_resource(show_spinner="Loading transport mode model...")
def _get_transport_resources():
    """Loaded lazily, only the first time predict_transport_mode() is actually called."""
    model = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_model.pkl"))
    scaler = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(PICKLES_DIR, "transport_mode_label_encoder.pkl"))
    return model, scaler, label_encoder


def predict_transport_mode(data: dict) -> dict:
    transport_model, transport_scaler, transport_label_encoder = _get_transport_resources()

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

CLIMATE_DATA_PATH = os.path.join(DATA_DIR, "Climate_Dataset_Final.csv")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"


def _find_logo_path() -> str | None:
    """Return the path to the sidebar logo in assets/ if present."""
    for ext in ("png", "jpg", "jpeg", "webp", "svg"):
        candidate = os.path.join(BASE_DIR, "assets", f"logo.{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def _sidebar_logo_html() -> str:
    """Base64-embed the sidebar logo as an <img> tag, or '' if no logo file exists."""
    logo_path = _find_logo_path()
    if not logo_path:
        return ""
    ext = logo_path.lower().rsplit(".", 1)[-1]
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(ext, "image/png")
    with open(logo_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"<img src='data:{mime};base64,{encoded}' class='sidebar-brand-logo' alt='SmartYatra' />"


def _rail_logo_html() -> str:
    """Return the Telangana mark used in the compact floating navigation rail."""
    logo_path = os.path.join(BASE_DIR, "assets", "telangana_neon_logo.png")
    if not os.path.exists(logo_path):
        return ""
    with open(logo_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("ascii")
    return f"<img src='data:image/png;base64,{encoded}' class='rail-logo' alt='Telangana location mark' />"


def _load_dropdown_options():
    spot_names = []
    crowd_districts = []
    climate_districts = []
    spot_to_district = {}
    district_to_spots = {}
    spot_to_category = {}

    if os.path.exists(CROWD_DATA_PATH):
        try:
            crowd_df = pd.read_csv(CROWD_DATA_PATH)
            spot_names = sorted(crowd_df["spot_name"].dropna().astype(str).unique().tolist())
            crowd_districts = sorted(crowd_df["district"].dropna().astype(str).unique().tolist())
            spot_to_district = (
                crowd_df.dropna(subset=["spot_name", "district"])
                .groupby("spot_name")["district"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )
            district_to_spots = {
                district: sorted(group["spot_name"].astype(str).unique().tolist())
                for district, group in crowd_df.dropna(subset=["spot_name", "district"]).groupby("district")
            }
            spot_to_category = (
                crowd_df.dropna(subset=["spot_name", "category"])
                .groupby("spot_name")["category"]
                .agg(lambda s: s.mode().iloc[0])
                .to_dict()
            )
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
            spot_names = []
            crowd_districts = []
            spot_to_district = {}
            district_to_spots = {}
            spot_to_category = {}

    if os.path.exists(CLIMATE_DATA_PATH):
        try:
            climate_df = pd.read_csv(CLIMATE_DATA_PATH)
            climate_districts = sorted(climate_df["District"].dropna().astype(str).unique().tolist())
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
            climate_districts = []

    return (
        spot_names,
        crowd_districts,
        climate_districts,
        spot_to_district,
        district_to_spots,
        spot_to_category,
    )


(
    SPOT_OPTIONS,
    CROWD_DISTRICT_OPTIONS,
    CLIMATE_DISTRICT_OPTIONS,
    SPOT_TO_DISTRICT,
    DISTRICT_TO_SPOTS,
    SPOT_TO_CATEGORY,
) = _load_dropdown_options()


def _inject_styles() -> None:
    theme = st.session_state.get("theme", "dark")
    bg = "#f8fafc" if theme == "light" else "#071124"
    fg = "#0f172a" if theme == "light" else "#f8fafc"

    st.markdown(
        f"""
        <style>
        :root {{ --app-bg: {bg}; --app-fg: {fg}; }}
        body {{ background: var(--app-bg) !important; color: var(--app-fg) !important; }}
        .report-card {{ border-radius: 20px; padding: 22px; color: #f8fafc; background: linear-gradient(135deg, #0f172a, #1e293b); margin-bottom: 18px; }}
        .report-card.light {{ background: linear-gradient(135deg, #f8fafc, #e2e8f0); color: #0f172a; }}
        .report-card .card-title {{ font-size: 0.83rem; text-transform: uppercase; letter-spacing: 0.14em; opacity: 0.75; margin-bottom: 10px; }}
        .report-card .card-value {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
        .report-card .card-note {{ font-size: 0.95rem; opacity: 0.82; line-height: 1.5; }}
        .summary-banner {{ border-radius: 20px; padding: 20px 24px; background: linear-gradient(135deg, #0f172a, #1f2937); color: var(--app-fg); margin-bottom: 18px; }}
        .summary-banner h4 {{ margin: 0; font-size: 1rem; font-weight: 700; }}
        .summary-banner p {{ margin: 8px 0 0; opacity: 0.82; line-height: 1.6; }}
        .disclaimer-card {{ display: flex; gap: 14px; align-items: flex-start; border: 1px solid #3b82f6; border-left: 4px solid #3b82f6; border-radius: 16px; padding: 15px 18px; margin: 8px 0 24px; background: linear-gradient(135deg, #0b1d3a, #12294a); color: #dbeafe; }}
        .disclaimer-card .disclaimer-icon {{ font-size: 1.25rem; line-height: 1.45; }}
        .disclaimer-card .disclaimer-title {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 3px; }}
        .disclaimer-card .disclaimer-text {{ font-size: 0.9rem; line-height: 1.55; opacity: 0.9; }}
        .climate-card {{ min-height: 176px; border: 1px solid rgba(96, 165, 250, 0.32); border-radius: 18px; padding: 20px; text-align: center; background: linear-gradient(145deg, rgba(30, 58, 138, 0.24), rgba(15, 23, 42, 0.12)); }}
        .climate-card-icon {{ font-size: 1.65rem; margin-bottom: 8px; }}
        .climate-card-label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--app-fg); opacity: 0.62; }}
        .climate-card-value {{ font-size: 2rem; font-weight: 800; margin: 6px 0; color: #38bdf8; }}
        .climate-card-note {{ font-size: 0.85rem; color: var(--app-fg); opacity: 0.7; }}
        .skeleton-box {{ border-radius: 12px; background: linear-gradient(90deg, #e2e8f0 25%, #f8fafc 50%, #e2e8f0 75%); background-size: 200% 100%; animation: shimmer 1.4s linear infinite; height: 80px; margin-bottom: 12px; }}
        @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}
        .travel-cta {{ border-radius: 16px; padding: 14px 16px; margin: 10px 0 10px; background: linear-gradient(120deg, rgba(56,189,248,0.18), rgba(14,165,233,0.08)); border: 1px solid rgba(56,189,248,0.35); }}
        .travel-cta-title {{ font-size: 0.82rem; letter-spacing: 0.13em; text-transform: uppercase; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
        .travel-cta-text {{ font-size: 0.9rem; opacity: 0.78; color: var(--app-fg); line-height: 1.5; }}
        .app-shell {{ position: relative; }}
        .app-shell::before {{ content: ""; position: fixed; inset: -20% -10% auto -10%; height: 420px; pointer-events: none; background: radial-gradient(circle at 18% 20%, rgba(56, 189, 248, 0.16) 0%, rgba(56, 189, 248, 0) 52%), radial-gradient(circle at 82% 8%, rgba(244, 114, 182, 0.10) 0%, rgba(244, 114, 182, 0) 50%); z-index: 0; }}
        .main .block-container > div {{ animation: fadeUp 0.5s ease both; }}
        @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .home-hero {{ border-radius: 22px; padding: 32px 28px; background: linear-gradient(135deg, #071124 0%, #0f172a 55%, #14213d 100%); border: 1px solid rgba(56,189,248,0.32); margin-bottom: 18px; }}
        .home-hero-tag {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); background: rgba(56,189,248,0.12); margin-bottom: 14px; }}
        .home-hero-title {{ font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 800; letter-spacing: -0.03em; color: #e2e8f0; margin-bottom: 10px; }}
        .home-hero-sub {{ font-size: 1rem; color: #cbd5e1; opacity: 0.88; line-height: 1.65; max-width: 760px; }}
        .hero-metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
        .hero-metric {{ border: 1px solid rgba(148,163,184,0.26); border-radius: 14px; background: rgba(15, 23, 42, 0.55); padding: 12px 14px; }}
        .hero-metric-value {{ font-size: 1.2rem; font-weight: 800; color: #7dd3fc; }}
        .hero-metric-label {{ font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.62; margin-top: 2px; }}
        .step-card {{ border: 1px solid rgba(125, 211, 252, 0.25); border-radius: 16px; padding: 16px 16px 8px; margin: 10px 0 14px; background: linear-gradient(135deg, rgba(11,20,40,0.92), rgba(15,23,42,0.78)); }}
        .step-header {{ display: flex; align-items: center; gap: 11px; margin-bottom: 5px; }}
        .step-tag {{ display: inline-flex; align-items: center; justify-content: center; flex: 0 0 30px; width: 30px; height: 30px; border: 1px solid rgba(186, 230, 253, 0.65); border-radius: 10px; box-sizing: border-box; font-size: 0.78rem; font-weight: 800; letter-spacing: 0; color: #f8fafc; background: linear-gradient(145deg, #38bdf8 0%, #2563eb 100%); box-shadow: 0 5px 12px rgba(37, 99, 235, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.28); }}
        .step-title {{ font-size: 1.05rem; font-weight: 700; color: var(--app-fg); }}
        .step-sub {{ font-size: 0.86rem; opacity: 0.68; color: var(--app-fg); line-height: 1.5; }}
        .results-hero {{ border-radius: 18px; padding: 20px 22px; margin: 8px 0 16px; background: linear-gradient(135deg, rgba(2,6,23,0.95), rgba(15,23,42,0.95)); border: 1px solid rgba(56,189,248,0.35); }}
        .results-hero-label {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.13em; color: #7dd3fc; margin-bottom: 8px; font-weight: 700; }}
        .results-hero-value {{ font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; line-height: 1.1; color: #e2e8f0; letter-spacing: -0.03em; }}
        .results-hero-status {{ margin-top: 10px; display: inline-flex; align-items: center; padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }}
        .results-hero-status.good {{ color: #22c55e; border: 1px solid rgba(34,197,94,0.4); background: rgba(34,197,94,0.1); }}
        .results-hero-status.warn {{ color: #f59e0b; border: 1px solid rgba(245,158,11,0.45); background: rgba(245,158,11,0.1); }}
        [data-testid="stSidebar"] {{
            min-width: 90px !important;
            max-width: 90px !important;
            background: linear-gradient(180deg, #0b1121 0%, #030712 100%) !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 30px !important;
            margin: 16px !important;
            height: fit-content !important;
            min-height: 400px !important;
            align-self: center !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
            transition: min-width 0.4s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            overflow-x: hidden !important;
        }}
        [data-testid="stSidebar"]:hover {{
            min-width: 320px !important;
            max-width: 320px !important;
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc;
            white-space: nowrap;
        }}
        .sidebar-brand {{
            padding: 24px 16px;
            background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(2,6,23,0.95));
            border-radius: 18px;
            margin: 10px 10px 24px 10px;
            border: 1px solid rgba(56, 189, 248, 0.4);
            text-align: center;
            box-shadow: 0 10px 32px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
            width: 270px;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }}
        [data-testid="stSidebar"]:hover .sidebar-brand {{
            opacity: 1;
            pointer-events: auto;
        }}
        .sidebar-brand::before {{
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(56,189,248,0.2) 0%, transparent 60%);
            z-index: 0;
            pointer-events: none;
        }}
        .sidebar-brand > * {{
            position: relative;
            z-index: 1;
        }}
        .sidebar-brand-title {{
            font-size: 1.6rem;
            font-weight: 900;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        .sidebar-brand-sub {{
            font-size: 0.82rem;
            color: #cbd5e1;
            font-weight: 500;
            letter-spacing: 0.04em;
            line-height: 1.5;
        }}
        .sidebar-brand-logo {{
            display: block;
            width: 85%;
            max-width: 150px;
            margin: 0 auto 16px;
            border-radius: 14px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.6);
            border: 2px solid rgba(56, 189, 248, 0.3);
            transition: transform 0.3s ease;
        }}
        .sidebar-brand-logo:hover {{
            transform: scale(1.05);
        }}
        .sidebar-shortcuts {{
            font-size: 0.8rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 800;
            color: #38bdf8;
            margin: 0 16px 16px;
            border-bottom: 1px solid rgba(56,189,248,0.3);
            padding-bottom: 10px;
            width: 270px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        [data-testid="stSidebar"]:hover .sidebar-shortcuts {{
            opacity: 1;
        }}
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
            background: linear-gradient(90deg, rgba(15,23,42,0.6) 0%, rgba(30,41,59,0.6) 100%) !important;
            border-radius: 12px;
            color: #e2e8f0 !important;
            font-weight: 700;
            letter-spacing: 0.08em;
            transition: all 0.3s ease;
            padding: 14px 16px 14px 18px !important;
            text-transform: uppercase;
            font-size: 0.85rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 8px;
            width: 100% !important;
            min-width: 0 !important;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            overflow: hidden !important;
            white-space: nowrap !important;
        }}
        [data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] {{
            width: 100% !important;
            min-width: 0 !important;
            overflow: hidden !important;
            text-overflow: clip !important;
        }}
        [data-testid="stSidebar"] .stButton > button p {{
            text-align: left !important;
            margin: 0 !important;
            width: 100% !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            border-color: rgba(56, 189, 248, 0.9) !important;
            background: linear-gradient(90deg, rgba(56,189,248,0.2) 0%, rgba(14,165,233,0.2) 100%) !important;
            color: #ffffff !important;
            transform: translateX(6px) translateY(-1px);
            box-shadow: 0 8px 24px rgba(56,189,248,0.4);
        }}
        [data-testid="stSidebar"] .stButton > button:active {{
            transform: translateX(3px) translateY(1px);
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(56, 189, 248, 0.2);
            margin: 24px 16px;
            width: 270px;
        }}
        .sidebar-filter-title {{
            margin-top: 8px;
            font-size: 0.78rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #7dd3fc;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .sidebar-filter-note {{
            font-size: 0.8rem;
            opacity: 0.72;
            line-height: 1.45;
            color: #cbd5e1;
            margin-bottom: 8px;
        }}
        .overview-hero {{
            border-radius: 22px;
            padding: 24px 24px;
            border: 1px solid rgba(56,189,248,0.35);
            background: linear-gradient(135deg, rgba(2,6,23,0.95), rgba(15,23,42,0.92));
            margin-bottom: 16px;
        }}
        .overview-hero-tag {{
            display: inline-flex;
            align-items: center;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #38bdf8;
            border: 1px solid rgba(56,189,248,0.42);
            background: rgba(56,189,248,0.14);
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .overview-hero-title {{
            font-size: clamp(1.55rem, 3.2vw, 2.25rem);
            line-height: 1.15;
            letter-spacing: -0.02em;
            color: #e2e8f0;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .overview-hero-sub {{
            font-size: 0.92rem;
            line-height: 1.6;
            color: #cbd5e1;
            opacity: 0.88;
        }}
        .overview-stat-card {{
            border-radius: 14px;
            padding: 14px 14px;
            border: 1px solid rgba(148,163,184,0.24);
            background: linear-gradient(135deg, rgba(11,20,40,0.95), rgba(30,41,59,0.95));
            min-height: 92px;
        }}
        .overview-stat-value {{
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.1;
            color: #7dd3fc;
            margin-bottom: 4px;
        }}
        .overview-stat-label {{
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #cbd5e1;
            opacity: 0.72;
        }}
        .overview-note {{
            border-left: 3px solid rgba(56,189,248,0.7);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 8px 0 14px;
            background: rgba(30, 64, 175, 0.12);
            color: var(--app-fg);
            font-size: 0.88rem;
            line-height: 1.55;
        }}
        .overview-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 2px 0 12px;
        }}
        .overview-chip {{
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #67e8f9;
            border: 1px solid rgba(103,232,249,0.34);
            background: rgba(6,182,212,0.1);
            font-weight: 700;
        }}
        .spot-browser-summary {{
            border-radius: 14px;
            padding: 12px 14px;
            margin: 8px 0 14px;
            border: 1px solid rgba(148,163,184,0.18);
            background: linear-gradient(135deg, rgba(11,20,40,0.9), rgba(30,41,59,0.75));
        }}
        .spot-browser-count {{
            font-size: 0.9rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 2px;
        }}
        .spot-browser-note {{
            font-size: 0.8rem;
            color: #cbd5e1;
            opacity: 0.72;
        }}
        .spot-card {{
            border-radius: 16px;
            padding: 16px 16px 14px;
            border: 1px solid rgba(148,163,184,0.16);
            background: linear-gradient(135deg, rgba(11,20,40,0.94), rgba(30,41,59,0.82));
            min-height: 176px;
            margin-bottom: 14px;
        }}
        .spot-card-title {{
            font-size: 1rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 8px;
            line-height: 1.35;
        }}
        .spot-card-meta {{
            font-size: 0.82rem;
            color: #cbd5e1;
            opacity: 0.8;
            line-height: 1.55;
            margin-bottom: 10px;
        }}
        .spot-card-tag-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .spot-card-tag {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #facc15;
            background: rgba(250,204,21,0.12);
            border: 1px solid rgba(250,204,21,0.32);
        }}
        .spot-card-fee {{
            font-size: 0.88rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 12px;
        }}
        .festival-showcase-head {{
            margin: 10px 0 8px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            color: #7dd3fc;
            font-weight: 700;
        }}
        .festival-showcase-card {{
            border-radius: 16px;
            min-height: 176px;
            padding: 14px 14px 12px;
            border: 1px solid rgba(251, 191, 36, 0.35);
            background: radial-gradient(circle at 90% 8%, rgba(251,191,36,0.12), rgba(251,191,36,0) 42%), linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.88));
            margin-bottom: 12px;
        }}
        .festival-date-pill {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 700;
            color: #78350f;
            background: #fef3c7;
            margin-bottom: 8px;
        }}
        .festival-title {{
            font-size: 0.98rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.35;
            margin-bottom: 6px;
        }}
        .festival-note {{
            font-size: 0.82rem;
            color: #cbd5e1;
            line-height: 1.5;
            margin-bottom: 8px;
        }}
        .festival-reco {{
            font-size: 0.8rem;
            color: #fde68a;
            line-height: 1.45;
            font-weight: 700;
        }}
        .action-panel {{
            border-radius: 18px;
            padding: 16px 16px 12px;
            border: 1px solid rgba(125, 211, 252, 0.32);
            background: radial-gradient(circle at 85% 15%, rgba(56,189,248,0.16), rgba(56,189,248,0) 42%), linear-gradient(135deg, rgba(8, 21, 42, 0.98), rgba(30, 41, 59, 0.9));
            margin-top: 14px;
            box-shadow: 0 14px 26px rgba(2, 6, 23, 0.32);
        }}
        .action-panel-title {{
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #67e8f9;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .action-panel-note {{
            font-size: 0.9rem;
            color: #dbeafe;
            opacity: 0.9;
            line-height: 1.55;
            margin-bottom: 12px;
        }}
        .action-hint {{
            font-size: 0.76rem;
            line-height: 1.45;
            color: #93c5fd;
            opacity: 0.86;
            margin-top: 6px;
            text-align: center;
        }}
        .stButton > button[kind="primary"] {{
            border: 1px solid rgba(251, 113, 133, 0.4);
            background: linear-gradient(90deg, #fb7185 0%, #ef4444 100%);
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0.01em;
            transition: transform 0.16s ease, box-shadow 0.16s ease;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(239, 68, 68, 0.34);
        }}
        .stButton > button[kind="secondary"] {{
            border: 1px solid rgba(56, 189, 248, 0.55);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 58, 138, 0.4));
            color: #e0f2fe;
            font-weight: 700;
            transition: transform 0.16s ease, box-shadow 0.16s ease;
        }}
        .stButton > button[kind="secondary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 18px rgba(14, 116, 144, 0.26);
        }}
        .ai-reco-card {{
            border-radius: 18px;
            padding: 18px 18px 14px;
            margin-top: 14px;
            border: 1px solid rgba(56, 189, 248, 0.45);
            background: radial-gradient(circle at 90% 18%, rgba(56,189,248,0.16), rgba(56,189,248,0) 38%), linear-gradient(135deg, rgba(8, 47, 73, 0.5), rgba(15, 23, 42, 0.98));
            box-shadow: 0 16px 28px rgba(2, 6, 23, 0.34);
        }}
        .ai-reco-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: #e0f2fe;
            margin-bottom: 8px;
        }}
        .ai-reco-note {{
            font-size: 0.9rem;
            color: #dbeafe;
            opacity: 0.92;
            line-height: 1.55;
            margin-bottom: 12px;
        }}
        .ai-reco-pill {{
            display: inline-flex;
            align-items: center;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 700;
            color: #0c4a6e;
            background: #bae6fd;
            margin-bottom: 10px;
        }}
        .ai-reco-list {{
            margin: 0;
            padding-left: 20px;
        }}
        .ai-reco-list li {{
            font-size: 0.9rem;
            color: #e2e8f0;
            margin: 8px 0;
            line-height: 1.55;
        }}
        .ai-reco-savings {{
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(125, 211, 252, 0.35);
            background: rgba(56, 189, 248, 0.1);
            color: #bae6fd;
            font-size: 0.86rem;
            font-weight: 700;
        }}
        .crowd-hero-card {{
            border-radius: 18px;
            padding: 18px 20px;
            border: 1px solid rgba(56, 189, 248, 0.38);
            background: radial-gradient(circle at 90% 10%, rgba(56,189,248,0.16), rgba(56,189,248,0) 40%), linear-gradient(135deg, rgba(2, 6, 23, 0.95), rgba(30, 41, 59, 0.88));
            margin-bottom: 12px;
        }}
        .crowd-hero-tag {{
            display: inline-flex;
            align-items: center;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 700;
            color: #082f49;
            background: #7dd3fc;
            margin-bottom: 8px;
        }}
        .crowd-hero-title {{
            font-size: 1.15rem;
            font-weight: 800;
            color: #e2e8f0;
            margin-bottom: 6px;
        }}
        .crowd-hero-sub {{
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.55;
            opacity: 0.9;
        }}
        .route-layers-card {{
            border-radius: 14px;
            padding: 12px 12px 8px;
            border: 1px solid rgba(56, 189, 248, 0.34);
            background: radial-gradient(circle at 88% 12%, rgba(56,189,248,0.12), rgba(56,189,248,0) 45%), linear-gradient(135deg, rgba(8, 21, 42, 0.96), rgba(30, 41, 59, 0.9));
            margin: 8px 0 10px;
        }}
        .route-layers-title {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #67e8f9;
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .route-layers-note {{
            font-size: 0.8rem;
            color: #cbd5e1;
            opacity: 0.85;
            line-height: 1.45;
            margin-bottom: 6px;
        }}
        .st-key-route_show_route,
        .st-key-route_show_accommodations,
        .st-key-route_show_amenities {{
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 12px;
            padding: 6px 10px;
            margin-bottom: 6px;
            background: rgba(15, 23, 42, 0.45);
        }}
        .st-key-reset_plan_fields_btn button {{
            border: 1px solid rgba(248, 113, 113, 0.62) !important;
            background: linear-gradient(135deg, rgba(69, 10, 10, 0.95), rgba(127, 29, 29, 0.95)) !important;
            color: #fee2e2 !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
            box-shadow: 0 8px 16px rgba(127, 29, 29, 0.24);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }}
        .st-key-reset_plan_fields_btn button:hover {{
            transform: translateY(-1px);
            border-color: rgba(252, 165, 165, 0.9) !important;
            box-shadow: 0 12px 20px rgba(153, 27, 27, 0.35);
        }}
        .st-key-reset_plan_fields_btn button:focus {{
            box-shadow: 0 0 0 0.2rem rgba(248, 113, 113, 0.28), 0 8px 16px rgba(127, 29, 29, 0.24) !important;
        }}
        @media (max-width: 900px) {{ .hero-metrics {{ grid-template-columns: 1fr; }} .home-hero {{ padding: 24px 18px; }} .step-card {{ padding: 14px 12px 6px; }} }}
        /* Floating hover navigation rail */
        [data-testid="stSidebar"] {{
            position: fixed !important;
            top: 50% !important;
            left: 18px !important;
            bottom: auto !important;
            transform: translateY(-50%) !important;
            min-width: 72px !important;
            max-width: 72px !important;
            width: 72px !important;
            height: auto !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 4px 10px !important;
            background: linear-gradient(160deg, rgba(15, 28, 52, 0.98), rgba(5, 12, 27, 0.98)) !important;
            border: 1px solid rgba(125, 211, 252, 0.35) !important;
            border-radius: 26px !important;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.4) !important;
            overflow: hidden !important;
            z-index: 1000000 !important;
            transition: width 240ms ease, min-width 240ms ease, max-width 240ms ease !important;
        }}
        [data-testid="stSidebar"]:hover {{
            min-width: 266px !important;
            max-width: 266px !important;
            width: 266px !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            width: 100% !important;
            padding: 0 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding: 0 !important;
        }}
        .rail-logo-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 66px;
            margin: 38px 0 0;
            overflow: hidden;
        }}
        .rail-logo-wrap .rail-logo {{
            width: 92px !important;
            height: 92px !important;
            object-fit: contain;
            transform: scale(1.14);
            transform-origin: center;
        }}
        .floating-nav-title {{
            display: none;
            padding: 7px 11px 13px;
            color: #d9fffb;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        [data-testid="stSidebar"]:hover .rail-logo-wrap {{
            justify-content: center;
            padding-left: 0;
        }}
        [data-testid="stSidebar"]:hover .floating-nav-title {{
            display: block;
            padding: 5px 0 12px;
            text-align: center;
        }}
        [data-testid="stSidebar"] .stButton {{
            margin: 0 0 8px !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_home"] {{
            margin-top: 22px !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_climate_festivals"] {{
            margin-bottom: 0 !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            height: 48px !important;
            padding: 0 !important;
            border: 1px solid transparent !important;
            border-radius: 15px !important;
            background: transparent !important;
            box-shadow: none !important;
            justify-content: center !important;
            font-size: 0 !important;
            transition: background 180ms ease, border-color 180ms ease, transform 180ms ease !important;
        }}
        [data-testid="stSidebar"] .stButton > button p {{
            font-size: 0 !important;
            text-align: center !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            transform: none !important;
            background: rgba(56, 189, 248, 0.12) !important;
            border-color: rgba(125, 211, 252, 0.26) !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"]:hover .stButton > button {{
            padding: 0 15px !important;
            justify-content: flex-start !important;
        }}
        [data-testid="stSidebar"]:hover .stButton > button p {{
            font-size: 0.91rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            text-align: left !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-nav_home"] button::before {{ content: "🏠"; }}
        [data-testid="stSidebar"] [class*="st-key-nav_overview"] button::before {{ content: "📋"; }}
        [data-testid="stSidebar"] [class*="st-key-nav_plan_trip"] button::before {{ content: "🧭"; }}
        [data-testid="stSidebar"] [class*="st-key-nav_climate_festivals"] button::before {{ content: "🌍"; }}
        [data-testid="stSidebar"] .stButton > button::before {{
            font-size: 1.35rem;
            line-height: 1;
        }}
        [data-testid="stSidebar"]:hover .stButton > button::before {{
            display: none;
        }}
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarHeader"],
        button[kind="header"] {{
            display: none !important;
        }}
        .main .block-container {{
            margin-left: 0 !important;
            padding-left: 0 !important;
        }}
        section[data-testid="stMain"] [data-testid="stMainBlockContainer"].block-container,
        .main [data-testid="stMainBlockContainer"].block-container {{
            box-sizing: border-box !important;
            margin-left: 140px !important;
            width: calc(100% - 140px) !important;
            padding-left: 0 !important;
            transition: margin-left 240ms ease, width 240ms ease;
        }}
        @media (max-width: 800px) {{
            [data-testid="stSidebar"] {{ left: 8px !important; }}
            [data-testid="stSidebar"]:hover {{ min-width: 240px !important; max-width: 240px !important; width: 240px !important; }}
            .main .block-container {{ margin-left: 0 !important; padding-left: 0 !important; }}
            section[data-testid="stMain"] [data-testid="stMainBlockContainer"].block-container,
            .main [data-testid="stMainBlockContainer"].block-container {{
                margin-left: 110px !important;
                width: calc(100% - 110px) !important;
                padding-left: 0 !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_step_card(step: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='step-card'>
            <div class='step-header'>
                <div class='step-tag'>{step}</div>
                <div class='step-title'>{title}</div>
            </div>
            <div class='step-sub'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _go_to_route_trip() -> None:
    _save_trip_plan_draft()
    st.session_state["page_navigation"] = "Route My Trip"


def _show_ai_recommendations() -> None:
    st.session_state["show_ai_recommendations"] = True


def _mark_duration_user_overridden() -> None:
    st.session_state["plan_duration_user_overridden"] = True
    _save_trip_plan_draft()


def _save_trip_plan_draft() -> None:
    """Keep planner inputs outside widget state while another page is open."""
    draft = dict(st.session_state.get("trip_plan_draft", {}))
    for key in (
        "plan_selected_district",
        "plan_start_date",
        "plan_duration_days",
        "plan_duration_user_overridden",
        "plan_num_travelers",
        "plan_user_budget",
    ):
        if key in st.session_state:
            draft[key] = st.session_state[key]
    st.session_state["trip_plan_draft"] = draft


def _reset_plan_fields() -> None:
    st.session_state["trip_cart"] = []

    static_keys = [
        "last_plan",
        "show_ai_recommendations",
        "spot_browser_signature",
        "spot_browser_page",
        "plan_selected_district",
        "plan_start_date",
        "plan_duration_days",
        "plan_duration_user_overridden",
        "plan_num_travelers",
        "plan_user_budget",
        "trip_plan_draft",
    ]
    for key in static_keys:
        st.session_state.pop(key, None)

    dynamic_prefixes = ("plan_category_filter_", "plan_sort_", "plan_search_")
    for key in list(st.session_state.keys()):
        if key.startswith(dynamic_prefixes):
            st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _geocode_location(location_query: str) -> dict | None:
    if not location_query.strip():
        return None

    query = location_query.strip()
    lowered_query = query.lower()
    if "india" not in lowered_query:
        if "telangana" not in lowered_query:
            query = f"{query}, Telangana, India"
        else:
            query = f"{query}, India"

    response = requests.get(
        NOMINATIM_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in",
        },
        headers={"User-Agent": "SmartYatra/1.0 (student project route planner)"},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    top_result = results[0]
    return {
        "label": top_result.get("display_name", location_query),
        "lat": float(top_result["lat"]),
        "lon": float(top_result["lon"]),
    }


@st.cache_data(show_spinner=False)
def _fetch_osrm_route(route_points: tuple[tuple[float, float], ...]) -> dict | None:
    if len(route_points) < 2:
        return None

    coordinates = ";".join(f"{lon},{lat}" for lat, lon in route_points)
    response = requests.get(
        f"{OSRM_ROUTE_URL}/{coordinates}",
        params={
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    routes = payload.get("routes", [])
    if not routes:
        return None

    route = routes[0]
    geometry = route["geometry"]["coordinates"]
    return {
        "distance_km": route["distance"] / 1000,
        "duration_min": route["duration"] / 60,
        "path": [[lat, lon] for lon, lat in geometry],
    }


def _build_leg_breakdown(route_points: list[dict]) -> list[dict]:
    leg_rows = []
    if len(route_points) < 2:
        return leg_rows

    for index in range(len(route_points) - 1):
        start_point = route_points[index]
        end_point = route_points[index + 1]
        try:
            leg_route = _fetch_osrm_route(((start_point["lat"], start_point["lon"]), (end_point["lat"], end_point["lon"])))
        except requests.RequestException:
            leg_route = None

        leg_rows.append({
            "Leg": index + 1,
            "From": start_point["name"],
            "To": end_point["name"],
            "Distance (km)": round(leg_route["distance_km"], 1) if leg_route else "Unavailable",
            "Duration (min)": round(leg_route["duration_min"], 0) if leg_route else "Unavailable",
        })

    return leg_rows


def _build_route_map(
    route_points: list[dict],
    route_data: dict | None,
    cart: list[dict],
    show_route_layer: bool = True,
    show_accommodation_layer: bool = True,
    show_amenities_layer: bool = True,
) -> folium.Map:
    center_lat = float(np.mean([point["lat"] for point in route_points]))
    center_lon = float(np.mean([point["lon"] for point in route_points]))
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")

    marker_colors = ["red"] + ["blue"] * max(0, len(route_points) - 2) + ["green"]
    for idx, point in enumerate(route_points, start=1):
        color = marker_colors[idx - 1] if idx - 1 < len(marker_colors) else "blue"
        latitude, longitude = float(point["lat"]), float(point["lon"])
        stop_name = html.escape(str(point["name"]))
        # When no origin is supplied, Google Maps uses the visitor's current
        # location (after its normal location-permission prompt).
        directions_url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&destination={latitude:.6f}%2C{longitude:.6f}&travelmode=driving"
        )
        popup_html = f"""
            <div style='width:230px; font-family:Arial,sans-serif; line-height:1.45;'>
                <strong>Stop {idx}: {stop_name}</strong><br>
                <span style='font-size:12px; color:#475569;'>Get directions from your current location.</span><br>
                <a href='{directions_url}' target='_blank' rel='noopener noreferrer'
                   style='display:inline-block; margin-top:9px; padding:7px 10px; background:#2563eb;
                   color:#ffffff; border-radius:6px; font-weight:700; text-decoration:none;'>
                    Redirect to Google Maps
                </a>
            </div>
        """
        folium.Marker(
            [latitude, longitude],
            tooltip=folium.Tooltip(f"Stop {idx}: {stop_name} — click for Google Maps directions", sticky=True),
            popup=folium.Popup(folium.IFrame(html=popup_html, width=250, height=140), max_width=250),
            icon=folium.Icon(color=color, icon="flag" if idx == 1 else "info-sign"),
        ).add_to(route_map)

    if show_route_layer and route_data and route_data.get("path"):
        folium.PolyLine(route_data["path"], color="#2563eb", weight=5, opacity=0.85).add_to(route_map)
        AntPath(route_data["path"], color="#38bdf8", weight=4, delay=800).add_to(route_map)

    route_names = {spot["name"] for spot in cart if isinstance(spot, dict) and spot.get("name")}
    route_districts = {spot["district"] for spot in cart if isinstance(spot, dict) and spot.get("district")}

    if show_accommodation_layer:
        accommodations_group = folium.FeatureGroup(name="Accommodations", show=True)
        if not ACCOMMODATIONS_DF.empty and route_districts:
            acc_df = ACCOMMODATIONS_DF[ACCOMMODATIONS_DF["district"].isin(route_districts)].copy()
            acc_df = acc_df.dropna(subset=["lat", "lon"]).head(20)
            for _, row in acc_df.iterrows():
                folium.Marker(
                    [float(row["lat"]), float(row["lon"])],
                    tooltip=f"Stay: {row['name']}",
                    popup=f"{row['name']} ({row['tier']})<br/>Approx cost: Rs {float(row['cost']):,.0f}",
                    icon=folium.Icon(color="cadetblue", icon="home", prefix="fa"),
                ).add_to(accommodations_group)
        accommodations_group.add_to(route_map)

    if show_amenities_layer:
        amenities_group = folium.FeatureGroup(name="Nearby Amenities", show=True)
        if not AMENITIES_DF.empty:
            amenity_df = AMENITIES_DF.dropna(subset=["lat", "lon"]).copy()
            if route_districts and "district" in amenity_df.columns:
                amenity_df = amenity_df[amenity_df["district"].isin(route_districts)]
            if route_names and "spot_name" in amenity_df.columns:
                amenity_df = amenity_df[amenity_df["spot_name"].isin(route_names)]

            amenity_color_map = {
                "restaurant": "#f59e0b",
                "atm": "#22c55e",
                "hospital": "#ef4444",
                "pharmacy": "#ef4444",
                "police": "#8b5cf6",
                "parking": "#06b6d4",
                "fuel": "#f97316",
            }
            for _, row in amenity_df.head(80).iterrows():
                amenity_type = str(row.get("amenity_type", "other")).lower()
                color = amenity_color_map.get(amenity_type, "#334155")
                folium.CircleMarker(
                    [float(row["lat"]), float(row["lon"])],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=f"Amenity: {row.get('amenity_name', 'Unknown')}",
                    popup=f"{row.get('amenity_name', 'Unknown')}<br/>Type: {amenity_type.title()}",
                ).add_to(amenities_group)
        amenities_group.add_to(route_map)

    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(125, 211, 252, 0.55);
        border-radius: 12px;
        padding: 10px 12px;
        color: #e2e8f0;
        font-size: 12px;
        min-width: 180px;
        box-shadow: 0 10px 22px rgba(2, 6, 23, 0.35);
    ">
        <div style="font-weight: 700; letter-spacing: 0.04em; margin-bottom: 6px; color: #7dd3fc;">Map Legend</div>
        <div><span style="color:#dc2626;">●</span> Trip start</div>
        <div><span style="color:#2563eb;">●</span> Trip stop</div>
        <div><span style="color:#16a34a;">●</span> Trip end</div>
        <div><span style="color:#0ea5a4;">●</span> Accommodation</div>
        <div><span style="color:#f59e0b;">●</span> Amenities</div>
        <div><span style="color:#38bdf8;">●</span> Route line</div>
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    route_map.get_root().add_child(legend)

    bounds = [[point["lat"], point["lon"]] for point in route_points]
    if show_route_layer and route_data and route_data.get("path"):
        bounds.extend(route_data["path"])
    route_map.fit_bounds(bounds, padding=(24, 24))
    return route_map


def _render_stat_card(title: str, value: str, note: str, theme: str = "dark") -> None:
    theme_class = "report-card light" if theme == "light" else "report-card"
    st.markdown(
        f"""
        <div class='{theme_class}'>
            <div class='card-title'>{title}</div>
            <div class='card-value'>{value}</div>
            <div class='card-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_climate_card(icon: str, label: str, value: str, note: str, card_class: str = "") -> None:
    st.markdown(
        f"""
        <div class='climate-card {card_class}'>
            <div class='climate-card-icon'>{icon}</div>
            <div class='climate-card-label'>{label}</div>
            <div class='climate-card-value'>{value}</div>
            <div class='climate-card-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_loading_skeletons(container) -> None:
    # Accept either a container or an empty placeholder
    cont = container.container() if hasattr(container, "container") else container
    with cont:
        cols = st.columns(3)
        for c in cols:
            c.markdown("<div class='skeleton-box' style='height:110px'></div>", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("<div class='skeleton-box' style='height:180px'></div>", unsafe_allow_html=True)
        c2.markdown("<div class='skeleton-box' style='height:180px'></div>", unsafe_allow_html=True)
        c3.markdown("<div class='skeleton-box' style='height:180px'></div>", unsafe_allow_html=True)

        st.markdown("---")
        c4, c5 = st.columns(2)
        c4.markdown("<div class='skeleton-box' style='height:320px'></div>", unsafe_allow_html=True)
        c5.markdown("<div class='skeleton-box' style='height:320px'></div>", unsafe_allow_html=True)


def _load_climate_history(district: str) -> pd.DataFrame:
    if not os.path.exists(CLIMATE_DATA_PATH):
        return pd.DataFrame()

    climate_df = pd.read_csv(CLIMATE_DATA_PATH)
    climate_df["Date"] = pd.to_datetime(climate_df["Date"], dayfirst=True, errors="coerce")
    climate_df = climate_df.dropna(subset=["Date"])
    district_df = climate_df[climate_df["District"] == district].copy()
    if district_df.empty:
        return pd.DataFrame()

    daily = (
        district_df.groupby("Date")[['Temperature_Max_C', 'Temperature_Min_C', 'Rainfall_mm']]
        .mean()
        .reset_index()
    )
    rain_chance = (
        (district_df['Rainfall_mm'] > 0)
        .groupby(district_df['Date'])
        .mean()
        .mul(100)
        .reset_index(name='RainChance')
    )
    daily = daily.merge(rain_chance, on='Date', how='left')
    return daily


# ============================================================
# NEW: LIVE / TRIP-DATE-AWARE CLIMATE HELPERS
# ============================================================
def _district_coords() -> dict:
    """District -> {lat, lon} centroid, averaged from other_spots.csv (Climate_Dataset_Final.csv
    has no lat/lon columns of its own)."""
    if SPOTS_MASTER_DF.empty:
        return {}
    return SPOTS_MASTER_DF.groupby("district")[["lat", "lon"]].mean().to_dict("index")


def _live_current_conditions(district: str) -> dict | None:
    """Genuinely live conditions via Open-Meteo (free, no API key). Returns None on any
    failure so the caller can degrade gracefully rather than crash."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "current": "temperature_2m,relative_humidity_2m,rain,precipitation",
                "timezone": "Asia/Kolkata",
            },
            timeout=5,
        )
        resp.raise_for_status()
        current = resp.json()["current"]
        return {
            "as_of": current["time"],
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "rainfall_mm": current.get("rain", current.get("precipitation", 0.0)),
        }
    except Exception:
        return None


def _openmeteo_daily_forecast(district: str, start_date, end_date) -> pd.DataFrame | None:
    """Real day-by-day forecast (Max/Min Temp, Rainfall mm, Humidity %) via Open-Meteo.
    Covers roughly the next 16 days — genuine meteorological forecast, not a model guess."""
    coords = DISTRICT_COORDS.get(district)
    if not coords:
        return None
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "hourly": "relative_humidity_2m",
                "start_date": str(start_date),
                "end_date": str(end_date),
                "timezone": "Asia/Kolkata",
            },
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()

        daily = pd.DataFrame({
            "forecast_date": pd.to_datetime(data["daily"]["time"]),
            "Temperature_Max_C": data["daily"]["temperature_2m_max"],
            "Temperature_Min_C": data["daily"]["temperature_2m_min"],
            "Rainfall_mm": data["daily"]["precipitation_sum"],
        })
        hourly_time = pd.to_datetime(data["hourly"]["time"])
        hourly_humidity = pd.Series(data["hourly"]["relative_humidity_2m"], index=hourly_time)
        daily_humidity = hourly_humidity.resample("D").mean()
        daily["Humidity_Percent"] = daily["forecast_date"].dt.normalize().map(daily_humidity)
        daily["source"] = "live_forecast"
        return daily
    except Exception:
        return None


def _climate_monthly_stats(district: str) -> pd.DataFrame:
    """Historical monthly averages for a district: avg temp, avg rainfall (mm), avg humidity."""
    if not os.path.exists(CLIMATE_DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CLIMATE_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    d = df[df["District"] == district].copy()
    if d.empty:
        return pd.DataFrame()
    d["AvgTemp"] = (d["Temperature_Max_C"] + d["Temperature_Min_C"]) / 2
    d["Month"] = d["Date"].dt.month
    monthly = d.groupby("Month").agg(
        AvgTemp=("AvgTemp", "mean"),
        AvgRainfall_mm=("Rainfall_mm", "mean"),
        AvgHumidity=("Humidity_Percent", "mean"),
    ).reset_index()
    month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                   7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    monthly["MonthName"] = monthly["Month"].map(month_names)
    return monthly.sort_values("Month")


def _month_recommendation(avg_temp: float, avg_rain: float) -> str:
    if avg_rain < 2:
        rain_desc = "a drier"
    elif avg_rain < 8:
        rain_desc = "a moderately rainy"
    else:
        rain_desc = "a wet"

    if avg_temp < 20:
        temp_desc, icon = "cool", "🧥"
    elif avg_temp < 28:
        temp_desc, icon = "pleasant", "☀️"
    elif avg_temp < 34:
        temp_desc, icon = "warm", "🌤️"
    else:
        temp_desc, icon = "hot", "🥵"

    if avg_rain < 2 and avg_temp < 34:
        suggestion = "Great time for sightseeing and outdoor activities!"
    elif avg_temp >= 34:
        suggestion = "Plan outdoor activities early in the day and carry plenty of water."
    elif avg_rain >= 8:
        suggestion = "Carry rain gear and keep indoor backup plans handy."
    else:
        suggestion = "A reasonable time to visit — check the day-by-day forecast for specifics."

    return f"{icon} This is typically {rain_desc}, {temp_desc} month (avg {avg_temp:.1f}°C, {avg_rain:.1f} mm/day rain). {suggestion}"


def _build_trip_climate_forecast(district: str, start_date, duration_days: int, lstm_result: dict) -> tuple[pd.DataFrame, str]:
    """Returns (day-by-day DataFrame, source_label) for the TRIP'S OWN dates — no
    separate district/date picker, uses whatever the user already chose in Plan Your Trip.

    Live Open-Meteo forecast when the trip falls within its ~16-day window; otherwise
    falls back to the trained LSTM's daily_forecast + climatological humidity, with
    rainfall left as N/A rather than fabricated (the LSTM was never trained to predict
    rainfall in mm, only rain chance %)."""
    end_date = start_date + pd.Timedelta(days=duration_days - 1)
    days_out = (end_date - datetime.now(timezone.utc).date()).days

    if days_out <= 15:
        live_df = _openmeteo_daily_forecast(district, start_date, end_date)
        if live_df is not None and not live_df.empty:
            return live_df, "🟢 Live day-by-day forecast (Open-Meteo)"

    daily = pd.DataFrame((lstm_result or {}).get("daily_forecast", []))
    if daily.empty:
        return pd.DataFrame(), "No forecast available"
    daily["forecast_date"] = pd.to_datetime(daily["forecast_date"])
    window = daily[
        (daily["forecast_date"] >= pd.Timestamp(start_date)) & (daily["forecast_date"] <= pd.Timestamp(end_date))
    ].copy()
    window["Rainfall_mm"] = None
    monthly = _climate_monthly_stats(district)
    window["Humidity_Percent"] = window["forecast_date"].dt.month.map(
        monthly.set_index("Month")["AvgHumidity"] if not monthly.empty else {}
    )
    window["source"] = "lstm_fallback"
    return window, "⚠️ Beyond live forecast range — showing LSTM trend + climatological humidity"


def _render_climate_charts(district: str, climate_result: dict, forecast_date: str) -> None:
    history = _load_climate_history(district)
    if history.empty:
        st.info("Historical climate data is unavailable for this district.")
        return

    history = history.sort_values("Date")
    recent = history.tail(14).copy()

    daily_forecast_df = pd.DataFrame(climate_result.get("daily_forecast", []))
    if not daily_forecast_df.empty:
        daily_forecast_df["Date"] = pd.to_datetime(daily_forecast_df["forecast_date"])

    date_axis = alt.Axis(title=None, format="%b %d", labelAngle=-30, grid=True)

    TEAL = "#2dd4bf"
    TEMP_FC = "#ef4444"
    RAIN_FC = "#60a5fa"
    GREY = "#94a3b8"
    SERIES = ["Actual (recent history)", "Forecast (LSTM)", "Trend"]
    dash_scale = alt.Scale(domain=SERIES, range=[[1, 0], [7, 4], [4, 4]])

    last_hist = recent.iloc[-1]

    def _forecast_series(col, fallback):
        if not daily_forecast_df.empty and col in daily_forecast_df.columns:
            return daily_forecast_df[["Date", col]].rename(columns={col: "Value"}).copy()
        if fallback is not None:
            try:
                return pd.DataFrame([{"Date": pd.to_datetime(forecast_date), "Value": float(fallback)}])
            except (TypeError, ValueError):
                return pd.DataFrame(columns=["Date", "Value"])
        return pd.DataFrame(columns=["Date", "Value"])

    def _widening_cone(anchor_val, fc_df, std):
        """Builds a cone that starts at zero width at anchor and widens over forecast steps."""
        rows = [{"Date": last_hist["Date"], "lower": anchor_val, "upper": anchor_val}]
        for i, r in enumerate(fc_df.itertuples(), 1):
            half = std * np.sqrt(i) * 0.35
            rows.append({"Date": r.Date, "lower": r.Value - half, "upper": r.Value + half})
        return pd.DataFrame(rows)

    def _rolling_trend(col, fc_df=None):
        t = history[["Date", col]].copy()
        t["Value"] = t[col].rolling(30, min_periods=1).mean()
        recent_part = t[t["Date"] >= recent.iloc[0]["Date"]][["Date", "Value"]].copy()
        if fc_df is not None and not fc_df.empty:
            last_val = recent_part.iloc[-1]["Value"]
            ext = fc_df[["Date", "Value"]].copy()
            ext["Value"] = np.linspace(last_val, fc_df["Value"].mean(), len(ext))
            recent_part = pd.concat([recent_part, ext], ignore_index=True)
        return recent_part

    # ── MAX TEMPERATURE ─────────────────────────────────────────────
    temp_hist_df = recent[["Date", "Temperature_Max_C"]].rename(columns={"Temperature_Max_C": "Value"}).copy()
    temp_fc_df = _forecast_series("Temperature_Max_C", climate_result.get("Temperature_Max_C"))
    temp_bridge = pd.concat([temp_hist_df.tail(1), temp_fc_df], ignore_index=True)
    temp_std = float(recent["Temperature_Max_C"].std() or 1.0)
    temp_cone_df = _widening_cone(float(last_hist["Temperature_Max_C"]), temp_fc_df, temp_std) if not temp_fc_df.empty else pd.DataFrame(columns=["Date", "lower", "upper"])
    temp_trend_df = _rolling_trend("Temperature_Max_C", temp_fc_df if not temp_fc_df.empty else None)

    temp_all = pd.concat([
        temp_hist_df.assign(Series="Actual (recent history)"),
        temp_bridge.assign(Series="Forecast (LSTM)"),
        temp_trend_df.assign(Series="Trend"),
    ], ignore_index=True)

    temp_lines = alt.Chart(temp_all).mark_line(strokeWidth=2.2).encode(
        x=alt.X("Date:T", axis=date_axis),
        y=alt.Y("Value:Q", title="°C"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=SERIES, range=[TEAL, TEMP_FC, GREY]),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=11)),
        strokeDash=alt.StrokeDash("Series:N", scale=dash_scale, legend=None),
        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".1f", title="°C")],
    )
    temp_cone_layer = alt.Chart(temp_cone_df).mark_area(color=TEMP_FC, opacity=0.20).encode(
        x=alt.X("Date:T", axis=date_axis),
        y=alt.Y("lower:Q", title=None),
        y2=alt.Y2("upper:Q"),
    ) if not temp_cone_df.empty else alt.Chart(pd.DataFrame({"Date": [], "lower": [], "upper": []})).mark_area()

    temp_chart = alt.layer(temp_cone_layer, temp_lines).properties(width=440, height=300).configure_view(strokeOpacity=0)

    # ── RAIN CHANCE ──────────────────────────────────────────────────
    rain_hist_df = recent[["Date", "RainChance"]].rename(columns={"RainChance": "Value"}).copy()
    rain_fc_df = _forecast_series("Rainfall_Percent", climate_result.get("Rainfall_Percent"))
    rain_bridge = pd.concat([rain_hist_df.tail(1), rain_fc_df], ignore_index=True)
    rain_std = float(recent["RainChance"].std() or 1.0)
    rain_cone_df = _widening_cone(float(last_hist["RainChance"]), rain_fc_df, rain_std) if not rain_fc_df.empty else pd.DataFrame(columns=["Date", "lower", "upper"])
    rain_trend_df = _rolling_trend("RainChance", rain_fc_df if not rain_fc_df.empty else None)

    rain_all = pd.concat([
        rain_hist_df.assign(Series="Actual (recent history)"),
        rain_bridge.assign(Series="Forecast (LSTM)"),
        rain_trend_df.assign(Series="Trend"),
    ], ignore_index=True)

    rain_lines = alt.Chart(rain_all).mark_line(strokeWidth=2.2).encode(
        x=alt.X("Date:T", axis=date_axis),
        y=alt.Y("Value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=SERIES, range=[TEAL, RAIN_FC, GREY]),
                        legend=alt.Legend(title=None, orient="top", labelFontSize=11)),
        strokeDash=alt.StrokeDash("Series:N", scale=dash_scale, legend=None),
        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".1f", title="%")],
    )
    rain_cone_layer = alt.Chart(rain_cone_df).mark_area(color=RAIN_FC, opacity=0.20).encode(
        x=alt.X("Date:T", axis=date_axis),
        y=alt.Y("lower:Q", title=None),
        y2=alt.Y2("upper:Q"),
    ) if not rain_cone_df.empty else alt.Chart(pd.DataFrame({"Date": [], "lower": [], "upper": []})).mark_area()

    rain_chart = alt.layer(rain_cone_layer, rain_lines).properties(width=440, height=300).configure_view(strokeOpacity=0)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Max temperature (°C)**")
        st.altair_chart(temp_chart, use_container_width=True)
    with c2:
        st.markdown("**Rain chance (%)**")
        st.altair_chart(rain_chart, use_container_width=True)

    n_fc = len(daily_forecast_df) if not daily_forecast_df.empty else 1
    fc_end = daily_forecast_df["Date"].max().strftime("%Y-%m-%d") if not daily_forecast_df.empty else forecast_date
    st.caption(
        f"Teal shows the last 14 day(s) of actual readings on record; the day-by-day LSTM forecast for "
        f"{forecast_date} → {fc_end} ({n_fc} day(s)) picks up from there, with a shaded band that widens "
        f"the further out the forecast reaches. Beyond the model's near-term window it's progressively blended toward "
        f"each month's historical average, so longer trips trend toward typical seasonal weather rather than a runaway forecast."
    )


def _render_budget_pie_chart(budget_result: dict) -> None:
    if not budget_result:
        st.info("No budget breakdown available.")
        return

    items = [
        ("Travel", float(budget_result.get("travel_cost_est", 0) or 0)),
        ("Stay", float(budget_result.get("stay_cost_est", 0) or 0)),
        ("Food", float(budget_result.get("food_cost_est", 0) or 0)),
        ("Entry Fees", float(budget_result.get("entry_fees_est", 0) or 0)),
        ("Tolls/Parking", float(budget_result.get("tolls_and_parking_est", 0) or 0)),
    ]
    df = pd.DataFrame(items, columns=["Item", "Amount"]).query("Amount > 0")
    if df.empty:
        st.info("Breakdown amounts are zero or unavailable.")
        return

    bar = alt.Chart(df).mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
    ).encode(
        x=alt.X("Amount:Q", title="Estimated Cost (₹)", axis=alt.Axis(format=",.0f")),
        y=alt.Y("Item:N", sort="-x", title=""),
        color=alt.Color(
            "Item:N",
            legend=None,
            scale=alt.Scale(scheme="tableau10"),
        ),
        tooltip=[alt.Tooltip("Item:N"), alt.Tooltip("Amount:Q", title="₹", format=",.2f")],
    ).properties(title="Budget breakdown", height=200)

    with st.container():
        st.altair_chart(bar, use_container_width=True)


def _prepare_result_frames(budget_result: dict, crowd_result: dict, climate_result: dict):
    budget_df = pd.DataFrame([budget_result]) if budget_result else pd.DataFrame()
    if crowd_result and isinstance(crowd_result, dict) and crowd_result.get("rows"):
        crowd_df = pd.DataFrame(crowd_result["rows"])
    else:
        crowd_df = pd.DataFrame([crowd_result]) if crowd_result else pd.DataFrame()
    climate_df = pd.DataFrame([climate_result]) if climate_result else pd.DataFrame()

    # budget breakdown table
    breakdown = []
    if budget_result:
        breakdown = [
            {"Item": "Travel", "Amount": float(budget_result.get('travel_cost_est', 0) or 0)},
            {"Item": "Stay", "Amount": float(budget_result.get('stay_cost_est', 0) or 0)},
            {"Item": "Food", "Amount": float(budget_result.get('food_cost_est', 0) or 0)},
            {"Item": "Entry Fees", "Amount": float(budget_result.get('entry_fees_est', 0) or 0)},
            {"Item": "Tolls/Parking", "Amount": float(budget_result.get('tolls_and_parking_est', 0) or 0)},
        ]
    breakdown_df = pd.DataFrame(breakdown)
    return budget_df, breakdown_df, crowd_df, climate_df


def _trip_summary_image(plan: dict) -> bytes:
    """Create a clean, shareable PNG snapshot of the current trip plan."""
    width, height = 1400, 1080
    image = Image.new("RGB", (width, height), "#071124")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    title_font, subtitle_font = font(46, True), font(22)
    metric_font, label_font, body_font = font(32, True), font(18, True), font(22)
    draw.rounded_rectangle((0, 0, width, 230), radius=0, fill="#0d2347")
    draw.ellipse((1030, -210, 1530, 290), fill="#123b70")
    draw.ellipse((1130, -100, 1420, 190), fill="#1a5794")
    draw.text((70, 53), "SMARTYATRA", font=label_font, fill="#7dd3fc")
    draw.text((70, 86), "Your Trip Summary", font=title_font, fill="#f8fafc")

    start_date = plan.get("start_date")
    date_text = start_date.strftime("%d %b %Y") if hasattr(start_date, "strftime") else str(start_date or "Date not set")
    draw.text((70, 153), f"{plan.get('primary_district', 'Telangana')}  •  {date_text}", font=subtitle_font, fill="#cbd5e1")

    budget = plan.get("budget_result") or {}
    transport = plan.get("transport_result") or {}
    crowd = plan.get("crowd_result") or {}
    climate = plan.get("climate_result") or {}
    predicted_cost = float(budget.get("predicted_total_cost", 0) or 0)
    user_budget = float(plan.get("user_budget", 0) or 0)
    budget_status = "Within budget" if predicted_cost <= user_budget else "Over budget"

    cards = [
        ("ESTIMATED COST", f"INR {predicted_cost:,.0f}", budget_status),
        ("TRANSPORT", str(transport.get("recommended_transport_mode", "N/A")).title(), "Recommended mode"),
        ("TRIP LENGTH", f"{plan.get('duration_days', 0)} days", f"{plan.get('num_travelers', 0)} traveler(s)"),
        ("ROUTE DISTANCE", f"{float(plan.get('total_distance', 0) or 0):.0f} km", "Estimated route span"),
    ]
    card_w, card_h, x0, gap = 300, 150, 70, 25
    for index, (label, value, note) in enumerate(cards):
        x = x0 + index * (card_w + gap)
        draw.rounded_rectangle((x, 265, x + card_w, 265 + card_h), radius=22, fill="#10203c", outline="#24466f", width=2)
        draw.text((x + 22, 288), label, font=label_font, fill="#7dd3fc")
        draw.text((x + 22, 322), value, font=metric_font, fill="#f8fafc")
        draw.text((x + 22, 370), note, font=body_font, fill="#a9bdd5")

    draw.text((70, 470), "Trip highlights", font=font(28, True), fill="#f8fafc")
    draw.rounded_rectangle((70, 520, 1330, 885), radius=22, fill="#0d1b34", outline="#1e3a5f", width=2)
    left_x, right_x = 105, 735
    lines = [
        ("Crowd outlook", str(crowd.get("crowd_level", "Unavailable"))),
        ("Peak location", str(crowd.get("peak_spot", "Unavailable"))),
        ("Stay recommendation", str((plan.get("accommodation") or {}).get("name", "Unavailable"))),
        ("Climate signal", "Unavailable" if climate.get("error") else "Forecast included"),
    ]
    for index, (label, value) in enumerate(lines):
        column_x = left_x if index % 2 == 0 else right_x
        y = 560 + (index // 2) * 145
        draw.text((column_x, y), label.upper(), font=label_font, fill="#7dd3fc")
        draw.text((column_x, y + 35), value, font=font(27, True), fill="#f8fafc")

    selected_spots = plan.get("selected_spots") or []
    spot_names = [str(spot.get("name", "")) for spot in selected_spots if spot.get("name")]
    spots_text = ", ".join(spot_names) if spot_names else "Your selected destinations"
    draw.text((70, 930), "DESTINATIONS", font=label_font, fill="#7dd3fc")
    for line_number, line in enumerate(_wrap_image_text(spots_text, 82)):
        draw.text((70, 962 + line_number * 30), line, font=body_font, fill="#dbeafe")
    draw.text((1120, 1025), "Generated by SmartYatra", font=font(16), fill="#7890ae")

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _wrap_image_text(text: str, max_chars: int) -> list[str]:
    words, lines, current = text.split(), [], []
    for word in words:
        if len(" ".join(current + [word])) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def _export_results_buttons(budget_result: dict, crowd_result: dict, climate_result: dict, plan: dict | None = None) -> None:
    budget_df, breakdown_df, crowd_df, climate_df = _prepare_result_frames(budget_result, crowd_result, climate_result)

    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if plan:
        st.markdown("### Shareable Trip Summary")
        st.caption("Download a polished image containing your trip details and key predictions.")
        st.download_button(
            "Download trip summary image",
            data=_trip_summary_image(plan),
            file_name=f"smart_yatra_trip_summary_{now}.png",
            mime="image/png",
            type="primary",
        )
        st.divider()

    cols = st.columns(4)
    if not budget_df.empty:
        csv = budget_df.to_csv(index=False).encode('utf-8')
        cols[0].download_button("Download budget CSV", data=csv, file_name=f"budget_{now}.csv", mime='text/csv')
    if not breakdown_df.empty:
        csv2 = breakdown_df.to_csv(index=False).encode('utf-8')
        cols[1].download_button("Download budget breakdown", data=csv2, file_name=f"budget_breakdown_{now}.csv", mime='text/csv')
    if not crowd_df.empty:
        csv3 = crowd_df.to_csv(index=False).encode('utf-8')
        cols[2].download_button("Download crowd CSV", data=csv3, file_name=f"crowd_{now}.csv", mime='text/csv')
    if not climate_df.empty:
        csv4 = climate_df.to_csv(index=False).encode('utf-8')
        cols[3].download_button("Download climate CSV", data=csv4, file_name=f"climate_{now}.csv", mime='text/csv')

    # combined JSON
    combined = {
        "budget": budget_result or {},
        "budget_breakdown": breakdown_df.to_dict(orient='records'),
        "crowd": crowd_result or {},
        "climate": climate_result or {},
    }
    json_bytes = json.dumps(combined, default=str, indent=2).encode('utf-8')
    st.download_button("Export all JSON", data=json_bytes, file_name=f"smart_yatra_export_{now}.json", mime='application/json')

CROWD_CATEGORIES = ["Temple", "Fort", "Lake", "Museum", "Park"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
SEASONS_CROWD = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

# NOTE: kept for reference — no longer used by the rewritten Plan Your Trip page,
# since accommodation tier and season are now derived automatically rather than
# picked from these dropdowns (see "why this changed" note below).
ACCOMMODATION_TIERS = ["Budget", "Standard", "Premium", "Luxury"]
TRANSPORT_MODES = ["Auto", "Bus", "Car", "Train", "Plane"]
SEASONS_BUDGET = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

LOW_CROWD_MAX = 1_000
BUSY_CROWD_MAX = 7_000

FESTIVAL_CALENDAR = [
    {"name": "Sankranti / Pongal", "month": 1, "day": 14, "festival": "Bhogi, Sankranti/Pongal, New Year's Day, Republic Day"},
    {"name": "Holi & Ugadi", "month": 3, "day": 20, "festival": "Holi, Ugadi, Eid-ul-Fitr"},
    {"name": "Sri Rama Navami", "month": 4, "day": 10, "festival": "Sri Rama Navami, Ambedkar Jayanti"},
    {"name": "Telangana Formation Day", "month": 6, "day": 2, "festival": "Bakrid/Eid-ul-Adha, Telangana Formation Day"},
    {"name": "Bonalu", "month": 7, "day": 15, "festival": "Bonalu"},
    {"name": "Independence Day", "month": 8, "day": 15, "festival": "Independence Day"},
    {"name": "Vinayaka Chavithi", "month": 9, "day": 10, "festival": "Ganesh Chaturthi, Engili Pula Bathukamma (start)"},
    {"name": "Bathukamma & Dussehra", "month": 10, "day": 12, "festival": "Bathukamma (Saddula Bathukamma), Dussehra/Vijayadashami"},
    {"name": "Diwali", "month": 11, "day": 1, "festival": "Diwali/Deepavali"},
    {"name": "Christmas", "month": 12, "day": 25, "festival": "Christmas"},
]

HOME_IMAGE_PATH = os.path.join(BASE_DIR, "assets", "home_banner.jpg")

DISCLAIMER_TEXT = (
    "Predictions shown here are generated by machine learning models trained on "
    "historical data and are estimates only — actual costs, crowd levels, and "
    "weather may vary. This tool is meant to assist trip planning, not replace "
    "official sources."
)


# ============================================================
# NEW: EXPLORE + TRIP PLANNER DATA & HELPERS
# ============================================================
OTHER_SPOTS_PATH = os.path.join(DATA_DIR, "other_spots.csv")
ACCOMMODATIONS_PATH = os.path.join(DATA_DIR, "accommodations.csv")
AMENITIES_PATH = os.path.join(DATA_DIR, "nearby_amenities.csv")

CATEGORY_LABELS = {"religious": "Religious", "heritage": "Heritage", "nature": "Nature",
                    "leisure": "Leisure", "other": "Other"}
CATEGORY_VISIT_HOURS = {"religious": 2.5, "heritage": 4.0, "nature": 5.0, "leisure": 4.0, "other": 2.5}
DAILY_SIGHTSEEING_HOURS = 8.0
CITY_TRAVEL_SPEED_KMPH = 25.0

# The Budget model was trained on THESE exact season labels — a different taxonomy
# than SEASONS_CROWD above, since it's a different model trained on different data.
BUDGET_SEASON_BY_MONTH = {
    12: "Winter_Picnic", 1: "Winter_Picnic", 2: "Winter_Picnic",
    3: "Summer", 4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Festival_Peak", 11: "Festival_Peak",
}

TRANSPORT_SPEED_KMPH = {"car": 45, "bus": 35, "auto": 25, "bike": 30, "train": 50}


def _load_spots_master():
    if not os.path.exists(OTHER_SPOTS_PATH):
        return pd.DataFrame()
    return pd.read_csv(OTHER_SPOTS_PATH).drop(columns=["text_vec"], errors="ignore")


def _load_accommodations():
    if not os.path.exists(ACCOMMODATIONS_PATH):
        return pd.DataFrame()
    return pd.read_csv(ACCOMMODATIONS_PATH)


def _load_amenities():
    if not os.path.exists(AMENITIES_PATH):
        return pd.DataFrame()
    return pd.read_csv(AMENITIES_PATH)


SPOTS_MASTER_DF = _load_spots_master()
ACCOMMODATIONS_DF = _load_accommodations()
AMENITIES_DF = _load_amenities()
EXPLORE_DISTRICTS = sorted(SPOTS_MASTER_DF["district"].dropna().unique().tolist()) if not SPOTS_MASTER_DF.empty else []
DISTRICT_COORDS = _district_coords()


def _init_trip_cart():
    if "trip_cart" not in st.session_state:
        st.session_state["trip_cart"] = []


def _add_to_cart(spot_row: dict):
    if not any(s["id"] == spot_row["id"] for s in st.session_state["trip_cart"]):
        st.session_state["trip_cart"].append(spot_row)


def _remove_from_cart(spot_id):
    st.session_state["trip_cart"] = [s for s in st.session_state["trip_cart"] if s["id"] != spot_id]


def _set_trip_cart(new_cart: list[dict]) -> None:
    st.session_state["trip_cart"] = new_cart
    if st.session_state.get("last_plan"):
        st.session_state["last_plan"]["total_distance"] = _total_trip_distance(new_cart)


def _move_trip_stop(index: int, direction: int) -> None:
    cart = list(st.session_state.get("trip_cart", []))
    swap_index = index + direction
    if index < 0 or swap_index < 0 or index >= len(cart) or swap_index >= len(cart):
        return
    cart[index], cart[swap_index] = cart[swap_index], cart[index]
    _set_trip_cart(cart)


def _optimized_trip_order(cart: list[dict], start_coords: tuple[float, float] | None = None) -> list[dict]:
    if len(cart) < 2:
        return list(cart)

    remaining = list(cart)
    ordered = []

    if start_coords is None:
        ordered.append(remaining.pop(0))
        current_lat = float(ordered[0]["lat"])
        current_lon = float(ordered[0]["lon"])
    else:
        current_lat, current_lon = start_coords

    while remaining:
        next_stop = min(
            remaining,
            key=lambda stop: _haversine_km(current_lat, current_lon, float(stop["lat"]), float(stop["lon"])),
        )
        ordered.append(next_stop)
        current_lat = float(next_stop["lat"])
        current_lon = float(next_stop["lon"])
        remaining.remove(next_stop)

    return ordered


def _festival_crowd_aware_trip_order(
    cart: list[dict],
    crowd_map: dict[str, float],
    start_coords: tuple[float, float] | None = None,
) -> list[dict]:
    if len(cart) < 2:
        return list(cart)

    remaining = list(cart)
    ordered = []

    if start_coords is None:
        # Start from the least crowded stop if no custom start is provided.
        seed = min(remaining, key=lambda stop: crowd_map.get(str(stop.get("name", "")), 5000.0))
        ordered.append(seed)
        remaining.remove(seed)
        current_lat = float(seed["lat"])
        current_lon = float(seed["lon"])
    else:
        current_lat, current_lon = start_coords

    while remaining:
        next_stop = min(
            remaining,
            key=lambda stop: (
                _haversine_km(current_lat, current_lon, float(stop["lat"]), float(stop["lon"]))
                + (crowd_map.get(str(stop.get("name", "")), 4000.0) / 8000.0)
            ),
        )
        ordered.append(next_stop)
        current_lat = float(next_stop["lat"])
        current_lon = float(next_stop["lon"])
        remaining.remove(next_stop)

    return ordered


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _total_trip_distance(cart):
    """Sequential straight-line distance in cart order — a placeholder until real
    road-routing (OSRM) replaces this in the Maps phase."""
    if len(cart) < 2:
        return 0.0
    return round(sum(
        _haversine_km(cart[i]["lat"], cart[i]["lon"], cart[i + 1]["lat"], cart[i + 1]["lon"])
        for i in range(len(cart) - 1)
    ), 2)


def _recommend_duration(cart):
    if not cart:
        return 1

    visit_hours = sum(CATEGORY_VISIT_HOURS.get(spot.get("category"), 3.0) for spot in cart)
    optimized_cart = _optimized_trip_order(cart) if len(cart) > 1 else cart
    straight_line_km = _total_trip_distance(optimized_cart)
    estimated_road_km = straight_line_km * 1.25
    travel_hours = estimated_road_km / CITY_TRAVEL_SPEED_KMPH
    stop_change_hours = 0.2 * max(0, len(cart) - 1)
    total_hours = visit_hours + travel_hours + stop_change_hours

    return max(1, int(np.ceil(total_hours / DAILY_SIGHTSEEING_HOURS)))


def _budget_season_for_date(d):
    return BUDGET_SEASON_BY_MONTH[d.month]


def _estimate_travel_time(distance_km, transport_mode):
    speed = TRANSPORT_SPEED_KMPH.get(transport_mode, 35)
    hours = max(distance_km, 1) / speed
    h, m = int(hours), round((hours - int(hours)) * 60)
    return f"{h}h {m}m"


def _recommend_accommodation(district, stay_budget):
    if ACCOMMODATIONS_DF.empty:
        return None
    df = ACCOMMODATIONS_DF[ACCOMMODATIONS_DF["district"] == district]
    if df.empty:
        df = ACCOMMODATIONS_DF
    affordable = df[df["cost"] <= stay_budget].sort_values("cost", ascending=False)
    row = affordable.iloc[0] if not affordable.empty else df.sort_values("cost").iloc[0]
    return {"name": row["name"], "tier": row["tier"], "cost_per_night": float(row["cost"])}


def _build_ai_recommendations(plan: dict, district: str, duration_days: int, num_travelers: int) -> dict:
    budget_result = plan.get("budget_result") or {}
    predicted_cost = float(budget_result.get("predicted_total_cost", 0.0) or 0.0)
    user_budget = float(plan.get("user_budget", 0.0) or 0.0)
    over_budget = predicted_cost > user_budget
    budget_gap = max(0.0, predicted_cost - user_budget)

    stay_cost_est = float(budget_result.get("stay_cost_est", 0.0) or 0.0)
    travel_cost_est = float(budget_result.get("travel_cost_est", 0.0) or 0.0)
    food_cost_est = float(budget_result.get("food_cost_est", 0.0) or 0.0)
    misc_cost_est = float(budget_result.get("tolls_and_parking_est", 0.0) or 0.0)

    recommendations = []
    total_savings = 0.0

    # Accommodation suggestion based on per-night affordability within district.
    nights = max(int(duration_days), 1)
    current_stay_per_night = stay_cost_est / nights if stay_cost_est > 0 else 0.0
    if not ACCOMMODATIONS_DF.empty and current_stay_per_night > 0:
        district_df = ACCOMMODATIONS_DF[ACCOMMODATIONS_DF["district"] == district]
        if district_df.empty:
            district_df = ACCOMMODATIONS_DF
        cheaper_stays = district_df[district_df["cost"] < current_stay_per_night].sort_values("cost", ascending=True)
        if not cheaper_stays.empty:
            alt_stay = cheaper_stays.iloc[0]
            alt_stay_total = float(alt_stay["cost"]) * nights
            stay_saving = max(0.0, stay_cost_est - alt_stay_total)
            if stay_saving > 0:
                total_savings += stay_saving
                recommendations.append(
                    f"Stay at {alt_stay['name']} ({alt_stay['tier']}) to save about Rs {stay_saving:,.0f}."
                )

    # Transport suggestion using a relative cost index.
    current_mode = str((plan.get("transport_result") or {}).get("recommended_transport_mode", "car")).lower()
    distance = float(plan.get("total_distance", 0.0) or 0.0)
    mode_cost_index = {
        "plane": 2.8,
        "car": 1.55,
        "auto": 1.28,
        "train": 0.82,
        "bus": 0.72,
        "bike": 0.62,
    }
    current_index = mode_cost_index.get(current_mode, 1.2)
    candidate_modes = ["train", "bus", "auto", "bike"]
    if num_travelers >= 4:
        candidate_modes = ["train", "bus"]
    elif num_travelers == 3:
        candidate_modes = ["train", "bus", "auto"]

    lower_cost_modes = [m for m in candidate_modes if mode_cost_index.get(m, 99) < current_index]
    if lower_cost_modes and travel_cost_est > 0:
        preferred_mode = "train" if distance >= 120 and "train" in lower_cost_modes else min(
            lower_cost_modes,
            key=lambda mode: mode_cost_index[mode],
        )
        projected_travel = travel_cost_est * (mode_cost_index[preferred_mode] / current_index)
        travel_saving = max(0.0, travel_cost_est - projected_travel)
        if travel_saving > 0:
            total_savings += travel_saving
            recommendations.append(
                f"Use {preferred_mode.title()} transport for this route and save around Rs {travel_saving:,.0f}."
            )

    # Food + misc savings fallback so users always receive actionable advice.
    meal_misc_saving = (food_cost_est * 0.12) + (misc_cost_est * 0.15)
    if meal_misc_saving > 0:
        total_savings += meal_misc_saving
        recommendations.append(
            f"Choose local eateries and pre-book parking/tickets to save roughly Rs {meal_misc_saving:,.0f}."
        )

    if not recommendations:
        recommendations.append("Keep your current plan and track daily expenses to stay within your target budget.")

    if over_budget:
        headline = "You are over budget - here is the best correction plan"
        summary = f"You are about Rs {budget_gap:,.0f} above your target. These changes can save about Rs {total_savings:,.0f}."
        status_label = "Budget Recovery"
    else:
        headline = "You are within budget - here are extra savings"
        summary = f"Your plan is under control. These tweaks can still save about Rs {total_savings:,.0f} more."
        status_label = "Savings Boost"

    return {
        "headline": headline,
        "summary": summary,
        "status_label": status_label,
        "lines": recommendations,
        "estimated_savings": total_savings,
    }


# ============================================================
# PREDICTION DISPATCH
# (previously an HTTP call to a separate FastAPI backend — now dispatches
#  directly to the in-process predict_* functions defined above. Same
#  function signature/return contract as before, so every call site
#  elsewhere in this file needed zero changes.)
# ============================================================
def _safe_insert_prediction(table_name: str, payload: dict) -> None:
    """Same safe-insert pattern the former backend used (main.py) — logs a
    prediction to Supabase, printing (not crashing) on failure."""
    if supabase_client is None:
        return
    try:
        supabase_client.table(table_name).insert(payload).execute()
    except Exception as exc:
        print(f"Skipping Supabase insert for {table_name}: {exc}")


def _log_prediction(module: str, payload: dict, result: dict) -> None:
    """Per-module Supabase logging — mirrors exactly what each /predict/*
    endpoint in the former main.py logged, field for field."""
    if module == "budget":
        _safe_insert_prediction("budget_predictions", {
            "duration_days": payload.get("duration_days"),
            "num_travelers": payload.get("num_travelers"),
            "route_distance_km": payload.get("route_distance_km"),
            "accommodation_tier": payload.get("accommodation_tier"),
            "transport_mode": payload.get("transport_mode"),
            "season": payload.get("season"),
            "predicted_total_cost": result["predicted_total_cost"],
        })
    elif module == "crowd":
        _safe_insert_prediction("crowd_predictions", {
            "spot_name": payload.get("spot_name"),
            "district": payload.get("district"),
            "category": payload.get("category"),
            "year": payload.get("year"),
            "month": payload.get("month"),
            "season": payload.get("season"),
            "festival": payload.get("festival"),
            "predicted_total_visitors": result["predicted_total_visitors"],
        })
    elif module == "climate":
        if "error" not in result:
            log_fields = {k: v for k, v in result.items() if k != "daily_forecast"}
            _safe_insert_prediction("climate_predictions", {
                "district": payload.get("district"),
                "forecast_date": payload.get("forecast_date"),
                **log_fields,
            })
    elif module == "transport-mode":
        _safe_insert_prediction("transport_predictions", {
            "distance_km": payload.get("distance_km"),
            "budget_limit": payload.get("budget_limit"),
            "num_people": payload.get("num_people"),
            "rainfall_mm": payload.get("rainfall_mm"),
            "road_access_rating": payload.get("road_access_rating"),
            "recommended_transport_mode": result["recommended_transport_mode"],
            "confidence": result["confidence"],
        })


def call_predict_endpoint(module: str, payload: dict) -> dict | None:
    """Runs the matching prediction function in-process, logs the prediction
    to Supabase (matching the former backend's per-endpoint logging), and
    returns the result — or None on failure."""
    dispatch = {
        "climate": predict_climate,
        "crowd": predict_crowd,
        "transport-mode": predict_transport_mode,
        "budget": predict_budget,
    }
    func = dispatch.get(module)
    if func is None:
        st.error(f"Unknown prediction module: {module}")
        return None
    try:
        result = func(payload)
        _log_prediction(module, payload, result)
        return result
    except Exception as e:
        st.error(f"Prediction failed for {module}: {e}")
        return None

def _log_trip_request(payload: dict) -> None:
    """Fire-and-forget log of the user's raw trip inputs — direct Supabase
    insert now that backend and frontend are combined (previously an HTTP
    call to the backend's /log-trip-request endpoint). Silently no-ops on
    failure so a logging hiccup never blocks the actual trip planning flow."""
    if supabase_client is None:
        return
    try:
        supabase_client.table("trip_requests").insert(payload).execute()
    except Exception:
        pass


def _get_crowd_level(visitor_count: float) -> str:
    if visitor_count < LOW_CROWD_MAX:
        return "Low"
    if visitor_count < BUSY_CROWD_MAX:
        return "Busy"
    return "Overcrowded"


def _crowd_season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def _upcoming_festivals(visit_date, limit: int = 5) -> list[dict]:
    upcoming = []
    for year in (visit_date.year, visit_date.year + 1):
        for festival in FESTIVAL_CALENDAR:
            festival_date = date(year, festival["month"], festival["day"])
            if festival_date >= visit_date:
                upcoming.append({**festival, "date": festival_date})

    return sorted(upcoming, key=lambda item: item["date"])[:limit]


def _festivals_during_trip(start_date, duration_days: int = 1) -> list[dict]:
    """Return calendar events that occur during the selected trip dates."""
    if not start_date:
        return []

    trip_end = start_date + timedelta(days=max(1, int(duration_days)) - 1)
    matches = []
    for festival in FESTIVAL_CALENDAR:
        festival_date = date(start_date.year, festival["month"], festival["day"])
        if start_date <= festival_date <= trip_end:
            matches.append({**festival, "date": festival_date})
    return matches


FESTIVAL_TRAVEL_GUIDE = {
    "Sankranti / Pongal": {
        "categories": ["religious", "heritage"],
        "note": "Temple rituals and local cultural gatherings are usually strongest around this period.",
    },
    "Holi & Ugadi": {
        "categories": ["leisure", "nature"],
        "note": "Open and festive places are ideal for colorful celebrations and family day outings.",
    },
    "Sri Rama Navami": {
        "categories": ["religious", "heritage"],
        "note": "Choose devotional routes with temple heritage and calmer evening experiences.",
    },
    "Telangana Formation Day": {
        "categories": ["heritage", "other"],
        "note": "City landmarks and cultural centers often host local pride events and themed activities.",
    },
    "Bonalu": {
        "categories": ["religious", "heritage"],
        "note": "Popular temple clusters become vibrant, making early visits and nearby exploration valuable.",
    },
    "Independence Day & Krishna Janmashtami": {
        "categories": ["religious", "leisure"],
        "note": "A mix of festive celebrations and family-friendly public spots works well for this period.",
    },
    "Vinayaka Chavithi": {
        "categories": ["religious", "other"],
        "note": "Ganesh festival season supports spiritually focused itineraries with nearby amenities.",
    },
    "Bathukamma & Dussehra": {
        "categories": ["heritage", "nature"],
        "note": "Floral and cultural celebrations pair nicely with scenic and heritage locations.",
    },
    "Diwali": {
        "categories": ["leisure", "heritage"],
        "note": "Evening-friendly spots and well-connected city attractions are generally preferred.",
    },
    "Christmas": {
        "categories": ["leisure", "nature"],
        "note": "Relaxed festive outings at open attractions are popular during holiday travel.",
    },
}


def _festival_recommendations_for_district(
    district_spots: pd.DataFrame,
    anchor_date: date,
    limit: int = 3,
) -> list[dict]:
    if district_spots.empty:
        return []

    recommendations = []
    upcoming = _upcoming_festivals(anchor_date, limit=limit)
    ranked_spots = district_spots.sort_values(["popularity", "rating"], ascending=[False, False])

    for festival in upcoming:
        guide = FESTIVAL_TRAVEL_GUIDE.get(
            festival["name"],
            {
                "categories": ["heritage", "nature"],
                "note": "Pick high-rated spots and travel earlier in the day for a smoother experience.",
            },
        )

        category_spots = ranked_spots[ranked_spots["category"].isin(guide["categories"])]
        chosen_spots = category_spots.head(2) if not category_spots.empty else ranked_spots.head(2)
        spot_names = ", ".join(chosen_spots["name"].astype(str).tolist())
        recommended_spots = chosen_spots.to_dict(orient="records")

        recommendations.append(
            {
                "festival_name": festival["name"],
                "festival_date": festival["date"].strftime("%d %b %Y"),
                "festival_context": festival["festival"],
                "note": guide["note"],
                "spots": spot_names or "Explore top-rated district spots",
                "recommended_spots": recommended_spots,
            }
        )

    return recommendations


# ============================================================
# PAGE: HOME
# ============================================================
def _go_to_trip_planner() -> None:
    st.session_state["page_navigation"] = "Plan Your Trip"


def render_home():
    total_spots = len(SPOTS_MASTER_DF) if not SPOTS_MASTER_DF.empty else 0
    district_count = len(EXPLORE_DISTRICTS)
    climate_count = len(CLIMATE_DISTRICT_OPTIONS)

    st.markdown(
        f"""
        <div class='home-hero'>
            <div class='home-hero-tag'>SmartYatra • AI Trip Planner</div>
            <div class='home-hero-title'>Plan Better Trips, Not Just Cheaper Ones</div>
            <div class='home-hero-sub'>Budget, crowd pressure, and climate signals are merged into one planning workspace so your itinerary is data-backed before you leave home.</div>
            <div class='hero-metrics'>
                <div class='hero-metric'>
                    <div class='hero-metric-value'>{district_count}</div>
                    <div class='hero-metric-label'>Districts Covered</div>
                </div>
                <div class='hero-metric'>
                    <div class='hero-metric-value'>{total_spots}</div>
                    <div class='hero-metric-label'>Discoverable Spots</div>
                </div>
                <div class='hero-metric'>
                    <div class='hero-metric-value'>{climate_count}</div>
                    <div class='hero-metric-label'>Climate Districts</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(HOME_IMAGE_PATH):
        st.image(HOME_IMAGE_PATH, use_container_width=True)
    else:
        st.info("Banner image placeholder — add one at " + HOME_IMAGE_PATH)

    # ── 4. SCOPE BADGE ────────────────────────────────────────────
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;'
        'background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.35);color:#38bdf8;'
        'font-size:0.82rem;font-weight:600;margin-bottom:18px;">📍 Coverage: Telangana, India</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
### About SmartYatra
Travelers often face fragmented, disconnected information — weather apps,
budget spreadsheets, and generic travel blogs that don't talk to each other.
SmartYatra brings trip budget, crowd, and climate predictions together in
one place, so planning a trip means checking one page instead of five.
    """)

    # ── 1. FEATURE CARDS ──────────────────────────────────────────
    def _fcard(icon, title, desc, accent):
        return (
            f'<div style="border-radius:16px;padding:22px 20px;height:100%;'
            f'background:linear-gradient(135deg,#0b1428,#1e293b);border:1px solid {accent};">'
            f'<div style="font-size:2rem;margin-bottom:10px;">{icon}</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#f8fafc;margin-bottom:6px;">{title}</div>'
            f'<div style="font-size:0.83rem;color:#f8fafc;opacity:0.62;line-height:1.55;">{desc}</div>'
            f'</div>'
        )
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(_fcard("💰", "Trip Budget",
        "Itemised cost estimate across travel, stay, food, entry fees, and tolls — broken down per person and per day.",
        "rgba(56,189,248,0.3)"), unsafe_allow_html=True)
    fc2.markdown(_fcard("👥", "Crowd Demand",
        "Expected visitor footfall for your chosen spot and date, with upcoming festival impact and a crowd level rating.",
        "rgba(167,139,250,0.3)"), unsafe_allow_html=True)
    fc3.markdown(_fcard("🌦️", "Climate Forecast",
        "LSTM-powered temperature and rainfall outlook showing recent history alongside a day-by-day predicted trend.",
        "rgba(52,211,153,0.3)"), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    st.divider()

    # ── 3. QUICK-START GUIDE ──────────────────────────────────────
    st.markdown("### Quick Start")
    _qs = [
        ("1", "Open <b>Plan Your Trip</b> from the sidebar."),
        ("2", "Select a <b>district</b>, filter by <b>category</b>, and add spots to your trip with <b>+ My Trip</b>."),
        ("3", "Set your <b>start date</b>, review the AI-recommended <b>duration</b>, and enter travelers + budget."),
        ("4", "Hit <b>Plan my Trip</b> for transport mode, stay tier, distance, and an itemised budget — all in one place."),
        ("5", "Plan a Fun and Smooth trip for you and your Friends/Family"),
    ]
    for _n, _text in _qs:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:14px;padding:10px 0;border-top:1px solid rgba(148,163,184,0.12);">'
            f'<div style="min-width:28px;height:28px;border-radius:50%;background:rgba(56,189,248,0.15);'
            f'border:1px solid rgba(56,189,248,0.38);color:#38bdf8;font-size:0.8rem;font-weight:700;'
            f'display:flex;align-items:center;justify-content:center;flex-shrink:0;">{_n}</div>'
            f'<div style="font-size:0.9rem;color:#f8fafc;opacity:0.82;padding-top:4px;line-height:1.55;">{_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='travel-cta'>
            <div class='travel-cta-title'>Ready To Build Your Itinerary?</div>
            <div class='travel-cta-text'>Jump straight to trip planning with your selected places, date, and budget.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("Lets Travel", type="primary", on_click=_go_to_trip_planner, use_container_width=True)
    st.divider()

    # ── 6. SEASONAL TIP ───────────────────────────────────────────
    _mo = datetime.now(timezone.utc).month
    if _mo in (12, 1, 2):
        _sname, _stip, _sc = "Winter (Dec – Feb)", "Peak sightseeing season — cooler temperatures (15–25 °C) and clear skies. Book early as popular spots fill up fast.", "#38bdf8"
    elif _mo in (3, 4, 5):
        _sname, _stip, _sc = "Summer (Mar – May)", "Intense heat expected (35–45 °C). Plan outdoor activities before 10 am, carry plenty of water, and consider hill-station destinations.", "#f97316"
    elif _mo in (6, 7, 8, 9):
        _sname, _stip, _sc = "Monsoon (Jun – Sep)", "Heavy rainfall season — waterfalls and lakes are at their most scenic. Pack rain gear and check road conditions before travelling.", "#60a5fa"
    else:
        _sname, _stip, _sc = "Post-Monsoon (Oct – Nov)", "Pleasant temperatures and low humidity. A great window for temples, forts, and outdoor parks before the winter rush.", "#4ade80"

    st.markdown(
        f'<div style="border-radius:14px;padding:16px 20px;margin-bottom:18px;'
        f'background:linear-gradient(135deg,#0b1428,#1e293b);border-left:4px solid {_sc};border:1px solid rgba(148,163,184,0.14);">'
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{_sc};margin-bottom:6px;">Seasonal tip · {_sname}</div>'
        f'<div style="font-size:0.9rem;color:#f8fafc;opacity:0.82;line-height:1.55;">{_stip}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 7. FESTIVAL SPOTLIGHT ─────────────────────────────────────
    _today = datetime.now(timezone.utc).date()
    _upcoming_f = []
    for _yr in (_today.year, _today.year + 1):
        for _fe in FESTIVAL_CALENDAR:
            _fd = date(_yr, _fe["month"], _fe["day"])
            if _fd >= _today:
                _upcoming_f.append((_fe["name"], _fd, (_fd - _today).days))
    _upcoming_f = sorted(_upcoming_f, key=lambda x: x[1])[:2]

    if _upcoming_f:
        st.markdown("### Upcoming Festivals in Telangana")
        _fest_cols = st.columns(len(_upcoming_f))
        for _col, (_fname, _fdate, _fdays) in zip(_fest_cols, _upcoming_f):
            _flabel = "Today!" if _fdays == 0 else f"In {_fdays} day{'s' if _fdays != 1 else ''}"
            _col.markdown(
                f'<div style="border-radius:14px;padding:20px;text-align:center;'
                f'background:linear-gradient(135deg,#0b1428,#1e293b);border:1px solid rgba(251,191,36,0.28);">'
                f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:#fbbf24;margin-bottom:8px;">{_flabel}</div>'
                f'<div style="font-size:1rem;font-weight:700;color:#f8fafc;margin-bottom:5px;">{_fname}</div>'
                f'<div style="font-size:0.82rem;color:#f8fafc;opacity:0.52;">{_fdate.strftime("%d %b %Y")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# PAGE: PROJECT OVERVIEW
# ============================================================
def render_overview():
    st.markdown(
        """
        <div class='overview-hero'>
            <div class='overview-hero-tag'>Academic Overview</div>
            <div class='overview-hero-title'>SmartYatra Project Documentation</div>
            <div class='overview-hero-sub'>A structured student-project view covering scope, data assets, feature engineering, model evaluation, API contracts, testing practices, and roadmap priorities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_catalog = [
        {"name": "Trip Budget", "file": "trip_budget_prediction_dataset.csv", "purpose": "Cost prediction training base", "focus": "duration, travelers, route, tier, transport, season"},
        {"name": "Crowd Demand", "file": "crowd_data.csv", "purpose": "Visitor demand modeling", "focus": "spot, district, category, month, season, festival"},
        {"name": "Climate", "file": "Climate_Dataset_Final.csv", "purpose": "LSTM weather forecasting", "focus": "district, date, temperature, humidity, rainfall, wind"},
        {"name": "Spots Master", "file": "other_spots.csv", "purpose": "Explore and trip curation inventory", "focus": "district, category, popularity, rating, entry fee, geo"},
        {"name": "Accommodations", "file": "accommodations.csv", "purpose": "Stay recommendation layer", "focus": "district, tier, cost"},
        {"name": "Festivals", "file": "festivals_geocoded.csv", "purpose": "Demand context and seasonal signal", "focus": "festival metadata and location"},
        {"name": "Nearby Amenities", "file": "nearby_amenities.csv", "purpose": "Locality quality expansion features", "focus": "amenity mix and proximity"},
        {"name": "Spot Visitors", "file": "spot_visitors.csv", "purpose": "Fine-grained demand references", "focus": "spot-level visitor trends"},
        {"name": "Transport Mode", "file": "transport_mode_dataset.csv", "purpose": "Mode recommendation training", "focus": "distance, budget, people, rainfall, road access"},
    ]

    profiles = []
    total_rows = 0
    available_count = 0
    for item in dataset_catalog:
        path = os.path.join(DATA_DIR, item["file"])
        if os.path.exists(path):
            try:
                frame = pd.read_csv(path)
                rows = len(frame)
                cols = frame.shape[1]
                total_rows += rows
                available_count += 1
                status = "Available"
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
                rows = 0
                cols = 0
                status = "Unreadable"
        else:
            rows = 0
            cols = 0
            status = "Missing"

        profiles.append({
            "Dataset": item["name"],
            "File": item["file"],
            "Rows": rows,
            "Columns": cols,
            "Purpose": item["purpose"],
            "Feature Focus": item["focus"],
            "Status": status,
        })

    profiles_df = pd.DataFrame(profiles)

    try:
        crowd_df = pd.read_csv(CROWD_DATA_PATH)
        crowd_districts = int(crowd_df["district"].nunique()) if "district" in crowd_df.columns else 0
        unique_spots = int(crowd_df["spot_name"].nunique()) if "spot_name" in crowd_df.columns else 0
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        crowd_districts = 0
        unique_spots = 0

    try:
        climate_df = pd.read_csv(CLIMATE_DATA_PATH)
        climate_districts = int(climate_df["District"].nunique()) if "District" in climate_df.columns else 0
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        climate_districts = 0

    st.markdown(
        """
        <div class='summary-banner'>
            <h4>Student Project Documentation</h4>
            <p>SmartYatra integrates data engineering, ML modeling, API serving, and UX into one decision-support application for Telangana trip planning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='overview-chip-row'>
            <span class='overview-chip'>ML</span>
            <span class='overview-chip'>Data Engineering</span>
            <span class='overview-chip'>FastAPI</span>
            <span class='overview-chip'>Streamlit UX</span>
            <span class='overview-chip'>Telangana Scope</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class='overview-stat-card'><div class='overview-stat-value'>{available_count}/{len(dataset_catalog)}</div><div class='overview-stat-label'>Datasets Available</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='overview-stat-card'><div class='overview-stat-value'>{total_rows:,}</div><div class='overview-stat-label'>Total Data Rows</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='overview-stat-card'><div class='overview-stat-value'>{crowd_districts}</div><div class='overview-stat-label'>Crowd Districts</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='overview-stat-card'><div class='overview-stat-value'>{unique_spots}</div><div class='overview-stat-label'>Mapped Spots</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class='overview-stat-card'><div class='overview-stat-value'>{climate_districts}</div><div class='overview-stat-label'>Climate Districts</div></div>""", unsafe_allow_html=True)

    st.divider()

    tab_scope, tab_data, tab_features, tab_models, tab_api, tab_testing, tab_roadmap = st.tabs([
        "Scope and Assumptions",
        "Data Assets",
        "Features and Preprocessing",
        "Modeling and Evaluation",
        "API and Tech Stack",
        "Testing and Reproducibility",
        "Limitations and Future Work",
    ])

    with tab_scope:
        st.markdown("<div class='overview-note'>This section defines academic boundaries for fair evaluation of model quality and product behavior.</div>", unsafe_allow_html=True)
        st.markdown("### Project Objective")
        st.markdown("Build an integrated AI trip-planning assistant that jointly predicts budget, crowd, climate, and transport recommendations in a single workflow.")

        st.markdown("### In Scope")
        st.markdown("""
- Telangana-focused planning workflow.
- Budget, crowd, climate, and transport prediction modules.
- Explore-first spot curation and itinerary-oriented planning UX.
- Export-ready outputs for analysis and reporting.
        """)

        st.markdown("### Out of Scope")
        st.markdown("""
- Live booking and real-time transaction systems.
- Pan-India generalization.
- Real-time weather/events streaming in current version.
        """)

        st.markdown("### Core Assumptions")
        assumptions_df = pd.DataFrame([
            {"Assumption": "Historical behavior remains informative", "Why": "Models are trained on historical records and synthetic rules"},
            {"Assumption": "District-level climate is adequate", "Why": "Forecasting currently operates at district granularity"},
            {"Assumption": "Route approximation is acceptable", "Why": "Distance currently uses haversine placeholder"},
        ])
        st.dataframe(assumptions_df, use_container_width=True, hide_index=True)

    with tab_data:
        st.markdown("<div class='overview-note'>Dataset health and coverage are tracked from your current workspace files for reproducible review.</div>", unsafe_allow_html=True)
        st.markdown("### Dataset Inventory")
        st.dataframe(profiles_df, use_container_width=True, hide_index=True)

        size_df = profiles_df[profiles_df["Status"] == "Available"][["Dataset", "Rows"]].copy()
        if not size_df.empty:
            size_chart = (
                alt.Chart(size_df)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X("Rows:Q", axis=alt.Axis(format=","), title="Rows"),
                    y=alt.Y("Dataset:N", sort="-x", title=""),
                    color=alt.value("#38bdf8"),
                    tooltip=["Dataset:N", alt.Tooltip("Rows:Q", format=",")],
                )
                .properties(height=320, title="Dataset size distribution")
            )
            st.altair_chart(size_chart, use_container_width=True)

    with tab_features:
        st.markdown("<div class='overview-note'>Feature pipeline is documented module-wise so preprocessing logic can be audited during demo and viva.</div>", unsafe_allow_html=True)
        st.markdown("### Preprocessing Pipeline")
        prep_df = pd.DataFrame([
            {"Stage": "Data cleaning", "Operation": "Missing handling, type fixes, standard column naming"},
            {"Stage": "Categorical encoding", "Operation": "One-hot and ordinal encoding for budget and crowd contexts"},
            {"Stage": "Scaling", "Operation": "Numeric scaling for models sensitive to feature scale"},
            {"Stage": "Temporal prep", "Operation": "Date-indexed sequence formation and differencing for climate LSTM"},
            {"Stage": "Planning heuristics", "Operation": "Distance aggregation, duration recommendation, stay budget slicing"},
        ])
        st.dataframe(prep_df, use_container_width=True, hide_index=True)

        st.markdown("### Module-wise Feature System")
        feature_df = pd.DataFrame([
            {"Module": "Budget", "Feature Strategy": "Tier/season/transport encoding plus scaled trip numerics", "Target": "5 cost components + total"},
            {"Module": "Crowd", "Feature Strategy": "Mean encoded spot/district/month/season with festival context", "Target": "Visitor forecast"},
            {"Module": "Climate", "Feature Strategy": "District-level sequential weather deltas", "Target": "Temperature and rainfall forecast"},
            {"Module": "Transport", "Feature Strategy": "Distance, budget, people, rainfall, road access", "Target": "Mode + confidence"},
        ])
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    with tab_models:
        st.markdown("<div class='overview-note'>Evaluation summaries are presented in student-friendly terms: protocol, baseline comparison, and error behavior.</div>", unsafe_allow_html=True)
        st.markdown("### Training Protocol")
        protocol_df = pd.DataFrame([
            {"Module": "Budget", "Split": "80/20", "Validation Style": "Holdout", "Primary Metric": "MAE"},
            {"Module": "Crowd", "Split": "80/20", "Validation Style": "Holdout", "Primary Metric": "MAE"},
            {"Module": "Climate", "Split": "Chronological", "Validation Style": "Time-aware", "Primary Metric": "RMSE/MAE"},
            {"Module": "Transport", "Split": "80/20 stratified", "Validation Style": "Classification", "Primary Metric": "Accuracy"},
        ])
        st.dataframe(protocol_df, use_container_width=True, hide_index=True)

        st.markdown("### Baseline vs Final")
        baseline_df = pd.DataFrame([
            {"Module": "Budget", "Baseline": "Global mean estimate", "Final Model": "XGBoost multi-output", "Observed Improvement": "Lower MAE"},
            {"Module": "Crowd", "Baseline": "Seasonal mean visitors", "Final Model": "Random Forest regressor", "Observed Improvement": "Better spot-level fit"},
            {"Module": "Climate", "Baseline": "Monthly climatology", "Final Model": "LSTM sequence model", "Observed Improvement": "Better short-range dynamics"},
            {"Module": "Transport", "Baseline": "Rule-based thresholds", "Final Model": "Random Forest classifier", "Observed Improvement": "Higher classification consistency"},
        ])
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)

        st.markdown("### Error Analysis Snapshot")
        error_df = pd.DataFrame([
            {"Module": "Budget", "Typical Failure": "Outlier itineraries with rare tier-season combinations", "Mitigation": "More diverse trip examples"},
            {"Module": "Crowd", "Typical Failure": "Sudden event spikes not represented in history", "Mitigation": "Live event feed integration"},
            {"Module": "Climate", "Typical Failure": "Far-horizon flattening", "Mitigation": "Frequent retraining + ensembling"},
            {"Module": "Transport", "Typical Failure": "Borderline medium-distance ambiguity", "Mitigation": "Additional route quality features"},
        ])
        st.dataframe(error_df, use_container_width=True, hide_index=True)

    with tab_api:
        st.markdown("<div class='overview-note'>API contracts and technology stack are grouped together to show implementation architecture clearly.</div>", unsafe_allow_html=True)
        st.markdown("### API Contract Summary")
        api_df = pd.DataFrame([
            {"Endpoint": "/predict/budget", "Request": "TripInput", "Response": "Cost components + total", "Failure Cases": "Invalid schema, model/data mismatch"},
            {"Endpoint": "/predict/crowd", "Request": "CrowdInput", "Response": "Predicted visitors", "Failure Cases": "Unknown category combinations"},
            {"Endpoint": "/predict/climate", "Request": "ClimateInput", "Response": "Forecast values (+optional daily)", "Failure Cases": "Date before training anchor, unknown district"},
            {"Endpoint": "/predict/transport-mode", "Request": "TransportInput", "Response": "Recommended mode + confidence", "Failure Cases": "Feature range anomalies"},
        ])
        st.dataframe(api_df, use_container_width=True, hide_index=True)

        st.markdown("### Tech Stack")
        tech_df = pd.DataFrame([
            {"Layer": "Frontend", "Technology": "Streamlit", "Purpose": "Interactive planning and visualization UI"},
            {"Layer": "Backend", "Technology": "FastAPI", "Purpose": "Prediction orchestration and API serving"},
            {"Layer": "ML", "Technology": "XGBoost, scikit-learn, PyTorch", "Purpose": "Regression, classification, and sequence forecasting"},
            {"Layer": "Data", "Technology": "pandas, NumPy", "Purpose": "Preprocessing and feature engineering"},
            {"Layer": "Visualization", "Technology": "Altair", "Purpose": "Charts for analysis and explanation"},
            {"Layer": "Storage", "Technology": "CSV, SQLite, Supabase", "Purpose": "Datasets, intermediate store, and prediction logs"},
        ])
        st.dataframe(tech_df, use_container_width=True, hide_index=True)

    with tab_testing:
        st.markdown("<div class='overview-note'>Testing section focuses on reproducibility and demonstration readiness for academic assessment.</div>", unsafe_allow_html=True)
        st.markdown("### Reproducibility Checklist")
        checklist_df = pd.DataFrame([
            {"Item": "Dataset availability", "Status": "Verified via inventory table"},
            {"Item": "Model artefacts present", "Status": "Referenced in backend prediction pipeline"},
            {"Item": "Backend run command", "Status": "python -m uvicorn App.Backend.main:app --reload"},
            {"Item": "Frontend run command", "Status": "python -m streamlit run App/Frontend/app.py"},
        ])
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

        st.markdown("### Functional Test Matrix")
        test_df = pd.DataFrame([
            {"Scenario": "Budget prediction", "Expected": "Returns total + component costs", "Status": "Pass criteria defined"},
            {"Scenario": "Crowd prediction", "Expected": "Returns visitor estimate for selected context", "Status": "Pass criteria defined"},
            {"Scenario": "Climate forecast", "Expected": "Returns forecast for valid district and date", "Status": "Pass criteria defined"},
            {"Scenario": "Transport recommendation", "Expected": "Returns mode with confidence", "Status": "Pass criteria defined"},
            {"Scenario": "Export outputs", "Expected": "CSV/JSON downloads generated", "Status": "Pass criteria defined"},
        ])
        st.dataframe(test_df, use_container_width=True, hide_index=True)

        st.markdown("### Demo Scenarios")
        demo_df = pd.DataFrame([
            {"Scenario": "Weekend family trip", "Focus": "Budget + mode sensitivity with 3 to 4 spots"},
            {"Scenario": "Festival period travel", "Focus": "Crowd stress and climate caution"},
            {"Scenario": "Low-budget student group", "Focus": "Transport and stay optimization"},
        ])
        st.dataframe(demo_df, use_container_width=True, hide_index=True)

    with tab_roadmap:
        st.markdown("<div class='overview-note'>Limitations are explicitly mapped to priority actions so future scope can be justified academically.</div>", unsafe_allow_html=True)
        st.markdown("### Current Limitations")
        limitations_df = pd.DataFrame([
            {"Limitation": "No live weather/event feed", "Impact": "Cannot react to sudden demand or climate shifts"},
            {"Limitation": "Haversine route approximation", "Impact": "Travel-time estimates may be optimistic"},
            {"Limitation": "Telangana-only scope", "Impact": "Limited generalization"},
            {"Limitation": "Static model artefacts", "Impact": "Potential drift without scheduled retraining"},
        ])
        st.dataframe(limitations_df, use_container_width=True, hide_index=True)

        st.markdown("### Priority Roadmap")
        roadmap_df = pd.DataFrame([
            {"Priority": "P1", "Work Item": "Integrate road routing engine", "Goal": "Improve route distance and ETA realism"},
            {"Priority": "P1", "Work Item": "Stabilize backend env and endpoint reliability", "Goal": "Consistent end-to-end runs"},
            {"Priority": "P2", "Work Item": "Integrate live weather/events signal", "Goal": "Improve crowd and climate responsiveness"},
            {"Priority": "P2", "Work Item": "Add model performance tracking dashboard", "Goal": "Track drift and quality over time"},
            {"Priority": "P3", "Work Item": "Expand beyond Telangana", "Goal": "Broader deployment readiness"},
        ])
        st.dataframe(roadmap_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: CLIMATE & FESTIVALS
# ============================================================
def render_climate_festivals():
    st.title("Climate & Festivals")
    st.caption("Explore historical climate patterns and upcoming Telangana festivals before choosing your travel dates.")

    district_options = CLIMATE_DISTRICT_OPTIONS or EXPLORE_DISTRICTS
    if not district_options:
        st.info("Climate data is not available right now.")
        return

    default_district = "Hyderabad" if "Hyderabad" in district_options else district_options[0]
    district = st.selectbox(
        "District",
        district_options,
        index=district_options.index(default_district),
        key="climate_festivals_district",
    )
    monthly = _climate_monthly_stats(district)
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    st.markdown(f"### Historical Average Temperature Trend for {district}")
    if monthly.empty:
        st.info("Historical climate data is unavailable for this district.")
    else:
        st.altair_chart(
            alt.Chart(monthly).mark_line(point=True, strokeWidth=2, color="#38bdf8").encode(
                x=alt.X("MonthName:N", sort=month_order, title="Month"),
                y=alt.Y("AvgTemp:Q", title="Temperature (°C)"),
                tooltip=[
                    alt.Tooltip("MonthName:N", title="Month"),
                    alt.Tooltip("AvgTemp:Q", format=".1f", title="Average temperature (°C)"),
                    alt.Tooltip("AvgRainfall_mm:Q", format=".1f", title="Average rainfall (mm/day)"),
                ],
            ).properties(height=280),
            use_container_width=True,
        )
        current_month = datetime.now(timezone.utc).month
        current_month_row = monthly[monthly["Month"] == current_month]
        if not current_month_row.empty:
            st.info(_month_recommendation(
                float(current_month_row["AvgTemp"].iloc[0]),
                float(current_month_row["AvgRainfall_mm"].iloc[0]),
            ))

    st.divider()
    st.markdown("### Upcoming Festivals")
    upcoming = _upcoming_festivals(datetime.now(timezone.utc).date(), limit=8)
    if upcoming:
        festival_rows = [
            {
                "Date": festival["date"].strftime("%d %b %Y"),
                "Festival": festival["name"],
                "Festival context": festival["festival"],
            }
            for festival in upcoming
        ]
        st.dataframe(pd.DataFrame(festival_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No upcoming festivals are configured yet.")


# ============================================================
# PAGE: PLAN YOUR TRIP  (Explore + Trip Planner + Predictions)
# ============================================================
def render_predictive():
    _init_trip_cart()
    st.title("Plan Your Trip")
    st.markdown(
        f"""
        <div class='disclaimer-card'>
            <div class='disclaimer-icon'>✦</div>
            <div>
                <div class='disclaimer-title'>Planning note</div>
                <div class='disclaimer-text'>{DISCLAIMER_TEXT}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    theme = st.session_state.get("theme", "dark")
    existing_plan = st.session_state.get("last_plan", {})
    saved_draft = st.session_state.get("trip_plan_draft", {})

    # Streamlit clears widget-bound values when their page is not rendered.
    # Restore them from the independent draft before re-creating the inputs.
    for draft_key, draft_value in saved_draft.items():
        if draft_key.startswith("plan_") and draft_key not in st.session_state:
            st.session_state[draft_key] = draft_value

    if "plan_selected_district" not in st.session_state:
        default_district = existing_plan.get("primary_district")
        if not default_district and st.session_state["trip_cart"]:
            default_district = st.session_state["trip_cart"][0].get("district")
        st.session_state["plan_selected_district"] = default_district if default_district in EXPLORE_DISTRICTS else None
    if "plan_start_date" not in st.session_state:
        st.session_state["plan_start_date"] = existing_plan.get("start_date")
    if "plan_duration_days" not in st.session_state:
        st.session_state["plan_duration_days"] = int(existing_plan.get("duration_days", 1) or 1)
    if "plan_duration_user_overridden" not in st.session_state:
        st.session_state["plan_duration_user_overridden"] = bool(existing_plan)
    if "plan_num_travelers" not in st.session_state:
        st.session_state["plan_num_travelers"] = existing_plan.get("num_travelers")
    if "plan_user_budget" not in st.session_state:
        st.session_state["plan_user_budget"] = existing_plan.get("user_budget")

    # ============================================================
    # STEP 1 — District, then Spots filtered by district + category
    # ============================================================
    _render_step_card("1", "Choose Your Destinations", "Pick a district and curate your stops by category.")

    if SPOTS_MASTER_DF.empty:
        st.error(f"other_spots.csv not found at {OTHER_SPOTS_PATH}")
        return

    district_default = st.session_state.get("plan_selected_district")
    district_index = EXPLORE_DISTRICTS.index(district_default) if district_default in EXPLORE_DISTRICTS else None
    selected_district = st.selectbox(
        "District",
        EXPLORE_DISTRICTS,
        index=district_index,
        placeholder="Select a district...",
        key="plan_selected_district",
        on_change=_save_trip_plan_draft,
    )

    if not selected_district and st.session_state["trip_cart"]:
        inferred_district = st.session_state["trip_cart"][0].get("district")
        if inferred_district in EXPLORE_DISTRICTS:
            st.session_state["plan_selected_district"] = inferred_district
            selected_district = inferred_district

    if not selected_district:
        st.caption("Select a district to browse and filter its available spots.")

    if selected_district:
        district_spots = SPOTS_MASTER_DF[SPOTS_MASTER_DF["district"] == selected_district]
        available_categories = sorted(district_spots["category"].unique().tolist())
        category_display = [CATEGORY_LABELS.get(c, c.title()) for c in available_categories]
        festival_anchor_date = st.session_state.get("plan_start_date") or datetime.now(timezone.utc).date()
        festival_recos = _festival_recommendations_for_district(district_spots, festival_anchor_date, limit=3)

        if festival_recos:
            cart_ids = {s["id"] for s in st.session_state["trip_cart"] if isinstance(s, dict) and "id" in s}
            st.markdown("<div class='festival-showcase-head'>Upcoming Festivals and Smart Spot Picks</div>", unsafe_allow_html=True)
            festival_cols = st.columns(len(festival_recos))
            for idx, festival_item in enumerate(festival_recos):
                with festival_cols[idx]:
                    st.markdown(
                        f"""
                        <div class='festival-showcase-card'>
                            <div class='festival-date-pill'>{festival_item['festival_date']}</div>
                            <div class='festival-title'>{festival_item['festival_name']}</div>
                            <div class='festival-note'>{festival_item['note']}</div>
                            <div class='festival-note'><strong>Festival context:</strong> {festival_item['festival_context']}</div>
                            <div class='festival-reco'>Recommended places: {festival_item['spots']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for rec_spot in festival_item.get("recommended_spots", []):
                        spot_name = str(rec_spot.get("name", "Recommended Spot"))
                        spot_id = rec_spot.get("id")
                        if spot_id in cart_ids:
                            st.caption(f"{spot_name} already in My Trip")
                        else:
                            if st.button(f"+ Add {spot_name}", key=f"festival_add_{idx}_{spot_id}", use_container_width=True):
                                _add_to_cart(rec_spot)
                                st.rerun()

        with st.expander("Browse filters", expanded=False):
            st.caption("Choose categories and a sort order to narrow the visible spots.")
            filter_key = f"plan_category_filter_{selected_district}"
            if hasattr(st, "pills"):
                selected_display = st.pills(
                    "Category",
                    category_display,
                    selection_mode="multi",
                    key=filter_key,
                )
            else:
                selected_display = st.multiselect(
                    "Category",
                    category_display,
                    key=filter_key,
                )
            sort_options = ["Popularity", "Rating", "Entry Fee: Low to High", "Entry Fee: High to Low", "Name A-Z"]
            sort_key = f"plan_sort_{selected_district}"
            if hasattr(st, "pills"):
                sort_option = st.pills(
                    "Sort",
                    sort_options,
                    key=sort_key,
                )
            else:
                sort_option = st.radio(
                    "Sort",
                    sort_options,
                    index=None,
                    key=sort_key,
                )
            if st.button("Reset all fields", use_container_width=True, key="reset_plan_fields_btn"):
                _reset_plan_fields()
                st.rerun()
        display_to_raw = {CATEGORY_LABELS.get(c, c.title()): c for c in available_categories}
        selected_raw = [display_to_raw[d] for d in selected_display]

        filtered = district_spots[district_spots["category"].isin(selected_raw)] if selected_raw else district_spots

        search_term = st.text_input(
            "Search spots in this district",
            placeholder="Type a spot name...",
            key=f"plan_search_{selected_district}",
        ).strip()

        if search_term:
            filtered = filtered[
                filtered["name"].astype(str).str.contains(search_term, case=False, na=False)
            ]

        if sort_option == "Popularity":
            filtered = filtered.sort_values("popularity", ascending=False)
        elif sort_option == "Rating":
            filtered = filtered.sort_values("rating", ascending=False)
        elif sort_option == "Entry Fee: Low to High":
            filtered = filtered.sort_values("entry_fee", ascending=True)
        elif sort_option == "Entry Fee: High to Low":
            filtered = filtered.sort_values("entry_fee", ascending=False)
        elif sort_option == "Name A-Z":
            filtered = filtered.sort_values("name", ascending=True)

        browser_signature = (
            selected_district,
            tuple(sorted(selected_display)),
            search_term.lower(),
            sort_option,
        )
        if st.session_state.get("spot_browser_signature") != browser_signature:
            st.session_state["spot_browser_signature"] = browser_signature
            st.session_state["spot_browser_page"] = 1

        page_size = 6
        total_spots = len(filtered)
        total_pages = max(1, (total_spots + page_size - 1) // page_size)
        current_page = min(st.session_state.get("spot_browser_page", 1), total_pages)
        st.session_state["spot_browser_page"] = current_page

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        visible_spots = filtered.iloc[start_idx:end_idx]

        cart_ids = {s["id"] for s in st.session_state["trip_cart"]}

        st.markdown(
            f"""
            <div class='spot-browser-summary'>
                <div class='spot-browser-count'>{total_spots} spot(s) found in {selected_district}</div>
                <div class='spot-browser-note'>Page {current_page} of {total_pages}. Use search, category, and sort to narrow the list faster.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Showing {start_idx + 1 if total_spots else 0}-{min(end_idx, total_spots)} of {total_spots} results")

        card_cols = st.columns(2)
        for idx, (_, spot) in enumerate(visible_spots.iterrows()):
            with card_cols[idx % 2]:
                fee = "Free entry" if spot["entry_fee"] == 0 else f"₹{spot['entry_fee']:.0f} entry fee"
                st.markdown(
                    f"""
                    <div class='spot-card'>
                        <div class='spot-card-title'>{spot['name']}</div>
                        <div class='spot-card-tag-row'>
                            <span class='spot-card-tag'>{CATEGORY_LABELS.get(spot['category'], spot['category'].title())}</span>
                            <span class='spot-card-tag'>{spot['rating']:.1f} ★</span>
                        </div>
                        <div class='spot-card-meta'>District: {spot['district']}<br/>Popularity score: {spot['popularity']:.0f}</div>
                        <div class='spot-card-fee'>{fee}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if spot["id"] in cart_ids:
                    if st.button("Remove", key=f"rm_{spot['id']}", use_container_width=True):
                        _remove_from_cart(spot["id"])
                        st.rerun()
                else:
                    if st.button("+ My Trip", key=f"add_{spot['id']}", use_container_width=True):
                        _add_to_cart(spot.to_dict())
                        st.rerun()

        if total_spots == 0:
            st.info("No spots matched your current search and filter combination.")
        else:
            _, nav_prev, nav_next, _ = st.columns([2, 1, 1, 2])
            with nav_prev:
                if st.button("Previous", use_container_width=True, disabled=current_page <= 1, key="spot_page_prev"):
                    st.session_state["spot_browser_page"] = max(1, current_page - 1)
                    st.rerun()
            with nav_next:
                if st.button("Next", use_container_width=True, disabled=current_page >= total_pages, key="spot_page_next"):
                    st.session_state["spot_browser_page"] = min(total_pages, current_page + 1)
                    st.rerun()
            st.markdown(
                f"<div style='text-align:center; font-size:0.82rem; color:#94a3b8; margin-top:6px;'>Page {current_page} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Select a district to browse spots.")
        return

    st.divider()

    # ============================================================
    # STEP 2 — Cart review + start date
    # ============================================================
    cart = st.session_state["trip_cart"]

    if not cart:
        return

    _render_step_card("2", f"Your Selected Spots ({len(cart)})", "Review what you have added and set trip date + duration.")

    st.table(pd.DataFrame(cart)[["name", "district", "category", "entry_fee"]].rename(
        columns={"name": "Spot", "district": "District", "category": "Category", "entry_fee": "Entry Fee (₹)"}
    ))

    st.caption("Need to adjust your itinerary? Remove a selected spot below.")
    remove_cols = st.columns(3)
    for index, spot in enumerate(cart):
        with remove_cols[index % 3]:
            if st.button(f"− Remove {spot['name']}", key=f"remove_selected_{spot['id']}", use_container_width=True):
                _remove_from_cart(spot["id"])
                st.rerun()

    recommended_days = _recommend_duration(cart)
    if not st.session_state.get("plan_duration_user_overridden", False):
        st.session_state["plan_duration_days"] = recommended_days

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input(
            "Trip start date",
            value=st.session_state.get("plan_start_date"),
            key="plan_start_date",
            on_change=_save_trip_plan_draft,
        )
    with d2:
        duration_days = st.number_input(
            "Trip duration (days)",
            min_value=1,
            max_value=15,
            key="plan_duration_days",
            help=f"AI-recommended: {recommended_days} day(s) based on your selected spots — feel free to adjust.",
            on_change=_mark_duration_user_overridden,
        )

    st.divider()

    # ============================================================
    # STEP 3 — Travelers + budget
    # ============================================================
    if len(cart) > 1 and duration_days < recommended_days:
        st.warning(
            f"**Your itinerary may be too tight.** You selected {len(cart)} spots in {duration_days} day(s), "
            f"but we recommend at least {recommended_days} day(s) to explore them at a comfortable pace. "
            "Consider extending your trip to allow for travel time, queues, and time at each destination."
        )

    _render_step_card("3", "Trip Details", "Enter traveler count and your target budget range.")
    p1, p2 = st.columns(2)
    with p1:
        num_travelers = st.number_input(
            "Number of travelers",
            min_value=1,
            max_value=10,
            value=st.session_state.get("plan_num_travelers"),
            placeholder="e.g. 2",
            key="plan_num_travelers",
            on_change=_save_trip_plan_draft,
        )
    with p2:
        user_budget = st.number_input(
            "Your budget (₹)",
            min_value=100.0,
            value=st.session_state.get("plan_user_budget"),
            placeholder="e.g. 15000",
            key="plan_user_budget",
            on_change=_save_trip_plan_draft,
        )

    missing = []
    if not start_date: missing.append("Start Date")
    if not num_travelers: missing.append("Number of Travelers")
    if not user_budget: missing.append("Budget")

    st.divider()

    # ============================================================
    # STEP 4 — Trip Summary
    # ============================================================
    _render_step_card("4", "Trip Summary", "Confirm your final inputs before running predictions.")
    if missing:
        st.info(f"Fill in all fields to continue. Still missing: {', '.join(missing)}")
        return

    primary_district = cart[0]["district"]
    st.table(pd.DataFrame({
        "Field": ["Spots Selected", "Primary District", "Start Date", "Duration (days)", "Travelers", "Your Budget"],
        "Value": [str(len(cart)), str(primary_district), str(start_date), str(duration_days), str(num_travelers), f"₹{user_budget:,.0f}"]
    }))

    # ============================================================
    # STEP 5 — Plan my Trip → all predictions
    # ============================================================
    if st.button("Plan my Trip", type="primary"):
        st.session_state["show_ai_recommendations"] = False
        total_distance = _total_trip_distance(cart)

        _log_trip_request({
            "district": primary_district,
            "number_of_travelers": int(num_travelers),
            "start_date": str(start_date),
            "budget_amount": float(user_budget),
            "selected_spots": [
                {"name": s.get("name"), "category": s.get("category"), "district": s.get("district")}
                for s in cart
            ],
            "accommodation_required": True,
        })

        budget_season = _budget_season_for_date(start_date)
        crowd_month = MONTHS[start_date.month - 1]
        crowd_season = _crowd_season_for_month(start_date.month)
        trip_festivals = _festivals_during_trip(start_date, duration_days)
        crowd_festival = ", ".join(festival["festival"] for festival in trip_festivals) or "No major festival"

        with st.spinner("Checking climate outlook..."):
            climate_result = call_predict_endpoint("climate", {
                "district": primary_district,
                "forecast_date": str(start_date),
                "include_daily": True,
            })
        rainfall_estimate = (
            climate_result.get("Rainfall_Percent", 10.0)
            if climate_result and "error" not in climate_result else 10.0
        )

        crowd_rows = []
        with st.spinner("Forecasting crowd levels..."):
            for spot in cart:
                planned_spot_name = str(spot.get("name", "")).strip()
                fallback_spot_pool = DISTRICT_TO_SPOTS.get(spot.get("district"), SPOT_OPTIONS)
                model_spot_name = (
                    planned_spot_name
                    if planned_spot_name in SPOT_OPTIONS
                    else (fallback_spot_pool[0] if fallback_spot_pool else planned_spot_name)
                )

                crowd_payload = {
                    "spot_name": model_spot_name,
                    "district": spot.get("district", primary_district),
                    "category": SPOT_TO_CATEGORY.get(model_spot_name, "Temple"),
                    "year": int(start_date.year),
                    "month": crowd_month,
                    "season": crowd_season,
                    "festival": crowd_festival,
                }
                crowd_response = call_predict_endpoint("crowd", crowd_payload)
                if crowd_response and crowd_response.get("predicted_total_visitors") is not None:
                    visitor_count = float(crowd_response["predicted_total_visitors"])
                    crowd_rows.append({
                        "planned_spot": planned_spot_name,
                        "model_spot": model_spot_name,
                        "district": crowd_payload["district"],
                        "category": crowd_payload["category"],
                        "predicted_total_visitors": round(visitor_count, 0),
                        "crowd_level": _get_crowd_level(visitor_count),
                    })

        crowd_result = {
            "rows": crowd_rows,
            "avg_visitors": float(np.mean([row["predicted_total_visitors"] for row in crowd_rows])) if crowd_rows else 0.0,
            "crowd_level": _get_crowd_level(float(np.mean([row["predicted_total_visitors"] for row in crowd_rows]))) if crowd_rows else "Unknown",
            "peak_spot": max(crowd_rows, key=lambda row: row["predicted_total_visitors"])["planned_spot"] if crowd_rows else "N/A",
            "peak_visitors": max(crowd_rows, key=lambda row: row["predicted_total_visitors"])["predicted_total_visitors"] if crowd_rows else 0.0,
            "month": crowd_month,
            "season": crowd_season,
            "festival": crowd_festival,
        }

        with st.spinner("Recommending transport mode..."):
            transport_result = call_predict_endpoint("transport-mode", {
                "distance_km": max(total_distance, 1.0),
                "budget_limit": user_budget,
                "num_people": num_travelers,
                "rainfall_mm": rainfall_estimate,
                "road_access_rating": 3,
            })
        recommended_transport = transport_result["recommended_transport_mode"] if transport_result else "car"

        stay_budget_slice = user_budget * 0.40
        accommodation = _recommend_accommodation(primary_district, stay_budget_slice)

        with st.spinner("Planning your trip"):
            budget_result = call_predict_endpoint("budget", {
                "duration_days": duration_days,
                "num_travelers": num_travelers,
                "route_distance_km": max(total_distance, 1.0),
                "accommodation_tier": accommodation["tier"] if accommodation else "Mid",
                "transport_mode": recommended_transport,
                "season": budget_season,
            })

        st.session_state["last_plan"] = {
            "total_distance": total_distance, "accommodation": accommodation,
            "transport_result": transport_result, "budget_result": budget_result,
            "crowd_result": crowd_result,
            "climate_result": climate_result, "user_budget": user_budget,
            "duration_days": int(duration_days), "num_travelers": int(num_travelers),
            "primary_district": primary_district,
            "start_date": start_date,
            "selected_spots": cart,
        }

    plan = st.session_state.get("last_plan")
    if plan and plan["budget_result"]:
        st.divider()
        st.markdown("## Results")

        predicted_cost = plan["budget_result"]["predicted_total_cost"]
        diff = predicted_cost - plan["user_budget"]
        travel_time = _estimate_travel_time(plan["total_distance"], plan["transport_result"]["recommended_transport_mode"])
        status_class = "warn" if diff > 0 else "good"
        status_text = "Over Budget" if diff > 0 else "Within Budget"

        st.markdown(
            f"""
            <div class='results-hero'>
                <div class='results-hero-label'>Predicted Total Cost</div>
                <div class='results-hero-value'>₹{predicted_cost:,.0f}</div>
                <div class='results-hero-status {status_class}'>{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            _render_stat_card("Predicted Mode", plan["transport_result"]["recommended_transport_mode"].upper(),f"{plan['transport_result']['confidence']*100:.0f}% confidence", theme=theme)
        with r2:
            acc = plan["accommodation"]
            _render_stat_card("Optimal Stay Tier", acc["tier"] if acc else "N/A", acc["name"] if acc else "", theme=theme)
        with r3:
            _render_stat_card("Total Distance", f"{plan['total_distance']:.1f} km", "Route span", theme=theme)
        with r4:
            _render_stat_card("Est. Travel Time", travel_time, "Drive duration", theme=theme)
        with r5:
            if diff > 0:
                _render_stat_card("Predicted Cost", f"₹{predicted_cost:,.0f}", f"Budget Deficit: ₹{diff:,.0f}", theme=theme)
            else:
                _render_stat_card("Predicted Cost", f"₹{predicted_cost:,.0f}", f"Budget Surplus: ₹{abs(diff):,.0f}", theme=theme)

        br = plan["budget_result"]
        tab_budget, tab_crowd, tab_climate, tab_export = st.tabs(["Budget Breakdown", "Crowd Insights", "Climate Outlook", "Export"])

        with tab_budget:
            st.markdown("### Detailed Itemized Cost Breakdown")
            st.table(pd.DataFrame([
                {"Expense Category": "Accommodation (Stay)", "Predicted Cost": f"₹{br.get('stay_cost_est', 0):,.0f}"},
                {"Expense Category": "Food & Dining", "Predicted Cost": f"₹{br.get('food_cost_est', 0):,.0f}"},
                {"Expense Category": "Travel & Transport", "Predicted Cost": f"₹{br.get('travel_cost_est', 0):,.0f}"},
                {"Expense Category": "Sightseeing & Entry Fees", "Predicted Cost": f"₹{br.get('entry_fees_est', 0):,.0f}"},
                {"Expense Category": "Tolls & Parking / Misc", "Predicted Cost": f"₹{br.get('tolls_and_parking_est', 0):,.0f}"},
            ]))
            _render_budget_pie_chart(br)

        with tab_climate:
            if plan["climate_result"] and "error" not in plan["climate_result"]:
                trip_district = plan.get("primary_district", primary_district)
                trip_start = plan.get("start_date", start_date)
                trip_duration = int(plan.get("duration_days", duration_days))

                # ---- Current Conditions — genuinely live, no picker needed ----
                current = _live_current_conditions(trip_district)
                st.markdown("### Current Conditions")
                if current:
                    st.caption(f"🟢 Live via Open-Meteo — as of {current['as_of']}")
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        _render_climate_card("🌡️", "Temperature", f"{current['temperature']:.1f} °C", "")
                    with cc2:
                        _render_climate_card("💧", "Humidity", f"{current['humidity']:.0f}%", "")
                    with cc3:
                        _render_climate_card("🌧️", "Rainfall", f"{current['rainfall_mm']:.1f} mm", "")
                else:
                    st.caption("Live weather unavailable right now.")

                # ---- Day-by-day forecast for the trip's own dates ----
                forecast_window, source_label = _build_trip_climate_forecast(
                    trip_district, trip_start, trip_duration, plan["climate_result"]
                )
                if not forecast_window.empty:
                    is_live_forecast = forecast_window.get("source", pd.Series(dtype=str)).eq("live_forecast").all()
                    forecast_title = "Trip Weather Forecast" if is_live_forecast else "Long-Range Climate Outlook"
                    st.markdown(f"### {forecast_title}")
                    st.caption(source_label)

                    avg_temp_pred = ((forecast_window["Temperature_Max_C"] + forecast_window["Temperature_Min_C"]) / 2).mean()
                    avg_rain_pred = forecast_window["Rainfall_mm"].mean() if forecast_window["Rainfall_mm"].notna().any() else None
                    avg_humidity_pred = forecast_window["Humidity_Percent"].mean() if "Humidity_Percent" in forecast_window else None

                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        _render_climate_card("📈", "Predicted Average Temp", f"{avg_temp_pred:.1f} °C", "Across your trip dates")
                    with pc2:
                        rain_display = f"{avg_rain_pred:.1f} mm/day" if avg_rain_pred is not None else "N/A"
                        _render_climate_card("🌧️", "Predicted Avg Rainfall", rain_display, "", card_class="rain")
                    with pc3:
                        hum_display = f"{avg_humidity_pred:.0f}%" if avg_humidity_pred is not None else "N/A"
                        _render_climate_card("💧", "Humidity", hum_display, "")

                    st.markdown("### Date-by-Date Trip Forecast Breakdown")
                    table_rows = []
                    for _, r in forecast_window.iterrows():
                        table_rows.append({
                            "Date / Day": r["forecast_date"].strftime("%a, %b %d"),
                            "Avg Temp (°C)": round((r["Temperature_Max_C"] + r["Temperature_Min_C"]) / 2, 1),
                            "Max Temp (°C)": round(r["Temperature_Max_C"], 1),
                            "Min Temp (°C)": round(r["Temperature_Min_C"], 1),
                            "Rainfall (mm)": round(r["Rainfall_mm"], 1) if pd.notna(r["Rainfall_mm"]) else "N/A",
                            "Humidity (%)": round(r["Humidity_Percent"], 0) if pd.notna(r.get("Humidity_Percent")) else "N/A",
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            else:
                st.info("Climate forecast is unavailable for this plan.")

        with tab_crowd:
            crowd_summary = plan.get("crowd_result") or {}
            crowd_rows = crowd_summary.get("rows", [])
            if crowd_rows:
                st.markdown(
                    f"""
                    <div class='crowd-hero-card'>
                        <div class='crowd-hero-tag'>Crowd Forecast</div>
                        <div class='crowd-hero-title'>{crowd_summary.get('crowd_level', 'Busy')} pressure expected in {crowd_summary.get('season', 'this season')}</div>
                        <div class='crowd-hero-sub'>Prediction window: {crowd_summary.get('month', '')} {start_date.year}. Closest festival context: {crowd_summary.get('festival', 'None')}.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    _render_stat_card(
                        "Avg Visitors",
                        f"{crowd_summary.get('avg_visitors', 0):,.0f}",
                        "Across selected stops",
                        theme=theme,
                    )
                with c2:
                    _render_stat_card(
                        "Peak Spot",
                        crowd_summary.get("peak_spot", "N/A"),
                        f"{crowd_summary.get('peak_visitors', 0):,.0f} expected visitors",
                        theme=theme,
                    )
                with c3:
                    _render_stat_card(
                        "Crowd Level",
                        crowd_summary.get("crowd_level", "Unknown"),
                        "Model-based crowd intensity",
                        theme=theme,
                    )

                crowd_df = pd.DataFrame(crowd_rows).rename(
                    columns={
                        "planned_spot": "Planned Spot",
                        "model_spot": "Reference Spot",
                        "district": "District",
                        "category": "Category",
                        "predicted_total_visitors": "Predicted Visitors",
                        "crowd_level": "Crowd Level",
                    }
                )
                chart_df = crowd_df[["Planned Spot", "Predicted Visitors", "Crowd Level"]].copy()
                crowd_chart = (
                    alt.Chart(chart_df)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                    .encode(
                        x=alt.X("Predicted Visitors:Q", title="Predicted Visitors", axis=alt.Axis(format=",")),
                        y=alt.Y("Planned Spot:N", sort="-x", title=""),
                        color=alt.Color(
                            "Crowd Level:N",
                            scale=alt.Scale(
                                domain=["Low", "Busy", "Overcrowded"],
                                range=["#22c55e", "#f59e0b", "#ef4444"],
                            ),
                            legend=alt.Legend(title="Crowd Level", orient="top"),
                        ),
                        tooltip=[
                            alt.Tooltip("Planned Spot:N"),
                            alt.Tooltip("Predicted Visitors:Q", format=","),
                            alt.Tooltip("Crowd Level:N"),
                        ],
                    )
                    .properties(height=max(220, 36 * len(chart_df)), title="Crowd Load by Selected Spot")
                )
                st.altair_chart(crowd_chart, use_container_width=True)
                st.markdown("### Spot-wise Crowd Forecast")
                st.dataframe(crowd_df, use_container_width=True, hide_index=True)
            else:
                st.info("Crowd prediction is unavailable for this plan right now.")

        with tab_export:
            _export_results_buttons(
                plan["budget_result"], plan.get("crowd_result"), plan["climate_result"], plan
            )

        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='action-panel'>
                <div class='action-panel-title'>Next Step</div>
                <div class='action-panel-note'>Continue to route planning or ask AI for cost-optimized recommendations before confirming your trip.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        action_left, action_right = st.columns(2)
        with action_left:
            st.button("Route My Trip", type="primary", use_container_width=True, on_click=_go_to_route_trip)
            st.markdown("<div class='action-hint'>Open interactive map, stop re-ordering, and road route flow.</div>", unsafe_allow_html=True)
        with action_right:
            st.button("Get AI Recommendations", use_container_width=True, on_click=_show_ai_recommendations)
            st.markdown("<div class='action-hint'>Get cost-focused stay, transport, and money-saving suggestions.</div>", unsafe_allow_html=True)

        if st.session_state.get("show_ai_recommendations", False):
            ai_reco = _build_ai_recommendations(
                plan,
                plan.get("primary_district", primary_district),
                int(plan.get("duration_days", duration_days)),
                int(plan.get("num_travelers", num_travelers)),
            )
            recommendation_items = "".join(
                f"<li>{line}</li>" for line in ai_reco["lines"]
            )
            st.markdown(
                f"""
                <div class='ai-reco-card'>
                    <div class='ai-reco-pill'>{ai_reco['status_label']}</div>
                    <div class='ai-reco-title'>{ai_reco['headline']}</div>
                    <div class='ai-reco-note'>{ai_reco['summary']}</div>
                    <ul class='ai-reco-list'>{recommendation_items}</ul>
                    <div class='ai-reco-savings'>Estimated savings impact: Rs {ai_reco['estimated_savings']:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_route_trip():
    st.title("Route My Trip")

    cart = st.session_state.get("trip_cart", [])
    plan = st.session_state.get("last_plan")

    with st.expander("Map layers", expanded=False):
        st.markdown(
            """
            <div class='route-layers-card'>
                <div class='route-layers-title'>Map Layers</div>
                <div class='route-layers-note'>Control what appears on the route map.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if hasattr(st, "toggle"):
            show_route_layer = st.toggle("Show Route Line", value=True, key="route_show_route")
            show_accommodation_layer = st.toggle("Show Accommodations", value=True, key="route_show_accommodations")
            show_amenities_layer = st.toggle("Show Amenities", value=True, key="route_show_amenities")
        else:
            show_route_layer = st.checkbox("Show Route Line", value=True, key="route_show_route")
            show_accommodation_layer = st.checkbox("Show Accommodations", value=True, key="route_show_accommodations")
            show_amenities_layer = st.checkbox("Show Amenities", value=True, key="route_show_amenities")
        st.markdown("---")

    if not cart:
        st.info("Build a trip in Plan My Trip first, then come back here for route mapping.")
        if st.button("Back to Plan My Trip", use_container_width=True):
            st.session_state["page_navigation"] = "Plan Your Trip"
            st.rerun()
        return

    st.markdown("### Build Your Road Route")
    start_location = st.text_input(
        "Optional custom start location",
        placeholder="Example: Hyderabad Railway Station, Hyderabad",
        help="Leave empty to start from your first selected spot.",
    ).strip()

    route_points = []
    if start_location:
        try:
            custom_start = _geocode_location(start_location)
            if custom_start:
                route_points.append({
                    "name": f"Start: {custom_start['label']}",
                    "lat": custom_start["lat"],
                    "lon": custom_start["lon"],
                })
            else:
                st.warning("Could not geocode the custom start location. The route will begin from your first selected spot.")
        except requests.RequestException as exc:
            st.warning(f"Geocoding service is unavailable right now: {exc}")

    route_points.extend([
        {
            "name": spot["name"],
            "lat": float(spot["lat"]),
            "lon": float(spot["lon"]),
        }
        for spot in cart
    ])

    route_data = None
    if len(route_points) >= 2:
        try:
            route_data = _fetch_osrm_route(tuple((point["lat"], point["lon"]) for point in route_points))
        except requests.RequestException as exc:
            st.warning(f"Routing service is unavailable right now: {exc}")

    leg_breakdown = _build_leg_breakdown(route_points)

    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown("### Selected Stops")
        for index, spot in enumerate(cart):
            st.markdown(f"**{index + 1}. {spot['name']}**")
            st.caption(f"{spot['district']} · {CATEGORY_LABELS.get(spot['category'], spot['category'].title())}")

    with summary_right:
        st.markdown("### Route Summary")
        distance = f"{route_data['distance_km']:.1f} km" if route_data else (f"{plan['total_distance']:.1f} km" if plan else "Not available yet")
        duration = f"{route_data['duration_min']:.0f} min" if route_data else "Not available yet"
        transport_mode = plan["transport_result"]["recommended_transport_mode"].upper() if plan and plan.get("transport_result") else "Not available yet"
        st.table(pd.DataFrame({
            "Field": ["Stops", "Estimated Road Distance", "Estimated Drive Time", "Suggested Transport"],
            "Value": [str(len(cart)), str(distance), str(duration), str(transport_mode)],
        }))

        if leg_breakdown:
            st.markdown("### Leg-by-Leg Breakdown")
            st.dataframe(pd.DataFrame(leg_breakdown), use_container_width=True, hide_index=True)

    if len(route_points) >= 2:
        st.markdown("### Route Map")
        route_map = _build_route_map(
            route_points,
            route_data,
            cart,
            show_route_layer=show_route_layer,
            show_accommodation_layer=show_accommodation_layer,
            show_amenities_layer=show_amenities_layer,
        )
        st_folium(route_map, use_container_width=True, height=520)
    else:
        st.info("Add at least two route points to generate a map.")

    back_col, _ = st.columns([1, 3])
    with back_col:
        if st.button("Back to Plan My Trip", use_container_width=True):
            st.session_state["page_navigation"] = "Plan Your Trip"
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="SmartYatra", layout="wide")
    _inject_styles()

    st.markdown("<div class='app-shell'></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<div class='rail-logo-wrap'>{_rail_logo_html()}</div>", unsafe_allow_html=True)
        st.markdown("<div class='floating-nav-title'>SmartYatra</div>", unsafe_allow_html=True)
        if st.button("🏠  Home", key="nav_home", use_container_width=True):
            st.session_state["page_navigation"] = "Home"
            st.rerun()
        if st.button("📋  Overview", key="nav_overview", use_container_width=True):
            st.session_state["page_navigation"] = "Project Overview"
            st.rerun()
        if st.button("🧭  Plan Your Trip", key="nav_plan_trip", use_container_width=True):
            st.session_state["page_navigation"] = "Plan Your Trip"
            st.rerun()
        if st.button("🌍  Climate & Festivals", key="nav_climate_festivals", use_container_width=True):
            st.session_state["page_navigation"] = "Climate & Festivals"
            st.rerun()
    if "page_navigation" not in st.session_state:
        st.session_state["page_navigation"] = "Home"

    page = st.session_state["page_navigation"]

    if page == "Home":
        render_home()
    elif page == "Project Overview":
        render_overview()
    elif page == "Route My Trip":
        render_route_trip()
    elif page == "Climate & Festivals":
        render_climate_festivals()
    else:
        render_predictive()