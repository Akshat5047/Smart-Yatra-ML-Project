# Smart-Yatra-ML-Project

**A multi-model AI trip planning assistant for Telangana, India.**

Smart Yatra tackles the "decision paralysis" travelers face when planning a trip — checking a weather app, a budget spreadsheet, a crowd-avoidance blog, and a maps app separately. It brings **budget prediction, crowd forecasting, climate outlook, and transport mode recommendation** together into one connected planning workflow, backed by four independently trained ML models running in a single Streamlit app.

---

## What It Does

A user picks a district, browses and adds tourist spots to their trip, sets a start date and budget — and the app runs all four models together, each feeding into the next:

```
Climate (rainfall outlook)
      ↓
Transport Mode (needs rainfall as an input)
      ↓
Accommodation Tier (rule-based, from budget)
      ↓
Budget Prediction (needs transport mode + tier + distance + season)
      ↓
One unified result: predicted cost, deficit/surplus vs. your budget,
itemized breakdown, crowd outlook, and a day-by-day climate forecast
```

---

## Features

- **Explore & Trip Planner** — browse 271+ spots across Telangana's districts, filter by category (religious, heritage, nature, leisure), add to a trip cart, and get an AI-recommended trip duration based on your selected spots' visit-time and travel distance.
- **Budget Prediction** — itemized cost estimate (travel, stay, food, entry fees, tolls/parking) with a clear deficit/surplus comparison against your entered budget.
- **Crowd Forecasting** — per-spot predicted visitor footfall, with festival-period awareness baked into the prediction inputs.
- **Climate Outlook** — live current conditions and a live day-by-day forecast (via Open-Meteo) for trips within ~16 days out; automatically falls back to a custom-trained LSTM forecast + historical climatology for longer-range trips, clearly labeled either way.
- **Transport Mode Recommendation** — classifies the best transport mode (car/bus/auto/bike) from distance, budget, group size, and rainfall.
- **Route My Trip** — interactive map (Folium + OSRM) with real road routing, nearest-neighbor and festival/crowd-aware stop reordering, nearby amenities (ATMs, restaurants, hospitals) overlaid, and an optional custom starting location (geocoded via Nominatim).
- **AI Recommendations** — rule-based savings suggestions (cheaper accommodation, better transport mode, budget-friendly tweaks) generated from the trip's own prediction results.
- **Shareable Trip Summary** — exports a polished PNG summary card of the planned trip.
- **Climate & Festivals** — a standalone explorer for historical monthly climate trends and Telangana's festival calendar, independent of an active trip plan.

---

## Architecture

Everything runs as **one Streamlit application** — model loading, prediction logic, and the UI all live in a single process. There's no separate API server to deploy or keep in sync.

```
User Input (Streamlit UI)
      ↓
Prediction functions called directly, in-process
(Budget · Crowd · Climate · Transport Mode)
      ↓
Results rendered back into the same app
      ↓
Predictions + raw trip inputs logged to Supabase
```

An earlier version of this project split the app into a FastAPI backend and a separate Streamlit frontend, communicating over HTTP. That backend code is still kept in `App/Backend/` for local reference and experimentation, but it is **not used in deployment** — `App/WebApp/app.py` is fully self-contained and includes everything needed to run and predict on its own.

---

## Tech Stack

| Layer | Technology |
|---|---|
| App | Streamlit, Altair (charts), Folium + streamlit-folium (maps) |
| ML | XGBoost, scikit-learn, PyTorch |
| Data | pandas, NumPy |
| Storage | SQLite (`smart_tourism.db`), CSV, Supabase (PostgreSQL — prediction & trip-request logging) |
| External APIs | Open-Meteo (live weather, free/no-key), OSRM (routing), Nominatim (geocoding) |
| Model serialization | joblib (sklearn/XGBoost), native PyTorch `.pt` |
| Deployment | Streamlit Community Cloud |

---

## ML Modules

| Module | Algorithm | Predicts | Notes |
|---|---|---|---|
| **Budget** | XGBoost (Multi-Output Regressor) | 5 cost components + total | Custom `ColumnTransformer` (ordinal + one-hot + scaling) rebuilt at runtime from source data, since the original saved encoder artifacts didn't persist correctly |
| **Crowd** | Random Forest Regressor | Visitor footfall per spot | Mean-encoding pipeline (spot/district/month/season → target means) + one-hot for category/festival |
| **Climate** | LSTM (PyTorch, 2-layer) | Max/Min temperature, rainfall chance | Trained on `Climate_Dataset_Final.csv`; autoregressive multi-day forecasting; supplemented by live Open-Meteo data for near-term trips |
| **Transport Mode** | Random Forest Classifier | Recommended mode (car/bus/auto/bike) | ~95.5% test accuracy; trained on `transport_mode_dataset.csv` |

---

## Project Structure

```
ML Project Batch-1/
├── App/
│   ├── WebApp/
│   │   ├── app.py                 # Full app — UI, model loading, and prediction logic
│   │   ├── requirements.txt
│   │   ├── assets/
│   │   │   ├── home_banner.jpg
│   │   │   ├── logo.png
│   │   │   ├── telangana_neon_logo.png
│   │   │   └── telangana_neon_logo_source.png
│   │   └── .streamlit/
│   │       └── secrets.toml       # Local only — never committed
│   ├── Backend/                   # Not used in deployment — kept for local reference
│   │   ├── main.py
│   │   ├── predict.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── config.py
│   │   └── requirements.txt
│   └── Pickles/
│       ├── Budget/
│       ├── Crowd/
│       ├── Climate/
│       └── transport_mode_*.pkl   # Sits flat here, no subfolder
├── Data/
│   ├── smart_tourism.db
│   ├── trip_budget_prediction_dataset.csv
│   ├── crowd_data.csv
│   ├── Climate_Dataset_Final.csv
│   ├── other_spots.csv
│   ├── accommodations.csv
│   ├── nearby_amenities.csv
│   ├── festivals_geocoded.csv
│   └── transport_mode_dataset.csv
├── Notebooks/                     # Model training notebooks
├── .env                            # Local only — never committed
├── .gitignore
└── README.md
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Akshat5047/Smart-Yatra-ML-Project.git
cd Smart-Yatra-ML-Project
```

### 2. Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r App/WebApp/requirements.txt
```

### 4. Set up environment variables
Create a `.env` file at the project root:
```
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-service-role-key
```

### 5. Run the app
```bash
cd App/WebApp
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run, model loading (XGBoost, Random Forest, and the PyTorch LSTM) takes a few seconds before the interface becomes responsive.

---

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (already done for this project)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repository, branch `main`
4. **Main file path**: `App/WebApp/app.py`
5. Under **Settings → Secrets**, add:
   ```toml
   SUPABASE_URL = "your-supabase-url"
   SUPABASE_KEY = "your-supabase-key"
   ```
6. Deploy — first build takes several minutes, since it installs PyTorch, XGBoost, and scikit-learn directly

No separate backend deployment is needed — the entire app deploys as one Streamlit service.

---

## Known Limitations

- Trip distance between selected spots is computed via straight-line (haversine) distance for budget prediction purposes — the interactive route map uses real OSRM road routing separately.
- Scope is limited to Telangana, India.
- No live booking/availability integration — accommodation and transport suggestions are estimates, not real-time inventory.
- Climate forecasts beyond ~16 days blend toward historical monthly averages rather than extending the live forecast indefinitely.
- Free-tier hosting means the app may sleep after inactivity — the first request after idle time can take longer to respond while it wakes up.

---

## Roadmap

- [ ] Factor predicted trip distance more directly into the Budget model's input
- [ ] Add a live event/festival feed beyond the static calendar
- [ ] Broaden dataset coverage beyond Telangana

---

## Author

**Akshat** — [github.com/Akshat5047](https://github.com/Akshat5047)
