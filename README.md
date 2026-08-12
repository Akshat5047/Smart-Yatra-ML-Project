# Smart-Yatra-ML-Project

**A multi-model AI trip planning assistant for Telangana, India.**

Smart Yatra tackles the "decision paralysis" travelers face when planning a trip — checking a weather app, a budget spreadsheet, a crowd-avoidance blog, and a maps app separately. It brings **budget prediction, crowd forecasting, climate outlook, and transport mode recommendation** together into one connected planning workflow, backed by four independently trained ML models working in sequence.

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

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Altair (charts), Folium + streamlit-folium (maps) |
| Backend | FastAPI, Uvicorn |
| ML | XGBoost, scikit-learn, PyTorch |
| Data | pandas, NumPy |
| Storage | SQLite (`smart_tourism.db`), CSV, Supabase (PostgreSQL — prediction & trip-request logging) |
| External APIs | Open-Meteo (live weather, free/no-key), OSRM (routing), Nominatim (geocoding) |
| Model serialization | joblib (sklearn/XGBoost), native PyTorch `.pt` |

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
│   ├── Backend/
│   │   ├── main.py            # FastAPI app, all /predict/* endpoints
│   │   ├── predict.py         # Model loading + prediction logic
│   │   ├── schemas.py         # Pydantic request models
│   │   ├── database.py        # Supabase client
│   │   ├── config.py          # .env loading
│   │   └── requirements.txt
│   ├── Frontend/
│   │   ├── app.py             # Streamlit app — all pages
│   │   ├── assets/            # Logos, banners
│   │   └── requirements.txt
│   └── Pickles/
│       ├── Budget/
│       ├── Crowd/
│       ├── Climate/
│       └── transport_mode_*.pkl   # Transport model artifacts sit flat here, no subfolder
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
├── Notebooks/                 # Model training notebooks
├── .gitignore
└── README.md
---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict/budget` | POST | Itemized trip cost prediction |
| `/predict/crowd` | POST | Visitor footfall prediction |
| `/predict/climate` | POST | Temperature & rainfall forecast |
| `/predict/transport-mode` | POST | Transport mode recommendation |
| `/log-trip-request` | POST | Logs raw user trip inputs to Supabase |

All prediction endpoints log to Supabase through a `_safe_insert_prediction()` wrapper — a failed or unreachable Supabase insert is caught and logged to the console rather than breaking the prediction response itself.

---

## Known Limitations

- Trip distance between selected spots is computed via straight-line (haversine) distance for budget prediction purposes — the interactive route map uses real OSRM road routing separately.
- Scope is limited to Telangana, India.
- No live booking/availability integration — accommodation and transport suggestions are estimates, not real-time inventory.
- Climate forecasts beyond ~16 days blend toward historical monthly averages rather than extending the live forecast indefinitely.

---

## Roadmap

- [ ] Expand routing to factor predicted trip distance directly into the Budget model's input
- [ ] Add live event/festival feed beyond the static calendar
- [ ] Broaden dataset coverage beyond Telangana

---

## License

*(Add your license here — MIT is a common choice for student/portfolio projects.)*

---

## Author

**Akshat** — [github.com/Akshat5047](https://github.com/Akshat5047)
