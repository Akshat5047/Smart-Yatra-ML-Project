from fastapi import FastAPI

try:
    from .schemas import TripInput, CrowdInput, ClimateInput, TransportInput
    from .predict import predict_budget, predict_crowd, predict_climate, predict_transport_mode
    from .database import supabase
except ImportError:
    from schemas import TripInput, CrowdInput, ClimateInput, TransportInput, TripRequestInput
    from predict import predict_budget, predict_crowd, predict_climate, predict_transport_mode
    from database import supabase

app = FastAPI(title="SmartYatra Prediction API")


def _safe_insert_prediction(table_name: str, payload: dict) -> None:
    try:
        supabase.table(table_name).insert(payload).execute()
    except Exception as exc:
        print(f"Skipping Supabase insert for {table_name}: {exc}")


@app.post("/predict/budget")
def predict_budget_endpoint(data: TripInput):
    result = predict_budget(data.dict())

    _safe_insert_prediction("budget_predictions", {
        "duration_days": data.duration_days,
        "num_travelers": data.num_travelers,
        "route_distance_km": data.route_distance_km,
        "accommodation_tier": data.accommodation_tier,
        "transport_mode": data.transport_mode,
        "season": data.season,
        "predicted_total_cost": result["predicted_total_cost"]
    })

    return result


@app.post("/predict/crowd")
def predict_crowd_endpoint(data: CrowdInput):
    result = predict_crowd(data.dict())

    _safe_insert_prediction("crowd_predictions", {
        "spot_name": data.spot_name,
        "district": data.district,
        "category": data.category,
        "year": data.year,
        "month": data.month,
        "season": data.season,
        "festival": data.festival,
        "predicted_total_visitors": result["predicted_total_visitors"]
    })

    return result


@app.post("/predict/climate")
def predict_climate_endpoint(data: ClimateInput):
    result = predict_climate(data.dict())

    if "error" not in result:
        log_fields = {k: v for k, v in result.items() if k != "daily_forecast"}
        _safe_insert_prediction("climate_predictions", {
            "district": data.district,
            "forecast_date": data.forecast_date,
            **log_fields
        })

    return result


@app.post("/predict/transport-mode")
def predict_transport_mode_endpoint(data: TransportInput):
    result = predict_transport_mode(data.dict())

    _safe_insert_prediction("transport_predictions", {
        "distance_km": data.distance_km,
        "budget_limit": data.budget_limit,
        "num_people": data.num_people,
        "rainfall_mm": data.rainfall_mm,
        "road_access_rating": data.road_access_rating,
        "recommended_transport_mode": result["recommended_transport_mode"],
        "confidence": result["confidence"]
    })

    return result


@app.post("/log-trip-request")
def log_trip_request(data: TripRequestInput):
    supabase.table("trip_requests").insert({
        "district": data.district,
        "number_of_travelers": data.number_of_travelers,
        "start_date": data.start_date,
        "budget_amount": data.budget_amount,
        "selected_spots": data.selected_spots,
        "accommodation_required": data.accommodation_required,
    }).execute()
    return {"status": "logged"}
