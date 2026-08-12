import os
import joblib
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PICKLES_DIR = os.path.join(PROJECT_ROOT, "Pickles")

files = [
    os.path.join(PICKLES_DIR, "Budget", "best_trip_cost_model.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "crowd_model.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "preprocessor.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "spot_mean_map.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "district_mean_map.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "month_mean_map.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "season_mean_map.pkl"),
    os.path.join(PICKLES_DIR, "Crowd", "global_visitor_mean.pkl"),
    os.path.join(PICKLES_DIR, "Climate", "best_climate_metadata.pkl"),
    os.path.join(PICKLES_DIR, "Climate", "best_climate_lstm_model.pt.zip"),
]

for path in files:
    if not os.path.exists(path):
        print("MISSING:", path)
        continue
    if path.lower().endswith((".pkl", ".joblib")):
        obj = joblib.load(path)
        joblib.dump(obj, path)
        print("Re-saved:", path)
    else:
        obj = torch.load(path, map_location="cpu")
        torch.save(obj, path)
        print("Re-saved:", path)