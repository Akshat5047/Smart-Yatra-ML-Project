from pydantic import BaseModel

class TripInput(BaseModel):
    duration_days: int
    num_travelers: int
    route_distance_km: float
    accommodation_tier: str
    transport_mode: str
    season: str

class CrowdInput(BaseModel):
    spot_name: str
    district: str
    category: str
    year: int
    month: str
    season: str
    festival: str

class ClimateInput(BaseModel):
    district: str
    forecast_date: str
    include_daily: bool = False

class TransportInput(BaseModel):
    distance_km: float
    budget_limit: float
    num_people: int
    rainfall_mm: float
    road_access_rating: int
    
class TripRequestInput(BaseModel):
    district: str
    number_of_travelers: int
    start_date: str
    budget_amount: float
    selected_spots: list[dict]
    accommodation_required: bool = True    
