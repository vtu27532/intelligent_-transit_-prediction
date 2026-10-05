"""FastAPI service for predictions, live vehicles, routes, and analytics."""

import json
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.model import DATA_PATH, METRICS_PATH, make_feature_frame, load_model
from backend.transit_data import ROUTES
from backend.traffic import get_tomtom_api_key, get_traffic_condition

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DB_PATH = ROOT / "data" / "transitpredict.sqlite3"

app = FastAPI(title="TransitPredict API", version="1.0.0", description="Arrival predictions for a synthetic Chennai transit demo network.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/css", StaticFiles(directory=FRONTEND / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND / "js"), name="js")


def initialize_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route TEXT NOT NULL,
            stop TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            predicted_delay REAL NOT NULL,
            predicted_arrival TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")


initialize_database()


class PredictionRequest(BaseModel):
    route: str
    stop: str
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    traffic: Literal["Light", "Moderate", "Heavy"]

    @field_validator("route")
    @classmethod
    def validate_route(cls, value: str) -> str:
        if value not in ROUTES:
            raise ValueError("Choose a route from the available route list.")
        return value

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str) -> str:
        if not any(value in config["stops"] for config in ROUTES.values()):
            raise ValueError("Choose a stop from the available stop list.")
        return value

    @model_validator(mode="after")
    def validate_route_stop_pair(self):
        if self.stop not in ROUTES[self.route]["stops"]:
            raise ValueError("Choose a stop served by the selected route.")
        return self


def prediction_model():
    try:
        return load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="The delay model is not trained yet. Run python train.py and restart the API.") from error


def display_time(value: datetime) -> str:
    """Use portable 12-hour labels on Windows and Unix."""
    return value.strftime("%I:%M %p").lstrip("0")


def map_weather_condition(weather_code: int, precipitation: float) -> str:
    if precipitation >= 7.5 or weather_code in {65, 67, 82, 95, 96, 99}:
        return "Heavy rain"
    if precipitation > 0 or weather_code in {51, 53, 55, 56, 57, 61, 63, 66, 80, 81}:
        return "Rain"
    if weather_code in {1, 2, 3, 45, 48}:
        return "Cloudy"
    return "Clear"


def get_current_weather(route: str, stop: str) -> dict:
    route_config = ROUTES[route]
    stop_index = route_config["stops"].index(stop)
    latitude, longitude = route_config["coordinates"][stop_index]
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,weather_code",
        "timezone": "auto",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read())
        current = payload["current"]
        temperature = float(current["temperature_2m"])
        precipitation = float(current["precipitation"])
        weather_code = int(current["weather_code"])
        if not math.isfinite(temperature) or not math.isfinite(precipitation) or precipitation < 0:
            raise ValueError("Open-Meteo returned invalid current weather values.")
        return {
            "condition": map_weather_condition(weather_code, precipitation),
            "source": "Open-Meteo",
            "is_live": True,
            "checked_at": current["time"],
            "temperature_c": round(temperature, 1),
            "precipitation_mm": round(precipitation, 1),
        }
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "condition": "Clear",
            "source": "Default weather (Open-Meteo unavailable)",
            "is_live": False,
            "temperature_c": None,
            "precipitation_mm": None,
        }


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/map")
def live_map_page():
    return FileResponse(FRONTEND / "map.html")


@app.get("/analytics")
def analytics_page():
    return FileResponse(FRONTEND / "analytics.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ROOT / "data" / "delay_model.joblib").exists()}


@app.get("/traffic/status")
def traffic_status():
    key_loaded = bool(get_tomtom_api_key())
    return {
        "key_loaded": key_loaded,
        "traffic_mode": "live" if key_loaded else "simulated",
    }


@app.get("/routes")
def get_routes():
    return [{"id": route_id, "name": config["name"], "stops": config["stops"]} for route_id, config in ROUTES.items()]


@app.get("/stops")
def get_stops(route: str | None = Query(default=None)):
    if route is not None and route not in ROUTES:
        raise HTTPException(status_code=404, detail="Route not found.")
    if route:
        return ROUTES[route]["stops"]
    return sorted({stop for config in ROUTES.values() for stop in config["stops"]})


@app.post("/predict")
def predict(request: PredictionRequest):
    hour, minute = map(int, request.time.split(":"))
    today = datetime.now().astimezone()
    scheduled = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
    weather = get_current_weather(request.route, request.stop)
    traffic = get_traffic_condition(request.route, request.stop, request.traffic)
    row = {
        "route": request.route,
        "stop": request.stop,
        "hour": hour,
        "day_of_week": scheduled.weekday(),
        "weather": weather["condition"],
        "traffic_level": traffic["condition"],
        "holiday_flag": 0,
    }
    try:
        delay = float(prediction_model().predict(make_feature_frame([row]))[0])
        metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail="Model metrics are unavailable. Run the training script again.") from error

    delay = max(-4.0, min(42.0, delay))
    uncertainty = float(metadata.get("confidence_minutes", 5.0))
    arrival = scheduled + timedelta(minutes=delay)
    low = arrival - timedelta(minutes=uncertainty)
    high = arrival + timedelta(minutes=uncertainty)
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute(
                "INSERT INTO predictions(route, stop, scheduled_time, predicted_delay, predicted_arrival, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (request.route, request.stop, scheduled.isoformat(), round(delay, 2), arrival.isoformat(), datetime.now().astimezone().isoformat()),
            )
    except sqlite3.Error as error:
        raise HTTPException(status_code=500, detail="Could not save this prediction.") from error

    return {
        "route": request.route,
        "route_name": ROUTES[request.route]["name"],
        "stop": request.stop,
        "scheduled_arrival": scheduled.isoformat(),
        "predicted_arrival": arrival.isoformat(),
        "predicted_arrival_label": display_time(arrival),
        "delay_minutes": round(delay, 1),
        "weather": weather,
        "traffic": traffic,
        "confidence_minutes": round(uncertainty, 1),
        "confidence_range": {"earliest": low.isoformat(), "latest": high.isoformat()},
        "confidence_labels": {"earliest": display_time(low), "latest": display_time(high)},
    }


def live_vehicle(route_id: str, vehicle_index: int, epoch: float) -> dict:
    config = ROUTES[route_id]
    coordinates = config["coordinates"]
    phase = (epoch / 55.0 + vehicle_index * 0.47 + list(ROUTES).index(route_id) * 0.21) % (len(coordinates) - 1)
    segment = int(phase)
    progress = phase - segment
    lat_a, lon_a = coordinates[segment]
    lat_b, lon_b = coordinates[segment + 1]
    latitude = lat_a + (lat_b - lat_a) * progress
    longitude = lon_a + (lon_b - lon_a) * progress
    eta = max(1, round((1.0 - progress) * 7 + vehicle_index * 1.5))
    next_stop = config["stops"][segment + 1]
    return {
        "id": f"{route_id}-{vehicle_index + 1}",
        "route": route_id,
        "route_name": config["name"],
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "next_stop": next_stop,
        "predicted_eta_minutes": eta,
        "updated_at": datetime.now().astimezone().isoformat(),
    }


@app.get("/live")
def get_live_vehicles():
    epoch = time.time()
    return {"updated_at": datetime.now().astimezone().isoformat(), "vehicles": [
        live_vehicle(route_id, vehicle_index, epoch)
        for route_id in ROUTES
        for vehicle_index in range(2)
    ]}


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Metrics are unavailable. Run python train.py first.")
    try:
        metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if DATA_PATH.exists():
            trips = pd.read_csv(DATA_PATH)
            by_hour = trips.groupby("hour")["delay_minutes"].mean().round(2)
            by_route = trips.groupby("route")["delay_minutes"].mean().round(2)
            metadata["delay_by_hour"] = [{"hour": int(hour), "delay": float(delay)} for hour, delay in by_hour.items()]
            metadata["delay_by_route"] = [{"route": route, "delay": float(delay)} for route, delay in by_route.items()]
        return metadata
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=500, detail="Could not read model analytics.") from error
