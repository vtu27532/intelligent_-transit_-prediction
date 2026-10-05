"""Create a realistic, repeatable synthetic dataset for model training."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from backend.transit_data import ROUTES, TRAFFIC_OPTIONS, WEATHER_OPTIONS
except ModuleNotFoundError:
    from transit_data import ROUTES, TRAFFIC_OPTIONS, WEATHER_OPTIONS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "trips.csv"


def generate_trips(rows: int = 24_000, seed: int = 42) -> pd.DataFrame:
    """Simulate service over many dates, routes, stops, and conditions."""
    rng = np.random.default_rng(seed)
    route_ids = list(ROUTES)
    route = rng.choice(route_ids, rows)
    stop_index = rng.integers(0, 5, rows)
    hour_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1.4, 1.5, 1.4, 1.2, 1.2, 1.3, 1.5, 1.7, 1.6, 1.1])
    hours = rng.choice(np.arange(5, 24), rows, p=hour_weights / hour_weights.sum())
    start = np.datetime64("2024-01-01T00:00")
    dates = start + rng.integers(0, 730, rows).astype("timedelta64[D]")
    weekdays = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    weather = rng.choice(WEATHER_OPTIONS, rows, p=[0.20, 0.20, 0.20, 0.27, 0.13])
    traffic = rng.choice(TRAFFIC_OPTIONS, rows, p=[0.35, 0.42, 0.23])
    holiday = (rng.random(rows) < 0.035).astype(int)

    # Rush hour, weather, and traffic create understandable delay patterns.
    rush = np.where(np.isin(hours, [7, 8, 9, 16, 17, 18]), 4.2, 0.0)
    traffic_effect = pd.Series(traffic).map({"Light": 0.3, "Moderate": 2.4, "Heavy": 6.8}).to_numpy()
    weather_effect = pd.Series(weather).map({"Clear": 0.0, "Cloudy": 0.5, "Humid": 1.0, "Rain": 2.2, "Heavy rain": 5.0}).to_numpy()
    route_effect = pd.Series(route).map({"R1": 2.0, "R2": 0.8, "R3": 1.6, "R4": 1.1}).to_numpy()
    stop_effect = stop_index * 0.55
    weekend_effect = np.where(weekdays >= 5, -0.8, 0.0)
    holiday_effect = holiday * -1.1
    noise = rng.normal(0, 2.8, rows)
    delay = np.clip(1.2 + rush + traffic_effect + weather_effect + route_effect + stop_effect + weekend_effect + holiday_effect + noise, -4, 42)

    minutes = rng.integers(0, 60, rows)
    scheduled = dates + hours.astype("timedelta64[h]") + minutes.astype("timedelta64[m]")
    actual = scheduled + np.rint(delay).astype("timedelta64[m]")
    stop_names = [ROUTES[route_id]["stops"][idx] for route_id, idx in zip(route, stop_index)]

    return pd.DataFrame({
        "route": route,
        "stop": stop_names,
        "scheduled_time": pd.to_datetime(scheduled).astype(str),
        "actual_arrival": pd.to_datetime(actual).astype(str),
        "hour": hours,
        "day_of_week": weekdays,
        "weather": weather,
        "traffic_level": traffic,
        "holiday_flag": holiday,
        "delay_minutes": np.round(delay, 2),
    })


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    trips = generate_trips()
    trips.to_csv(OUTPUT, index=False)
    print(f"Created {len(trips):,} trips at {OUTPUT}")
    print(f"Average delay: {trips['delay_minutes'].mean():.1f} min")


if __name__ == "__main__":
    main()
