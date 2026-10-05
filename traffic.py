"""TomTom live traffic integration."""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv

from backend.transit_data import ROUTES

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)
logger.info("TOMTOM_API_KEY found: %s", bool(os.getenv("TOMTOM_API_KEY")))


def get_tomtom_api_key() -> str | None:
    load_dotenv(ENV_PATH)
    return os.getenv("TOMTOM_API_KEY")


def get_traffic_condition(route: str, stop: str, fallback: str) -> dict:
    api_key = get_tomtom_api_key()
    if not api_key:
        return {
            "condition": fallback,
            "source": "Manual selection (TOMTOM_API_KEY is not set)",
            "is_live": False,
        }

    route_config = ROUTES[route]
    stop_index = route_config["stops"].index(stop)
    latitude, longitude = route_config["coordinates"][stop_index]
    query = urlencode({
        "key": api_key,
        "point": f"{latitude},{longitude}",
        "unit": "kmph",
    })
    url = (
        "https://api.tomtom.com/traffic/services/4/"
        f"flowSegmentData/absolute/10/json?{query}"
    )
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("TomTom returned an unexpected response.")
        segment = payload.get("flowSegmentData", payload)
        if not isinstance(segment, dict):
            raise ValueError("TomTom returned an unexpected flow segment.")
        current_speed = float(segment["currentSpeed"])
        free_flow_speed = float(segment["freeFlowSpeed"])
        if (
            not math.isfinite(current_speed)
            or not math.isfinite(free_flow_speed)
            or current_speed < 0
            or free_flow_speed <= 0
        ):
            raise ValueError("TomTom returned invalid traffic speeds.")
        speed_ratio = current_speed / free_flow_speed
        is_closed = segment.get("roadClosure", False) is True
        condition = "Heavy" if is_closed or speed_ratio < 0.5 else "Moderate" if speed_ratio <= 0.8 else "Light"
        return {
            "condition": condition,
            "source": "TomTom Traffic API",
            "is_live": True,
            "checked_at": datetime.now().astimezone().isoformat(),
            "current_speed_kmph": round(current_speed, 1),
            "free_flow_speed_kmph": round(free_flow_speed, 1),
        }
    except HTTPError as error:
        reason = f"TomTom Traffic API returned HTTP {error.code}"
    except URLError:
        reason = "Could not connect to TomTom Traffic API"
    except (TimeoutError, OSError):
        reason = "TomTom Traffic API request timed out or failed"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        reason = "TomTom Traffic API returned invalid traffic data"

    return {
        "condition": fallback,
        "source": f"Manual selection (fallback: {reason})",
        "is_live": False,
    }
