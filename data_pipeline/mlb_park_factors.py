from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import requests


STATCAST_PARK_FACTORS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?batSide=&condition=All&parks=mlb&rolling=1&stat=index_wOBA&type=distance-all&year={year}"
)


@lru_cache(maxsize=4)
def fetch_statcast_park_carry(year: int) -> dict[str, dict[str, Any]]:
    response = requests.get(STATCAST_PARK_FACTORS_URL.format(year=year), timeout=20)
    response.raise_for_status()
    text = response.text
    match = re.search(r"\[(\{\"venue_id\".+?\})\]", text)
    if not match:
        return {}
    rows = json.loads(f"[{match.group(1)}]")
    result = {}
    for row in rows:
        venue_name = row.get("venue_name")
        if not venue_name:
            continue
        distance_key = f"extra_distance_{year}"
        extra_distance = row.get(distance_key)
        result[str(venue_name)] = {
            "venue_id": row.get("venue_id"),
            "team_name": row.get("name_display_club"),
            "extra_distance": float(extra_distance) if extra_distance not in (None, "") else 0.0,
        }
    return result


def park_run_factor(extra_distance: float) -> float:
    return max(0.90, min(1.18, 1.0 + (extra_distance / 120.0)))


def park_hr_factor(extra_distance: float) -> float:
    return max(0.85, min(1.28, 1.0 + (extra_distance / 80.0)))


def venue_park_factors(venue_name: str | None, year: int) -> dict[str, float]:
    if not venue_name:
        return {"run_factor": 1.0, "hr_factor": 1.0, "extra_distance": 0.0}
    row = fetch_statcast_park_carry(year).get(str(venue_name))
    if row is None:
        return {"run_factor": 1.0, "hr_factor": 1.0, "extra_distance": 0.0}
    extra_distance = float(row.get("extra_distance", 0.0))
    return {
        "run_factor": park_run_factor(extra_distance),
        "hr_factor": park_hr_factor(extra_distance),
        "extra_distance": extra_distance,
    }
