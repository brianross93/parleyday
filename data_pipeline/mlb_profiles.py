from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import requests

from monte_carlo.mlb import BatterProfile, PitcherProfile, TeamContext


LEAGUE_HITTING_PRIORS = {
    "strikeout_rate": 0.225,
    "walk_rate": 0.085,
    "hbp_rate": 0.010,
    "single_rate": 0.155,
    "double_rate": 0.045,
    "triple_rate": 0.005,
    "home_run_rate": 0.032,
}
LEAGUE_PITCHING_PRIORS = {
    "strikeout_rate": 0.225,
    "walk_rate": 0.085,
    "hbp_rate": 0.010,
    "single_rate": 0.155,
    "double_rate": 0.045,
    "triple_rate": 0.005,
    "home_run_rate": 0.032,
}
HITTER_PRIOR_PA = 450.0
PITCHER_PRIOR_BF = 500.0
BATTER_PROFILE_FIELDS = {
    "player_id",
    "name",
    "hand",
    "pa_share",
    "strikeout_rate",
    "walk_rate",
    "hbp_rate",
    "single_rate",
    "double_rate",
    "triple_rate",
    "home_run_rate",
    "speed_factor",
    "vs_left_factor",
    "vs_right_factor",
}
PITCHER_PROFILE_FIELDS = {
    "player_id",
    "name",
    "hand",
    "strikeout_rate",
    "walk_rate",
    "hbp_rate",
    "single_rate",
    "double_rate",
    "triple_rate",
    "home_run_rate",
    "fatigue_start",
    "fatigue_full",
}


def build_bullpen_profiles_payload(team_code: str, bullpen_snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    relievers = (bullpen_snapshot or {}).get("relievers", []) or []
    if not relievers:
        return []
    payloads: list[dict[str, Any]] = []
    for idx, reliever in enumerate(relievers[:4]):
        recent_pitches = float(reliever.get("pitches_last_3_days", 0) or 0.0)
        appearances = float(reliever.get("appearances", 0) or 0.0)
        freshness = float(np.clip(1.0 - (recent_pitches / 70.0) - (appearances * 0.03), 0.72, 1.08))
        leverage = max(0.0, 1.0 - (idx * 0.06))
        quality = float(np.clip((freshness * 0.7) + (leverage * 0.3), 0.72, 1.08))
        profile = PitcherProfile(
            player_id=str(reliever.get("player_id") or f"{team_code}-rp-{idx}"),
            name=str(reliever.get("player_name", f"{team_code} Bullpen {idx + 1}")),
            hand=str(reliever.get("hand") or "R"),
            strikeout_rate=float(np.clip(LEAGUE_PITCHING_PRIORS["strikeout_rate"] * (1.02 + (quality - 1.0) * 0.55), 0.17, 0.34)),
            walk_rate=float(np.clip(LEAGUE_PITCHING_PRIORS["walk_rate"] * (0.98 + (1.0 - quality) * 0.65), 0.05, 0.13)),
            hbp_rate=LEAGUE_PITCHING_PRIORS["hbp_rate"],
            single_rate=float(np.clip(LEAGUE_PITCHING_PRIORS["single_rate"] * (1.0 + (1.0 - quality) * 0.35), 0.12, 0.19)),
            double_rate=float(np.clip(LEAGUE_PITCHING_PRIORS["double_rate"] * (1.0 + (1.0 - quality) * 0.25), 0.03, 0.06)),
            triple_rate=LEAGUE_PITCHING_PRIORS["triple_rate"],
            home_run_rate=float(np.clip(LEAGUE_PITCHING_PRIORS["home_run_rate"] * (1.0 + (1.0 - quality) * 0.45), 0.022, 0.05)),
            fatigue_start=max(12, int(18 + freshness * 10.0)),
            fatigue_full=max(20, int(27 + freshness * 10.0)),
        )
        payload = asdict(profile)
        payload["recent_pitches"] = recent_pitches
        payload["appearances_last_3_days"] = appearances
        payload["quality_score"] = quality
        payloads.append(payload)
    return payloads


def safe_rate(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator <= 0:
        return fallback
    return max(0.0, numerator / denominator)


def blend_rate(raw_rate: float, sample: float, prior_rate: float, prior_sample: float) -> float:
    return ((raw_rate * sample) + (prior_rate * prior_sample)) / max(sample + prior_sample, 1.0)


def fetch_player_people_stats(player_id: int, group: str, season: int) -> dict[str, Any]:
    response = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{player_id}",
        params={"hydrate": f"stats(group=[{group}],type=[season],season={season})"},
        timeout=20,
    )
    response.raise_for_status()
    people = response.json().get("people", [])
    if not people:
        raise RuntimeError(f"No player data for MLB player id {player_id}")
    return people[0]


def fetch_player_hand(player_id: int, hand_key: str) -> str:
    response = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{player_id}",
        timeout=20,
    )
    response.raise_for_status()
    people = response.json().get("people", [])
    if not people:
        return "R"
    return str((people[0].get(hand_key) or {}).get("code") or "R")


def fetch_batter_split_factors(player_id: int, season: int) -> tuple[float, float]:
    response = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{player_id}",
        params={"hydrate": f"stats(group=[hitting],type=[statSplits],sitCodes=[vl,vr],season={season})"},
        timeout=20,
    )
    response.raise_for_status()
    people = response.json().get("people", [])
    if not people:
        return 1.0, 1.0
    stats = (people[0].get("stats") or [{}])[0]
    splits = stats.get("splits") or []
    factors = {"vl": 1.0, "vr": 1.0}
    split_values: dict[str, tuple[float, float]] = {}
    for split in splits:
        code = str((split.get("split") or {}).get("code") or "").lower()
        if code not in {"vl", "vr"}:
            continue
        stat = split.get("stat") or {}
        pa = float(stat.get("plateAppearances", 0) or 0.0)
        try:
            ops = float(stat.get("ops", 0.0) or 0.0)
        except (TypeError, ValueError):
            ops = 0.0
        if pa > 0 and ops > 0:
            split_values[code] = (ops, pa)
    if not split_values:
        return 1.0, 1.0
    overall_ops = sum(ops * pa for ops, pa in split_values.values()) / max(sum(pa for _, pa in split_values.values()), 1.0)
    for code, (ops, pa) in split_values.items():
        raw = ops / max(overall_ops, 0.001)
        shrink = pa / (pa + 140.0)
        factors[code] = float(np.clip(1.0 + ((raw - 1.0) * shrink), 0.82, 1.18))
    return factors["vl"], factors["vr"]


def build_batter_profile_payload(player_id: int, name: str, season: int, lineup_index: int) -> dict[str, Any]:
    person = fetch_player_people_stats(player_id, "hitting", season)
    hand = (person.get("batSide") or {}).get("code", "R")
    vs_left_factor, vs_right_factor = fetch_batter_split_factors(player_id, season)
    splits = ((person.get("stats") or [{}])[0].get("splits") or [])
    stat = (splits[0].get("stat") if splits else {}) or {}
    pa = float(stat.get("plateAppearances", 0) or 0)
    hits = float(stat.get("hits", 0) or 0)
    doubles = float(stat.get("doubles", 0) or 0)
    triples = float(stat.get("triples", 0) or 0)
    home_runs = float(stat.get("homeRuns", 0) or 0)
    singles = max(0.0, hits - doubles - triples - home_runs)
    strikeouts = float(stat.get("strikeOuts", 0) or 0)
    walks = float(stat.get("baseOnBalls", 0) or 0)
    hbp = float(stat.get("hitByPitch", 0) or 0)
    stolen_bases = float(stat.get("stolenBases", 0) or 0)
    speed_factor = min(1.3, max(0.85, 1.0 + (stolen_bases / max(pa, 1.0))))

    profile = BatterProfile(
        player_id=str(player_id),
        name=name,
        hand=hand,
        pa_share=max(0.08, 0.13 - (0.004 * lineup_index)),
        strikeout_rate=blend_rate(safe_rate(strikeouts, pa), pa, LEAGUE_HITTING_PRIORS["strikeout_rate"], HITTER_PRIOR_PA),
        walk_rate=blend_rate(safe_rate(walks, pa), pa, LEAGUE_HITTING_PRIORS["walk_rate"], HITTER_PRIOR_PA),
        hbp_rate=blend_rate(safe_rate(hbp, pa), pa, LEAGUE_HITTING_PRIORS["hbp_rate"], HITTER_PRIOR_PA),
        single_rate=blend_rate(safe_rate(singles, pa), pa, LEAGUE_HITTING_PRIORS["single_rate"], HITTER_PRIOR_PA),
        double_rate=blend_rate(safe_rate(doubles, pa), pa, LEAGUE_HITTING_PRIORS["double_rate"], HITTER_PRIOR_PA),
        triple_rate=blend_rate(safe_rate(triples, pa), pa, LEAGUE_HITTING_PRIORS["triple_rate"], HITTER_PRIOR_PA),
        home_run_rate=blend_rate(safe_rate(home_runs, pa), pa, LEAGUE_HITTING_PRIORS["home_run_rate"], HITTER_PRIOR_PA),
        speed_factor=speed_factor,
        vs_left_factor=vs_left_factor,
        vs_right_factor=vs_right_factor,
    )
    payload = asdict(profile)
    payload["season"] = season
    payload["sample_pa"] = pa
    return payload


def build_pitcher_profile_payload(player_id: int, name: str, season: int) -> dict[str, Any]:
    person = fetch_player_people_stats(player_id, "pitching", season)
    hand = (person.get("pitchHand") or {}).get("code", "R")
    splits = ((person.get("stats") or [{}])[0].get("splits") or [])
    stat = (splits[0].get("stat") if splits else {}) or {}
    bf = float(stat.get("battersFaced", 0) or 0)
    hits = float(stat.get("hits", 0) or 0)
    doubles = float(stat.get("doubles", 0) or 0)
    triples = float(stat.get("triples", 0) or 0)
    home_runs = float(stat.get("homeRuns", 0) or 0)
    singles = max(0.0, hits - doubles - triples - home_runs)
    strikeouts = float(stat.get("strikeOuts", 0) or 0)
    walks = float(stat.get("baseOnBalls", 0) or 0)
    hbp = float(stat.get("hitByPitch", 0) or 0)
    number_of_pitches = float(stat.get("numberOfPitches", 0) or 0)
    games_started = float(stat.get("gamesStarted", 0) or 0)
    avg_pitches = number_of_pitches / max(games_started, 1.0)

    profile = PitcherProfile(
        player_id=str(player_id),
        name=name,
        hand=hand,
        strikeout_rate=blend_rate(safe_rate(strikeouts, bf), bf, LEAGUE_PITCHING_PRIORS["strikeout_rate"], PITCHER_PRIOR_BF),
        walk_rate=blend_rate(safe_rate(walks, bf), bf, LEAGUE_PITCHING_PRIORS["walk_rate"], PITCHER_PRIOR_BF),
        hbp_rate=blend_rate(safe_rate(hbp, bf), bf, LEAGUE_PITCHING_PRIORS["hbp_rate"], PITCHER_PRIOR_BF),
        single_rate=blend_rate(safe_rate(singles, bf), bf, LEAGUE_PITCHING_PRIORS["single_rate"], PITCHER_PRIOR_BF),
        double_rate=blend_rate(safe_rate(doubles, bf), bf, LEAGUE_PITCHING_PRIORS["double_rate"], PITCHER_PRIOR_BF),
        triple_rate=blend_rate(safe_rate(triples, bf), bf, LEAGUE_PITCHING_PRIORS["triple_rate"], PITCHER_PRIOR_BF),
        home_run_rate=blend_rate(safe_rate(home_runs, bf), bf, LEAGUE_PITCHING_PRIORS["home_run_rate"], PITCHER_PRIOR_BF),
        fatigue_start=max(60, int(avg_pitches * 0.78)) if avg_pitches else 75,
        fatigue_full=max(85, int(avg_pitches * 1.02)) if avg_pitches else 100,
    )
    payload = asdict(profile)
    payload["season"] = season
    payload["sample_bf"] = bf
    return payload


def team_context_from_cached_payload(
    team_code: str,
    lineup_payload: list[dict],
    pitcher_payload: dict | None,
    bullpen_payload: list[dict] | None = None,
) -> TeamContext:
    lineup = tuple(BatterProfile(**{key: item[key] for key in BATTER_PROFILE_FIELDS if key in item}) for item in lineup_payload)
    starter_payload = _normalize_pitcher_payload(team_code, pitcher_payload)
    starter = PitcherProfile(**{key: starter_payload[key] for key in PITCHER_PROFILE_FIELDS})
    bullpen = tuple(
        PitcherProfile(**{key: item[key] for key in PITCHER_PROFILE_FIELDS})
        for item in (bullpen_payload or [])
        if PITCHER_PROFILE_FIELDS.issubset(item.keys())
    )
    return TeamContext(team_code=team_code, lineup=lineup, starter=starter, bullpen=bullpen)


def _normalize_pitcher_payload(team_code: str, pitcher_payload: dict | None) -> dict[str, Any]:
    payload = dict(pitcher_payload or {})
    if payload and PITCHER_PROFILE_FIELDS.issubset(payload.keys()):
        return payload
    normalized = {key: payload.get(key) for key in PITCHER_PROFILE_FIELDS}
    normalized["player_id"] = str(normalized.get("player_id") or f"{team_code}-sp-fallback")
    normalized["name"] = str(normalized.get("name") or f"{team_code} Starter")
    normalized["hand"] = str(normalized.get("hand") or "R")
    normalized["strikeout_rate"] = float(normalized.get("strikeout_rate") or LEAGUE_PITCHING_PRIORS["strikeout_rate"])
    normalized["walk_rate"] = float(normalized.get("walk_rate") or LEAGUE_PITCHING_PRIORS["walk_rate"])
    normalized["hbp_rate"] = float(normalized.get("hbp_rate") or LEAGUE_PITCHING_PRIORS["hbp_rate"])
    normalized["single_rate"] = float(normalized.get("single_rate") or LEAGUE_PITCHING_PRIORS["single_rate"])
    normalized["double_rate"] = float(normalized.get("double_rate") or LEAGUE_PITCHING_PRIORS["double_rate"])
    normalized["triple_rate"] = float(normalized.get("triple_rate") or LEAGUE_PITCHING_PRIORS["triple_rate"])
    normalized["home_run_rate"] = float(normalized.get("home_run_rate") or LEAGUE_PITCHING_PRIORS["home_run_rate"])
    normalized["fatigue_start"] = int(normalized.get("fatigue_start") or 75)
    normalized["fatigue_full"] = int(normalized.get("fatigue_full") or 100)
    return normalized
