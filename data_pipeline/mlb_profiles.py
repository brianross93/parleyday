from __future__ import annotations

from dataclasses import asdict
from typing import Any

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


def build_batter_profile_payload(player_id: int, name: str, season: int, lineup_index: int) -> dict[str, Any]:
    person = fetch_player_people_stats(player_id, "hitting", season)
    hand = (person.get("batSide") or {}).get("code", "R")
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


def team_context_from_cached_payload(team_code: str, lineup_payload: list[dict], pitcher_payload: dict) -> TeamContext:
    lineup = tuple(BatterProfile(**{key: item[key] for key in BATTER_PROFILE_FIELDS}) for item in lineup_payload)
    starter = PitcherProfile(**{key: pitcher_payload[key] for key in PITCHER_PROFILE_FIELDS})
    return TeamContext(team_code=team_code, lineup=lineup, starter=starter)
