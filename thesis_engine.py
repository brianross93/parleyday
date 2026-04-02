from __future__ import annotations

from typing import Any, Callable


GameContextLoader = Callable[[str, str, str], dict[str, Any] | None]
ProfileLoader = Callable[[str, str], dict[str, Any] | None]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_confidence(value: float) -> float:
    return max(0.0, min(1.0, round(value, 2)))


def _base_thesis(
    *,
    thesis_id: str,
    source: str,
    thesis_type: str,
    sport: str,
    games: list[str],
    summary: str,
    confidence: float,
    supporting_facts: list[str],
    missing_information: list[str] | None = None,
    verification_targets: list[str] | None = None,
    kill_conditions: list[str] | None = None,
    candidate_leg_types: list[str] | None = None,
    focus_players: list[str] | None = None,
    fade_players: list[str] | None = None,
    focus_stats: list[str] | None = None,
    focus_teams: list[str] | None = None,
    thesis_direction: str | None = None,
) -> dict[str, Any]:
    return {
        "thesis_id": thesis_id,
        "source": source,
        "type": thesis_type,
        "sport": sport,
        "games": games,
        "summary": summary,
        "confidence": _clip_confidence(confidence),
        "supporting_facts": supporting_facts,
        "missing_information": missing_information or [],
        "verification_targets": verification_targets or [],
        "kill_conditions": kill_conditions or [],
        "candidate_leg_types": candidate_leg_types or [],
        "focus_players": focus_players or [],
        "fade_players": fade_players or [],
        "focus_stats": focus_stats or [],
        "focus_teams": focus_teams or [],
        "thesis_direction": thesis_direction or "",
    }


def _mlb_games(legs: list[Any]) -> list[str]:
    return sorted({str(leg.game) for leg in legs if getattr(leg, "sport", "") == "mlb"})


def _nba_games(legs: list[Any]) -> list[str]:
    return sorted({str(leg.game) for leg in legs if getattr(leg, "sport", "") == "nba"})


def detect_mlb_bullpen_exhaustion(
    *,
    date_str: str,
    legs: list[Any],
    load_game_context: GameContextLoader,
) -> list[dict[str, Any]]:
    theses: list[dict[str, Any]] = []
    for game in _mlb_games(legs):
        context = load_game_context(date_str, "mlb", game)
        if not context:
            continue
        bullpen = context.get("bullpen") or {}
        away_fatigue = _safe_float((bullpen.get("away") or {}).get("fatigue_score"))
        home_fatigue = _safe_float((bullpen.get("home") or {}).get("fatigue_score"))
        high_relief_load = max(away_fatigue, home_fatigue)
        if high_relief_load < 1.1:
            continue
        facts = []
        if away_fatigue >= 1.1:
            facts.append(f"Away bullpen fatigue score {away_fatigue:.2f}")
        if home_fatigue >= 1.1:
            facts.append(f"Home bullpen fatigue score {home_fatigue:.2f}")
        probable_pitchers = context.get("probable_pitchers") or {}
        away_pitcher = str((probable_pitchers.get("away") or {}).get("fullName") or "").strip()
        home_pitcher = str((probable_pitchers.get("home") or {}).get("fullName") or "").strip()
        if away_pitcher or home_pitcher:
            facts.append(f"Probables: {away_pitcher or 'TBD'} vs {home_pitcher or 'TBD'}")
        theses.append(
            _base_thesis(
                thesis_id=f"mlb_bullpen_exhaustion_{game.lower().replace('@', '_')}_{date_str.replace('-', '_')}",
                source="structured",
                thesis_type="bullpen_exhaustion",
                sport="mlb",
                games=[game],
                summary="Bullpen fatigue creates a fragile late-game scoring environment.",
                confidence=0.45 + min(high_relief_load * 0.12, 0.3),
                supporting_facts=facts,
                missing_information=[],
                verification_targets=["probable pitchers", "active roster changes"],
                kill_conditions=["starter scratch", "multiple core bats absent"],
                candidate_leg_types=["total", "hits", "hr"],
            )
        )
    return theses


def detect_mlb_run_environment(
    *,
    date_str: str,
    legs: list[Any],
    load_game_context: GameContextLoader,
    load_matchup_profile: ProfileLoader,
) -> list[dict[str, Any]]:
    theses: list[dict[str, Any]] = []
    for game in _mlb_games(legs):
        context = load_game_context(date_str, "mlb", game)
        if not context:
            continue
        weather = context.get("weather") or {}
        temp = _safe_float(weather.get("temperature_f"))
        wind = _safe_float(weather.get("wind_speed_mph"))
        profile = load_matchup_profile(date_str, game) or {}
        lineup_count = len(profile.get("away_lineup", []) or []) + len(profile.get("home_lineup", []) or [])
        environment_score = 0.0
        facts: list[str] = []
        if temp >= 78.0:
            environment_score += 0.18
            facts.append(f"Warm weather {temp:.0f}F")
        if wind >= 10.0:
            environment_score += 0.18
            facts.append(f"Wind {wind:.0f} mph")
        if lineup_count >= 12:
            environment_score += 0.10
            facts.append(f"{lineup_count} tracked lineup slots in matchup profile")
        if environment_score < 0.18:
            continue
        theses.append(
            _base_thesis(
                thesis_id=f"mlb_run_environment_{game.lower().replace('@', '_')}_{date_str.replace('-', '_')}",
                source="structured",
                thesis_type="run_environment",
                sport="mlb",
                games=[game],
                summary="Weather and matchup context support a more offense-friendly run environment.",
                confidence=0.36 + environment_score,
                supporting_facts=facts,
                missing_information=[],
                verification_targets=["weather recheck near first pitch", "probable pitchers"],
                kill_conditions=["weather shifts materially", "pitching change"],
                candidate_leg_types=["total", "hits", "hr"],
            )
        )
    return theses


def detect_nba_thin_rotation(
    *,
    date_str: str,
    legs: list[Any],
    load_game_context: GameContextLoader,
    load_nba_matchup_profile: ProfileLoader,
) -> list[dict[str, Any]]:
    theses: list[dict[str, Any]] = []
    for game in _nba_games(legs):
        context = load_game_context(date_str, "nba", game)
        if not context:
            continue
        profile = load_nba_matchup_profile(date_str, game) or {}
        availability = context.get("availability") or {}
        for side in ("away", "home"):
            team_entries = availability.get(side) or []
            profiles = profile.get(f"{side}_profiles", []) or []
            tracked_profiles = len(profiles)
            if tracked_profiles == 0 and not team_entries:
                continue
            out_count = sum(
                1
                for entry in team_entries
                if str(entry.get("status") or "").strip().lower() in {"out", "doubtful"}
            )
            playable = max(0, tracked_profiles - out_count)
            submitted = bool(availability.get(f"{side}_submitted", False))
            if playable > 8 and submitted:
                continue
            facts = [
                f"{tracked_profiles} tracked profiles",
                f"{out_count} out or doubtful entries",
                f"Estimated playable rotation {playable}",
            ]
            missing = []
            if not submitted:
                missing.append("Official NBA injury report submission")
            theses.append(
                _base_thesis(
                    thesis_id=f"nba_thin_rotation_{game.lower().replace('@', '_')}_{side}_{date_str.replace('-', '_')}",
                    source="structured",
                    thesis_type="thin_rotation",
                    sport="nba",
                    games=[game],
                    summary="Rotation compression may create concentrated usage and non-linear team outcomes.",
                    confidence=0.4 + max(0.0, (8 - playable) * 0.06) + (0.12 if not submitted else 0.0),
                    supporting_facts=facts,
                    missing_information=missing,
                    verification_targets=["official injury report", "confirmed active list"],
                    kill_conditions=["late active status flip", "unexpected starting lineup change"],
                    candidate_leg_types=["ml", "total", "points", "rebounds", "assists"],
                )
            )
    return theses


def detect_model_market_divergence(
    *,
    date_str: str,
    legs: list[Any],
    activation: list[float] | Any | None,
    pricing_details: dict[int, dict[str, str]] | None,
) -> list[dict[str, Any]]:
    if activation is None:
        return []
    theses: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[Any, float, float, dict[str, str] | None]]] = {}
    for idx, leg in enumerate(legs):
        if idx >= len(activation):
            break
        activation_value = _safe_float(activation[idx])
        edge = activation_value - _safe_float(getattr(leg, "implied_prob", 0.0))
        if abs(edge) < 0.08:
            continue
        grouped.setdefault(str(leg.game), []).append((leg, activation_value, edge, (pricing_details or {}).get(idx)))
    for game, entries in grouped.items():
        top = sorted(entries, key=lambda item: abs(item[2]), reverse=True)[:3]
        positive = [entry for entry in top if entry[2] > 0]
        negative = [entry for entry in top if entry[2] < 0]
        facts = [
            f"{entry[0].label}: model {entry[1]:.3f} vs market {_safe_float(getattr(entry[0], 'implied_prob', 0.0)):.3f}"
            for entry in top
        ]
        sport = str(getattr(top[0][0], "sport", "") or "")
        max_gap = max(abs(entry[2]) for entry in top)
        focus_players = sorted(
            {
                str(entry[0].label).split(" O ", 1)[0].strip()
                for entry in positive
                if getattr(entry[0], "category", "") == "prop" and " O " in str(entry[0].label)
            }
        )
        fade_players = sorted(
            {
                str(entry[0].label).split(" O ", 1)[0].strip()
                for entry in negative
                if getattr(entry[0], "category", "") == "prop" and " O " in str(entry[0].label)
            }
        )
        focus_stats = sorted(
            {
                (
                    "hits"
                    if str(entry[0].label).upper().endswith(" O 1 H")
                    else "hr"
                    if str(entry[0].label).upper().endswith("HR")
                    else "rebounds"
                    if " REB" in str(entry[0].label).upper()
                    else "assists"
                    if " AST" in str(entry[0].label).upper()
                    else "points"
                    if " PTS" in str(entry[0].label).upper()
                    else "prop"
                )
                for entry in positive
                if getattr(entry[0], "category", "") == "prop"
            }
        )
        theses.append(
            _base_thesis(
                thesis_id=f"model_market_divergence_{game.lower().replace('@', '_')}_{date_str.replace('-', '_')}",
                source="structured",
                thesis_type="model_market_divergence",
                sport=sport,
                games=[game],
                summary="The model and the current market disagree materially on this matchup cluster.",
                confidence=0.4 + min(max_gap * 0.9, 0.35),
                supporting_facts=facts,
                missing_information=[],
                verification_targets=["check current injury/lineup state", "verify market freshness"],
                kill_conditions=["market reprices after fresh news", "model input corrected by updated context"],
                candidate_leg_types=["ml", "total", "prop"],
                focus_players=focus_players,
                fade_players=fade_players,
                focus_stats=focus_stats,
                thesis_direction="positive_model_divergence",
            )
        )
    return theses


def build_structured_theses(
    *,
    date_str: str,
    legs: list[Any],
    activation: list[float] | Any | None,
    pricing_details: dict[int, dict[str, str]] | None,
    load_game_context: GameContextLoader,
    load_matchup_profile: ProfileLoader,
    load_nba_matchup_profile: ProfileLoader,
) -> list[dict[str, Any]]:
    theses: list[dict[str, Any]] = []
    theses.extend(
        detect_mlb_bullpen_exhaustion(
            date_str=date_str,
            legs=legs,
            load_game_context=load_game_context,
        )
    )
    theses.extend(
        detect_mlb_run_environment(
            date_str=date_str,
            legs=legs,
            load_game_context=load_game_context,
            load_matchup_profile=load_matchup_profile,
        )
    )
    theses.extend(
        detect_nba_thin_rotation(
            date_str=date_str,
            legs=legs,
            load_game_context=load_game_context,
            load_nba_matchup_profile=load_nba_matchup_profile,
        )
    )
    theses.extend(
        detect_model_market_divergence(
            date_str=date_str,
            legs=legs,
            activation=activation,
            pricing_details=pricing_details,
        )
    )

    deduped: dict[str, dict[str, Any]] = {}
    for thesis in theses:
        existing = deduped.get(thesis["thesis_id"])
        if existing is None or float(thesis["confidence"]) > float(existing["confidence"]):
            deduped[thesis["thesis_id"]] = thesis
    return sorted(deduped.values(), key=lambda item: (-float(item["confidence"]), item["thesis_id"]))
