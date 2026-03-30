from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from env_config import load_local_env

load_local_env()

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None

from quantum_parlay_oracle import (
    load_game_context_snapshot,
    load_matchup_profile_snapshot,
    load_nba_matchup_profile_snapshot,
)


DEFAULT_OPENAI_MODEL = os.getenv("PARLEYDAY_LLM_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = os.getenv("PARLEYDAY_LLM_REASONING_EFFORT", "medium")
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "minimal", "none"}

JUDGMENT_INSTRUCTIONS = """You are the final judgment layer for a local sports market tool.

Use only the slate data, model outputs, and cached roster/matchup context provided.
Do not invent injuries, lineups, prices, or market conditions that are not explicitly present.
If the data is thin, stale, contradictory, or weak, say so directly and recommend a pass.
Prefer concise, decision-oriented output over broad commentary.

Return plain text with exactly these sections:
Call:
Recommended Action:
Why:
Main Risks:
Passes:
"""

CHAT_INSTRUCTIONS = """You are a contextual assistant inside a local sports market dashboard.

Answer the user's question using only the slate data, model outputs, cached roster context, injury state, and matchup context provided.
Do not invent injuries, prices, lineups, or confidence that are not present in the supplied payload.
If the answer depends on missing or stale data, say that directly.
Be concise, concrete, and decision-oriented.
"""


def dependency_status(api_key: str | None = None) -> tuple[bool, str]:
    if OpenAI is None:
        return False, "The `openai` package is not installed. Run `pip install -r requirements.txt`."
    if api_key or os.getenv("OPENAI_API_KEY"):
        return True, ""
    return False, "Missing `OPENAI_API_KEY`."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trim_list(items: list[Any], limit: int = 5) -> list[Any]:
    return items[:limit]


def _player_name(entry: dict[str, Any]) -> str:
    return str(entry.get("player_name") or entry.get("name") or "").strip()


def _extract_response_text(response: Any) -> str:
    direct = str(getattr(response, "output_text", "") or "").strip()
    if direct:
        return direct

    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
        except Exception:
            payload = None
    else:
        payload = response

    fragments: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = str(node.get("type") or "")
            if node_type in {"output_text", "text"}:
                value = node.get("text")
                if isinstance(value, str) and value.strip():
                    fragments.append(value.strip())
                elif isinstance(value, dict):
                    nested_value = value.get("value") or value.get("text")
                    if isinstance(nested_value, str) and nested_value.strip():
                        fragments.append(nested_value.strip())
                plain_value = node.get("value")
                if isinstance(plain_value, str) and plain_value.strip():
                    fragments.append(plain_value.strip())
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)

    deduped: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment in seen:
            continue
        seen.add(fragment)
        deduped.append(fragment)
    return "\n\n".join(deduped).strip()


def _response_debug_details(response: Any) -> str:
    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
        except Exception:
            payload = {}
    else:
        payload = response if isinstance(response, dict) else {}
    status = payload.get("status")
    incomplete = payload.get("incomplete_details")
    error = payload.get("error")
    output_items = payload.get("output") or []
    parts = [f"status={status or 'unknown'}", f"output_items={len(output_items)}"]
    if incomplete:
        parts.append(f"incomplete={json.dumps(incomplete, default=str)}")
    if error:
        parts.append(f"error={json.dumps(error, default=str)}")
    return "; ".join(parts)


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact["top_legs"] = compact.get("top_legs", [])[:5]
    compact["fades"] = compact.get("fades", [])[:3]
    compact["game_contexts"] = compact.get("game_contexts", [])[:4]
    compact["recommendations"] = [
        {
            **recommendation,
            "legs": (recommendation.get("legs") or [])[:2],
        }
        for recommendation in (compact.get("recommendations") or [])[:3]
    ]
    return compact


def _summarize_mlb_context(game: str, date_str: str) -> dict[str, Any] | None:
    game_context = load_game_context_snapshot(date_str, "mlb", game)
    if game_context is None:
        return None
    profile = load_matchup_profile_snapshot(date_str, game) or {}
    availability = game_context.get("availability") or {}
    bullpen = game_context.get("bullpen") or {}
    weather = game_context.get("weather") or {}
    probable_pitchers = game_context.get("probable_pitchers") or {}
    lineup_status = game_context.get("lineup_status") or {}

    def team_summary(side: str) -> dict[str, Any]:
        side_availability = availability.get(side) or {}
        side_bullpen = bullpen.get(side) or {}
        return {
            "lineup_confirmed": bool(lineup_status.get(f"{side}_confirmed")),
            "lineup_head": _trim_list(list((game_context.get("lineups") or {}).get(side, []) or [])),
            "unavailable": _trim_list(
                [
                    {
                        "name": _player_name(entry),
                        "status": str(entry.get("status_description") or entry.get("status_code") or "").strip(),
                    }
                    for entry in side_availability.get("unavailable_players", []) or []
                    if _player_name(entry)
                ]
            ),
            "recent_transactions": _trim_list(
                [
                    str(item.get("description") or item.get("typeDesc") or item.get("typeCode") or "").strip()
                    for item in side_availability.get("transactions", []) or []
                    if str(item.get("description") or item.get("typeDesc") or item.get("typeCode") or "").strip()
                ],
                limit=3,
            ),
            "bullpen_fatigue_score": _safe_float(side_bullpen.get("fatigue_score")),
            "relievers": _trim_list(
                [
                    {
                        "name": _player_name(entry),
                        "hand": str(entry.get("hand") or "").strip(),
                        "pitches_last_3_days": int(entry.get("pitches_last_3_days") or 0),
                    }
                    for entry in side_bullpen.get("relievers", []) or []
                    if _player_name(entry)
                ],
                limit=4,
            ),
        }

    return {
        "sport": "mlb",
        "game": game,
        "status": str(game_context.get("status") or "").strip(),
        "game_time": game_context.get("game_time"),
        "venue": (game_context.get("venue") or {}).get("name"),
        "probable_pitchers": {
            "away": str((probable_pitchers.get("away") or {}).get("fullName") or ""),
            "home": str((probable_pitchers.get("home") or {}).get("fullName") or ""),
        },
        "weather": {
            "temperature_f": weather.get("temperature_f"),
            "wind_speed_mph": weather.get("wind_speed_mph"),
            "humidity_pct": weather.get("humidity_pct"),
        },
        "tracked_profiles": {
            "away_lineup": len(profile.get("away_lineup", []) or []),
            "home_lineup": len(profile.get("home_lineup", []) or []),
        },
        "away": team_summary("away"),
        "home": team_summary("home"),
    }


def _status_bucket(entries: list[dict[str, Any]], wanted: str) -> list[str]:
    results: list[str] = []
    for entry in entries or []:
        status = str(entry.get("status") or "").strip().lower()
        if status != wanted:
            continue
        name = _player_name(entry)
        if name:
            results.append(name)
    return _trim_list(results)


def _summarize_nba_context(game: str, date_str: str) -> dict[str, Any] | None:
    game_context = load_game_context_snapshot(date_str, "nba", game)
    if game_context is None:
        return None
    profile = load_nba_matchup_profile_snapshot(date_str, game) or {}
    availability = game_context.get("availability") or {}

    def team_summary(side: str) -> dict[str, Any]:
        entries = availability.get(side) or []
        profiles = profile.get(f"{side}_profiles", []) or []
        return {
            "injury_report_submitted": bool(availability.get(f"{side}_submitted", False)),
            "outs": _status_bucket(entries, "out"),
            "doubtful": _status_bucket(entries, "doubtful"),
            "questionable": _status_bucket(entries, "questionable"),
            "tracked_profiles": len(profiles),
            "top_profiles": _trim_list(
                [
                    {
                        "name": str(player.get("name") or "").strip(),
                        "minutes": _safe_float(player.get("minutes_per_game")),
                        "points": _safe_float(player.get("points_per_game")),
                    }
                    for player in profiles
                    if str(player.get("name") or "").strip()
                ]
            ),
        }

    return {
        "sport": "nba",
        "game": game,
        "status": str(game_context.get("status") or "").strip(),
        "game_time": game_context.get("game_time"),
        "injury_report_source": str(availability.get("source") or "").strip(),
        "away": team_summary("away"),
        "home": team_summary("home"),
    }


def extract_judgment_games(result: dict[str, Any], max_games: int = 6) -> list[tuple[str, str]]:
    config = result.get("config") or {}
    fallback_sport = str(config.get("sport") or "").strip().lower()
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []

    def maybe_add(leg: dict[str, Any]) -> None:
        sport = str(leg.get("sport") or fallback_sport).strip().lower()
        game = str(leg.get("game") or "").strip()
        if not sport or sport == "both" or not game:
            return
        key = (sport, game)
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    for parlay in result.get("tier_parlays", []) or []:
        for leg in parlay.get("legs", []) or []:
            maybe_add(leg)
    for leg in result.get("top_legs", []) or []:
        maybe_add(leg)
    return ordered[:max_games]


def build_judgment_payload(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config") or {}
    date_str = str(config.get("date") or "")

    def leg_summary(leg: dict[str, Any]) -> dict[str, Any]:
        return {
            "sport": leg.get("sport"),
            "label": leg.get("label"),
            "game": leg.get("game"),
            "category": leg.get("category"),
            "model_prob": round(_safe_float(leg.get("activation")), 3),
            "market_prob": round(_safe_float(leg.get("implied_prob")), 3),
            "edge": round(_safe_float(leg.get("score_delta")), 3),
            "trust": round(_safe_float(leg.get("trust_score")), 3),
            "pricing": leg.get("pricing_label"),
            "notes": leg.get("notes"),
        }

    payload: dict[str, Any] = {
        "config": {
            "date": config.get("date"),
            "sport": config.get("sport"),
            "slate_mode": config.get("slate_mode"),
            "score_source": config.get("score_source"),
        },
        "meta": {
            "entropy_source": (result.get("meta") or {}).get("entropy_source"),
            "kalshi_markets": (result.get("meta") or {}).get("kalshi_markets"),
            "games": (result.get("meta") or {}).get("games"),
            "pricing_summary": (result.get("meta") or {}).get("pricing_summary"),
        },
        "refresh": result.get("refresh") or {},
        "recommendations": [
            {
                "key": parlay.get("key"),
                "label": parlay.get("label"),
                "actual_size": parlay.get("actual_size"),
                "payout_estimate": round(_safe_float(parlay.get("payout_estimate")), 2)
                if parlay.get("payout_estimate") is not None
                else None,
                "model_joint_prob": round(_safe_float(parlay.get("model_joint_prob")), 3)
                if parlay.get("model_joint_prob") is not None
                else None,
                "market_joint_prob": round(_safe_float(parlay.get("market_joint_prob")), 3)
                if parlay.get("market_joint_prob") is not None
                else None,
                "average_edge": round(_safe_float(parlay.get("average_edge")), 3)
                if parlay.get("average_edge") is not None
                else None,
                "average_trust": round(_safe_float(parlay.get("average_trust")), 3)
                if parlay.get("average_trust") is not None
                else None,
                "legs": [leg_summary(leg) for leg in (parlay.get("legs") or [])],
            }
            for parlay in result.get("tier_parlays", []) or []
        ],
        "top_legs": [leg_summary(leg) for leg in (result.get("top_legs") or [])[:8]],
        "fades": [leg_summary(leg) for leg in (result.get("fades") or [])[:5]],
        "moonshot": leg_summary(result["moonshot"]) if result.get("moonshot") else None,
        "game_contexts": [],
    }

    for sport, game in extract_judgment_games(result):
        if sport == "mlb":
            context = _summarize_mlb_context(game, date_str)
        elif sport == "nba":
            context = _summarize_nba_context(game, date_str)
        else:
            context = None
        if context is not None:
            payload["game_contexts"].append(context)
    return payload


def build_judgment_prompt(result: dict[str, Any], *, compact: bool = False) -> str:
    payload = build_judgment_payload(result)
    if compact:
        payload = _compact_payload(payload)
    return (
        "Assess this slate and make a conservative but useful judgment call. "
        "You may recommend a pass if the slate quality is weak.\n\n"
        "Slate payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_chat_prompt(
    result: dict[str, Any],
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    compact: bool = False,
) -> str:
    payload = build_judgment_payload(result)
    if compact:
        payload = _compact_payload(payload)
    history_lines = []
    for item in history or []:
        role = str(item.get("role") or "user").strip().title()
        content = str(item.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"
    return (
        "Current slate payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        "Conversation so far:\n"
        f"{history_block}\n\n"
        "User question:\n"
        f"{question.strip()}"
    )


def generate_slate_judgment(
    result: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    ready, reason = dependency_status(api_key)
    if not ready:
        return {
            "status": "unavailable",
            "model": model_name,
            "reasoning_effort": effort,
            "message": reason,
            "text": "",
            "generated_at": None,
        }

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    prompt = build_judgment_prompt(result)
    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=JUDGMENT_INSTRUCTIONS,
            input=prompt,
            max_output_tokens=700,
        )
    except Exception as exc:  # pragma: no cover - network/runtime failure
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "message": str(exc),
            "text": "",
            "generated_at": None,
        }

    text = _extract_response_text(response)
    if not text:
        retry_response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=JUDGMENT_INSTRUCTIONS,
            input=build_judgment_prompt(result, compact=True),
            max_output_tokens=700,
        )
        text = _extract_response_text(retry_response)
        if text:
            response = retry_response
    if not text:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "message": f"OpenAI returned no judgment text. {_response_debug_details(response)}",
            "text": "",
            "generated_at": None,
        }
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "message": "",
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_chat_reply(
    result: dict[str, Any],
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    if not question.strip():
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "message": "Ask a question first.",
            "text": "",
            "generated_at": None,
        }

    ready, reason = dependency_status(api_key)
    if not ready:
        return {
            "status": "unavailable",
            "model": model_name,
            "reasoning_effort": effort,
            "message": reason,
            "text": "",
            "generated_at": None,
        }

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    prompt = build_chat_prompt(result, question, history=history)
    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=CHAT_INSTRUCTIONS,
            input=prompt,
            max_output_tokens=700,
        )
    except Exception as exc:  # pragma: no cover - network/runtime failure
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "message": str(exc),
            "text": "",
            "generated_at": None,
        }

    text = _extract_response_text(response)
    if not text:
        retry_response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=CHAT_INSTRUCTIONS,
            input=build_chat_prompt(result, question, history=history, compact=True),
            max_output_tokens=700,
        )
        text = _extract_response_text(retry_response)
        if text:
            response = retry_response
    if not text:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "message": f"OpenAI returned no chat text. {_response_debug_details(response)}",
            "text": "",
            "generated_at": None,
        }
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "message": "",
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
