from __future__ import annotations

import json
import os
import warnings
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
DEFAULT_WEB_SEARCH = os.getenv("PARLEYDAY_LLM_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "minimal", "none"}

JUDGMENT_INSTRUCTIONS = """You are the final judgment layer for a local sports market tool.

Use only the slate data, model outputs, and cached roster/matchup context provided.
Do not invent injuries, lineups, prices, or market conditions that are not explicitly present.
If the data is thin, stale, contradictory, or weak, say so directly, but still identify the best available options when the slate contains any usable candidates.
Prefer concise, decision-oriented output over broad commentary.
Do not turn every imperfect slate into a blanket pass. Distinguish between:
- full pass
- wait for confirmation but keep 1-2 leans ready
- small-stake best-available action

If recommendations exist in the payload, refer to them explicitly.
If score_source is implied, remember there is no independent model edge in that mode. Do not speak as if the market was disproven or validated by a separate model. In implied mode, focus on trust, context quality, stale or missing inputs, and best-available ranking rather than on edge-vs-market language.
When data quality is weak, you may still name a best available single and best available parlay, provided you describe them as conditional or low-conviction rather than pretending they are strong.
If web search is available and the supplied slate context is missing or stale, you may use it to verify missing injuries, lineups, pitching changes, scratches, or official status updates. Prefer official league/team sources and reliable current reporting, and mention the sourced update succinctly when it materially changes the recommendation.
Treat the supplied roster and matchup assignments as authoritative by default, even when they look surprising or unfamiliar. Human teams make strange trades, waivers, and late-season moves. Do not call a roster or matchup mapping contaminated merely because a player appears on an unexpected team. If the supplied matchup profile and sourced roster both place the player on that team, treat it as valid and move on. Only call mapping contamination when the payload and current sourced roster or matchup assignment directly conflict.

Return plain text with exactly these sections:
Call:
Recommended Action:
Why:
Main Risks:
Passes:
"""

THESIS_JUDGE_INSTRUCTIONS = """You are ranking thesis bundles for a sports longshot search tool.

Use only the supplied thesis payload, candidate expressions, and any web-search verification you explicitly perform.
Do not invent injuries, lineups, prices, market conditions, or candidate legs that are not explicitly present or sourced.
Your job is to judge the theses and their candidate expressions directly. Do not center your answer on generic parlay profiles.

Return JSON only in this exact top-level shape:
{
  "call": "one-sentence overall slate call",
  "portfolio_note": "one concise note on how to handle the slate as a whole",
  "ranked_theses": [
    {
      "thesis_id": "existing thesis id",
      "verdict": "back or conditional or lean or pass",
      "confidence": 0.0,
      "best_candidate_index": 0,
      "reason": "one concise paragraph",
      "risks": ["risk 1", "risk 2"],
      "kill_conditions": ["kill 1", "kill 2"],
      "dfs_guidance": {
        "stack_targets": ["player 1", "player 2"],
        "bring_back_targets": ["player 3"],
        "one_off_targets": ["player 4"],
        "avoid_chalk": ["player 5"],
        "max_players_from_game": 3,
        "preferred_salary_shape": "balanced or stars_and_scrubs or leave_salary"
      }
    }
  ]
}

Rules:
- Only rank thesis ids that already exist in the supplied payload.
- Use `best_candidate_index` to point at the thesis candidate list already supplied for that thesis.
- Confidence must be between 0 and 1.
- Return at most 6 ranked theses.
- If nothing is usable, return {"call":"Pass the slate.","portfolio_note":"No thesis is credible enough right now.","ranked_theses":[]}.
- Prefer `back` only for the strongest thesis expressions. Use `conditional` when the thesis is real but should wait on a confirmation or trigger, `lean` for lower-conviction but currently playable ideas, and `pass` for fragile or broken theses.
- Emit `dfs_guidance` only when it helps translate the thesis into DraftKings lineup construction. Keep it concise and bounded to players already present in the supplied payload.
- `max_players_from_game` must be an integer between 1 and 5 when present.
- `preferred_salary_shape` must be one of `balanced`, `stars_and_scrubs`, or `leave_salary` when present.
- Treat the supplied roster and matchup assignments as authoritative by default, even when they look surprising or unfamiliar. Human teams make strange trades, waivers, and late-season moves. Do not call a roster or matchup mapping contaminated merely because a player appears on an unexpected team. If the supplied matchup profile and sourced roster both place the player on that team, treat it as valid and move on. Only call mapping contamination when the payload and current sourced roster or matchup assignment directly conflict.
- Do not emit any explanation outside the JSON.
"""

INTUITION_THESIS_INSTRUCTIONS = """You are generating bounded intuition theses for a sports longshot search tool.

Use only the supplied slate payload and any web-search verification you explicitly perform.
Do not invent injuries, lineups, prices, or market conditions that are not explicitly present or sourced.
Your job is to propose a few structured hypotheses that feel interesting, unusual, or potentially misread by the market, but only when they are grounded in the supplied payload.

Return JSON only in this exact top-level shape:
{
  "theses": [
    {
      "type": "short_snake_case_type",
      "sport": "mlb or nba",
      "games": ["GAME1", "GAME2"],
      "summary": "one-sentence thesis summary",
      "supporting_facts": ["fact 1", "fact 2"],
      "missing_information": ["missing item 1"],
      "verification_targets": ["target 1"],
      "confidence": 0.0,
      "kill_conditions": ["kill 1"],
      "candidate_leg_types": ["ml", "total", "prop"]
    }
  ]
}

Rules:
- Return at most 5 theses.
- Confidence must be between 0 and 1.
- If nothing interesting is present, return {"theses": []}.
- Keep summaries concise and specific.
- Do not emit freeform explanation outside the JSON.
"""

THESIS_VERIFICATION_INSTRUCTIONS = """You are verifying structured and intuition theses for a sports longshot search tool.

Use only the supplied thesis payload and any web-search verification you explicitly perform.
Do not invent injuries, lineups, prices, or market conditions that are not explicitly present or sourced.

Return JSON only in this exact top-level shape:
{
  "verified_theses": [
    {
      "thesis_id": "existing thesis id",
      "verification_status": "verified or partially_verified or unverified or contradicted",
      "updated_confidence": 0.0,
      "verification_notes": ["note 1", "note 2"],
      "sources": [
        {"title": "source title", "url": "https://example.com"}
      ]
    }
  ]
}

Rules:
- Only verify thesis ids that already exist in the supplied payload.
- updated_confidence must be between 0 and 1.
- If nothing can be verified, return {"verified_theses": []}.
- Keep notes concise and evidence-oriented.
- Use `contradicted` when current sourced information materially breaks the thesis.
- Do not emit any explanation outside the JSON.
"""

CHAT_INSTRUCTIONS = """You are a contextual assistant inside a local sports market dashboard.

Answer the user's question using only the slate data, model outputs, cached roster context, injury state, and matchup context provided.
Do not invent injuries, prices, lineups, or confidence that are not present in the supplied payload.
If the answer depends on missing or stale data, say that directly.
Be concise, concrete, and decision-oriented.
If web search is available and the cached slate data is missing or stale, you may use it to verify current injuries, lineups, pitching changes, scratches, or official status updates. Prefer official league/team sources and reliable current reporting, and mention sourced updates succinctly.
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


def _iter_object_values(node: Any) -> list[Any]:
    values: list[Any] = []
    for name in (
        "output",
        "content",
        "text",
        "value",
        "action",
        "sources",
        "results",
        "items",
        "data",
        "annotations",
        "url",
        "title",
        "name",
        "status",
        "incomplete_details",
        "error",
        "type",
    ):
        try:
            value = getattr(node, name)
        except Exception:
            continue
        if value is not None:
            values.append(value)
    return values


def _quiet_model_dump(node: Any) -> Any:
    if not hasattr(node, "model_dump"):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return node.model_dump()
    except Exception:
        return None


def _extract_response_text(response: Any) -> str:
    direct = str(getattr(response, "output_text", "") or "").strip()
    if direct:
        return direct

    fragments: list[str] = []
    seen_nodes: set[int] = set()

    def walk(node: Any) -> None:
        node_id = id(node)
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
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
        elif isinstance(node, tuple):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            return
        else:
            node_type = str(getattr(node, "type", "") or "")
            if node_type in {"output_text", "text"}:
                value = getattr(node, "text", None)
                if isinstance(value, str) and value.strip():
                    fragments.append(value.strip())
                elif value is not None:
                    nested_value = getattr(value, "value", None) or getattr(value, "text", None)
                    if isinstance(nested_value, str) and nested_value.strip():
                        fragments.append(nested_value.strip())
                plain_value = getattr(node, "value", None)
                if isinstance(plain_value, str) and plain_value.strip():
                    fragments.append(plain_value.strip())
            for child in _iter_object_values(node):
                walk(child)

    walk(response)
    if not fragments:
        walk(_quiet_model_dump(response))

    deduped: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment in seen:
            continue
        seen.add(fragment)
        deduped.append(fragment)
    return "\n\n".join(deduped).strip()


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```" in raw:
        parts = [part.strip() for part in raw.split("```") if part.strip()]
        candidates = parts + candidates
    for candidate in candidates:
        cleaned = candidate
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            parsed = json.loads(cleaned)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _response_debug_details(response: Any) -> str:
    payload = _quiet_model_dump(response) or {}
    status = getattr(response, "status", None) or payload.get("status")
    incomplete = getattr(response, "incomplete_details", None) or payload.get("incomplete_details")
    error = getattr(response, "error", None) or payload.get("error")
    output_items = getattr(response, "output", None) or payload.get("output") or []
    parts = [f"status={status or 'unknown'}", f"output_items={len(output_items)}"]
    if incomplete:
        parts.append(f"incomplete={json.dumps(incomplete, default=str)}")
    if error:
        parts.append(f"error={json.dumps(error, default=str)}")
    return "; ".join(parts)


def _extract_web_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_nodes: set[int] = set()

    def maybe_add(source: Any) -> None:
        if isinstance(source, dict):
            url = str(source.get("url") or source.get("link") or "").strip()
            title = str(source.get("title") or source.get("name") or url).strip()
        else:
            url = str(getattr(source, "url", "") or getattr(source, "link", "") or "").strip()
            title = str(getattr(source, "title", "") or getattr(source, "name", "") or url).strip()
        if not url:
            return
        key = (title, url)
        if key in seen:
            return
        seen.add(key)
        sources.append({"title": title, "url": url})

    def walk(node: Any) -> None:
        node_id = id(node)
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        if isinstance(node, dict):
            node_type = str(node.get("type") or "")
            if node_type == "web_search_call":
                action = node.get("action")
                if isinstance(action, dict):
                    for source in action.get("sources") or []:
                        maybe_add(source)
            for key, value in node.items():
                if key == "sources" and isinstance(value, list):
                    for source in value:
                        maybe_add(source)
                walk(value)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, tuple):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            return
        else:
            node_type = str(getattr(node, "type", "") or "")
            if node_type == "web_search_call":
                action = getattr(node, "action", None)
                if action is not None:
                    for source in getattr(action, "sources", None) or []:
                        maybe_add(source)
            for child in _iter_object_values(node):
                if isinstance(child, list):
                    for source in child:
                        maybe_add(source)
                walk(child)

    walk(response)
    if not sources:
        walk(_quiet_model_dump(response))
    return sources


def _response_options(*, web_search: bool) -> dict[str, Any]:
    if not web_search:
        return {}
    return {
        "tools": [
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                },
            }
        ],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
    }


def _normalize_intuition_thesis(date_str: str, thesis: dict[str, Any], index: int) -> dict[str, Any] | None:
    if not isinstance(thesis, dict):
        return None
    summary = str(thesis.get("summary") or "").strip()
    if not summary:
        return None
    thesis_type = str(thesis.get("type") or "intuition").strip().lower().replace(" ", "_")
    sport = str(thesis.get("sport") or "").strip().lower()
    games = [
        str(game).strip()
        for game in (thesis.get("games") or [])
        if str(game).strip()
    ]
    confidence = _safe_float(thesis.get("confidence"))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "thesis_id": f"intuition_{thesis_type}_{date_str.replace('-', '_')}_{index}",
        "source": "intuition",
        "type": thesis_type or "intuition",
        "sport": sport,
        "games": games,
        "summary": summary,
        "confidence": round(confidence, 2),
        "supporting_facts": [
            str(item).strip() for item in (thesis.get("supporting_facts") or []) if str(item).strip()
        ][:6],
        "missing_information": [
            str(item).strip() for item in (thesis.get("missing_information") or []) if str(item).strip()
        ][:6],
        "verification_targets": [
            str(item).strip() for item in (thesis.get("verification_targets") or []) if str(item).strip()
        ][:6],
        "kill_conditions": [
            str(item).strip() for item in (thesis.get("kill_conditions") or []) if str(item).strip()
        ][:6],
        "candidate_leg_types": [
            str(item).strip() for item in (thesis.get("candidate_leg_types") or []) if str(item).strip()
        ][:6],
    }


def _normalize_verified_thesis(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    thesis_id = str(item.get("thesis_id") or "").strip()
    if not thesis_id:
        return None
    status = str(item.get("verification_status") or "").strip().lower()
    if status not in {"verified", "partially_verified", "unverified", "contradicted"}:
        status = "unverified"
    confidence = _safe_float(item.get("updated_confidence"))
    confidence = max(0.0, min(1.0, confidence))
    sources = []
    for source in item.get("sources") or []:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        title = str(source.get("title") or url).strip()
        sources.append({"title": title, "url": url})
    return {
        "thesis_id": thesis_id,
        "verification_status": status,
        "updated_confidence": round(confidence, 2),
        "verification_notes": [
            str(note).strip() for note in (item.get("verification_notes") or []) if str(note).strip()
        ][:6],
        "sources": sources[:6],
    }


def _normalize_thesis_judgment(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    thesis_id = str(item.get("thesis_id") or "").strip()
    if not thesis_id:
        return None
    verdict = str(item.get("verdict") or "").strip().lower()
    if verdict not in {"back", "conditional", "lean", "pass"}:
        verdict = "pass"
    confidence = max(0.0, min(1.0, _safe_float(item.get("confidence"))))
    dfs_guidance = item.get("dfs_guidance") or {}
    preferred_salary_shape = str(dfs_guidance.get("preferred_salary_shape") or "").strip().lower()
    if preferred_salary_shape not in {"balanced", "stars_and_scrubs", "leave_salary"}:
        preferred_salary_shape = ""
    return {
        "thesis_id": thesis_id,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "best_candidate_index": max(0, int(_safe_float(item.get("best_candidate_index"), 0))),
        "reason": str(item.get("reason") or "").strip(),
        "risks": [str(risk).strip() for risk in (item.get("risks") or []) if str(risk).strip()][:6],
        "kill_conditions": [
            str(condition).strip()
            for condition in (item.get("kill_conditions") or [])
            if str(condition).strip()
        ][:6],
        "dfs_guidance": {
            "stack_targets": [
                str(name).strip()
                for name in (dfs_guidance.get("stack_targets") or [])
                if str(name).strip()
            ][:6],
            "bring_back_targets": [
                str(name).strip()
                for name in (dfs_guidance.get("bring_back_targets") or [])
                if str(name).strip()
            ][:6],
            "one_off_targets": [
                str(name).strip()
                for name in (dfs_guidance.get("one_off_targets") or [])
                if str(name).strip()
            ][:6],
            "avoid_chalk": [
                str(name).strip()
                for name in (dfs_guidance.get("avoid_chalk") or [])
                if str(name).strip()
            ][:6],
            "max_players_from_game": min(5, max(1, int(_safe_float(dfs_guidance.get("max_players_from_game"), 0))))
            if dfs_guidance.get("max_players_from_game") is not None
            else None,
            "preferred_salary_shape": preferred_salary_shape or None,
        },
    }


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact["top_legs"] = compact.get("top_legs", [])[:10]
    compact["fades"] = compact.get("fades", [])[:6]
    compact["thesis_candidates"] = compact.get("thesis_candidates", [])[:8]
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
        lineup_available = bool(lineup_status.get(f"{side}_available")) or bool(lineup_status.get(f"{side}_confirmed"))
        return {
            "lineup_source": str(lineup_status.get(f"{side}_source") or "missing"),
            "lineup_available": lineup_available,
            "lineup_confirmed": lineup_available,
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


def extract_judgment_games(result: dict[str, Any], max_games: int | None = None) -> list[tuple[str, str]]:
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

    for thesis in result.get("thesis_candidates", []) or []:
        for candidate in thesis.get("candidates", []) or []:
            for leg in candidate.get("legs", []) or []:
                maybe_add(leg)
    for parlay in result.get("tier_parlays", []) or []:
        for leg in parlay.get("legs", []) or []:
            maybe_add(leg)
    for leg in result.get("top_legs", []) or []:
        maybe_add(leg)
    for leg in result.get("fades", []) or []:
        maybe_add(leg)
    moonshot = result.get("moonshot")
    if isinstance(moonshot, dict):
        maybe_add(moonshot)
    if max_games is None:
        meta_games = result.get("meta", {}).get("games")
        try:
            max_games = max(12, int(meta_games) + 4) if meta_games is not None else 20
        except (TypeError, ValueError):
            max_games = 20
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

    def thesis_summary(thesis: dict[str, Any]) -> dict[str, Any]:
        return {
            "thesis_id": thesis.get("thesis_id"),
            "type": thesis.get("type"),
            "source": thesis.get("source"),
            "sport": thesis.get("sport"),
            "games": thesis.get("games") or [],
            "summary": thesis.get("summary"),
            "confidence": round(
                _safe_float(thesis.get("verified_confidence", thesis.get("confidence"))),
                2,
            ),
            "verification_status": thesis.get("verification_status", "unverified"),
            "supporting_facts": _trim_list(list(thesis.get("supporting_facts") or []), limit=5),
            "missing_information": _trim_list(list(thesis.get("missing_information") or []), limit=5),
            "kill_conditions": _trim_list(list(thesis.get("kill_conditions") or []), limit=5),
        }

    def candidate_summary(entry: dict[str, Any]) -> dict[str, Any]:
        candidates = []
        for index, candidate in enumerate(entry.get("candidates") or []):
            candidates.append(
                {
                    "candidate_index": index,
                    "actual_size": candidate.get("actual_size"),
                    "payout_estimate": round(_safe_float(candidate.get("payout_estimate")), 2)
                    if candidate.get("payout_estimate") is not None
                    else None,
                    "model_joint_prob": round(_safe_float(candidate.get("model_joint_prob")), 3)
                    if candidate.get("model_joint_prob") is not None
                    else None,
                    "market_joint_prob": round(_safe_float(candidate.get("market_joint_prob")), 3)
                    if candidate.get("market_joint_prob") is not None
                    else None,
                    "expression_score": round(_safe_float(candidate.get("expression_score")), 3),
                    "correlation_flags": candidate.get("correlation_flags") or [],
                    "data_quality_flags": candidate.get("data_quality_flags") or [],
                    "kill_conditions": candidate.get("kill_conditions") or [],
                    "legs": [leg_summary(leg) for leg in (candidate.get("legs") or [])],
                }
            )
        return {
            "thesis_id": entry.get("thesis_id"),
            "type": entry.get("type"),
            "source": entry.get("source"),
            "sport": entry.get("sport"),
            "games": entry.get("games") or [],
            "summary": entry.get("summary"),
            "confidence": round(_safe_float(entry.get("confidence")), 2),
            "verification_status": entry.get("verification_status", "unverified"),
            "supporting_legs": [leg_summary(leg) for leg in (entry.get("supporting_legs") or [])[:8]],
            "candidates": candidates,
        }

    payload: dict[str, Any] = {
        "config": {
            "date": config.get("date"),
            "sport": config.get("sport"),
            "slate_mode": config.get("slate_mode"),
            "score_source": config.get("score_source"),
            "props_only": bool(config.get("props_only")),
        },
        "meta": {
            "entropy_source": (result.get("meta") or {}).get("entropy_source"),
            "kalshi_markets": (result.get("meta") or {}).get("kalshi_markets"),
            "games": (result.get("meta") or {}).get("games"),
            "pricing_summary": (result.get("meta") or {}).get("pricing_summary"),
        },
        "refresh": result.get("refresh") or {},
        "structured_theses": [thesis_summary(thesis) for thesis in (result.get("theses") or [])],
        "intuition_theses": [thesis_summary(thesis) for thesis in (result.get("intuition_theses") or [])],
        "thesis_candidates": [candidate_summary(item) for item in (result.get("thesis_candidates") or [])],
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
    score_source = str((result.get("config") or {}).get("score_source") or "")
    mode_note = ""
    if score_source == "implied":
        mode_note = (
            "Special note: this slate is running in pure market implied mode. "
            "There is no independent model to compare against the market in this mode. "
            "Do not frame the answer as if zero edge means the slate failed a model check. "
            "Instead, treat this as a market-and-context review of the thesis board: identify the best available thesis expressions, explain the confidence limits, and say what fresh lineup or injury information would upgrade or kill the lean.\n\n"
        )
    return (
        "Assess this slate by judging the thesis bundles and their candidate expressions directly. "
        "Do not center your answer on generic recommendation buckets. "
        "Identify the strongest thesis, the best expression of that thesis, and the main fragile or broken theses.\n\n"
        f"{mode_note}"
        "Slate payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_thesis_judge_prompt(result: dict[str, Any], *, compact: bool = False) -> str:
    payload = build_judgment_payload(result)
    if compact:
        payload = _compact_payload(payload)
    return (
        "Rank the supplied thesis bundles and their candidate expressions. "
        "Pick the best candidates only from the candidates already attached to each thesis.\n\n"
        "Slate payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_intuition_thesis_prompt(result: dict[str, Any], *, compact: bool = False) -> str:
    payload = build_judgment_payload(result)
    if compact:
        payload = _compact_payload(payload)
    return (
        "Generate bounded intuition theses for this slate. "
        "Use the structured theses already present as context, but feel free to propose different hypotheses if the payload supports them.\n\n"
        "Slate payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def build_thesis_verification_prompt(result: dict[str, Any], *, compact: bool = False) -> str:
    payload = build_judgment_payload(result)
    if compact:
        payload = _compact_payload(payload)
    return (
        "Verify the supplied theses against the current slate payload. "
        "Use web search only when the payload is stale or missing key facts, and return per-thesis verification statuses.\n\n"
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
    web_search: bool | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    allow_web_search = DEFAULT_WEB_SEARCH if web_search is None else bool(web_search)
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
            **_response_options(web_search=allow_web_search),
        )
    except Exception as exc:  # pragma: no cover - network/runtime failure
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
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
            **_response_options(web_search=allow_web_search),
        )
        text = _extract_response_text(retry_response)
        if text:
            response = retry_response
    if not text:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": _extract_web_sources(response),
            "message": f"OpenAI returned no judgment text. {_response_debug_details(response)}",
            "text": "",
            "generated_at": None,
        }
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "web_search_enabled": allow_web_search,
        "sources": _extract_web_sources(response),
        "message": "",
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_thesis_judgment(
    result: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    web_search: bool | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    allow_web_search = DEFAULT_WEB_SEARCH if web_search is None else bool(web_search)
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    ready, reason = dependency_status(api_key)
    if not ready:
        return {
            "status": "unavailable",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": reason,
            "call": "",
            "portfolio_note": "",
            "ranked_theses": [],
            "generated_at": None,
        }

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    prompt = build_thesis_judge_prompt(result)
    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=THESIS_JUDGE_INSTRUCTIONS,
            input=prompt,
            **_response_options(web_search=allow_web_search),
        )
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": str(exc),
            "call": "",
            "portfolio_note": "",
            "ranked_theses": [],
            "generated_at": None,
        }

    text = _extract_response_text(response)
    parsed = _extract_json_payload(text) if text else None
    if not parsed:
        retry_response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=THESIS_JUDGE_INSTRUCTIONS,
            input=build_thesis_judge_prompt(result, compact=True),
            **_response_options(web_search=allow_web_search),
        )
        retry_text = _extract_response_text(retry_response)
        retry_parsed = _extract_json_payload(retry_text) if retry_text else None
        if retry_parsed:
            response = retry_response
            parsed = retry_parsed
    if not parsed:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": _extract_web_sources(response),
            "message": f"OpenAI returned no thesis judgment JSON. {_response_debug_details(response)}",
            "call": "",
            "portfolio_note": "",
            "ranked_theses": [],
            "generated_at": None,
        }

    candidate_lookup = {item.get("thesis_id"): item for item in (result.get("thesis_candidates") or [])}
    ranked_theses = []
    for item in parsed.get("ranked_theses") or []:
        normalized = _normalize_thesis_judgment(item)
        if normalized is None:
            continue
        thesis_bundle = candidate_lookup.get(normalized["thesis_id"], {})
        candidates = thesis_bundle.get("candidates") or []
        index = normalized["best_candidate_index"]
        normalized["best_candidate"] = candidates[index] if 0 <= index < len(candidates) else None
        ranked_theses.append(normalized)
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "web_search_enabled": allow_web_search,
        "sources": _extract_web_sources(response),
        "message": "",
        "call": str(parsed.get("call") or "").strip(),
        "portfolio_note": str(parsed.get("portfolio_note") or "").strip(),
        "ranked_theses": ranked_theses,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_intuition_theses(
    result: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    web_search: bool | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    allow_web_search = DEFAULT_WEB_SEARCH if web_search is None else bool(web_search)
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    ready, reason = dependency_status(api_key)
    if not ready:
        return {
            "status": "unavailable",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": reason,
            "theses": [],
            "generated_at": None,
        }

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    prompt = build_intuition_thesis_prompt(result)
    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=INTUITION_THESIS_INSTRUCTIONS,
            input=prompt,
            **_response_options(web_search=allow_web_search),
        )
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": str(exc),
            "theses": [],
            "generated_at": None,
        }

    text = _extract_response_text(response)
    parsed = _extract_json_payload(text) if text else None
    if not parsed:
        retry_response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=INTUITION_THESIS_INSTRUCTIONS,
            input=build_intuition_thesis_prompt(result, compact=True),
            **_response_options(web_search=allow_web_search),
        )
        retry_text = _extract_response_text(retry_response)
        retry_parsed = _extract_json_payload(retry_text) if retry_text else None
        if retry_parsed:
            response = retry_response
            parsed = retry_parsed
    if not parsed:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": _extract_web_sources(response),
            "message": f"OpenAI returned no intuition thesis JSON. {_response_debug_details(response)}",
            "theses": [],
            "generated_at": None,
        }

    date_str = str((result.get("config") or {}).get("date") or datetime.now().strftime("%Y-%m-%d"))
    theses = []
    for index, thesis in enumerate(parsed.get("theses") or [], start=1):
        normalized = _normalize_intuition_thesis(date_str, thesis, index)
        if normalized is not None:
            theses.append(normalized)
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "web_search_enabled": allow_web_search,
        "sources": _extract_web_sources(response),
        "message": "",
        "theses": theses,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_verified_theses(
    result: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    web_search: bool | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    allow_web_search = DEFAULT_WEB_SEARCH if web_search is None else bool(web_search)
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    ready, reason = dependency_status(api_key)
    if not ready:
        return {
            "status": "unavailable",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": reason,
            "verified_theses": [],
            "generated_at": None,
        }

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    prompt = build_thesis_verification_prompt(result)
    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=THESIS_VERIFICATION_INSTRUCTIONS,
            input=prompt,
            **_response_options(web_search=allow_web_search),
        )
    except Exception as exc:  # pragma: no cover
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
            "message": str(exc),
            "verified_theses": [],
            "generated_at": None,
        }

    text = _extract_response_text(response)
    parsed = _extract_json_payload(text) if text else None
    if not parsed:
        retry_response = client.responses.create(
            model=model_name,
            reasoning={"effort": effort},
            instructions=THESIS_VERIFICATION_INSTRUCTIONS,
            input=build_thesis_verification_prompt(result, compact=True),
            **_response_options(web_search=allow_web_search),
        )
        retry_text = _extract_response_text(retry_response)
        retry_parsed = _extract_json_payload(retry_text) if retry_text else None
        if retry_parsed:
            response = retry_response
            parsed = retry_parsed
    if not parsed:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": _extract_web_sources(response),
            "message": f"OpenAI returned no thesis verification JSON. {_response_debug_details(response)}",
            "verified_theses": [],
            "generated_at": None,
        }

    verified_theses = []
    for item in parsed.get("verified_theses") or []:
        normalized = _normalize_verified_thesis(item)
        if normalized is not None:
            verified_theses.append(normalized)
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "web_search_enabled": allow_web_search,
        "sources": _extract_web_sources(response),
        "message": "",
        "verified_theses": verified_theses,
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
    web_search: bool | None = None,
) -> dict[str, Any]:
    model_name = str(model or DEFAULT_OPENAI_MODEL).strip()
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    allow_web_search = DEFAULT_WEB_SEARCH if web_search is None else bool(web_search)
    if effort not in SUPPORTED_REASONING_EFFORTS:
        effort = DEFAULT_REASONING_EFFORT

    if not question.strip():
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
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
            **_response_options(web_search=allow_web_search),
        )
    except Exception as exc:  # pragma: no cover - network/runtime failure
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": [],
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
            **_response_options(web_search=allow_web_search),
        )
        text = _extract_response_text(retry_response)
        if text:
            response = retry_response
    if not text:
        return {
            "status": "error",
            "model": model_name,
            "reasoning_effort": effort,
            "web_search_enabled": allow_web_search,
            "sources": _extract_web_sources(response),
            "message": f"OpenAI returned no chat text. {_response_debug_details(response)}",
            "text": "",
            "generated_at": None,
        }
    return {
        "status": "ok",
        "model": model_name,
        "reasoning_effort": effort,
        "web_search_enabled": allow_web_search,
        "sources": _extract_web_sources(response),
        "message": "",
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
