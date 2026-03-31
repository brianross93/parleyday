from __future__ import annotations

from itertools import combinations
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _market_parlay_payout(legs: list[Any], indices: list[int]) -> float | None:
    if not indices:
        return None
    entry_cost = 1.0
    for idx in indices:
        leg = legs[idx]
        price = getattr(leg, "entry_price", None)
        if price is None:
            price = getattr(leg, "implied_prob", None)
        price = _safe_float(price)
        if price <= 0:
            return None
        entry_cost *= price
    return (1.0 / entry_cost) if entry_cost > 0 else None


def _model_parlay_probability(activation: Any, co_activation: Any, indices: list[int]) -> float:
    if not indices:
        return 0.0
    if len(indices) == 1:
        return _safe_float(activation[indices[0]])
    base_prob = 1.0
    pairwise_ratios = []
    for idx in indices:
        base_prob *= _safe_float(activation[idx], 1.0)
    for i, first in enumerate(indices):
        for second in indices[i + 1 :]:
            independent = _safe_float(activation[first]) * _safe_float(activation[second])
            if independent <= 0:
                continue
            joint = _safe_float(co_activation[first, second])
            pairwise_ratios.append(joint / independent)
    if not pairwise_ratios:
        return max(0.0, min(0.999, base_prob))
    mean_ratio = sum(pairwise_ratios) / len(pairwise_ratios)
    return max(0.0, min(0.999, base_prob * max(0.6, min(1.6, mean_ratio))))


def _serialize_leg(leg: Any, activation_value: float, pricing_meta: dict[str, str] | None) -> dict[str, Any]:
    pricing_meta = pricing_meta or {}
    return {
        "label": getattr(leg, "label", ""),
        "sport": getattr(leg, "sport", ""),
        "category": getattr(leg, "category", ""),
        "game": getattr(leg, "game", ""),
        "activation": activation_value,
        "implied_prob": _safe_float(getattr(leg, "implied_prob", 0.0)),
        "entry_price": _safe_float(getattr(leg, "entry_price", getattr(leg, "implied_prob", 0.0))),
        "score_delta": activation_value - _safe_float(getattr(leg, "implied_prob", 0.0)),
        "trust_score": 0.0,
        "pricing_source": pricing_meta.get("pricing_source", "market"),
        "pricing_label": pricing_meta.get("pricing_label", "Market implied"),
        "pricing_reason": pricing_meta.get("pricing_reason"),
        "notes": getattr(leg, "notes", ""),
    }


def _player_name_from_label(label: str) -> str:
    raw = str(label or "").strip()
    if " O " in raw:
        return raw.split(" O ", 1)[0].strip()
    return ""


def _stat_from_label(label: str) -> str:
    raw = str(label or "").strip().upper()
    if raw.endswith(" O 1 H"):
        return "hits"
    if raw.endswith("HR"):
        return "hr"
    if " REB" in raw:
        return "rebounds"
    if " AST" in raw:
        return "assists"
    if " PTS" in raw:
        return "points"
    if " O " in raw:
        return "prop"
    return str(raw).lower()


def _is_suspect_leg(leg: Any, activation_value: float, pricing_meta: dict[str, str] | None) -> bool:
    pricing_meta = pricing_meta or {}
    source = str(pricing_meta.get("pricing_source", "market") or "").strip().lower()
    category = str(getattr(leg, "category", "") or "").strip().lower()
    implied_prob = _safe_float(getattr(leg, "implied_prob", 0.0))
    entry_price = _safe_float(getattr(leg, "entry_price", implied_prob))
    if source in {"market_fallback", "implied_fallback"} and category == "prop":
        if implied_prob <= 0.02 or entry_price <= 0.02 or activation_value <= 0.03:
            return True
    return False


def _thesis_matches_leg(thesis: dict[str, Any], leg: Any) -> bool:
    sport = str(thesis.get("sport") or "").strip().lower()
    if sport and str(getattr(leg, "sport", "")).strip().lower() != sport:
        return False
    games = [str(game).strip() for game in (thesis.get("games") or []) if str(game).strip()]
    if games and str(getattr(leg, "game", "")) not in games:
        return False
    candidate_leg_types = {str(item).strip().lower() for item in (thesis.get("candidate_leg_types") or [])}
    if not candidate_leg_types:
        return True
    category = str(getattr(leg, "category", "")).strip().lower()
    label = str(getattr(leg, "label", "")).strip().upper()
    if category in candidate_leg_types:
        return True
    if "prop" in candidate_leg_types and category == "prop":
        return True
    if category == "prop":
        if "hits" in candidate_leg_types and label.endswith(" O 1 H"):
            return True
        if "hr" in candidate_leg_types and label.endswith("HR"):
            return True
        if "points" in candidate_leg_types and " PTS" in label:
            return True
        if "rebounds" in candidate_leg_types and " REB" in label:
            return True
        if "assists" in candidate_leg_types and " AST" in label:
            return True
    return False


def _leg_matches_intent(intent_type: str, leg: Any) -> bool:
    intent = str(intent_type or "").strip().lower()
    category = str(getattr(leg, "category", "")).strip().lower()
    label = str(getattr(leg, "label", "")).strip().upper()
    if intent in {"ml", "total"}:
        return category == intent
    if intent == "prop":
        return category == "prop"
    if category != "prop":
        return False
    if intent == "hits":
        return label.endswith(" O 1 H")
    if intent == "hr":
        return label.endswith("HR")
    if intent == "points":
        return " PTS" in label
    if intent == "rebounds":
        return " REB" in label
    if intent == "assists":
        return " AST" in label
    return False


def _thesis_leg_score(
    thesis: dict[str, Any],
    leg: Any,
    activation_value: float,
    pricing_meta: dict[str, str] | None,
) -> float:
    score = activation_value * 3.0
    category = str(getattr(leg, "category", "")).strip().lower()
    thesis_type = str(thesis.get("type") or "").strip().lower()
    edge = activation_value - _safe_float(getattr(leg, "implied_prob", 0.0))
    player_name = _player_name_from_label(getattr(leg, "label", ""))
    stat_name = _stat_from_label(getattr(leg, "label", ""))
    focus_players = {str(item).strip().lower() for item in (thesis.get("focus_players") or []) if str(item).strip()}
    fade_players = {str(item).strip().lower() for item in (thesis.get("fade_players") or []) if str(item).strip()}
    focus_stats = {str(item).strip().lower() for item in (thesis.get("focus_stats") or []) if str(item).strip()}
    if player_name and player_name.lower() in focus_players:
        score += 3.25
    if player_name and player_name.lower() in fade_players:
        score -= 2.5
    if stat_name and stat_name.lower() in focus_stats:
        score += 1.4
    if thesis_type == "model_market_divergence":
        score += max(edge, 0.0) * 10.0
        score -= abs(min(edge, 0.0)) * 2.0
    if thesis_type == "bullpen_exhaustion":
        if category == "total":
            score += 0.6
        if category == "prop":
            score += 0.18
    elif thesis_type == "run_environment":
        if category == "total":
            score += 0.5
        if category == "prop":
            score += 0.16
    elif thesis_type == "thin_rotation":
        if category == "prop":
            score += 0.42
        if category == "ml":
            score += 0.22
    else:
        score += abs(edge) * 1.2
    return score


def _data_quality_flags(
    thesis: dict[str, Any],
    legs: list[Any],
    indices: list[int],
    activation: Any,
    pricing_details: dict[int, dict[str, str]] | None,
) -> list[str]:
    flags: list[str] = []
    if thesis.get("missing_information"):
        flags.extend([f"missing:{item}" for item in thesis.get("missing_information", [])[:3]])
    for idx in indices:
        pricing = (pricing_details or {}).get(idx, {})
        source = pricing.get("pricing_source", "market")
        if source in {"market_fallback", "implied_fallback"} and "fallback_pricing" not in flags:
            flags.append("fallback_pricing")
    return flags[:6]


def _correlation_flags(legs: list[Any], indices: list[int]) -> list[str]:
    flags: list[str] = []
    games = [str(getattr(legs[idx], "game", "")) for idx in indices]
    if len(set(games)) < len(games):
        flags.append("same_game")
    categories = [str(getattr(legs[idx], "category", "")) for idx in indices]
    if len(set(categories)) < len(categories):
        flags.append("duplicate_category")
    return flags


def _has_direct_contradiction(legs: list[Any], indices: list[int]) -> bool:
    for i, first in enumerate(indices):
        for second in indices[i + 1 :]:
            a = legs[first]
            b = legs[second]
            if str(getattr(a, "game", "")) != str(getattr(b, "game", "")):
                continue
            if str(getattr(a, "category", "")) == "ml" and str(getattr(b, "category", "")) == "ml":
                return True
            if str(getattr(a, "category", "")) == "total" and str(getattr(b, "category", "")) == "total":
                return True
    return False


def _intent_pools(
    thesis: dict[str, Any],
    legs: list[Any],
    activation: Any,
    pricing_details: dict[int, dict[str, str]] | None,
) -> tuple[list[str], dict[str, list[int]]]:
    focus_stats = [str(item).strip().lower() for item in (thesis.get("focus_stats") or []) if str(item).strip()]
    intents = focus_stats + [
        str(item).strip().lower() for item in (thesis.get("candidate_leg_types") or []) if str(item).strip()
    ]
    intents = list(dict.fromkeys(intents))
    if not intents:
        intents = ["ml", "total", "prop"]
    pools: dict[str, list[int]] = {}
    for intent in intents:
        matches = [
            idx
            for idx, leg in enumerate(legs)
            if _thesis_matches_leg(thesis, leg)
            and _leg_matches_intent(intent, leg)
            and not _is_suspect_leg(leg, _safe_float(activation[idx]), (pricing_details or {}).get(idx))
        ]
        matches.sort(
            key=lambda idx: _thesis_leg_score(thesis, legs[idx], _safe_float(activation[idx]), (pricing_details or {}).get(idx)),
            reverse=True,
        )
        pools[intent] = matches[:4]
    return intents, pools


def _candidate_alignment_flags(thesis: dict[str, Any], legs: list[Any], indices: list[int]) -> list[str]:
    flags: list[str] = []
    focus_players = {str(item).strip().lower() for item in (thesis.get("focus_players") or []) if str(item).strip()}
    focus_stats = {str(item).strip().lower() for item in (thesis.get("focus_stats") or []) if str(item).strip()}
    if focus_players:
        aligned_player = any(_player_name_from_label(getattr(legs[idx], "label", "")).lower() in focus_players for idx in indices)
        if not aligned_player:
            flags.append("focus_player_mismatch")
    if focus_stats:
        aligned_stat = any(_stat_from_label(getattr(legs[idx], "label", "")).lower() in focus_stats for idx in indices)
        if not aligned_stat:
            flags.append("focus_stat_mismatch")
    return flags


def build_thesis_candidates(
    *,
    theses: list[dict[str, Any]],
    legs: list[Any],
    activation: Any,
    co_activation: Any,
    pricing_details: dict[int, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    pricing_details = pricing_details or {}
    results: list[dict[str, Any]] = []

    for thesis in theses:
        intents, pools = _intent_pools(thesis, legs, activation, pricing_details)
        ranked = []
        seen_ranked: set[int] = set()
        for intent in intents:
            for idx in pools.get(intent, []):
                if idx in seen_ranked:
                    continue
                seen_ranked.add(idx)
                ranked.append(idx)
        supporting_legs = [
            {
                **_serialize_leg(legs[idx], _safe_float(activation[idx]), pricing_details.get(idx)),
                "intent_type": next(
                    (
                        intent
                        for intent in intents
                        if idx in (pools.get(intent) or [])
                    ),
                    "mixed",
                ),
            }
            for idx in ranked[:8]
        ]
        candidate_pool: list[dict[str, Any]] = []
        seed_combos: list[tuple[int, ...]] = []
        for intent in intents:
            pool = pools.get(intent) or []
            for idx in pool[:2]:
                seed_combos.append((idx,))
        if len(intents) >= 2:
            first_pool = pools.get(intents[0]) or []
            second_pool = pools.get(intents[1]) or []
            for left in first_pool[:2]:
                for right in second_pool[:2]:
                    if left != right:
                        seed_combos.append(tuple(sorted((left, right))))
        unique_seeds: list[tuple[int, ...]] = []
        seen_seeds: set[tuple[int, ...]] = set()
        for seed in seed_combos:
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            unique_seeds.append(seed)
        for seed in unique_seeds or [tuple()]:
            available = [idx for idx in ranked if idx not in seed]
            seed_len = len(seed)
            max_size = min(5, len(ranked))
            start_size = max(1, seed_len)
            for size in range(start_size, max_size + 1):
                extra_needed = size - seed_len
                extra_combos = [tuple()] if extra_needed == 0 else combinations(available, extra_needed)
                for extra in extra_combos:
                    combo = tuple(dict.fromkeys(seed + tuple(extra)))
                    if not combo:
                        continue
                    indices = list(combo)
                    if len(indices) != size:
                        continue
                    if any(
                        _is_suspect_leg(legs[idx], _safe_float(activation[idx]), pricing_details.get(idx))
                        for idx in indices
                    ):
                        continue
                    if _has_direct_contradiction(legs, indices):
                        continue
                    payout = _market_parlay_payout(legs, indices)
                    if payout is None:
                        continue
                    model_prob = _model_parlay_probability(activation, co_activation, indices)
                    market_prob = 1.0 / payout if payout > 0 else 0.0
                    average_edge = sum(
                        _safe_float(activation[idx]) - _safe_float(getattr(legs[idx], "implied_prob", 0.0))
                        for idx in indices
                    ) / len(indices)
                    mean_leg_score = sum(
                        _thesis_leg_score(thesis, legs[idx], _safe_float(activation[idx]), pricing_details.get(idx))
                        for idx in indices
                    ) / len(indices)
                    intent_coverage = 0
                    for intent in intents:
                        if any(idx in (pools.get(intent) or []) for idx in indices):
                            intent_coverage += 1
                    if str(thesis.get("type") or "").strip().lower() == "model_market_divergence":
                        expression_score = (
                            (mean_leg_score * 3.0)
                            + (max(average_edge, 0.0) * 18.0)
                            + (model_prob * 6.0)
                            + (intent_coverage * 0.45)
                        )
                    else:
                        expression_score = (
                            (model_prob * 28.0)
                            + (mean_leg_score * 0.7)
                            + (intent_coverage * 0.45)
                        )
                    candidate_pool.append(
                        {
                            "actual_size": len(indices),
                            "payout_estimate": payout,
                            "model_joint_prob": model_prob,
                            "market_joint_prob": market_prob,
                            "average_edge": average_edge,
                            "expression_score": expression_score,
                            "correlation_flags": _correlation_flags(legs, indices),
                            "data_quality_flags": _data_quality_flags(thesis, legs, indices, activation, pricing_details)
                            + _candidate_alignment_flags(thesis, legs, indices),
                            "kill_conditions": thesis.get("kill_conditions", []),
                            "legs": [
                                _serialize_leg(legs[idx], _safe_float(activation[idx]), pricing_details.get(idx))
                                for idx in indices
                            ],
                        }
                    )

        candidate_pool.sort(key=lambda item: item["expression_score"], reverse=True)
        natural_candidates: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for candidate in candidate_pool:
            signature = tuple(sorted(str(leg.get("label") or "") for leg in candidate.get("legs", [])))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            natural_candidates.append(candidate)
            if len(natural_candidates) >= 3:
                break

        results.append(
            {
                "thesis_id": thesis.get("thesis_id"),
                "summary": thesis.get("summary"),
                "type": thesis.get("type"),
                "source": thesis.get("source"),
                "sport": thesis.get("sport"),
                "games": thesis.get("games", []),
                "confidence": thesis.get("verified_confidence", thesis.get("confidence")),
                "verification_status": thesis.get("verification_status", "unverified"),
                "supporting_legs": supporting_legs,
                "best_candidate": (
                    next(
                        (
                            candidate
                            for candidate in natural_candidates
                            if "focus_player_mismatch" not in (candidate.get("data_quality_flags") or [])
                        ),
                        natural_candidates[0],
                    )
                    if natural_candidates
                    else None
                ),
                "candidates": natural_candidates,
            }
        )

    return results
