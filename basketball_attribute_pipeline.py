from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


RowKey = tuple[str, str]
RowData = dict[str, object]


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    components: dict[str, float]
    base: float = 0.0
    position_baselines: dict[str, float] | None = None
    allowed_sources: tuple[str, ...] = ("base",)
    curve_strength: float = 6.0
    max_rating: float = 20.0


def build_attribute_scores(rows: list[RowData]) -> dict[RowKey, dict[str, float]]:
    feature_values = []
    for row in rows:
        feature_row = _build_feature_row(row)
        feature_row["_row_key"] = (str(row.get("team_code") or ""), str(row.get("name_key") or ""))
        feature_values.append(feature_row)
    percentiles = _build_percentiles(feature_values)

    specs = (
        AttributeSpec(
            name="ball_handle",
            components={
                "tracking:dribbles_per_touch": 0.24,
                "tracking:time_of_poss_per_touch": 0.18,
                "tracking:drives_per_touch": 0.08,
                "tracking:drive_tov_rate_inv": 0.08,
                "pbp:bh_tov_rate_inv": 0.15,
                "base:turnover_control_inv": 0.08,
                "base:usage_load": 0.07,
                "base:assist_load": 0.07,
                "base:creation_load": 0.05,
            },
            base=0.02,
            position_baselines={"PG": 0.14, "SG": 0.10, "SF": 0.03, "PF": -0.05, "C": -0.14},
            allowed_sources=("base", "tracking", "pbp"),
            curve_strength=5.4,
            max_rating=19.5,
        ),
        AttributeSpec(
            name="pullup_shooting",
            components={
                "tracking:pullup_fga": 0.24,
                "tracking:pullup_fg_pct": 0.16,
                "tracking:pullup_fg3a": 0.10,
                "tracking:pullup_fg3_pct": 0.08,
                "tracking:pullup_efg_pct": 0.12,
                "base:shot_creation_load": 0.12,
                "base:free_throw_pressure": 0.08,
                "base:usage_load": 0.05,
                "base:scoring_load": 0.05,
            },
            base=0.02,
            position_baselines={"PG": 0.05, "SG": 0.05, "SF": 0.02, "PF": -0.04, "C": -0.10},
            allowed_sources=("base", "tracking"),
            curve_strength=5.2,
            max_rating=19.4,
        ),
        AttributeSpec(
            name="catch_shoot",
            components={
                "tracking:catch_shoot_fga": 0.12,
                "tracking:catch_shoot_fg_pct": 0.14,
                "tracking:catch_shoot_fg3a": 0.24,
                "tracking:catch_shoot_fg3_pct": 0.14,
                "tracking:catch_shoot_efg_pct": 0.10,
                "base:three_volume": 0.18,
                "base:minutes_load": 0.05,
            },
            base=0.04,
            position_baselines={"PG": 0.03, "SG": 0.08, "SF": 0.06, "PF": 0.0, "C": -0.08},
            allowed_sources=("base", "tracking"),
            curve_strength=5.0,
            max_rating=19.3,
        ),
        AttributeSpec(
            name="finishing",
            components={
                "scoring_load": 0.20,
                "free_throw_pressure": 0.24,
                "oreb_rate": 0.16,
                "usage_load": 0.10,
                "minutes_load": 0.08,
            },
            base=0.06,
            position_baselines={"PG": -0.02, "SG": 0.02, "SF": 0.08, "PF": 0.14, "C": 0.18},
            allowed_sources=("base", "tracking"),
        ),
        AttributeSpec(
            name="pass_vision",
            components={
                "tracking:potential_assists": 0.24,
                "tracking:secondary_assists": 0.08,
                "tracking:ast_points_created": 0.20,
                "tracking:ast_to_pass_pct": 0.08,
                "tracking:usage_pct": 0.10,
                "tracking:passes_made": 0.03,
                "base:assist_load": 0.16,
                "base:creation_load": 0.09,
                "base:minutes_load": 0.02,
                "pbp:pass_tov_rate_inv": 0.03,
            },
            base=0.02,
            position_baselines={"PG": 0.12, "SG": 0.04, "SF": 0.0, "PF": -0.04, "C": -0.10},
            allowed_sources=("base", "tracking", "pbp"),
            curve_strength=4.6,
            max_rating=19.7,
        ),
        AttributeSpec(
            name="pass_accuracy",
            components={
                "tracking:ast_to_pass_pct": 0.22,
                "tracking:ast_to_ratio": 0.18,
                "tracking:ast_points_created": 0.06,
                "tracking:usage_pct": 0.04,
                "pbp:pass_tov_rate_inv": 0.18,
                "base:turnover_control_inv": 0.18,
                "base:assist_load": 0.08,
                "base:creation_load": 0.08,
            },
            base=0.05,
            position_baselines={"PG": 0.08, "SG": 0.03, "SF": 0.0, "PF": -0.04, "C": -0.08},
            allowed_sources=("base", "tracking", "pbp"),
            curve_strength=4.8,
            max_rating=19.5,
        ),
        AttributeSpec(
            name="screen_setting",
            components={
                "rebound_load": 0.18,
                "oreb_rate": 0.18,
                "minutes_load": 0.08,
                "start_rate": 0.10,
            },
            base=0.12,
            position_baselines={"PG": -0.08, "SG": -0.04, "SF": 0.02, "PF": 0.12, "C": 0.20},
            allowed_sources=("base", "tracking"),
        ),
        AttributeSpec(
            name="stamina",
            components={
                "minutes_load": 0.44,
                "games_sample": 0.16,
                "start_rate": 0.14,
                "recent_form": 0.08,
            },
            base=0.10,
            position_baselines={"PG": 0.02, "SG": 0.02, "SF": 0.01, "PF": 0.0, "C": -0.01},
            allowed_sources=("base",),
        ),
    )

    raw_scores_by_spec: dict[str, dict[RowKey, float]] = {}
    for spec in specs:
        spec_raw_scores: dict[RowKey, float] = {}
        for row in rows:
            row_key = (str(row.get("team_code") or ""), str(row.get("name_key") or ""))
            position = str(row.get("position") or "")
            raw = spec.base
            for feature_name, weight in spec.components.items():
                resolved_feature_name = _resolve_feature_name(feature_name, spec.allowed_sources)
                if resolved_feature_name is None:
                    continue
                raw += percentiles.get(resolved_feature_name, {}).get(row_key, 0.5) * weight
            if spec.position_baselines:
                raw += spec.position_baselines.get(position, 0.0)
            spec_raw_scores[row_key] = raw
        raw_scores_by_spec[spec.name] = spec_raw_scores

    results: dict[RowKey, dict[str, float]] = {
        (str(row.get("team_code") or ""), str(row.get("name_key") or "")): {}
        for row in rows
    }
    for spec in specs:
        raw_percentiles = _build_scalar_percentiles(raw_scores_by_spec[spec.name])
        for row_key, percentile in raw_percentiles.items():
            results.setdefault(row_key, {})[spec.name] = _scale_unit_to_rating(
                percentile,
                curve_strength=spec.curve_strength,
                max_rating=spec.max_rating,
            )
    return results


def _build_feature_row(row: RowData) -> dict[str, float]:
    games_sample = _safe_float(row.get("games_sample"))
    minutes = _safe_float(row.get("minutes"))
    points = _safe_float(row.get("points"))
    rebounds = _safe_float(row.get("rebounds"))
    assists = _safe_float(row.get("assists"))
    recent_fpts_avg = _safe_float(row.get("recent_fpts_avg"))
    recent_fpts_weighted = _safe_float(row.get("recent_fpts_weighted"))
    recent_form_delta = _safe_float(row.get("recent_form_delta"))
    extra_stats = row.get("extra_stats")
    if not isinstance(extra_stats, dict):
        extra_stats = {}
    starts = _safe_float(extra_stats.get("starts"))
    turnovers = _safe_float(extra_stats.get("turnovers"))
    fga = _safe_float(extra_stats.get("fga"))
    three_pa = _safe_float(extra_stats.get("three_pa"))
    fta = _safe_float(extra_stats.get("fta"))
    oreb = _safe_float(extra_stats.get("oreb"))
    dreb = _safe_float(extra_stats.get("dreb"))

    mpg = minutes / max(games_sample, 1.0)
    apg = assists / max(games_sample, 1.0)
    ppg = points / max(games_sample, 1.0)
    topg = turnovers / max(games_sample, 1.0)
    fgapg = fga / max(games_sample, 1.0)
    three_pag = three_pa / max(games_sample, 1.0)
    ftapg = fta / max(games_sample, 1.0)
    orpg = oreb / max(games_sample, 1.0)
    drpg = dreb / max(games_sample, 1.0)

    usage_load = (fgapg * 0.55) + (ftapg * 0.35) + (apg * 0.85)
    creation_load = (apg * 0.72) + (topg * 0.18)
    shot_creation_load = (ppg * 0.42) + (three_pag * 0.90) + (ftapg * 0.45)
    turnover_control = apg / max(topg, 0.5)
    recent_form = (recent_fpts_weighted / max(recent_fpts_avg, 1.0)) + (recent_form_delta / max(recent_fpts_avg, 12.0))
    ball_handle_tracking = _extract_source_stats(row, "ball_handle", "tracking", "tracking_stats")
    passing_tracking = _extract_source_stats(row, "pass_vision", "tracking", "tracking_stats")
    passing_accuracy_tracking = _extract_source_stats(row, "pass_accuracy", "tracking", "tracking_stats")
    pullup_tracking = _extract_source_stats(row, "pullup_shooting", "tracking", "tracking_stats")
    catch_tracking = _extract_source_stats(row, "catch_shoot", "tracking", "tracking_stats")
    ball_handle_pbp = _extract_source_stats(row, "ball_handle", "pbp", "pbp_stats")
    passing_pbp = _extract_source_stats(row, "pass_accuracy", "pbp", "pbp_stats")

    features = {
        "base:assist_load": apg,
        "base:turnover_control_inv": turnover_control,
        "base:usage_load": usage_load,
        "base:creation_load": creation_load,
        "base:minutes_load": mpg,
        "base:shot_creation_load": shot_creation_load,
        "base:scoring_load": ppg,
        "base:three_volume": three_pag,
        "base:free_throw_pressure": ftapg,
        "base:oreb_rate": orpg,
        "base:rebound_load": drpg + orpg + (rebounds / max(games_sample, 1.0) * 0.25),
        "base:games_sample": games_sample,
        "base:start_rate": starts / max(games_sample, 1.0),
        "base:recent_form": recent_form,
    }
    dribbles_per_touch = _safe_float(ball_handle_tracking.get("dribbles_per_touch"))
    time_of_poss_per_touch = _safe_float(
        ball_handle_tracking.get("time_of_poss_per_touch", ball_handle_tracking.get("avg_sec_per_touch"))
    )
    drives_per_touch = _safe_float(
        ball_handle_tracking.get("drives_per_touch", ball_handle_tracking.get("drives_per_game"))
    )
    drive_tov_rate_inv = ball_handle_tracking.get("drive_tov_rate_inv")
    if drive_tov_rate_inv is None and (
        "drive_tov_rate" in ball_handle_tracking or "drive_tov_frac" in ball_handle_tracking
    ):
        drive_tov_rate_inv = 1.0 - _safe_float(
            ball_handle_tracking.get("drive_tov_rate", ball_handle_tracking.get("drive_tov_frac"))
        )

    if dribbles_per_touch:
        features["tracking:dribbles_per_touch"] = dribbles_per_touch
    if time_of_poss_per_touch:
        features["tracking:time_of_poss_per_touch"] = time_of_poss_per_touch
    if drives_per_touch:
        features["tracking:drives_per_touch"] = drives_per_touch
    if drive_tov_rate_inv is not None:
        features["tracking:drive_tov_rate_inv"] = _safe_float(drive_tov_rate_inv)

    bh_tov_rate_inv = ball_handle_pbp.get("bh_tov_rate_inv")
    if bh_tov_rate_inv is None and "bh_tov_frac" in ball_handle_pbp:
        bh_tov_rate_inv = 1.0 - _safe_float(ball_handle_pbp.get("bh_tov_frac"))
    if bh_tov_rate_inv is not None:
        features["pbp:bh_tov_rate_inv"] = _safe_float(bh_tov_rate_inv)

    pass_tov_rate_inv = passing_pbp.get("pass_tov_rate_inv")
    if pass_tov_rate_inv is None and "pass_tov_frac" in passing_pbp:
        pass_tov_rate_inv = 1.0 - _safe_float(passing_pbp.get("pass_tov_frac"))
    if pass_tov_rate_inv is not None:
        features["pbp:pass_tov_rate_inv"] = _safe_float(pass_tov_rate_inv)

    combined_passing_tracking = {}
    combined_passing_tracking.update(
        {key: value for key, value in passing_tracking.items() if key not in combined_passing_tracking}
    )
    combined_passing_tracking.update(
        {key: value for key, value in passing_accuracy_tracking.items() if key not in combined_passing_tracking}
    )
    for key in (
        "passes_made",
        "passes_received",
        "secondary_assists",
        "potential_assists",
        "ast_points_created",
        "ast_to_pass_pct",
        "usage_pct",
        "ast_to_ratio",
    ):
        if key in combined_passing_tracking:
            features[f"tracking:{key}"] = _safe_float(combined_passing_tracking.get(key))

    for key in (
        "pullup_fga",
        "pullup_fg_pct",
        "pullup_fg3a",
        "pullup_fg3_pct",
        "pullup_efg_pct",
    ):
        if key in pullup_tracking:
            features[f"tracking:{key}"] = _safe_float(pullup_tracking.get(key))

    for key in (
        "catch_shoot_fga",
        "catch_shoot_fg_pct",
        "catch_shoot_fg3a",
        "catch_shoot_fg3_pct",
        "catch_shoot_efg_pct",
    ):
        if key in catch_tracking:
            features[f"tracking:{key}"] = _safe_float(catch_tracking.get(key))
    return features


def _build_percentiles(feature_rows: list[dict[str, float]]) -> dict[str, dict[RowKey, float]]:
    if not feature_rows:
        return {}
    feature_names = sorted(
        name
        for name in set().union(*(feature_row.keys() for feature_row in feature_rows))
        if name != "_row_key"
    )
    percentiles: dict[str, dict[RowKey, float]] = {name: {} for name in feature_names}
    keyed_rows = list(enumerate(feature_rows))
    for feature_name in feature_names:
        sorted_rows = sorted(keyed_rows, key=lambda item: float(item[1].get(feature_name, 0.0)))
        if len(sorted_rows) == 1:
            row_key = sorted_rows[0][1]["_row_key"]  # type: ignore[index]
            percentiles[feature_name][row_key] = 1.0
            continue
        total = len(sorted_rows) - 1
        for rank, (idx, feature_map) in enumerate(sorted_rows):
            row_key = feature_map["_row_key"]  # type: ignore[index]
            percentiles[feature_name][row_key] = rank / total
    return percentiles


def _build_scalar_percentiles(values_by_row: dict[RowKey, float]) -> dict[RowKey, float]:
    if not values_by_row:
        return {}
    sorted_rows = sorted(values_by_row.items(), key=lambda item: item[1])
    if len(sorted_rows) == 1:
        return {sorted_rows[0][0]: 1.0}
    total = len(sorted_rows) - 1
    return {
        row_key: rank / total
        for rank, (row_key, _) in enumerate(sorted_rows)
    }


def _scale_unit_to_rating(raw: float, *, curve_strength: float = 6.0, max_rating: float = 20.0) -> float:
    clipped = max(0.0, min(1.0, raw))
    curved = 1.0 / (1.0 + math.exp(-curve_strength * (clipped - 0.5)))
    return round(1.0 + (curved * max(0.0, max_rating - 1.0)), 2)


def _safe_float(value: object) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def _resolve_feature_name(feature_name: str, allowed_sources: Iterable[str]) -> str | None:
    if ":" not in feature_name:
        # Bare feature names currently resolve to the first allowed source, which keeps
        # existing base-only specs simple. If we later want "prefer tracking, fall back
        # to base" semantics for the same logical feature, this resolver will need to
        # look at available feature keys instead of choosing the first source eagerly.
        for source_name in allowed_sources:
            candidate = f"{source_name}:{feature_name}"
            return candidate
        return feature_name
    source_name, _ = feature_name.split(":", 1)
    return feature_name if source_name in set(allowed_sources) else None


def _extract_source_stats(
    row: RowData,
    attribute_name: str,
    source_name: str,
    direct_key: str,
) -> dict[str, object]:
    direct_stats = row.get(direct_key)
    if isinstance(direct_stats, dict):
        return direct_stats
    supplemental_sources = row.get("supplemental_sources")
    if not isinstance(supplemental_sources, dict):
        return {}
    attribute_sources = supplemental_sources.get(attribute_name)
    if not isinstance(attribute_sources, dict):
        return {}
    source_stats = attribute_sources.get(source_name)
    return source_stats if isinstance(source_stats, dict) else {}
