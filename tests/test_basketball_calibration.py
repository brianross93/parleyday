from basketball_calibration import (
    assert_calibration_targets,
    evaluate_calibration_targets,
    measure_fta_by_archetype,
    measure_possession_mix,
    measure_usage_concentration,
    run_calibration_suite,
)


def test_run_calibration_suite_returns_expected_sections() -> None:
    suite = run_calibration_suite(
        possession_samples=80,
        usage_samples=80,
        archetype_games=8,
        game_count=8,
        rng_seed=5,
    )
    assert set(suite) == {"possession_mix", "usage_concentration", "fta_by_archetype", "game_variance"}
    assert "points_per_possession" in suite["possession_mix"]
    assert "top_scorer_possession_share" in suite["usage_concentration"]
    assert "historic_drawer" in suite["fta_by_archetype"]
    assert "game_total_std" in suite["game_variance"]


def test_fta_by_archetype_increases_with_foul_drawing() -> None:
    results = measure_fta_by_archetype(games_per_archetype=80, rng_seed=13)
    ordered = [
        results["spot_up_role"],
        results["average_starter"],
        results["aggressive_driver"],
        results["elite_drawer"],
        results["historic_drawer"],
    ]
    assert ordered == sorted(ordered)


def test_usage_concentration_prefers_primary_option() -> None:
    usage = measure_usage_concentration(samples=150, rng_seed=17)
    assert usage["top_scorer_possession_share"] > 0.20
    assert usage["top_2_usage_share"] >= usage["top_scorer_possession_share"]
    assert usage["top_3_usage_share"] >= usage["top_2_usage_share"]


def test_possession_mix_outputs_bounded_rates() -> None:
    metrics = measure_possession_mix(samples=120, rng_seed=19)
    assert 0.0 <= metrics["turnover_rate"] <= 1.0
    assert 0.0 <= metrics["shooting_foul_rate"] <= 1.0
    assert 0.0 <= metrics["oreb_rate_on_misses"] <= 1.0
    assert 0.0 <= metrics["three_pa_share"] <= 1.0


def test_evaluate_calibration_targets_reports_out_of_band_metrics() -> None:
    report = {
        "possession_mix": {"points_per_possession": 0.9},
        "usage_concentration": {},
        "fta_by_archetype": {},
        "game_variance": {},
    }
    violations = evaluate_calibration_targets(report)
    assert violations
    assert any("possession_mix.points_per_possession" in violation for violation in violations)


def test_assert_calibration_targets_raises_on_failures() -> None:
    report = {
        "possession_mix": {"points_per_possession": 0.9},
        "usage_concentration": {},
        "fta_by_archetype": {},
        "game_variance": {},
    }
    try:
        assert_calibration_targets(report)
    except AssertionError as exc:
        assert "Calibration target failures" in str(exc)
    else:
        raise AssertionError("Expected calibration assertions to fail")
