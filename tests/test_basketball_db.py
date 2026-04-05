from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from basketball_db import (
    ensure_basketball_schema,
    generate_nba_player_profiles,
    generate_nba_player_profiles_for_season,
    import_draftkings_slate_to_db,
    list_nba_player_profiles,
    load_draftkings_slate_from_db,
    load_nba_sim_profiles_for_slate,
    seed_nba_player_season_advanced_history_from_nba_api,
    seed_nba_player_stats_from_bref,
    seed_nba_player_season_history_from_nba_api,
    seed_nba_player_stats_from_stathead_dump,
    upsert_player_attribute_source_features,
    upsert_nba_player_stats,
)
from basketball_attribute_pipeline import build_attribute_scores
from basketball_sim_builder import build_nba_sim_inputs_from_db_slate, build_nba_sim_inputs_from_dk_csv


class BasketballDbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.csv_path = root / "DKSalaries.csv"
        self.db_path = root / "parleyday.sqlite"
        self.csv_path.write_text(
            "\n".join(
                [
                    "Position,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                    "PG,AAA One,1,PG/G/UTIL,8400,AAA@BBB 04/02/2026 7:00PM ET,AAA,42.5",
                    "SG,AAA Two,2,SG/G/UTIL,7600,AAA@BBB 04/02/2026 7:00PM ET,AAA,35.0",
                    "SF,AAA Three,3,SF/F/UTIL,7100,AAA@BBB 04/02/2026 7:00PM ET,AAA,33.2",
                    "PF,AAA Four,4,PF/F/UTIL,6900,AAA@BBB 04/02/2026 7:00PM ET,AAA,31.5",
                    "C,AAA Five,5,C/UTIL,6700,AAA@BBB 04/02/2026 7:00PM ET,AAA,30.1",
                    "PG,BBB One,6,PG/G/UTIL,8300,AAA@BBB 04/02/2026 7:00PM ET,BBB,41.8",
                    "SG,BBB Two,7,SG/G/UTIL,7500,AAA@BBB 04/02/2026 7:00PM ET,BBB,34.0",
                    "SF,BBB Three,8,SF/F/UTIL,7000,AAA@BBB 04/02/2026 7:00PM ET,BBB,32.0",
                    "PF,BBB Four,9,PF/F/UTIL,6800,AAA@BBB 04/02/2026 7:00PM ET,BBB,30.2",
                    "C,BBB Five,10,C/UTIL,6600,AAA@BBB 04/02/2026 7:00PM ET,BBB,29.9",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _insert_season_stat_row(
        self,
        *,
        db_path: Path,
        season: str,
        player_id: str,
        name: str,
        team_code: str,
        age: float,
        position: str,
        assists: float,
        points: float = 1500.0,
        minutes: float = 2400.0,
        games_sample: float = 72.0,
    ) -> None:
        ensure_basketball_schema(str(db_path))
        name_key = name.lower().replace(" ", "-")
        extra_stats_json = json.dumps(
            {
                "starts": 72.0,
                "turnovers": 180.0 if assists >= 500.0 else 120.0,
                "fouls": 110.0,
                "fga": 1100.0,
                "three_pa": 320.0,
                "fta": 260.0,
                "oreb": 30.0,
                "dreb": 220.0,
            }
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO basketball_player_season_stats (
                    season, player_key, player_id, name, name_key, team_code, team_id,
                    age, position, games_sample, minutes, points, rebounds, assists,
                    recent_fpts_avg, recent_fpts_weighted, recent_form_delta,
                    extra_stats_json, source, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season,
                    f"{team_code}:{name_key}",
                    player_id,
                    name,
                    name_key,
                    team_code,
                    "100",
                    age,
                    position,
                    games_sample,
                    minutes,
                    points,
                    320.0,
                    assists,
                    38.0,
                    39.0,
                    1.0,
                    extra_stats_json,
                    "test_history",
                    "2026-04-04T00:00:00+00:00",
                ),
            )

    def test_import_round_trips_slate_from_sqlite(self) -> None:
        imported = import_draftkings_slate_to_db("2026-04-02", str(self.csv_path), db_path=str(self.db_path))
        loaded = load_draftkings_slate_from_db("2026-04-02", db_path=str(self.db_path), source_path=str(self.csv_path))

        self.assertEqual(imported.sport, "nba")
        self.assertEqual(len(loaded.players), 10)
        self.assertEqual(loaded.players[0].game, "AAA@BBB")
        self.assertEqual({player.team for player in loaded.players}, {"AAA", "BBB"})

    def test_builder_reads_imported_slate_from_sqlite(self) -> None:
        imported = import_draftkings_slate_to_db("2026-04-02", str(self.csv_path), db_path=str(self.db_path))
        stat_rows = []
        for idx, player in enumerate(imported.players):
            stat_rows.append(
                {
                    "player_id": f"espn-{player.player_id}",
                    "name": player.name,
                    "team_code": player.team,
                    "team_id": "100" if player.team == "AAA" else "200",
                    "position": player.positions[0],
                    "status": "active",
                    "games_sample": 40.0,
                    "starts": 32.0,
                    "minutes": 30.0,
                    "points": 18.0 + idx,
                    "rebounds": 5.0 if player.positions[0] != "C" else 9.0,
                    "assists": 6.0 if player.positions[0] == "PG" else 3.0,
                    "turnovers": 2.1,
                    "fouls": 2.4,
                    "fga": 14.0,
                    "three_pa": 5.0,
                    "fta": 4.5,
                    "oreb": 1.1,
                    "dreb": 4.8,
                    "recent_fpts_avg": 32.0 + idx,
                    "recent_fpts_weighted": 33.5 + idx,
                    "recent_form_delta": 1.5,
                    "injuries_json": json.dumps([]),
                }
            )
        upsert_nba_player_stats("2026-04-02", stat_rows, db_path=str(self.db_path), source="test")
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))

        loaded = load_draftkings_slate_from_db("2026-04-02", db_path=str(self.db_path), source_path=str(self.csv_path))
        profiles = load_nba_sim_profiles_for_slate("2026-04-02", loaded, db_path=str(self.db_path))
        with patch(
            "basketball_sim_builder.load_nba_matchup_features",
            lambda date_str, teams: {
                "AAA": {"recent_pace": 100.0, "opp_orb_rate_allowed": 0.29},
                "BBB": {"recent_pace": 98.0, "opp_orb_rate_allowed": 0.27},
            },
        ):
            results = build_nba_sim_inputs_from_db_slate("2026-04-02", loaded, db_path=str(self.db_path))

        self.assertEqual(len(profiles), 10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].game_id, "AAA@BBB")
        self.assertEqual(len(results[0].players), 10)
        self.assertGreater(results[0].players[0].traits.separation, 1.0)
        listed = list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        self.assertEqual(len(listed), 10)
        self.assertAlmostEqual(listed[0]["fga"], 14.0)
        self.assertIn("offensive_role", listed[0])
        self.assertIn("traits", listed[0])

    def test_csv_entrypoint_uses_db_profiles(self) -> None:
        imported = import_draftkings_slate_to_db("2026-04-02", str(self.csv_path), db_path=str(self.db_path))
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": f"espn-{player.player_id}",
                    "name": player.name,
                    "team_code": player.team,
                    "team_id": "100" if player.team == "AAA" else "200",
                    "position": player.positions[0],
                    "status": "active",
                    "games_sample": 50.0,
                    "starts": 45.0,
                    "minutes": 31.0,
                    "points": 20.0,
                    "rebounds": 6.0,
                    "assists": 4.0,
                    "turnovers": 2.0,
                    "fouls": 2.5,
                    "fga": 15.0,
                    "three_pa": 5.5,
                    "fta": 4.2,
                    "oreb": 1.0,
                    "dreb": 5.0,
                    "recent_fpts_avg": 34.0,
                    "recent_fpts_weighted": 35.0,
                    "recent_form_delta": 1.0,
                    "injuries_json": json.dumps([]),
                }
                for player in imported.players
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        with patch(
            "basketball_sim_builder.load_nba_matchup_features",
            lambda date_str, teams: {
                "AAA": {"recent_pace": 100.0, "opp_orb_rate_allowed": 0.29},
                "BBB": {"recent_pace": 98.0, "opp_orb_rate_allowed": 0.27},
            },
        ):
            results = build_nba_sim_inputs_from_dk_csv("2026-04-02", str(self.csv_path), db_path=str(self.db_path))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].home_team_code, "BBB")

    def test_players_route_renders_db_backed_profiles(self) -> None:
        imported = import_draftkings_slate_to_db("2026-04-02", str(self.csv_path), db_path=str(self.db_path))
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": f"espn-{player.player_id}",
                    "name": player.name,
                    "team_code": player.team,
                    "team_id": "100" if player.team == "AAA" else "200",
                    "position": player.positions[0],
                    "status": "active",
                    "games_sample": 50.0,
                    "starts": 40.0,
                    "minutes": 30.5,
                    "points": 19.5,
                    "rebounds": 6.5,
                    "assists": 4.5,
                    "turnovers": 2.0,
                    "fouls": 2.2,
                    "fga": 15.3,
                    "three_pa": 5.7,
                    "fta": 4.8,
                    "oreb": 1.2,
                    "dreb": 5.3,
                    "recent_fpts_avg": 34.0,
                    "recent_fpts_weighted": 35.2,
                    "recent_form_delta": 1.2,
                    "injuries_json": json.dumps([]),
                }
                for player in imported.players
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        os.environ["PARLEYDAY_DB_PATH"] = str(self.db_path)
        from dashboard_app import app

        response = app.test_client().get("/basketball-players?date=2026-04-02")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Basketball Player View", html)
        self.assertIn("AAA One", html)
        self.assertIn("Technical", html)
        self.assertIn("Physical", html)
        self.assertIn("Defensive", html)

    def test_elite_shooting_profiles_can_reach_real_top_end(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "elite-1",
                    "name": "Elite Guard",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 70.0,
                    "starts": 70.0,
                    "minutes": 35.5,
                    "points": 31.0,
                    "rebounds": 5.2,
                    "assists": 6.4,
                    "turnovers": 2.7,
                    "fouls": 2.1,
                    "fga": 22.5,
                    "three_pa": 10.6,
                    "fta": 8.7,
                    "oreb": 0.6,
                    "dreb": 4.6,
                    "recent_fpts_avg": 50.0,
                    "recent_fpts_weighted": 52.0,
                    "recent_form_delta": 2.0,
                    "injuries_json": json.dumps([]),
                }
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-02", db_path=str(self.db_path), search="Elite Guard")
        self.assertEqual(len(listed), 1)
        self.assertGreaterEqual(float(listed[0]["traits"]["pullup_shooting"]), 15.0)
        self.assertGreaterEqual(float(listed[0]["traits"]["catch_shoot"]), 15.0)

    def test_elite_finishing_profiles_can_reach_real_top_end(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "finisher-1",
                    "name": "Elite Finisher",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "SF",
                    "status": "active",
                    "games_sample": 72.0,
                    "starts": 72.0,
                    "minutes": 34.0,
                    "points": 29.5,
                    "rebounds": 7.2,
                    "assists": 4.1,
                    "turnovers": 2.8,
                    "fouls": 2.0,
                    "fga": 20.0,
                    "three_pa": 2.5,
                    "fta": 9.2,
                    "oreb": 2.4,
                    "dreb": 4.8,
                    "recent_fpts_avg": 47.0,
                    "recent_fpts_weighted": 49.0,
                    "recent_form_delta": 1.8,
                    "injuries_json": json.dumps([]),
                }
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-02", db_path=str(self.db_path), search="Elite Finisher")
        self.assertEqual(len(listed), 1)
        self.assertGreaterEqual(float(listed[0]["traits"]["finishing"]), 14.0)

    def test_screen_setting_stays_on_scale_and_rewards_frontcourt_roles(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "center-1",
                    "name": "Screen Big",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "C",
                    "status": "active",
                    "games_sample": 70.0,
                    "starts": 68.0,
                    "minutes": 2100.0,
                    "points": 980.0,
                    "rebounds": 760.0,
                    "assists": 210.0,
                    "turnovers": 110.0,
                    "fouls": 190.0,
                    "fga": 720.0,
                    "three_pa": 40.0,
                    "fta": 300.0,
                    "oreb": 210.0,
                    "dreb": 550.0,
                    "recent_fpts_avg": 33.0,
                    "recent_fpts_weighted": 34.0,
                    "recent_form_delta": 1.0,
                    "injuries_json": json.dumps([]),
                },
                {
                    "player_id": "guard-1",
                    "name": "Shooter Wing",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "SG",
                    "status": "active",
                    "games_sample": 70.0,
                    "starts": 66.0,
                    "minutes": 2200.0,
                    "points": 1320.0,
                    "rebounds": 280.0,
                    "assists": 240.0,
                    "turnovers": 120.0,
                    "fouls": 140.0,
                    "fga": 980.0,
                    "three_pa": 520.0,
                    "fta": 190.0,
                    "oreb": 35.0,
                    "dreb": 245.0,
                    "recent_fpts_avg": 31.0,
                    "recent_fpts_weighted": 31.5,
                    "recent_form_delta": 0.5,
                    "injuries_json": json.dumps([]),
                },
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = {row["name"]: row for row in list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))}
        self.assertLessEqual(float(listed["Screen Big"]["traits"]["screen_setting"]), 20.0)
        self.assertLessEqual(float(listed["Shooter Wing"]["traits"]["screen_setting"]), 20.0)
        self.assertGreater(
            float(listed["Screen Big"]["traits"]["screen_setting"]),
            float(listed["Shooter Wing"]["traits"]["screen_setting"]),
        )

    def test_stamina_separates_workhorse_from_bench_player(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "starter-1",
                    "name": "Ironman Starter",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 76.0,
                    "starts": 76.0,
                    "minutes": 2660.0,
                    "points": 1500.0,
                    "rebounds": 340.0,
                    "assists": 520.0,
                    "turnovers": 170.0,
                    "fouls": 145.0,
                    "fga": 1180.0,
                    "three_pa": 460.0,
                    "fta": 320.0,
                    "oreb": 45.0,
                    "dreb": 295.0,
                    "recent_fpts_avg": 38.0,
                    "recent_fpts_weighted": 39.5,
                    "recent_form_delta": 1.2,
                    "injuries_json": json.dumps([]),
                },
                {
                    "player_id": "bench-1",
                    "name": "Bench Guard",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 50.0,
                    "starts": 0.0,
                    "minutes": 620.0,
                    "points": 240.0,
                    "rebounds": 90.0,
                    "assists": 110.0,
                    "turnovers": 42.0,
                    "fouls": 60.0,
                    "fga": 230.0,
                    "three_pa": 95.0,
                    "fta": 44.0,
                    "oreb": 12.0,
                    "dreb": 78.0,
                    "recent_fpts_avg": 12.0,
                    "recent_fpts_weighted": 12.4,
                    "recent_form_delta": 0.3,
                    "injuries_json": json.dumps([]),
                },
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = {row["name"]: row for row in list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))}
        starter_stamina = float(listed["Ironman Starter"]["traits"]["stamina"])
        bench_stamina = float(listed["Bench Guard"]["traits"]["stamina"])
        self.assertGreater(starter_stamina, bench_stamina)
        self.assertLess(starter_stamina, 20.0)

    def test_ball_handle_uses_tracking_and_pbp_supplemental_sources(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "guard-1",
                    "name": "On Ball Guard",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 70.0,
                    "starts": 70.0,
                    "minutes": 2380.0,
                    "points": 1320.0,
                    "rebounds": 280.0,
                    "assists": 510.0,
                    "turnovers": 170.0,
                    "fouls": 120.0,
                    "fga": 940.0,
                    "three_pa": 310.0,
                    "fta": 250.0,
                    "oreb": 28.0,
                    "dreb": 252.0,
                    "recent_fpts_avg": 38.0,
                    "recent_fpts_weighted": 39.0,
                    "recent_form_delta": 1.0,
                    "injuries_json": json.dumps([]),
                },
                {
                    "player_id": "big-1",
                    "name": "Safe Hands Big",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "C",
                    "status": "active",
                    "games_sample": 70.0,
                    "starts": 70.0,
                    "minutes": 2100.0,
                    "points": 980.0,
                    "rebounds": 760.0,
                    "assists": 150.0,
                    "turnovers": 92.0,
                    "fouls": 170.0,
                    "fga": 700.0,
                    "three_pa": 18.0,
                    "fta": 260.0,
                    "oreb": 210.0,
                    "dreb": 550.0,
                    "recent_fpts_avg": 31.0,
                    "recent_fpts_weighted": 31.5,
                    "recent_form_delta": 0.4,
                    "injuries_json": json.dumps([]),
                },
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        base_profiles = {
            row["name"]: row for row in list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        }
        base_handle = float(base_profiles["On Ball Guard"]["traits"]["ball_security"])

        upsert_player_attribute_source_features(
            "2026-04-02",
            [
                {
                    "name": "On Ball Guard",
                    "team_code": "AAA",
                    "attribute_name": "ball_handle",
                    "source": "tracking",
                    "features": {
                        "dribbles_per_touch": 6.8,
                        "time_of_poss_per_touch": 5.6,
                        "drives_per_touch": 0.18,
                        "drive_tov_rate_inv": 0.84,
                    },
                },
                {
                    "name": "On Ball Guard",
                    "team_code": "AAA",
                    "attribute_name": "ball_handle",
                    "source": "pbp",
                    "features": {
                        "bh_tov_rate_inv": 0.81,
                    },
                },
                {
                    "name": "Safe Hands Big",
                    "team_code": "AAA",
                    "attribute_name": "ball_handle",
                    "source": "tracking",
                    "features": {
                        "dribbles_per_touch": 1.2,
                        "time_of_poss_per_touch": 1.4,
                        "drives_per_touch": 0.02,
                        "drive_tov_rate_inv": 0.63,
                    },
                },
                {
                    "name": "Safe Hands Big",
                    "team_code": "AAA",
                    "attribute_name": "ball_handle",
                    "source": "pbp",
                    "features": {
                        "bh_tov_rate_inv": 0.46,
                    },
                },
            ],
            db_path=str(self.db_path),
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        enriched_profiles = {
            row["name"]: row for row in list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        }
        enriched_handle = float(enriched_profiles["On Ball Guard"]["traits"]["ball_security"])
        big_handle = float(enriched_profiles["Safe Hands Big"]["traits"]["ball_security"])

        self.assertGreaterEqual(enriched_handle, base_handle)
        self.assertGreater(enriched_handle, big_handle)
        self.assertGreaterEqual(enriched_handle, 15.0)

    def test_attribute_pipeline_accepts_direct_tracking_and_pbp_row_data(self) -> None:
        scores = build_attribute_scores(
            [
                {
                    "team_code": "AAA",
                    "name_key": "onballguard",
                    "position": "PG",
                    "games_sample": 70.0,
                    "minutes": 2380.0,
                    "points": 1320.0,
                    "rebounds": 280.0,
                    "assists": 510.0,
                    "recent_fpts_avg": 38.0,
                    "recent_fpts_weighted": 39.0,
                    "recent_form_delta": 1.0,
                    "extra_stats": {
                        "starts": 70.0,
                        "turnovers": 170.0,
                        "fga": 940.0,
                        "three_pa": 310.0,
                        "fta": 250.0,
                        "oreb": 28.0,
                        "dreb": 252.0,
                    },
                    "tracking_stats": {
                        "dribbles_per_touch": 6.8,
                        "avg_sec_per_touch": 5.6,
                        "drives_per_game": 0.18,
                        "drive_tov_rate": 0.16,
                        "potential_assists": 14.0,
                        "secondary_assists": 1.9,
                        "ast_points_created": 24.0,
                        "passes_made": 58.0,
                        "ast_to_pass_pct": 0.14,
                        "usage_pct": 0.31,
                        "ast_to_ratio": 2.6,
                        "pullup_fga": 6.2,
                        "pullup_fg_pct": 0.46,
                        "pullup_fg3a": 2.6,
                        "pullup_fg3_pct": 0.39,
                        "pullup_efg_pct": 0.56,
                        "catch_shoot_fga": 4.4,
                        "catch_shoot_fg_pct": 0.43,
                        "catch_shoot_fg3a": 3.7,
                        "catch_shoot_fg3_pct": 0.41,
                        "catch_shoot_efg_pct": 0.60,
                    },
                    "pbp_stats": {
                        "bh_tov_frac": 0.19,
                        "pass_tov_frac": 0.12,
                    },
                },
                {
                    "team_code": "AAA",
                    "name_key": "safehandsbig",
                    "position": "C",
                    "games_sample": 70.0,
                    "minutes": 2100.0,
                    "points": 980.0,
                    "rebounds": 760.0,
                    "assists": 150.0,
                    "recent_fpts_avg": 31.0,
                    "recent_fpts_weighted": 31.5,
                    "recent_form_delta": 0.4,
                    "extra_stats": {
                        "starts": 70.0,
                        "turnovers": 92.0,
                        "fga": 700.0,
                        "three_pa": 18.0,
                        "fta": 260.0,
                        "oreb": 210.0,
                        "dreb": 550.0,
                    },
                    "tracking_stats": {
                        "dribbles_per_touch": 1.2,
                        "avg_sec_per_touch": 1.4,
                        "drives_per_game": 0.02,
                        "drive_tov_rate": 0.37,
                        "potential_assists": 2.1,
                        "secondary_assists": 0.2,
                        "ast_points_created": 3.8,
                        "passes_made": 18.0,
                        "ast_to_pass_pct": 0.04,
                        "usage_pct": 0.18,
                        "ast_to_ratio": 1.1,
                        "pullup_fga": 0.1,
                        "pullup_fg_pct": 0.10,
                        "pullup_fg3a": 0.0,
                        "pullup_fg3_pct": 0.0,
                        "pullup_efg_pct": 0.10,
                        "catch_shoot_fga": 0.5,
                        "catch_shoot_fg_pct": 0.22,
                        "catch_shoot_fg3a": 0.1,
                        "catch_shoot_fg3_pct": 0.10,
                        "catch_shoot_efg_pct": 0.24,
                    },
                    "pbp_stats": {
                        "bh_tov_frac": 0.54,
                        "pass_tov_frac": 0.31,
                    },
                },
            ]
        )
        self.assertGreater(scores[("AAA", "onballguard")]["ball_handle"], scores[("AAA", "safehandsbig")]["ball_handle"])
        self.assertGreater(scores[("AAA", "onballguard")]["pass_vision"], scores[("AAA", "safehandsbig")]["pass_vision"])
        self.assertGreater(scores[("AAA", "onballguard")]["pass_accuracy"], scores[("AAA", "safehandsbig")]["pass_accuracy"])
        self.assertGreater(scores[("AAA", "onballguard")]["pullup_shooting"], scores[("AAA", "safehandsbig")]["pullup_shooting"])
        self.assertGreater(scores[("AAA", "onballguard")]["catch_shoot"], scores[("AAA", "safehandsbig")]["catch_shoot"])

    def test_elite_creator_can_reach_top_end_passing_traits(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "creator-1",
                    "name": "Elite Creator",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 74.0,
                    "starts": 74.0,
                    "minutes": 2550.0,
                    "points": 1700.0,
                    "rebounds": 320.0,
                    "assists": 760.0,
                    "turnovers": 180.0,
                    "fouls": 120.0,
                    "fga": 1180.0,
                    "three_pa": 390.0,
                    "fta": 420.0,
                    "oreb": 35.0,
                    "dreb": 285.0,
                    "recent_fpts_avg": 46.0,
                    "recent_fpts_weighted": 47.5,
                    "recent_form_delta": 1.3,
                    "injuries_json": json.dumps([]),
                }
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-02", db_path=str(self.db_path), search="Elite Creator")
        self.assertEqual(len(listed), 1)
        self.assertGreaterEqual(float(listed[0]["traits"]["pass_vision"]), 14.0)
        self.assertGreaterEqual(float(listed[0]["traits"]["pass_accuracy"]), 13.0)

    def test_interior_defense_separates_anchor_big_from_forward(self) -> None:
        upsert_nba_player_stats(
            "2026-04-02",
            [
                {
                    "player_id": "anchor-1",
                    "name": "Anchor Big",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "C",
                    "status": "active",
                    "games_sample": 72.0,
                    "starts": 72.0,
                    "minutes": 2220.0,
                    "points": 920.0,
                    "rebounds": 760.0,
                    "assists": 140.0,
                    "turnovers": 95.0,
                    "fouls": 180.0,
                    "fga": 700.0,
                    "three_pa": 20.0,
                    "fta": 280.0,
                    "oreb": 210.0,
                    "dreb": 550.0,
                    "recent_fpts_avg": 33.0,
                    "recent_fpts_weighted": 34.5,
                    "recent_form_delta": 1.1,
                    "injuries_json": json.dumps([]),
                },
                {
                    "player_id": "forward-1",
                    "name": "Combo Forward",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PF",
                    "status": "active",
                    "games_sample": 72.0,
                    "starts": 72.0,
                    "minutes": 2220.0,
                    "points": 1100.0,
                    "rebounds": 470.0,
                    "assists": 210.0,
                    "turnovers": 105.0,
                    "fouls": 150.0,
                    "fga": 820.0,
                    "three_pa": 210.0,
                    "fta": 230.0,
                    "oreb": 70.0,
                    "dreb": 400.0,
                    "recent_fpts_avg": 29.0,
                    "recent_fpts_weighted": 29.5,
                    "recent_form_delta": 0.8,
                    "injuries_json": json.dumps([]),
                },
            ],
            db_path=str(self.db_path),
            source="test",
        )
        generate_nba_player_profiles("2026-04-02", db_path=str(self.db_path))
        listed = {row["name"]: row for row in list_nba_player_profiles("2026-04-02", db_path=str(self.db_path))}
        self.assertGreater(
            float(listed["Anchor Big"]["traits"]["interior_def"]),
            float(listed["Combo Forward"]["traits"]["interior_def"]),
        )
        self.assertGreater(
            float(listed["Anchor Big"]["traits"]["rim_protect"]),
            float(listed["Combo Forward"]["traits"]["rim_protect"]),
        )

    def test_bref_seed_maps_per_game_stats_into_sqlite(self) -> None:
        class FakeFrame:
            def __init__(self, rows):
                self._rows = rows
                self.empty = not rows

            def to_dict(self, orient="records"):
                self.last_orient = orient
                return list(self._rows)

        def fake_get_roster_stats(team_code, season_end_year, data_format="PER_GAME"):
            if team_code != "ATL":
                return FakeFrame([])
            self.assertEqual(season_end_year, 2026)
            self.assertEqual(data_format, "PER_GAME")
            return FakeFrame(
                [
                    {
                        "PLAYER": "Test Guard",
                        "POS": "PG",
                        "G": 70,
                        "GS": 68,
                        "MP": 34.1,
                        "PTS": 27.4,
                        "TRB": 5.6,
                        "AST": 7.8,
                        "STL": 1.4,
                        "BLK": 0.3,
                        "TOV": 2.9,
                        "PF": 2.1,
                        "FGA": 20.2,
                        "3PA": 8.4,
                        "FTA": 7.1,
                        "ORB": 0.7,
                        "DRB": 4.9,
                    }
                ]
            )

        with patch("basketball_db._load_bref_get_roster_stats", return_value=fake_get_roster_stats):
            count = seed_nba_player_stats_from_bref("2026-04-03", db_path=str(self.db_path))

        self.assertEqual(count, 1)
        generate_nba_player_profiles("2026-04-03", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-03", db_path=str(self.db_path), search="Test Guard")
        self.assertEqual(len(listed), 1)
        self.assertAlmostEqual(float(listed[0]["fga"]), 20.2)
        self.assertAlmostEqual(float(listed[0]["three_pa"]), 8.4)
        self.assertAlmostEqual(float(listed[0]["fta"]), 7.1)

    def test_stathead_dump_import_handles_repeated_headers_and_team_normalization(self) -> None:
        dump_path = Path(self.tempdir.name) / "stathead_dump.md"
        dump_path.write_text(
            "\n".join(
                [
                    "--- When using SR data, please cite us and provide a link and/or a mention.",
                    "",
                    "Rk,Player,WS,Season,Age,Team,G,GS,AS,MP,FG,FGA,2P,2PA,3P,3PA,FT,FTA,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,FG%,2P%,3P%,FT%,TS%,eFG%,Pos,Player-additional",
                    "1,LaMelo Ball,5.5,2025-26,24,CHO,67,64,0,1858,461,1144,218,477,243,667,145,162,58,263,321,477,79,15,182,181,1310,.403,.457,.364,.895,.539,.509,G,ballla01",
                    "2,James Harden,7.5,2025-26,36,CLELAC,67,67,0,2345,467,1070,263,524,204,546,445,505,39,288,327,541,74,27,236,131,1583,.436,.502,.374,.881,.613,.532,G,hardeja01",
                    "",
                    "--- When using SR data, please cite us and provide a link and/or a mention.",
                    "Rk,Player,WS,Season,Age,Team,G,GS,AS,MP,FG,FGA,2P,2PA,3P,3PA,FT,FTA,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,FG%,2P%,3P%,FT%,TS%,eFG%,Pos,Player-additional",
                    "3,Devin Booker,5.9,2025-26,29,PHO,61,61,1,2035,517,1133,402,782,115,351,421,485,46,190,236,365,50,18,193,160,1570,.456,.514,.328,.868,.583,.507,G,bookede01",
                ]
            ),
            encoding="utf-8",
        )
        count = seed_nba_player_stats_from_stathead_dump("2026-04-03", str(dump_path), db_path=str(self.db_path))
        self.assertEqual(count, 3)
        generate_nba_player_profiles("2026-04-03", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-03", db_path=str(self.db_path))
        by_name = {row["name"]: row for row in listed}
        self.assertAlmostEqual(float(by_name["LaMelo Ball"]["fga"]), 1144.0)
        self.assertEqual(by_name["LaMelo Ball"]["team_code"], "CHA")
        self.assertEqual(by_name["Devin Booker"]["team_code"], "PHX")
        self.assertEqual(by_name["James Harden"]["team_code"], "LAC")

    def test_stathead_dump_replaces_existing_rows_for_date(self) -> None:
        upsert_nba_player_stats(
            "2026-04-03",
            [
                {
                    "player_id": "espn-old",
                    "name": "Old ESPN Row",
                    "team_code": "AAA",
                    "team_id": "100",
                    "position": "PG",
                    "status": "active",
                    "games_sample": 10.0,
                    "starts": 10.0,
                    "minutes": 20.0,
                    "points": 10.0,
                    "rebounds": 2.0,
                    "assists": 3.0,
                    "turnovers": 1.0,
                    "fouls": 2.0,
                    "fga": 9.0,
                    "three_pa": 3.0,
                    "fta": 2.0,
                    "oreb": 0.5,
                    "dreb": 1.5,
                    "recent_fpts_avg": 18.0,
                    "recent_fpts_weighted": 18.0,
                    "recent_form_delta": 0.0,
                    "injuries_json": json.dumps([]),
                }
            ],
            db_path=str(self.db_path),
            source="espn_team_profiles",
        )
        dump_path = Path(self.tempdir.name) / "stathead_dump_replace.md"
        dump_path.write_text(
            "\n".join(
                [
                    "Rk,Player,WS,Season,Age,Team,G,GS,AS,MP,FG,FGA,2P,2PA,3P,3PA,FT,FTA,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,FG%,2P%,3P%,FT%,TS%,eFG%,Pos,Player-additional",
                    "1,New Canonical,5.0,2025-26,25,PHO,60,60,0,1800,300,700,200,400,100,300,200,240,50,200,250,300,40,10,120,140,900,.429,.500,.333,.833,.555,.500,G,newcano01",
                ]
            ),
            encoding="utf-8",
        )
        count = seed_nba_player_stats_from_stathead_dump("2026-04-03", str(dump_path), db_path=str(self.db_path))
        self.assertEqual(count, 1)
        generate_nba_player_profiles("2026-04-03", db_path=str(self.db_path))
        listed = list_nba_player_profiles("2026-04-03", db_path=str(self.db_path))
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "New Canonical")

    def test_nba_api_history_seed_builds_season_profiles(self) -> None:
        class FakeFrame:
            def __init__(self, rows):
                self._rows = rows

            def to_dict(self, orient="records"):
                self.last_orient = orient
                return list(self._rows)

        class FakeLeagueDashPlayerStats:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_data_frames(self):
                return [
                    FakeFrame(
                        [
                            {
                                "PLAYER_ID": 201939,
                                "PLAYER_NAME": "Stephen Curry",
                                "TEAM_ID": 1610612744,
                                "TEAM_ABBREVIATION": "GSW",
                                "AGE": 27.0,
                                "GP": 79,
                                "MIN": 34.2,
                                "FGA": 20.2,
                                "FG_PCT": 0.504,
                                "FG3A": 11.2,
                                "FG3_PCT": 0.454,
                                "FTA": 5.1,
                                "FT_PCT": 0.908,
                                "OREB": 0.9,
                                "DREB": 4.5,
                                "REB": 5.4,
                                "AST": 6.7,
                                "TOV": 3.3,
                                "STL": 2.1,
                                "BLK": 0.2,
                                "PF": 2.0,
                                "PFD": 4.6,
                                "PTS": 30.1,
                                "NBA_FANTASY_PTS": 49.3,
                            }
                        ]
                    )
                ]

        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", FakeLeagueDashPlayerStats):
            count = seed_nba_player_season_history_from_nba_api(["2015-16"], db_path=str(self.db_path))
        self.assertEqual(count, 1)
        generated = generate_nba_player_profiles_for_season("2015-16", db_path=str(self.db_path))
        self.assertEqual(generated, 1)
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            season_row = conn.execute(
                "SELECT * FROM basketball_player_season_stats WHERE season = ? AND name_key = ?",
                ("2015-16", "stephen curry"),
            ).fetchone()
            profile_row = conn.execute(
                "SELECT * FROM basketball_player_season_profiles WHERE season = ? AND name_key = ?",
                ("2015-16", "stephen curry"),
            ).fetchone()
        self.assertIsNotNone(season_row)
        self.assertIsNotNone(profile_row)
        self.assertEqual(str(season_row["team_code"]), "GS")

    def test_season_profile_generation_blends_prior_history_for_sticky_traits(self) -> None:
        history_db = Path(self.tempdir.name) / "history.sqlite"
        isolated_db = Path(self.tempdir.name) / "isolated.sqlite"

        comparison_players = (
            ("lead-guard-1", "Lead Guard", "BBB", 27.0, "PG", 760.0, 1660.0),
            ("combo-wing-1", "Combo Wing", "CCC", 26.0, "SG", 180.0, 1080.0),
            ("screen-big-1", "Screen Big", "DDD", 29.0, "C", 110.0, 920.0),
        )

        self._insert_season_stat_row(
            db_path=history_db,
            season="2023-24",
            player_id="vet-guard-1",
            name="Veteran Guard",
            team_code="AAA",
            age=33.0,
            position="PG",
            assists=640.0,
            points=1520.0,
        )
        self._insert_season_stat_row(
            db_path=history_db,
            season="2024-25",
            player_id="vet-guard-1",
            name="Veteran Guard",
            team_code="AAA",
            age=34.0,
            position="PG",
            assists=300.0,
            points=1180.0,
        )
        for player_id, name, team_code, age, position, assists, points in comparison_players:
            self._insert_season_stat_row(
                db_path=history_db,
                season="2024-25",
                player_id=player_id,
                name=name,
                team_code=team_code,
                age=age,
                position=position,
                assists=assists,
                points=points,
            )
        generate_nba_player_profiles_for_season("2023-24", db_path=str(history_db))
        generate_nba_player_profiles_for_season("2024-25", db_path=str(history_db))

        self._insert_season_stat_row(
            db_path=isolated_db,
            season="2024-25",
            player_id="vet-guard-1",
            name="Veteran Guard",
            team_code="AAA",
            age=34.0,
            position="PG",
            assists=300.0,
            points=1180.0,
        )
        for player_id, name, team_code, age, position, assists, points in comparison_players:
            self._insert_season_stat_row(
                db_path=isolated_db,
                season="2024-25",
                player_id=player_id,
                name=name,
                team_code=team_code,
                age=age,
                position=position,
                assists=assists,
                points=points,
            )
        generate_nba_player_profiles_for_season("2024-25", db_path=str(isolated_db))

        with sqlite3.connect(history_db) as conn:
            history_traits = json.loads(
                conn.execute(
                    "SELECT traits_json FROM basketball_player_season_profiles WHERE season = ? AND name_key = ?",
                    ("2024-25", "veteran-guard"),
                ).fetchone()[0]
            )
        with sqlite3.connect(isolated_db) as conn:
            isolated_traits = json.loads(
                conn.execute(
                    "SELECT traits_json FROM basketball_player_season_profiles WHERE season = ? AND name_key = ?",
                    ("2024-25", "veteran-guard"),
                ).fetchone()[0]
            )

        pass_vision_delta = float(history_traits["pass_vision"]) - float(isolated_traits["pass_vision"])
        self.assertGreater(pass_vision_delta, 0.0)
        self.assertGreater(float(history_traits["pass_accuracy"]), float(isolated_traits["pass_accuracy"]))
        self.assertLess(
            abs(float(history_traits["burst"]) - float(isolated_traits["burst"])),
            pass_vision_delta,
        )

    def test_nba_api_advanced_history_seed_stores_metrics(self) -> None:
        class FakeFrame:
            def __init__(self, rows):
                self._rows = rows

            def to_dict(self, orient="records"):
                self.last_orient = orient
                return list(self._rows)

        class FakeLeagueDashPlayerStats:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_data_frames(self):
                return [
                    FakeFrame(
                        [
                            {
                                "PLAYER_ID": 201939,
                                "PLAYER_NAME": "Stephen Curry",
                                "TEAM_ID": 1610612744,
                                "TEAM_ABBREVIATION": "GSW",
                                "AGE": 27.0,
                                "GP": 79,
                                "MIN": 34.2,
                                "OFF_RATING": 121.9,
                                "DEF_RATING": 103.8,
                                "NET_RATING": 18.1,
                                "AST_PCT": 0.334,
                                "AST_TO": 2.12,
                                "AST_RATIO": 19.6,
                                "OREB_PCT": 1.2,
                                "DREB_PCT": 13.1,
                                "REB_PCT": 7.2,
                                "TM_TOV_PCT": 12.3,
                                "E_TOV_PCT": 10.8,
                                "EFG_PCT": 0.630,
                                "TS_PCT": 0.669,
                                "USG_PCT": 0.323,
                                "PACE": 99.3,
                                "PIE": 0.208,
                            }
                        ]
                    )
                ]

        with patch("nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats", FakeLeagueDashPlayerStats):
            count = seed_nba_player_season_advanced_history_from_nba_api(["2015-16"], db_path=str(self.db_path))
        self.assertEqual(count, 1)
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM basketball_player_season_advanced_stats WHERE season = ? AND name_key = ?",
                ("2015-16", "stephen curry"),
            ).fetchone()
        self.assertIsNotNone(row)
        metrics = json.loads(str(row["metrics_json"] or "{}"))
        self.assertAlmostEqual(float(metrics["usg_pct"]), 0.323)
        self.assertAlmostEqual(float(metrics["ts_pct"]), 0.669)


if __name__ == "__main__":
    unittest.main()
