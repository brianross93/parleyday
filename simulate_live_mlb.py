import argparse
import json
from datetime import datetime

from data_pipeline import SnapshotStore
from monte_carlo.mlb import MLBGameConfig, MLBGameSimulator
from quantum_parlay_oracle import build_calibrated_live_mlb_contexts


def load_matchup_profile(store: SnapshotStore, date_str: str, matchup: str) -> dict:
    snapshot = store.get_snapshot(
        source="mlb_refresh",
        sport="mlb",
        entity_type="matchup_profile",
        entity_key=matchup,
        as_of_date=date_str,
    )
    if snapshot is None:
        raise RuntimeError(
            f"No cached matchup profile for {matchup} on {date_str}. "
            f"Run refresh_slate.py first."
        )
    return snapshot["payload"]
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live MLB Monte Carlo sim from cached player profiles")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--matchup", required=True, help="Matchup token like NYY@BOS")
    parser.add_argument("--db-path", default=SnapshotStore().db_path)
    parser.add_argument("--sims", type=int, default=2000)
    args = parser.parse_args()

    store = SnapshotStore(args.db_path)
    payload = load_matchup_profile(store, args.date, args.matchup)
    away_code, home_code = args.matchup.split("@")
    away, home = build_calibrated_live_mlb_contexts(args.date, args.matchup, payload)
    simulator = MLBGameSimulator(MLBGameConfig(n_simulations=args.sims, random_seed=17))
    result = simulator.simulate_game(away=away, home=home)

    summary = {
        "date": args.date,
        "matchup": args.matchup,
        "simulations": args.sims,
        "average_score": {
            away_code: sum(result.away_scores) / len(result.away_scores),
            home_code: sum(result.home_scores) / len(result.home_scores),
        },
        "win_rate": {
            away_code: sum(1 for winner in result.winners if winner == away_code) / len(result.winners),
            home_code: sum(1 for winner in result.winners if winner == home_code) / len(result.winners),
        },
        "top_hitters": [
            {
                "player": player_name,
                "stat": stat,
                "mean": distribution.mean,
            }
            for (player_name, stat), distribution in result.player_props.items()
            if stat in {"hits", "home_runs", "total_bases"} and distribution.mean > 0.2
        ][:12],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
