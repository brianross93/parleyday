import argparse
import json

from backtest_monte_carlo_2025 import run_backtest as run_mlb_backtest, write_outputs as write_mlb_outputs
from backtest_nba_2025 import run_backtest as run_nba_backtest, write_outputs as write_nba_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2025 Monte Carlo backtests for MLB, NBA, or both")
    parser.add_argument("--sport", choices=["mlb", "nba", "both"], default="both")
    parser.add_argument("--mlb-start-date", default="2025-04-01")
    parser.add_argument("--mlb-end-date", default="2025-10-01")
    parser.add_argument("--nba-start-date", default="2025-01-01")
    parser.add_argument("--nba-end-date", default="2025-12-31")
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--run-name-prefix", default="full_2025")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = {}
    if args.sport in {"mlb", "both"}:
        mlb = run_mlb_backtest(
            start_date=args.mlb_start_date,
            end_date=args.mlb_end_date,
            n_simulations=args.n_simulations,
            seed=args.seed,
        )
        mlb_run = f"{args.run_name_prefix}_mlb"
        write_mlb_outputs(mlb_run, mlb["game_rows"], mlb["summary"])
        output["mlb"] = mlb["summary"]
    if args.sport in {"nba", "both"}:
        nba = run_nba_backtest(
            start_date=args.nba_start_date,
            end_date=args.nba_end_date,
            n_simulations=args.n_simulations,
            seed=args.seed + 1000,
        )
        nba_run = f"{args.run_name_prefix}_nba"
        write_nba_outputs(nba_run, nba["game_rows"], nba["summary"])
        output["nba"] = nba["summary"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
