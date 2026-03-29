import argparse
import json
from pathlib import Path

from phase3_backtest import RESULTS_DIR, run_backtest, write_outputs


CANDIDATE_CONFIGS = [
    {"strategy": "standard", "score_source": "implied"},
    {"strategy": "standard", "score_source": "heuristic"},
    {"strategy": "standard", "score_source": "residual"},
    {"strategy": "totals_focus", "score_source": "implied"},
    {"strategy": "totals_focus", "score_source": "heuristic"},
    {"strategy": "totals_focus", "score_source": "residual"},
]


def total_profit(summary: dict) -> float:
    return sum(float(value) for value in summary.get("oracle_profit_units", {}).values())


def total_roi(summary: dict) -> float:
    days = max(int(summary.get("days_processed", 0)), 1)
    return total_profit(summary) / (days * 3.0)


def config_label(config: dict) -> str:
    return f"{config['strategy']}::{config['score_source']}"


def run_split(
    *,
    start_date: str,
    end_date: str,
    strategy: str,
    score_source: str,
    seed: int,
    samples_per_beta: int,
    warmup: int,
    thin: int,
    learning_rate: float,
    coupling_prior: float,
    max_coupling_magnitude: float,
) -> dict:
    return run_backtest(
        start_date=start_date,
        end_date=end_date,
        samples_per_beta=samples_per_beta,
        warmup=warmup,
        thin=thin,
        learning_rate=learning_rate,
        seed=seed,
        strategy=strategy,
        coupling_prior=coupling_prior,
        max_coupling_magnitude=max_coupling_magnitude,
        score_source=score_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/holdout parlay profitability backtest")
    parser.add_argument("--train-start-date", default="2025-04-01")
    parser.add_argument("--train-end-date", default="2025-08-31")
    parser.add_argument("--holdout-start-date", default="2025-09-01")
    parser.add_argument("--holdout-end-date", default="2025-10-01")
    parser.add_argument("--samples-per-beta", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--thin", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coupling-prior", type=float, default=12.0)
    parser.add_argument("--max-coupling-magnitude", type=float, default=1.5)
    parser.add_argument("--run-name", default="train_holdout_2025")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaderboard = []

    for index, config in enumerate(CANDIDATE_CONFIGS):
        train = run_split(
            start_date=args.train_start_date,
            end_date=args.train_end_date,
            strategy=config["strategy"],
            score_source=config["score_source"],
            seed=args.seed + index,
            samples_per_beta=args.samples_per_beta,
            warmup=args.warmup,
            thin=args.thin,
            learning_rate=args.learning_rate,
            coupling_prior=args.coupling_prior,
            max_coupling_magnitude=args.max_coupling_magnitude,
        )
        leaderboard.append(
            {
                "config": config,
                "label": config_label(config),
                "train_summary": train["summary"],
                "train_total_profit": total_profit(train["summary"]),
                "train_total_roi": total_roi(train["summary"]),
            }
        )

    leaderboard.sort(key=lambda item: (item["train_total_roi"], item["train_total_profit"]), reverse=True)
    winner = leaderboard[0]
    best_config = winner["config"]

    holdout = run_split(
        start_date=args.holdout_start_date,
        end_date=args.holdout_end_date,
        strategy=best_config["strategy"],
        score_source=best_config["score_source"],
        seed=args.seed + 999,
        samples_per_beta=args.samples_per_beta,
        warmup=args.warmup,
        thin=args.thin,
        learning_rate=args.learning_rate,
        coupling_prior=args.coupling_prior,
        max_coupling_magnitude=args.max_coupling_magnitude,
    )

    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        f"{args.run_name}_holdout_best",
        holdout["daily_rows"],
        holdout["summary"],
        holdout["oracle_leg_rows"],
    )

    report = {
        "train_window": {"start": args.train_start_date, "end": args.train_end_date},
        "holdout_window": {"start": args.holdout_start_date, "end": args.holdout_end_date},
        "leaderboard": [
            {
                "label": item["label"],
                "strategy": item["config"]["strategy"],
                "score_source": item["config"]["score_source"],
                "train_days": item["train_summary"]["days_processed"],
                "train_total_profit": item["train_total_profit"],
                "train_total_roi": item["train_total_roi"],
                "train_oracle_roi": item["train_summary"]["oracle_roi"],
            }
            for item in leaderboard
        ],
        "best_config": best_config,
        "holdout_summary": {
            "days_processed": holdout["summary"]["days_processed"],
            "oracle_hits": holdout["summary"]["oracle_hits"],
            "oracle_profit_units": holdout["summary"]["oracle_profit_units"],
            "oracle_roi": holdout["summary"]["oracle_roi"],
            "implied_profit_units": holdout["summary"]["implied_profit_units"],
            "implied_roi": holdout["summary"]["implied_roi"],
            "random_profit_units": holdout["summary"]["random_profit_units"],
            "random_roi": holdout["summary"]["random_roi"],
            "edge_profit_units": holdout["summary"]["edge_profit_units"],
            "edge_roi": holdout["summary"]["edge_roi"],
        },
        "notes": "Profitability is a proxy based on synthesized market-implied probabilities, not archived sportsbook prices.",
    }

    report_path = run_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
