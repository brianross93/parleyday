import argparse
import csv
import json
from pathlib import Path


RESULTS_DIR = Path("results") / "phase3_backtest"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def bucket_label(lower: float, upper: float) -> str:
    return f"{lower:.1f}-{upper:.1f}"


def analyze_leg_rows(leg_rows: list[dict]) -> dict:
    total = len(leg_rows)
    category_counts = {}
    category_hits = {}
    size_counts = {}
    size_hits = {}
    bucket_counts = {}
    bucket_hits = {}

    bucket_edges = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for row in leg_rows:
        category = row["category"]
        parlay_size = int(row["parlay_size"])
        activation = float(row["activation"])
        hit = int(row["hit"])

        category_counts[category] = category_counts.get(category, 0) + 1
        category_hits[category] = category_hits.get(category, 0) + hit
        size_counts[parlay_size] = size_counts.get(parlay_size, 0) + 1
        size_hits[parlay_size] = size_hits.get(parlay_size, 0) + hit

        lower = 0.0
        upper = 1.0
        for edge in bucket_edges:
            if activation < edge:
                upper = edge
                break
            lower = edge
        label = bucket_label(lower, upper)
        bucket_counts[label] = bucket_counts.get(label, 0) + 1
        bucket_hits[label] = bucket_hits.get(label, 0) + hit

    return {
        "total_rows": total,
        "by_category": [
            {
                "category": category,
                "count": category_counts[category],
                "hit_rate": rate(category_hits[category], category_counts[category]),
            }
            for category in sorted(category_counts)
        ],
        "by_parlay_size": [
            {
                "parlay_size": size,
                "count": size_counts[size],
                "hit_rate": rate(size_hits[size], size_counts[size]),
            }
            for size in sorted(size_counts)
        ],
        "by_activation_bucket": [
            {
                "bucket": label,
                "count": bucket_counts[label],
                "hit_rate": rate(bucket_hits[label], bucket_counts[label]),
            }
            for label in sorted(bucket_counts)
        ],
    }


def summarize_run(run_dir: Path) -> dict:
    summary = load_json(run_dir / "summary.json")
    daily_rows = load_csv_rows(run_dir / "daily_results.csv")
    leg_rows = load_csv_rows(run_dir / "oracle_legs.csv")
    days = len(daily_rows)
    coupling_report = summary.get("coupling_report", [])

    parlay_summary = []
    for size in (3, 4, 5):
        oracle_hits = int(summary["oracle_hits"][str(size)])
        random_hits = int(summary["random_hits"][str(size)])
        implied_hits = int(summary["implied_hits"][str(size)])
        edge_hits = int(summary["edge_hits"][str(size)])
        parlay_summary.append(
            {
                "size": size,
                "oracle_rate": rate(oracle_hits, days),
                "random_rate": rate(random_hits, days),
                "implied_rate": rate(implied_hits, days),
                "edge_rate": rate(edge_hits, days),
                "oracle_minus_random": rate(oracle_hits, days) - rate(random_hits, days),
                "oracle_minus_implied": rate(oracle_hits, days) - rate(implied_hits, days),
                "oracle_minus_edge": rate(oracle_hits, days) - rate(edge_hits, days),
            }
        )

    return {
        "run_dir": str(run_dir),
        "days_processed": days,
        "score_source": summary.get("score_source", "ising"),
        "oracle_leg_hit_rate": float(summary["oracle_leg_hit_rate"]),
        "oracle_avg_activation": float(summary["oracle_avg_activation"]),
        "oracle_avg_implied_prob": float(summary["oracle_avg_implied_prob"]),
        "coupling_prior": float(summary.get("coupling_prior", 0.0)),
        "max_coupling_magnitude": float(summary.get("max_coupling_magnitude", 0.0)),
        "parlay_summary": parlay_summary,
        "leg_analysis": analyze_leg_rows(leg_rows),
        "top_repulsive_couplings": coupling_report[:5],
        "top_attractive_couplings": sorted(coupling_report, key=lambda row: float(row["adjustment"]))[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a Phase 3 backtest run")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = RESULTS_DIR / args.run_name
    report = summarize_run(run_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
