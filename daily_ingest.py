import argparse
from datetime import datetime

from data_pipeline import SnapshotStore
from quantum_parlay_oracle import (
    fetch_live_mlb_team_form,
    fetch_live_nba_team_form,
    fetch_schedule_for_sport,
)


def ingest_daily_baseline(date_str: str, db_path: str) -> dict:
    store = SnapshotStore(db_path)

    mlb_schedule = [game.__dict__ for game in fetch_schedule_for_sport(date_str, "mlb")]
    nba_schedule = [game.__dict__ for game in fetch_schedule_for_sport(date_str, "nba")]
    mlb_team_form = fetch_live_mlb_team_form(date_str)
    nba_team_form = fetch_live_nba_team_form(date_str)

    store.upsert_snapshot(
        source="mlb_statsapi",
        sport="mlb",
        entity_type="schedule",
        entity_key="daily",
        as_of_date=date_str,
        payload=mlb_schedule,
        is_volatile=False,
    )
    store.upsert_snapshot(
        source="mlb_statsapi",
        sport="mlb",
        entity_type="team_form",
        entity_key="daily",
        as_of_date=date_str,
        payload=mlb_team_form,
        is_volatile=False,
    )
    store.upsert_snapshot(
        source="nba_scoreboard",
        sport="nba",
        entity_type="schedule",
        entity_key="daily",
        as_of_date=date_str,
        payload=nba_schedule,
        is_volatile=False,
    )
    store.upsert_snapshot(
        source="nba_scoreboard",
        sport="nba",
        entity_type="team_form",
        entity_key="daily",
        as_of_date=date_str,
        payload=nba_team_form,
        is_volatile=False,
    )

    return {
        "date": date_str,
        "db_path": db_path,
        "mlb_games": len(mlb_schedule),
        "nba_games": len(nba_schedule),
        "mlb_teams": len(mlb_team_form),
        "nba_teams": len(nba_team_form),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull and cache baseline daily MLB/NBA data")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--db-path", default=SnapshotStore().db_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ingest_daily_baseline(args.date, args.db_path)
    print(summary)


if __name__ == "__main__":
    main()
