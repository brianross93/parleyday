from pathlib import Path

from dfs_ingest import NBA_CLASSIC_SLOTS, parse_draftkings_salary_csv


def test_parse_draftkings_salary_csv_infers_nba_slate(tmp_path: Path) -> None:
    csv_path = tmp_path / "DKSalaries.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                "PF,Giannis Antetokounmpo (42464713),Giannis Antetokounmpo,42464713,PF/F/UTIL,11000,DAL@MIL 03/31/2026 08:00PM ET,MIL,50.60",
                "PG/SG,Devin Booker (42464725),Devin Booker,42464725,PG/SG/G/UTIL,9400,PHX@ORL 03/31/2026 07:00PM ET,PHX,40.11",
            ]
        ),
        encoding="utf-8",
    )

    slate = parse_draftkings_salary_csv(csv_path)

    assert slate.site == "draftkings"
    assert slate.sport == "nba"
    assert slate.salary_cap == 50000
    assert slate.roster_slots == NBA_CLASSIC_SLOTS
    assert len(slate.players) == 2
    assert slate.players[0].team == "MIL"
    assert slate.players[0].opponent == "DAL"
    assert slate.players[0].game == "DAL@MIL"
    assert slate.players[0].salary == 11000
    assert slate.players[1].positions == ("PG", "SG")
    assert slate.players[1].roster_positions == ("PG", "SG", "G", "UTIL")


def test_parse_real_salary_file_shape() -> None:
    path = Path(r"C:\Users\brssn\Downloads\DKSalaries.csv")
    if not path.exists():
        return
    slate = parse_draftkings_salary_csv(path)
    assert slate.site == "draftkings"
    assert slate.sport == "nba"
    assert slate.roster_slots == NBA_CLASSIC_SLOTS
    assert len(slate.players) > 50
