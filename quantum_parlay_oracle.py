import argparse
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None

try:
    import statsapi
except ImportError:  # pragma: no cover - optional dependency
    statsapi = None

try:
    from data_pipeline.cache import DEFAULT_DB_PATH, SnapshotStore
except ImportError:  # pragma: no cover - optional dependency at runtime
    DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "parleyday.sqlite")
    SnapshotStore = None

try:
    from data_pipeline.mlb_profiles import team_context_from_cached_payload
except ImportError:  # pragma: no cover - optional dependency at runtime
    team_context_from_cached_payload = None

try:
    from monte_carlo.mlb import MLBGameConfig, MLBGameSimulator
except ImportError:  # pragma: no cover - optional dependency at runtime
    MLBGameConfig = None
    MLBGameSimulator = None


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
DEFAULT_BETAS = [0.8, 1.0, 1.2, 1.5, 2.0]
DATE_CODE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})(\d{4})?([A-Z0-9]+)?$")
TEAM_WORD_RE = re.compile(r"[^a-z0-9]+")
EASTERN_TZ = ZoneInfo("America/New_York")
LEAGUE_AVG_RUNS = 4.5
NBA_BASELINE_POINTS = 112.0
NBA_HOME_COURT_POINTS = 2.5
SIMULATION_RUNS = 4000
DEFAULT_BASELINE_CACHE_MAX_AGE_HOURS = 36.0
DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS = 6.0

TEAM_CODES = {
    "Arizona Diamondbacks": ["AZ", "ARI"],
    "Atlanta Braves": ["ATL"],
    "Baltimore Orioles": ["BAL"],
    "Boston Red Sox": ["BOS"],
    "Chicago Cubs": ["CHC"],
    "Chicago White Sox": ["CWS", "CHW", "Chicago WS"],
    "Cincinnati Reds": ["CIN"],
    "Cleveland Guardians": ["CLE"],
    "Colorado Rockies": ["COL"],
    "Detroit Tigers": ["DET"],
    "Houston Astros": ["HOU"],
    "Kansas City Royals": ["KC", "KCR"],
    "Los Angeles Angels": ["LAA", "ANA", "Angels"],
    "Los Angeles Dodgers": ["LAD", "Dodgers"],
    "Miami Marlins": ["MIA"],
    "Milwaukee Brewers": ["MIL"],
    "Minnesota Twins": ["MIN"],
    "New York Mets": ["NYM", "Mets"],
    "New York Yankees": ["NYY", "Yankees"],
    "Athletics": ["ATH", "OAK", "A's"],
    "Philadelphia Phillies": ["PHI"],
    "Pittsburgh Pirates": ["PIT"],
    "San Diego Padres": ["SD", "SDP", "Padres"],
    "San Francisco Giants": ["SF", "SFG", "Giants"],
    "Seattle Mariners": ["SEA"],
    "St. Louis Cardinals": ["STL"],
    "Tampa Bay Rays": ["TB", "TBR", "Rays"],
    "Texas Rangers": ["TEX"],
    "Toronto Blue Jays": ["TOR"],
    "Washington Nationals": ["WSH", "WAS"],
    "Atlanta Hawks": ["ATL"],
    "Boston Celtics": ["BOS"],
    "Brooklyn Nets": ["BKN", "BRK"],
    "Charlotte Hornets": ["CHA"],
    "Chicago Bulls": ["CHI"],
    "Cleveland Cavaliers": ["CLE"],
    "Dallas Mavericks": ["DAL"],
    "Denver Nuggets": ["DEN"],
    "Detroit Pistons": ["DET"],
    "Golden State Warriors": ["GSW", "GS"],
    "Houston Rockets": ["HOU"],
    "Indiana Pacers": ["IND"],
    "LA Clippers": ["LAC", "Clippers"],
    "Los Angeles Lakers": ["LAL", "Lakers"],
    "Memphis Grizzlies": ["MEM"],
    "Miami Heat": ["MIA"],
    "Milwaukee Bucks": ["MIL"],
    "Minnesota Timberwolves": ["MIN"],
    "New Orleans Pelicans": ["NOP", "NO"],
    "New York Knicks": ["NYK"],
    "Oklahoma City Thunder": ["OKC"],
    "Orlando Magic": ["ORL"],
    "Philadelphia 76ers": ["PHI", "Sixers"],
    "Phoenix Suns": ["PHX"],
    "Portland Trail Blazers": ["POR"],
    "Sacramento Kings": ["SAC"],
    "San Antonio Spurs": ["SAS", "SA"],
    "Toronto Raptors": ["TOR"],
    "Utah Jazz": ["UTA"],
    "Washington Wizards": ["WAS", "WSH"],
}

NBA_CODE_TO_NAME = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def cache_db_path() -> str:
    return os.getenv("PARLEYDAY_DB_PATH", DEFAULT_DB_PATH)


def load_cached_payload(
    *,
    source: str,
    sport: str,
    entity_type: str,
    entity_key: str,
    as_of_date: str,
    max_age_hours: float,
):
    if SnapshotStore is None:
        return None
    snapshot = SnapshotStore(cache_db_path()).get_snapshot(
        source=source,
        sport=sport,
        entity_type=entity_type,
        entity_key=entity_key,
        as_of_date=as_of_date,
        max_age_hours=max_age_hours,
    )
    return None if snapshot is None else snapshot["payload"]


def store_cached_payload(
    *,
    source: str,
    sport: str,
    entity_type: str,
    entity_key: str,
    as_of_date: str,
    payload,
    is_volatile: bool,
) -> None:
    if SnapshotStore is None:
        return
    SnapshotStore(cache_db_path()).upsert_snapshot(
        source=source,
        sport=sport,
        entity_type=entity_type,
        entity_key=entity_key,
        as_of_date=as_of_date,
        payload=payload,
        is_volatile=is_volatile,
    )


class QuantumEntropySource:
    """
    Fetches random bytes from ANU's QRNG API and falls back to OS-backed entropy.
    """

    ANU_URL = "https://qrng.anu.edu.au/API/jsonI.php"
    MAX_BATCH_SIZE = 1024

    def __init__(self, n_bytes: int = 65536, fallback: bool = False):
        self.buffer = np.array([], dtype=np.uint8)
        self.index = 0
        self.total_consumed = 0
        self.source = "unknown"
        self.fallback = fallback
        self._fetch(n_bytes)

    def _fetch(self, n_bytes: int) -> None:
        if self.fallback:
            self._fetch_os_entropy(n_bytes)
            return

        if requests is None:
            print("  requests is unavailable, falling back to OS entropy")
            self._fetch_os_entropy(n_bytes)
            return

        try:
            print(f"  Fetching {n_bytes} bytes from ANU QRNG...")
            all_bytes = []
            remaining = n_bytes
            batch = 0

            while remaining > 0:
                chunk = min(remaining, self.MAX_BATCH_SIZE)
                response = requests.get(
                    self.ANU_URL,
                    params={"length": chunk, "type": "uint8"},
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()

                if not payload.get("success"):
                    raise RuntimeError(f"ANU QRNG returned non-success: {payload}")

                data = payload.get("data", [])
                if not data:
                    raise RuntimeError("ANU QRNG returned an empty payload")

                all_bytes.extend(data)
                remaining -= len(data)
                batch += 1
                print(
                    f"    Batch {batch}: received {len(data)} bytes "
                    f"({n_bytes - remaining}/{n_bytes})"
                )

                if remaining > 0:
                    time.sleep(0.25)

            self.buffer = np.array(all_bytes, dtype=np.uint8)
            self.source = "ANU QRNG"
            print(f"  Loaded {len(self.buffer)} quantum bytes")
        except Exception as exc:
            print(f"  ANU QRNG unavailable: {exc}")
            print("  Falling back to OS entropy")
            self._fetch_os_entropy(n_bytes)

    def _fetch_os_entropy(self, n_bytes: int) -> None:
        rng = np.random.default_rng()
        self.buffer = np.frombuffer(rng.bytes(n_bytes), dtype=np.uint8)
        self.source = "OS entropy fallback"

    def _ensure_capacity(self, n: int) -> None:
        if self.index + n <= len(self.buffer):
            return
        needed = max(len(self.buffer), n, 1024)
        self._fetch_os_entropy(needed)
        self.index = 0

    def next_float(self) -> float:
        self._ensure_capacity(1)
        value = float(self.buffer[self.index]) / 256.0
        self.index += 1
        self.total_consumed += 1
        return value

    def next_floats(self, n: int) -> np.ndarray:
        self._ensure_capacity(n)
        values = self.buffer[self.index : self.index + n].astype(np.float64) / 256.0
        self.index += n
        self.total_consumed += n
        return values


class StaticEntropySource:
    def __init__(self, source: str):
        self.source = source
        self.total_consumed = 0


@dataclass(frozen=True)
class GameInfo:
    sport: str
    game_pk: int
    date: str
    game_time: str
    game_state: str
    away_name: str
    home_name: str
    away_code: str
    home_code: str
    away_detail: str
    home_detail: str

    @property
    def matchup(self) -> str:
        return f"{self.away_code}@{self.home_code}"

    @property
    def title_key(self) -> tuple[str, str]:
        return normalize_name(self.away_name), normalize_name(self.home_name)

    @property
    def kalshi_tokens(self) -> set[str]:
        away_aliases = TEAM_CODES.get(self.away_name, [self.away_code])
        home_aliases = TEAM_CODES.get(self.home_name, [self.home_code])
        tokens = set()
        for away in away_aliases:
            for home in home_aliases:
                tokens.add(compact_token(f"{away}{home}"))
        return tokens

    @property
    def matchup_note(self) -> str:
        if self.sport == "mlb":
            return f"{self.away_detail} vs {self.home_detail}"
        return f"{self.away_name} at {self.home_name}"


@dataclass(frozen=True)
class Leg:
    id: int
    label: str
    category: str
    game: str
    implied_prob: float
    notes: str = ""
    sport: str = "mlb"


STATIC_LEGS = [
    Leg(0, "MIL ML", "ml", "CWS@MIL", 0.66, "Static fallback slate"),
    Leg(1, "CWS ML", "ml", "CWS@MIL", 0.34, "Static fallback slate"),
    Leg(2, "CHC ML", "ml", "WSH@CHC", 0.69, "Static fallback slate"),
    Leg(3, "WSH ML", "ml", "WSH@CHC", 0.31, "Static fallback slate"),
    Leg(4, "BAL ML", "ml", "MIN@BAL", 0.59, "Static fallback slate"),
    Leg(5, "MIN ML", "ml", "MIN@BAL", 0.41, "Static fallback slate"),
    Leg(6, "HOU ML", "ml", "LAA@HOU", 0.65, "Static fallback slate"),
    Leg(7, "LAA ML", "ml", "LAA@HOU", 0.35, "Static fallback slate"),
    Leg(8, "SD ML", "ml", "DET@SD", 0.60, "Static fallback slate"),
    Leg(9, "DET ML", "ml", "DET@SD", 0.40, "Static fallback slate"),
    Leg(10, "CIN ML", "ml", "BOS@CIN", 0.41, "Static fallback slate"),
    Leg(11, "BOS ML", "ml", "BOS@CIN", 0.59, "Static fallback slate"),
    Leg(12, "PHI ML", "ml", "TEX@PHI", 0.62, "Static fallback slate"),
    Leg(13, "TEX ML", "ml", "TEX@PHI", 0.38, "Static fallback slate"),
    Leg(14, "STL ML", "ml", "TB@STL", 0.54, "Static fallback slate"),
    Leg(15, "TB ML", "ml", "TB@STL", 0.46, "Static fallback slate"),
    Leg(16, "LAD ML", "ml", "AZ@LAD", 0.72, "Static fallback slate"),
    Leg(17, "AZ ML", "ml", "AZ@LAD", 0.28, "Static fallback slate"),
    Leg(18, "SEA ML", "ml", "CLE@SEA", 0.65, "Static fallback slate"),
    Leg(19, "CLE ML", "ml", "CLE@SEA", 0.35, "Static fallback slate"),
    Leg(20, "CWS@MIL U8", "total", "CWS@MIL", 0.48, "Static fallback slate"),
    Leg(21, "CWS@MIL O8", "total", "CWS@MIL", 0.52, "Static fallback slate"),
    Leg(22, "BOS@CIN U8.5", "total", "BOS@CIN", 0.50, "Static fallback slate"),
    Leg(23, "BOS@CIN O8.5", "total", "BOS@CIN", 0.50, "Static fallback slate"),
    Leg(24, "CLE@SEA U6.5", "total", "CLE@SEA", 0.55, "Static fallback slate"),
    Leg(25, "CLE@SEA O6.5", "total", "CLE@SEA", 0.45, "Static fallback slate"),
    Leg(26, "AZ@LAD U8", "total", "AZ@LAD", 0.45, "Static fallback slate"),
    Leg(27, "AZ@LAD O8", "total", "AZ@LAD", 0.55, "Static fallback slate"),
    Leg(28, "TB@STL U8", "total", "TB@STL", 0.45, "Static fallback slate"),
    Leg(29, "TB@STL O8", "total", "TB@STL", 0.55, "Static fallback slate"),
    Leg(30, "Skubal O K's", "prop", "DET@SD", 0.50, "Static fallback slate"),
    Leg(31, "Sanchez O K's", "prop", "TEX@PHI", 0.48, "Static fallback slate"),
    Leg(32, "Gilbert O K's", "prop", "CLE@SEA", 0.50, "Static fallback slate"),
    Leg(33, "Misiorowski O K's", "prop", "CWS@MIL", 0.50, "Static fallback slate"),
    Leg(34, "Crochet O K's", "prop", "BOS@CIN", 0.52, "Static fallback slate"),
    Leg(35, "Yamamoto O K's", "prop", "AZ@LAD", 0.48, "Static fallback slate"),
    Leg(36, "Alvarez HR", "prop", "LAA@HOU", 0.15, "Static fallback slate"),
    Leg(37, "Suarez HR", "prop", "DET@SD", 0.14, "Static fallback slate"),
    Leg(38, "Tucker HR", "prop", "AZ@LAD", 0.13, "Static fallback slate"),
]


def normalize_name(value: str) -> str:
    return TEAM_WORD_RE.sub("", value.lower())


def compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def parse_market_price(market: dict) -> float | None:
    for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"):
        raw = market.get(key)
        if raw in (None, "", "0.0000", "0", 0):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 < value < 1.0:
            return value
    return None


def market_midpoint(market: dict) -> float | None:
    values = []
    for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"):
        raw = market.get(key)
        if raw in (None, "", "0.0000", "0", 0):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 < value < 1.0:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def structured_threshold_label(
    market: dict, metric_code: str, metric_word: str
) -> str | None:
    title = market.get("title", "")
    yes_subtitle = market.get("yes_sub_title") or ""
    if ":" in title:
        player = title.split(":", 1)[0].strip()
    else:
        player = yes_subtitle.split(":", 1)[0].strip()

    if not player:
        return None

    threshold = None
    if ":" in yes_subtitle:
        rhs = yes_subtitle.split(":", 1)[1].strip()
        if rhs.endswith("+"):
            threshold = rhs[:-1].strip()

    if not threshold:
        floor = market.get("floor_strike")
        if floor is not None:
            threshold = str(int(float(floor) + 0.5))

    if not threshold:
        rules = market.get("rules_primary", "")
        match = re.search(r"records\s+(\d+)\+\s+" + re.escape(metric_word), rules, re.IGNORECASE)
        if match:
            threshold = match.group(1)

    if not threshold:
        return None

    return f"{player} O {threshold} {metric_code}"


def canonical_team_code(team_name: str) -> str:
    aliases = TEAM_CODES.get(team_name)
    if aliases:
        return compact_token(aliases[0])
    parts = team_name.replace(".", "").split()
    return compact_token("".join(part[:3] for part in parts[-2:]))


def fetch_mlb_schedule(date_str: str) -> list[GameInfo]:
    games = []

    if statsapi is not None:
        try:
            schedule = statsapi.schedule(date=date_str)
            for item in schedule:
                away_name = item.get("away_name")
                home_name = item.get("home_name")
                if not away_name or not home_name:
                    continue
                games.append(
                    GameInfo(
                        sport="mlb",
                        game_pk=int(item["game_id"]),
                        date=item.get("game_date", date_str),
                        game_time=item.get("game_datetime", ""),
                        game_state=item.get("status", ""),
                        away_name=away_name,
                        home_name=home_name,
                        away_code=canonical_team_code(away_name),
                        home_code=canonical_team_code(home_name),
                        away_detail=item.get("away_probable_pitcher", "TBD"),
                        home_detail=item.get("home_probable_pitcher", "TBD"),
                    )
                )
            if games:
                return games
        except Exception:
            games = []

    if requests is None:
        raise RuntimeError("Live schedule loading requires requests")

    response = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    for date_block in payload.get("dates", []):
        for item in date_block.get("games", []):
            away = item["teams"]["away"]
            home = item["teams"]["home"]
            away_name = away["team"]["name"]
            home_name = home["team"]["name"]
            games.append(
                GameInfo(
                    sport="mlb",
                    game_pk=int(item["gamePk"]),
                    date=item.get("officialDate", date_str),
                    game_time=item.get("gameDate", ""),
                    game_state=item.get("status", {}).get("detailedState", ""),
                    away_name=away_name,
                    home_name=home_name,
                    away_code=canonical_team_code(away_name),
                    home_code=canonical_team_code(home_name),
                    away_detail=away.get("probablePitcher", {}).get("fullName", "TBD"),
                    home_detail=home.get("probablePitcher", {}).get("fullName", "TBD"),
                )
            )
    return games


def fetch_nba_schedule(date_str: str) -> list[GameInfo]:
    if requests is None:
        raise RuntimeError("Live NBA loading requires requests")

    response = requests.get(NBA_SCOREBOARD_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    board = payload.get("scoreboard", {})

    if board.get("gameDate") != date_str:
        return fetch_nba_schedule_from_kalshi(date_str)

    games = []
    for item in board.get("games", []):
        away = item.get("awayTeam", {})
        home = item.get("homeTeam", {})
        away_name = f"{away.get('teamCity', '').strip()} {away.get('teamName', '').strip()}".strip()
        home_name = f"{home.get('teamCity', '').strip()} {home.get('teamName', '').strip()}".strip()
        if not away_name or not home_name:
            continue
        games.append(
            GameInfo(
                sport="nba",
                game_pk=int(item.get("gameId", 0)),
                date=board.get("gameDate", date_str),
                game_time=item.get("gameTimeUTC", ""),
                game_state=item.get("gameStatusText", ""),
                away_name=away_name,
                home_name=home_name,
                away_code=compact_token(away.get("teamTricode", canonical_team_code(away_name))),
                home_code=compact_token(home.get("teamTricode", canonical_team_code(home_name))),
                away_detail=f"{away.get('wins', 0)}-{away.get('losses', 0)}",
                home_detail=f"{home.get('wins', 0)}-{home.get('losses', 0)}",
            )
        )
    return games


def fetch_nba_schedule_from_kalshi(date_str: str) -> list[GameInfo]:
    if requests is None:
        raise RuntimeError("Kalshi NBA schedule loading requires requests")

    target_code = datetime.strptime(date_str, "%Y-%m-%d").strftime("%y%b%d").upper()
    response = requests.get(
        f"{KALSHI_API_BASE}/events",
        params={"limit": 200, "series_ticker": "KXNBAGAME"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    games = []

    for event in payload.get("events", []):
        ticker = event.get("event_ticker", "")
        if target_code not in ticker:
            continue
        suffix = ticker.split("-")[-1]
        if len(suffix) < 13:
            continue
        matchup = suffix[7:]
        away_code = matchup[:3]
        home_code = matchup[3:6]
        away_name = NBA_CODE_TO_NAME.get(away_code, away_code)
        home_name = NBA_CODE_TO_NAME.get(home_code, home_code)
        games.append(
            GameInfo(
                sport="nba",
                game_pk=0,
                date=date_str,
                game_time=f"{date_str}T00:00:00Z",
                game_state="Scheduled",
                away_name=away_name,
                home_name=home_name,
                away_code=away_code,
                home_code=home_code,
                away_detail="TBD",
                home_detail="TBD",
            )
        )

    return games


def fetch_schedule_for_sport(date_str: str, sport: str) -> list[GameInfo]:
    if sport == "mlb":
        return fetch_mlb_schedule(date_str)
    if sport == "nba":
        return fetch_nba_schedule(date_str)
    raise ValueError(f"Unsupported sport: {sport}")


def fetch_kalshi_sport_markets(
    target_date: datetime, sport: str, max_pages: int = 25, page_limit: int = 1000
) -> list[dict]:
    if requests is None:
        raise RuntimeError("Live Kalshi loading requires requests")

    target_code = target_date.strftime("%y%b%d").upper()
    ticker_prefix = "KXMLB" if sport == "mlb" else "KXNBA"
    markets = []
    cursor = None
    seen = set()

    for _ in range(max_pages):
        params = {"limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(f"{KALSHI_API_BASE}/markets", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for market in payload.get("markets", []):
            event_ticker = market.get("event_ticker", "")
            ticker = market.get("ticker", "")
            if not (event_ticker.startswith(ticker_prefix) or ticker.startswith(ticker_prefix)):
                continue
            if target_code not in event_ticker and target_code not in ticker:
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            markets.append(market)

        cursor = payload.get("cursor")
        if not cursor:
            break

    return markets


def kalshi_game_token(game: GameInfo) -> str:
    if game.sport == "nba":
        dt = datetime.strptime(game.date, "%Y-%m-%d")
        return f"{dt.strftime('%y%b%d').upper()}{game.away_code}{game.home_code}"
    dt = datetime.fromisoformat(game.game_time.replace("Z", "+00:00")).astimezone(EASTERN_TZ)
    return f"{dt.strftime('%y%b%d').upper()}{dt.strftime('%H%M')}{game.away_code}{game.home_code}"


def fetch_kalshi_event(event_ticker: str) -> list[dict]:
    if requests is None:
        raise RuntimeError("Kalshi event lookup requires requests")

    response = requests.get(f"{KALSHI_API_BASE}/events/{event_ticker}", timeout=20)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    return payload.get("markets", [])


def fetch_kalshi_markets_for_game(game: GameInfo) -> list[dict]:
    token = kalshi_game_token(game)
    if game.sport == "mlb":
        event_prefixes = [
            "KXMLBGAME",
            "KXMLBTOTAL",
            "KXMLBSPREAD",
            "KXMLBHR",
            "KXMLBHIT",
            "KXMLBTB",
            "KXMLBHRR",
        ]
    elif game.sport == "nba":
        event_prefixes = [
            "KXNBAGAME",
            "KXNBATOTAL",
            "KXNBAPTS",
            "KXNBAREB",
            "KXNBAAST",
        ]
    else:
        return []

    markets = []
    seen = set()
    for prefix in event_prefixes:
        for market in fetch_kalshi_event(f"{prefix}-{token}"):
            ticker = market.get("ticker", "")
            if ticker in seen:
                continue
            seen.add(ticker)
            markets.append(market)
    return markets


def build_game_lookup(games: list[GameInfo]) -> tuple[dict[str, GameInfo], list[GameInfo]]:
    token_lookup = {}
    for game in games:
        for token in game.kalshi_tokens:
            token_lookup[token] = game
    return token_lookup, games


def extract_matchup_token(event_ticker: str, ticker: str) -> str | None:
    for candidate in (event_ticker, ticker):
        match = DATE_CODE_RE.search(candidate)
        if not match:
            continue
        suffix = match.group(3)
        if suffix:
            return compact_token(suffix)
    return None


def match_market_to_game(market: dict, games: list[GameInfo], token_lookup: dict[str, GameInfo]) -> GameInfo | None:
    token = extract_matchup_token(market.get("event_ticker", ""), market.get("ticker", ""))
    if token and token in token_lookup:
        return token_lookup[token]

    title = normalize_name(market.get("title", ""))
    for game in games:
        away_key, home_key = game.title_key
        if away_key and home_key and away_key in title and home_key in title:
            return game
    return None


def clean_market_team_label(value: str) -> str:
    value = value.replace("Chicago WS", "CWS")
    value = value.replace("Chicago C", "CHC")
    value = value.replace("LA Angels", "LAA")
    value = value.replace("LA Dodgers", "LAD")
    value = value.replace("Tampa Bay", "TB")
    words = value.split()
    if not words:
        return value
    normalized = compact_token("".join(words))
    for aliases in TEAM_CODES.values():
        for alias in aliases:
            if compact_token(alias) == normalized:
                return compact_token(aliases[0])
    return value.strip()


def total_line_bounds(sport: str) -> tuple[float, float]:
    if sport == "mlb":
        return (5.5, 11.5)
    if sport == "nba":
        return (190.5, 260.5)
    return (0.0, 1e9)


def extract_total_line_value(label: str) -> float | None:
    match = re.search(r"[OU](\d+(?:\.\d+)?)", label)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def total_leg_in_bounds(leg: Leg) -> bool:
    if leg.category != "total":
        return True
    line = extract_total_line_value(leg.label)
    if line is None:
        return False
    lower, upper = total_line_bounds(leg.sport)
    return lower <= line <= upper


def fetch_mlb_team_code_lookup(season: int) -> dict[int, str]:
    if requests is None:
        return {}
    response = requests.get(
        MLB_TEAMS_URL,
        params={"sportId": 1, "season": season},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    lookup = {}
    for team in payload.get("teams", []):
        team_id = team.get("id")
        abbreviation = team.get("abbreviation")
        if team_id and abbreviation:
            lookup[int(team_id)] = compact_token(str(abbreviation))
    return lookup


def parse_split_record(records: dict, split_type: str) -> tuple[int, int]:
    for record in records.get("splitRecords", []):
        if record.get("type") == split_type:
            return int(record.get("wins", 0)), int(record.get("losses", 0))
    return 0, 0


def fetch_live_mlb_team_form(date_str: str) -> dict[str, dict]:
    if requests is None:
        return {}
    season = int(date_str[:4])
    code_lookup = fetch_mlb_team_code_lookup(season)
    response = requests.get(
        MLB_STANDINGS_URL,
        params={"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    team_form = {}
    for record_group in payload.get("records", []):
        for record in record_group.get("teamRecords", []):
            team_id = record.get("team", {}).get("id")
            code = code_lookup.get(int(team_id)) if team_id is not None else None
            if not code:
                continue
            games_played = max(int(record.get("gamesPlayed", 0)), 1)
            runs_scored = float(record.get("runsScored", 0.0))
            runs_allowed = float(record.get("runsAllowed", 0.0))
            run_diff = float(record.get("runDifferential", 0.0))
            home_wins, home_losses = parse_split_record(record.get("records", {}), "home")
            away_wins, away_losses = parse_split_record(record.get("records", {}), "away")
            last_ten_wins, last_ten_losses = parse_split_record(record.get("records", {}), "lastTen")
            last_ten_games = max(last_ten_wins + last_ten_losses, 1)
            team_form[code] = {
                "win_pct": float(record.get("winningPercentage", 0.5) or 0.5),
                "run_diff_pg": run_diff / games_played,
                "runs_scored_pg": runs_scored / games_played,
                "runs_allowed_pg": runs_allowed / games_played,
                "recent_win_pct": last_ten_wins / last_ten_games,
                "home_win_pct": home_wins / max(home_wins + home_losses, 1),
                "away_win_pct": away_wins / max(away_wins + away_losses, 1),
            }
    return team_form


def fetch_live_nba_team_form(date_str: str) -> dict[str, dict]:
    games = fetch_nba_schedule(date_str)
    team_form = {}
    for game in games:
        for team_code, detail in ((game.away_code, game.away_detail), (game.home_code, game.home_detail)):
            if team_code in team_form:
                continue
            wins = 0
            losses = 0
            match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", detail or "")
            if match:
                wins = int(match.group(1))
                losses = int(match.group(2))
            games_played = max(wins + losses, 1)
            win_pct = wins / games_played if games_played else 0.5
            net_rating_proxy = (win_pct - 0.5) * 24.0
            team_form[team_code] = {
                "win_pct": win_pct,
                "games_played": games_played,
                "net_rating_proxy": net_rating_proxy,
            }
    return team_form


def load_team_form_snapshot(date_str: str, sport: str) -> dict[str, dict]:
    if sport == "mlb":
        cached = load_cached_payload(
            source="mlb_statsapi",
            sport="mlb",
            entity_type="team_form",
            entity_key="daily",
            as_of_date=date_str,
            max_age_hours=DEFAULT_BASELINE_CACHE_MAX_AGE_HOURS,
        )
        if cached is not None:
            return cached
        payload = fetch_live_mlb_team_form(date_str)
        store_cached_payload(
            source="mlb_statsapi",
            sport="mlb",
            entity_type="team_form",
            entity_key="daily",
            as_of_date=date_str,
            payload=payload,
            is_volatile=False,
        )
        return payload

    if sport == "nba":
        cached = load_cached_payload(
            source="nba_scoreboard",
            sport="nba",
            entity_type="team_form",
            entity_key="daily",
            as_of_date=date_str,
            max_age_hours=DEFAULT_BASELINE_CACHE_MAX_AGE_HOURS,
        )
        if cached is not None:
            return cached
        payload = fetch_live_nba_team_form(date_str)
        store_cached_payload(
            source="nba_scoreboard",
            sport="nba",
            entity_type="team_form",
            entity_key="daily",
            as_of_date=date_str,
            payload=payload,
            is_volatile=False,
        )
        return payload

    raise ValueError(f"Unsupported sport for team-form snapshot: {sport}")


def load_game_context_snapshot(date_str: str, sport: str, matchup: str) -> dict | None:
    if sport == "mlb":
        return load_cached_payload(
            source="mlb_refresh",
            sport="mlb",
            entity_type="game_context",
            entity_key=matchup,
            as_of_date=date_str,
            max_age_hours=DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS,
        )
    if sport == "nba":
        return load_cached_payload(
            source="nba_refresh",
            sport="nba",
            entity_type="game_context",
            entity_key=matchup,
            as_of_date=date_str,
            max_age_hours=DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS,
        )
    return None


def load_matchup_profile_snapshot(date_str: str, matchup: str) -> dict | None:
    return load_cached_payload(
        source="mlb_refresh",
        sport="mlb",
        entity_type="matchup_profile",
        entity_key=matchup,
        as_of_date=date_str,
        max_age_hours=DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS,
    )


def load_nba_matchup_profile_snapshot(date_str: str, matchup: str) -> dict | None:
    return load_cached_payload(
        source="nba_refresh",
        sport="nba",
        entity_type="matchup_profile",
        entity_key=matchup,
        as_of_date=date_str,
        max_age_hours=DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS,
    )


def direct_total_line(leg: Leg) -> float | None:
    return extract_total_line_value(leg.label)


def live_mlb_residual_adjustment(leg: Leg, team_form: dict[str, dict]) -> float:
    if leg.sport != "mlb":
        return 0.0
    away_code, home_code = leg.game.split("@")
    away = team_form.get(away_code)
    home = team_form.get(home_code)
    if away is None or home is None:
        return 0.0

    if leg.category == "ml":
        selected = leg.label.split()[0]
        selected_state = away if selected == away_code else home
        opponent_state = home if selected == away_code else away
        venue_edge = (
            float(selected_state["home_win_pct"]) - float(opponent_state["away_win_pct"])
            if selected == home_code
            else float(selected_state["away_win_pct"]) - float(opponent_state["home_win_pct"])
        )
        adjustment = (
            0.05 * np.tanh((float(selected_state["win_pct"]) - float(opponent_state["win_pct"])) / 0.10)
            + 0.04 * np.tanh((float(selected_state["run_diff_pg"]) - float(opponent_state["run_diff_pg"])) / 1.5)
            + 0.025 * np.tanh((float(selected_state["recent_win_pct"]) - float(opponent_state["recent_win_pct"])) / 0.12)
            + 0.015 * np.tanh(venue_edge / 0.12)
        )
        return float(adjustment)

    if leg.category == "total":
        line = direct_total_line(leg)
        if line is None:
            return 0.0
        projected_total = (
            float(away["runs_scored_pg"])
            + float(home["runs_scored_pg"])
            + float(away["runs_allowed_pg"])
            + float(home["runs_allowed_pg"])
        ) / 2.0
        recent_total = (
            float(away["runs_scored_pg"])
            + float(away["runs_allowed_pg"])
            + float(home["runs_scored_pg"])
            + float(home["runs_allowed_pg"])
        ) / 2.0
        total_edge = projected_total - line
        environment_edge = recent_total - (LEAGUE_AVG_RUNS * 2.0)
        adjustment = (
            0.05 * np.tanh(total_edge / 1.25)
            + 0.02 * np.tanh(environment_edge / 2.0)
        )
        if is_under_leg(leg):
            adjustment *= -1.0
        return float(adjustment)

    return 0.0


def expected_mlb_runs(
    away_code: str,
    home_code: str,
    team_form: dict[str, dict],
    game_context: dict | None = None,
) -> tuple[float, float]:
    away = team_form.get(away_code)
    home = team_form.get(home_code)
    if away is None or home is None:
        return LEAGUE_AVG_RUNS, LEAGUE_AVG_RUNS
    away_mean = (
        float(away["runs_scored_pg"]) + float(home["runs_allowed_pg"])
    ) / 2.0
    home_mean = (
        float(home["runs_scored_pg"]) + float(away["runs_allowed_pg"])
    ) / 2.0
    away_recent = float(away["recent_win_pct"]) - 0.5
    home_recent = float(home["recent_win_pct"]) - 0.5
    away_mean += 0.35 * away_recent - 0.20 * home_recent
    home_mean += 0.35 * home_recent - 0.20 * away_recent + 0.15
    if game_context:
        weather = game_context.get("weather") or {}
        temperature_f = weather.get("temperature_f")
        wind_speed_mph = weather.get("wind_speed_mph")
        if temperature_f is not None:
            if temperature_f >= 82:
                away_mean += 0.20
                home_mean += 0.20
            elif temperature_f <= 52:
                away_mean -= 0.20
                home_mean -= 0.20
        if wind_speed_mph is not None and wind_speed_mph >= 12:
            away_mean += 0.12
            home_mean += 0.12
        lineup_status = game_context.get("lineup_status") or {}
        if not (lineup_status.get("away_confirmed") and lineup_status.get("home_confirmed")):
            away_mean = (away_mean * 0.75) + (LEAGUE_AVG_RUNS * 0.25)
            home_mean = (home_mean * 0.75) + (LEAGUE_AVG_RUNS * 0.25)
        availability = game_context.get("availability") or {}
        bullpen = game_context.get("bullpen") or {}
        away_unavailable = availability.get("away", {}).get("unavailable_players", [])
        home_unavailable = availability.get("home", {}).get("unavailable_players", [])
        away_lineup = set(game_context.get("lineups", {}).get("away", []))
        home_lineup = set(game_context.get("lineups", {}).get("home", []))
        away_missing_core = sum(1 for player in away_unavailable if player.get("player_name") in away_lineup)
        home_missing_core = sum(1 for player in home_unavailable if player.get("player_name") in home_lineup)
        away_mean -= 0.20 * away_missing_core
        home_mean -= 0.20 * home_missing_core
        away_bullpen_fatigue = float(bullpen.get("away", {}).get("fatigue_score", 0.0) or 0.0)
        home_bullpen_fatigue = float(bullpen.get("home", {}).get("fatigue_score", 0.0) or 0.0)
        home_mean += 0.08 * away_bullpen_fatigue
        away_mean += 0.08 * home_bullpen_fatigue
        away_pitcher = (game_context.get("probable_pitchers") or {}).get("away", {}).get("fullName")
        home_pitcher = (game_context.get("probable_pitchers") or {}).get("home", {}).get("fullName")
        away_pitcher_unavailable = any(
            player.get("player_name") == away_pitcher
            for player in away_unavailable
        )
        home_pitcher_unavailable = any(
            player.get("player_name") == home_pitcher
            for player in home_unavailable
        )
        if away_pitcher and away_pitcher_unavailable:
            home_mean += 0.45
        if home_pitcher and home_pitcher_unavailable:
            away_mean += 0.45
    return max(2.5, away_mean), max(2.5, home_mean)


def nba_availability_penalty(entries: list[dict]) -> float:
    penalty = 0.0
    for entry in entries:
        status = str(entry.get("status", "")).strip().lower()
        if status == "out":
            penalty += 1.75
        elif status == "doubtful":
            penalty += 1.0
        elif status == "questionable":
            penalty += 0.55
        elif status == "probable":
            penalty += 0.15
    return penalty


def expected_nba_points(
    away_code: str,
    home_code: str,
    team_form: dict[str, dict],
    game_context: dict | None = None,
) -> tuple[float, float]:
    away = team_form.get(away_code)
    home = team_form.get(home_code)
    if away is None or home is None:
        return NBA_BASELINE_POINTS - 1.0, NBA_BASELINE_POINTS + 1.0
    away_strength = float(away["net_rating_proxy"])
    home_strength = float(home["net_rating_proxy"])
    away_mean = NBA_BASELINE_POINTS + (away_strength * 0.8) - (home_strength * 0.45) - NBA_HOME_COURT_POINTS
    home_mean = NBA_BASELINE_POINTS + (home_strength * 0.8) - (away_strength * 0.45) + NBA_HOME_COURT_POINTS
    if game_context:
        availability = game_context.get("availability") or {}
        away_penalty = nba_availability_penalty(availability.get("away", []))
        home_penalty = nba_availability_penalty(availability.get("home", []))
        away_mean -= away_penalty
        home_mean -= home_penalty
        if not availability.get("away_submitted", True):
            away_mean = (away_mean * 0.7) + (NBA_BASELINE_POINTS * 0.3)
        if not availability.get("home_submitted", True):
            home_mean = (home_mean * 0.7) + (NBA_BASELINE_POINTS * 0.3)
    return float(np.clip(away_mean, 96.0, 132.0)), float(np.clip(home_mean, 96.0, 132.0))


def simulate_game_distributions(
    sport: str,
    away_code: str,
    home_code: str,
    date_str: str,
    n_sims: int = SIMULATION_RUNS,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(abs(hash((sport, away_code, home_code, date_str))) % (2**32))
    if sport == "mlb":
        team_form = load_team_form_snapshot(date_str, "mlb")
        game_context = load_game_context_snapshot(date_str, "mlb", f"{away_code}@{home_code}")
        away_mean, home_mean = expected_mlb_runs(away_code, home_code, team_form, game_context)
        away_scores = rng.poisson(lam=away_mean, size=n_sims)
        home_scores = rng.poisson(lam=home_mean, size=n_sims)
        tie_mask = away_scores == home_scores
        while np.any(tie_mask):
            away_scores[tie_mask] += rng.poisson(lam=0.6, size=int(np.sum(tie_mask)))
            home_scores[tie_mask] += rng.poisson(lam=0.6, size=int(np.sum(tie_mask)))
            tie_mask = away_scores == home_scores
        return away_scores.astype(np.int16), home_scores.astype(np.int16)

    if sport == "nba":
        team_form = load_team_form_snapshot(date_str, "nba")
        game_context = load_game_context_snapshot(date_str, "nba", f"{away_code}@{home_code}")
        away_mean, home_mean = expected_nba_points(away_code, home_code, team_form, game_context)
        away_scores = np.rint(rng.normal(loc=away_mean, scale=11.5, size=n_sims)).astype(np.int16)
        home_scores = np.rint(rng.normal(loc=home_mean, scale=11.0, size=n_sims)).astype(np.int16)
        away_scores = np.clip(away_scores, 75, 170)
        home_scores = np.clip(home_scores, 75, 170)
        tie_mask = away_scores == home_scores
        while np.any(tie_mask):
            away_scores[tie_mask] += rng.integers(0, 8, size=int(np.sum(tie_mask)), endpoint=False)
            home_scores[tie_mask] += rng.integers(0, 8, size=int(np.sum(tie_mask)), endpoint=False)
            tie_mask = away_scores == home_scores
        return away_scores, home_scores

    raise ValueError(f"Unsupported sport for simulation: {sport}")


def is_pregame_game(game: GameInfo) -> bool:
    state = (game.game_state or "").strip().lower()
    if not state:
        return True

    blocked_terms = {
        "final",
        "final/so",
        "completed",
        "game over",
        "postponed",
        "cancelled",
        "canceled",
        "suspended",
        "in progress",
        "mid 1st",
        "mid 2nd",
        "mid 3rd",
        "mid 4th",
        "mid 5th",
        "mid 6th",
        "mid 7th",
        "mid 8th",
        "mid 9th",
        "top 1st",
        "top 2nd",
        "top 3rd",
        "top 4th",
        "top 5th",
        "top 6th",
        "top 7th",
        "top 8th",
        "top 9th",
        "bot 1st",
        "bot 2nd",
        "bot 3rd",
        "bot 4th",
        "bot 5th",
        "bot 6th",
        "bot 7th",
        "bot 8th",
        "bot 9th",
        "halftime",
    }
    if state in blocked_terms:
        return False

    live_fragments = (
        "quarter",
        "1st",
        "2nd",
        "3rd",
        "4th",
        "inning",
        "live",
        "delayed",
        "rain delay",
    )
    return not any(fragment in state for fragment in live_fragments)


def market_is_actionable(market: dict) -> bool:
    status = str(market.get("status", "")).strip().lower()
    if status in {"finalized", "settled", "closed", "expired"}:
        return False
    result = str(market.get("result", "")).strip().lower()
    if result in {"yes", "no"}:
        return False
    return True


def classify_market(market: dict, game: GameInfo) -> Leg | None:
    implied_prob = parse_market_price(market)
    if implied_prob is None:
        return None

    title = market.get("title", "")
    yes_subtitle = market.get("yes_sub_title") or ""
    event_ticker = market.get("event_ticker", "")
    title_lower = title.lower()
    note = game.matchup_note

    if "winner?" in title_lower or event_ticker.endswith("GAME") or "game winner" in title_lower:
        team = clean_market_team_label(yes_subtitle or title.replace(" Winner?", ""))
        return Leg(-1, f"{team} ML", "ml", game.matchup, implied_prob, note, game.sport)

    if game.sport == "mlb":
        if event_ticker.startswith("KXMLBHR-") or "home runs?" in title_lower:
            if market.get("floor_strike") not in (None, 0.5):
                return None
            label = structured_threshold_label(market, "HR", "home runs")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)
            player = title.split(":", 1)[0].strip()
            return Leg(-1, f"{player} O 1 HR", "prop", game.matchup, implied_prob, note, game.sport)

        if event_ticker.startswith("KXMLBHIT-") or " hits?" in title_lower:
            label = structured_threshold_label(market, "H", "hits")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)

        if "strikeout" in title_lower or "strikeouts" in title_lower or event_ticker.startswith("KXMLBSO"):
            label = structured_threshold_label(market, "K", "strikeouts")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)
            player = title.split(":")[0].strip()
            return Leg(-1, f"{player} O 1 K", "prop", game.matchup, implied_prob, note, game.sport)

        if event_ticker.startswith("KXMLBTOTAL") or "total runs?" in title_lower:
            strike = market.get("floor_strike")
            if strike is None:
                raw = yes_subtitle or title
                cleaned = raw.replace("Over", "O").replace("over", "O").strip()
                leg = Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note, game.sport)
                return leg if total_leg_in_bounds(leg) else None
            leg = Leg(-1, f"{game.matchup} O{strike + 0.0:g}", "total", game.matchup, implied_prob, note, game.sport)
            return leg if total_leg_in_bounds(leg) else None

        if event_ticker.startswith("KXMLBSPREAD") or "wins by over" in title_lower:
            return None

    if game.sport == "nba":
        if event_ticker.startswith("KXNBAPTS") or " points?" in title_lower:
            label = structured_threshold_label(market, "PTS", "Points")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)
        if event_ticker.startswith("KXNBAREB") or " rebounds?" in title_lower:
            label = structured_threshold_label(market, "REB", "Rebounds")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)
        if event_ticker.startswith("KXNBAAST") or " assists?" in title_lower:
            label = structured_threshold_label(market, "AST", "Assists")
            if label:
                return Leg(-1, label, "prop", game.matchup, implied_prob, note, game.sport)
        if "total points?" in title_lower or event_ticker.startswith("KXNBATOTAL") or event_ticker.startswith("KXNBAOU"):
            raw = yes_subtitle or title
            if "over" in raw.lower():
                cleaned = raw.replace("Over", "O").replace("over", "O").strip()
                leg = Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note, game.sport)
                return leg if total_leg_in_bounds(leg) else None
            if "under" in raw.lower():
                cleaned = raw.replace("Under", "U").replace("under", "U").strip()
                leg = Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note, game.sport)
                return leg if total_leg_in_bounds(leg) else None

    return None


def dedupe_legs(legs: list[Leg]) -> list[Leg]:
    deduped = {}
    for leg in legs:
        if leg.category == "total":
            if not total_leg_in_bounds(leg):
                continue
            key = (leg.game, leg.category)
        elif leg.category == "spread":
            key = (leg.game, leg.label)
        elif leg.category == "prop":
            if " O " in leg.label:
                player, metric = leg.label.split(" O ", 1)
            elif leg.label.endswith(" HR"):
                player = leg.label[: -3]
                metric = "HR"
            else:
                player = leg.label
                metric = leg.label
            metric = re.sub(r"^\d+\s+", "", metric)
            key = (leg.game, player.strip(), metric.strip())
        else:
            key = (leg.game, leg.label)

        existing = deduped.get(key)
        current_distance = abs(leg.implied_prob - 0.5)
        existing_distance = abs(existing.implied_prob - 0.5) if existing else None

        if existing is None or current_distance < existing_distance:
            deduped[key] = leg
    ordered = sorted(deduped.values(), key=lambda leg: (leg.game, leg.category, leg.label))
    return [replace(leg, id=index) for index, leg in enumerate(ordered)]


def load_live_legs(date_str: str, sports: list[str], kalshi_pages: int) -> tuple[list[Leg], dict]:
    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    all_games = []
    all_markets = []
    legs = []

    for sport in sports:
        games = [game for game in fetch_schedule_for_sport(date_str, sport) if is_pregame_game(game)]
        token_lookup, game_list = build_game_lookup(games)
        if sport in {"mlb", "nba"}:
            markets = []
            for game in games:
                markets.extend(
                    market for market in fetch_kalshi_markets_for_game(game) if market_is_actionable(market)
                )
        else:
            markets = [
                market
                for market in fetch_kalshi_sport_markets(target_date, sport=sport, max_pages=kalshi_pages)
                if market_is_actionable(market)
            ]
        all_games.extend(games)
        all_markets.extend(markets)

        for market in markets:
            game = match_market_to_game(market, game_list, token_lookup)
            if game is None:
                continue
            leg = classify_market(market, game)
            if leg is not None:
                legs.append(leg)

    legs = dedupe_legs(legs)
    meta = {
        "date": date_str,
        "sports": ",".join(sports),
        "games": len(all_games),
        "kalshi_markets": len(all_markets),
        "recognized_legs": len(legs),
    }
    return legs, meta


def load_cached_recognized_legs(date_str: str, sports: list[str]) -> tuple[list[Leg], dict] | None:
    cached_legs: list[Leg] = []
    cached_meta: dict | None = None
    for sport in sports:
        payload = load_cached_payload(
            source="kalshi",
            sport=sport,
            entity_type="recognized_legs",
            entity_key="daily",
            as_of_date=date_str,
            max_age_hours=DEFAULT_VOLATILE_CACHE_MAX_AGE_HOURS,
        )
        if payload is None:
            continue
        cached_meta = payload.get("meta", {}) or {}
        for item in payload.get("legs", []):
            cached_legs.append(
                Leg(
                    id=len(cached_legs),
                    label=item["label"],
                    category=item["category"],
                    game=item["game"],
                    implied_prob=float(item["implied_prob"]),
                    notes=item.get("notes", ""),
                    sport=item.get("sport", sport),
                )
            )
    if not cached_legs:
        return None
    meta = {
        "date": date_str,
        "sports": ",".join(sports),
        "games": cached_meta.get("games", 0) if cached_meta else 0,
        "kalshi_markets": cached_meta.get("kalshi_markets", 0) if cached_meta else 0,
        "recognized_legs": len(cached_legs),
        "source": "cache",
    }
    return cached_legs, meta


def load_legs(date_str: str, mode: str, sports: list[str], kalshi_pages: int) -> tuple[list[Leg], str, dict]:
    cached = load_cached_recognized_legs(date_str, sports)
    if cached is not None:
        cached_legs, meta = cached
        return cached_legs, "cached", meta
    try:
        live_legs, meta = load_live_legs(date_str, sports=sports, kalshi_pages=kalshi_pages)
        if live_legs:
            return live_legs, "live", meta
        raise RuntimeError("No recognizable Kalshi sports legs found for the target date")
    except Exception as exc:
        fallback_cached = load_cached_recognized_legs(date_str, sports)
        if fallback_cached is not None:
            cached_legs, meta = fallback_cached
            meta["fallback_reason"] = str(exc)
            return cached_legs, "cached", meta
        raise RuntimeError(
            f"Live loader unavailable for {date_str}: {exc}. "
            "No fallback slate is used."
        )


def is_over_leg(leg: Leg) -> bool:
    return leg.category == "total" and " O" in f" {leg.label}"


def is_under_leg(leg: Leg) -> bool:
    return leg.category == "total" and " U" in f" {leg.label}"


def is_k_prop(leg: Leg) -> bool:
    return leg.category == "prop" and "O K" in leg.label


def is_hr_prop(leg: Leg) -> bool:
    return leg.category == "prop" and leg.label.endswith("HR")


def compute_biases(legs: list[Leg]) -> np.ndarray:
    biases = np.zeros(len(legs), dtype=np.float64)
    for leg in legs:
        if leg.category == "ml":
            value = 0.5 - leg.implied_prob
            biases[leg.id] = -0.15 * value
        elif leg.category == "total":
            biases[leg.id] = 0.0
        elif is_hr_prop(leg):
            biases[leg.id] = 0.2
        else:
            biases[leg.id] = -0.12
    return biases


def build_coupling_matrix(legs: list[Leg]) -> np.ndarray:
    coupling = np.zeros((len(legs), len(legs)), dtype=np.float64)
    games = {}
    for leg in legs:
        games.setdefault(leg.game, []).append(leg)

    for game_legs in games.values():
        mls = [leg for leg in game_legs if leg.category == "ml"]
        if len(mls) >= 2:
            for idx, first in enumerate(mls):
                for second in mls[idx + 1 :]:
                    coupling[first.id, second.id] = 2.5
                    coupling[second.id, first.id] = 2.5

        overs = [leg for leg in game_legs if is_over_leg(leg)]
        unders = [leg for leg in game_legs if is_under_leg(leg)]
        for over in overs:
            for under in unders:
                coupling[over.id, under.id] = 2.5
                coupling[under.id, over.id] = 2.5

    k_props = [leg for leg in legs if is_k_prop(leg)]
    for idx, first in enumerate(k_props):
        for second in k_props[idx + 1 :]:
            coupling[first.id, second.id] = -0.3
            coupling[second.id, first.id] = -0.3

    unders = [leg for leg in legs if is_under_leg(leg)]
    for idx, first in enumerate(unders):
        for second in unders[idx + 1 :]:
            coupling[first.id, second.id] = -0.15
            coupling[second.id, first.id] = -0.15

    for game_legs in games.values():
        k_over = [leg for leg in game_legs if is_k_prop(leg)]
        game_unders = [leg for leg in game_legs if is_under_leg(leg)]
        for k_leg in k_over:
            for under_leg in game_unders:
                coupling[k_leg.id, under_leg.id] = -0.4
                coupling[under_leg.id, k_leg.id] = -0.4

        mls = [leg for leg in game_legs if leg.category == "ml"]
        if mls and k_over:
            favorite = max(mls, key=lambda leg: leg.implied_prob)
            for k_leg in k_over:
                coupling[favorite.id, k_leg.id] = -0.25
                coupling[k_leg.id, favorite.id] = -0.25

    hr_props = [leg for leg in legs if is_hr_prop(leg)]
    for idx, first in enumerate(hr_props):
        for second in hr_props[idx + 1 :]:
            coupling[first.id, second.id] = 0.2
            coupling[second.id, first.id] = 0.2

    home_dogs = [
        leg for leg in legs if leg.category == "ml" and 0.25 < leg.implied_prob < 0.45
    ]
    for idx, first in enumerate(home_dogs):
        for second in home_dogs[idx + 1 :]:
            coupling[first.id, second.id] = -0.2
            coupling[second.id, first.id] = -0.2

    return coupling


def gibbs_sample(
    biases: np.ndarray,
    coupling: np.ndarray,
    beta: float,
    qrng: QuantumEntropySource,
    n_warmup: int = 300,
    n_samples: int = 1500,
    thin: int = 3,
) -> np.ndarray:
    n_spins = len(biases)
    initial = qrng.next_floats(n_spins)
    spins = np.where(initial > 0.5, 1.0, -1.0)

    samples = np.zeros((n_samples, n_spins), dtype=np.float64)
    total_steps = n_warmup + (n_samples * thin)
    sample_index = 0

    for step in range(total_steps):
        for idx in range(n_spins):
            local_field = -(biases[idx] + np.dot(coupling[idx], spins))
            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
            q = qrng.next_float()
            spins[idx] = 1.0 if q < p_up else -1.0

        if step >= n_warmup and (step - n_warmup) % thin == 0:
            samples[sample_index] = spins.copy()
            sample_index += 1

    return samples


def incompatible_pair(a: Leg, b: Leg) -> bool:
    if a.game != b.game:
        return False
    if a.category == "ml" and b.category == "ml":
        return True
    if a.category == "total" and b.category == "total":
        return True
    return False


def standalone_leg_score(leg: Leg, activation_value: float) -> float:
    edge = activation_value - leg.implied_prob
    score = (activation_value * 0.9) + (edge * 1.8)
    if is_hr_prop(leg):
        score -= 0.08
    return score


def candidate_leg_score(
    candidate_idx: int,
    parlay: list[int],
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    available_sports: set[str],
) -> float:
    candidate = legs[candidate_idx]
    activation_value = float(activation[candidate_idx])
    edge = activation_value - candidate.implied_prob
    score = (activation_value * 0.9) + (edge * 1.8)

    if parlay:
        pair_fit = []
        for pick in parlay:
            pair_fit.append(
                float(co_activation[candidate_idx, pick] - (activation[candidate_idx] * activation[pick]))
            )
        score += (sum(pair_fit) / len(pair_fit)) * 1.5

    same_game_count = sum(1 for pick in parlay if legs[pick].game == candidate.game)
    if same_game_count:
        score -= 0.22 * same_game_count

    same_category_count = sum(1 for pick in parlay if legs[pick].category == candidate.category)
    if candidate.category == "ml":
        score -= 0.18 * same_category_count
    elif candidate.category == "total":
        score -= 0.12 * same_category_count
    elif candidate.category == "prop" and is_hr_prop(candidate):
        score -= 0.08 * same_category_count

    if len(available_sports) > 1:
        sport_counts = {}
        for pick in parlay:
            sport_counts[legs[pick].sport] = sport_counts.get(legs[pick].sport, 0) + 1
        candidate_count = sport_counts.get(candidate.sport, 0)
        other_count = sum(sport_counts.values()) - candidate_count
        if candidate_count < other_count:
            score += 0.26
        elif candidate_count > other_count:
            score -= 0.16

    return score


def leg_allowed_for_parlay_size(
    leg: Leg,
    activation_value: float,
    requested_size: int,
) -> bool:
    if requested_size == 3 and activation_value < 0.54:
        return False
    if requested_size == 4 and activation_value < 0.52:
        return False
    if requested_size <= 4:
        if is_hr_prop(leg):
            return activation_value >= 0.62
        if leg.category == "prop":
            return activation_value >= 0.54
    if requested_size == 3 and leg.category == "prop" and activation_value < 0.58:
        return False
    return True


def candidate_pool_limit(size: int) -> int:
    if size <= 3:
        return 18
    if size == 4:
        return 22
    return 26


def beam_width_for_size(size: int) -> int:
    if size <= 3:
        return 32
    if size == 4:
        return 40
    return 56


def state_score_profile(target_size: int) -> dict[str, float]:
    return {
        "activation": 1.15,
        "edge": 2.6,
        "pair_fit": 2.2,
        "same_game": 0.24,
        "future": 0.35,
        "single_sport": 0.22,
        "sport_gap": 0.06,
        "ml_penalty": 0.28,
        "total_penalty": 0.10,
        "prop_penalty": 0.12,
        "hr_short": 0.18,
        "hr_long": 0.12,
    }


def partial_state_score(
    parlay: list[int],
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    available_sports: set[str],
    target_size: int,
) -> float:
    if not parlay:
        return -np.inf

    profile = state_score_profile(target_size)
    parlay_set = set(parlay)
    activations = [float(activation[idx]) for idx in parlay]
    edges = [float(activation[idx] - legs[idx].implied_prob) for idx in parlay]

    pair_terms = []
    same_game_penalty = 0.0
    for i, first in enumerate(parlay):
        for second in parlay[i + 1 :]:
            fit = float(co_activation[first, second] - (activation[first] * activation[second]))
            pair_terms.append(fit)
            if legs[first].game == legs[second].game:
                same_game_penalty += profile["same_game"]

    mean_activation = sum(activations) / len(activations)
    mean_edge = sum(edges) / len(edges)
    mean_pair_fit = (sum(pair_terms) / len(pair_terms)) if pair_terms else 0.0

    score = (
        (mean_activation * profile["activation"])
        + (mean_edge * profile["edge"])
        + (mean_pair_fit * profile["pair_fit"])
    )

    category_counts = {}
    sport_counts = {}
    for idx in parlay:
        category = legs[idx].category
        category_counts[category] = category_counts.get(category, 0) + 1
        sport = legs[idx].sport
        sport_counts[sport] = sport_counts.get(sport, 0) + 1

    for category, count in category_counts.items():
        if category == "ml":
            score -= max(0, count - 2) * profile["ml_penalty"]
        elif category == "total":
            score -= max(0, count - 3) * profile["total_penalty"]
        elif category == "prop":
            score -= max(0, count - 2) * profile["prop_penalty"]

    hr_count = sum(1 for idx in parlay if is_hr_prop(legs[idx]))
    if target_size <= 4:
        score -= hr_count * profile["hr_short"]
    else:
        score -= max(0, hr_count - 1) * profile["hr_long"]

    if len(available_sports) > 1 and len(parlay) >= 2:
        if len(sport_counts) == 1:
            score -= profile["single_sport"] * min(len(parlay), target_size - 1)
        else:
            balance_gap = max(sport_counts.values()) - min(sport_counts.values())
            score -= profile["sport_gap"] * balance_gap

    score -= same_game_penalty

    if len(parlay) < target_size:
        candidate_indices = [
            idx
            for idx in range(len(legs))
            if idx not in parlay_set
            and leg_allowed_for_parlay_size(legs[idx], float(activation[idx]), target_size)
            and not any(incompatible_pair(legs[idx], legs[pick]) for pick in parlay)
        ]
        if not candidate_indices:
            score -= 1.25
        else:
            frontier = sorted(
                candidate_indices,
                key=lambda idx: candidate_leg_score(
                    candidate_idx=idx,
                    parlay=parlay,
                    legs=legs,
                    activation=activation,
                    co_activation=co_activation,
                    available_sports=available_sports,
                ),
                reverse=True,
            )[: max(1, target_size - len(parlay))]
            future_bonus = sum(
                standalone_leg_score(legs[idx], float(activation[idx])) for idx in frontier
            ) / len(frontier)
            score += future_bonus * profile["future"]

    return score


def build_state_parlay(
    legs: list[Leg], activation: np.ndarray, co_activation: np.ndarray, size: int
) -> list[int]:
    target_size = min(size, len(legs))
    if target_size <= 0:
        return []

    ranked = sorted(
        range(len(legs)),
        key=lambda idx: standalone_leg_score(legs[idx], float(activation[idx])),
        reverse=True,
    )
    ranked = [
        idx for idx in ranked if leg_allowed_for_parlay_size(legs[idx], float(activation[idx]), size)
    ]
    if not ranked:
        return []

    candidate_pool = ranked[: candidate_pool_limit(size)]
    available_sports = {leg.sport for leg in legs}

    starters = [idx for idx in candidate_pool if not is_hr_prop(legs[idx])]
    seed_pool = starters[:beam_width_for_size(size)] if starters else candidate_pool[:beam_width_for_size(size)]
    beam = [[idx] for idx in seed_pool]
    best_complete = []
    best_complete_score = -np.inf

    for depth in range(1, target_size + 1):
        scored_states = []
        for parlay in beam:
            score = partial_state_score(
                parlay=parlay,
                legs=legs,
                activation=activation,
                co_activation=co_activation,
                available_sports=available_sports,
                target_size=target_size,
            )
            scored_states.append((score, parlay))
            if len(parlay) == target_size and score > best_complete_score:
                best_complete_score = score
                best_complete = parlay

        if depth == target_size:
            break

        next_states = []
        seen = set()
        for _, parlay in sorted(scored_states, key=lambda item: item[0], reverse=True)[: beam_width_for_size(size)]:
            for candidate in candidate_pool:
                if candidate in parlay:
                    continue
                if any(incompatible_pair(legs[candidate], legs[pick]) for pick in parlay):
                    continue
                new_state = tuple(sorted(parlay + [candidate]))
                if new_state in seen:
                    continue
                seen.add(new_state)
                next_states.append(list(new_state))

        if not next_states:
            break

        next_scored = [
            (
                partial_state_score(
                    parlay=state,
                    legs=legs,
                    activation=activation,
                    co_activation=co_activation,
                    available_sports=available_sports,
                    target_size=target_size,
                ),
                state,
            )
            for state in next_states
        ]
        next_scored.sort(key=lambda item: item[0], reverse=True)
        beam = [state for _, state in next_scored[: beam_width_for_size(size)]]

    if best_complete:
        return best_complete

    best_partial = max(
        beam,
        key=lambda state: partial_state_score(
            parlay=state,
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            available_sports=available_sports,
            target_size=target_size,
        ),
        default=[],
    )
    return best_partial


def build_greedy_parlay(
    legs: list[Leg], activation: np.ndarray, co_activation: np.ndarray, size: int
) -> list[int]:
    return build_state_parlay(legs, activation, co_activation, size)


TIER_DEFINITIONS = [
    {
        "key": "cash",
        "label": "Cash",
        "size": 3,
        "min_prob": 0.70,
        "max_prob": 0.98,
        "target_payout_min": 1.5,
        "target_payout_max": 3.0,
        "bankroll_hint": "50%",
        "description": "High-probability grinder built to cash regularly.",
    },
    {
        "key": "decent",
        "label": "Decent Bet",
        "size": 4,
        "min_prob": 0.50,
        "max_prob": 0.70,
        "target_payout_min": 5.0,
        "target_payout_max": 15.0,
        "bankroll_hint": "30%",
        "description": "Mid-range state-search ticket where combination fit matters most.",
    },
    {
        "key": "longshot",
        "label": "Longshot",
        "size": 5,
        "min_prob": 0.30,
        "max_prob": 0.50,
        "target_payout_min": 25.0,
        "target_payout_max": 100.0,
        "bankroll_hint": "20%",
        "description": "Asymmetric upside ticket built from lower-probability legs.",
    },
]


def direct_leg_tag(leg: Leg) -> str:
    if leg.category == "ml":
        side = leg.label.split()[0]
        away_code, home_code = leg.game.split("@")
        is_home = side == home_code
        role = "fav" if leg.implied_prob >= 0.5 else "dog"
        location = "home" if is_home else "away"
        return f"ml:{location}:{role}"
    if leg.category == "total":
        return "total:over" if is_over_leg(leg) else "total:under"
    return f"prop:{leg.category}"


def direct_pair_bonus(first: Leg, second: Leg, mode: str) -> float:
    if first.game == second.game and first.category == second.category:
        return -0.95

    bonus = 0.0
    first_tag = direct_leg_tag(first)
    second_tag = direct_leg_tag(second)
    tags = {first_tag, second_tag}

    if mode == "heuristic":
        if first.category == "total" and second.category == "total":
            if tags == {"total:under"}:
                bonus += 0.04
            elif tags == {"total:over"}:
                bonus += 0.015
            else:
                bonus -= 0.04
        if first.category == "ml" and second.category == "ml":
            if "ml:away:dog" in tags and len(tags) == 1:
                bonus += 0.01
            if "ml:home:fav" in tags and len(tags) == 1:
                bonus += 0.01
        if tags == {"ml:home:fav", "total:under"}:
            bonus += 0.02
        if tags == {"ml:away:dog", "total:over"}:
            bonus += 0.01

    return bonus


def simulated_leg_probability(
    leg: Leg,
    away_scores: np.ndarray,
    home_scores: np.ndarray,
) -> float | None:
    total_scores = away_scores + home_scores
    away_code, home_code = leg.game.split("@")

    if leg.category == "ml":
        side = leg.label.split()[0]
        if side == away_code:
            return float(np.mean(away_scores > home_scores))
        if side == home_code:
            return float(np.mean(home_scores > away_scores))
        return None

    if leg.category == "total":
        line = direct_total_line(leg)
        if line is None:
            return None
        if is_over_leg(leg):
            return float(np.mean(total_scores > line))
        if is_under_leg(leg):
            return float(np.mean(total_scores < line))
    return None


def simulation_activation_and_coactivation(
    legs: list[Leg],
    date_str: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, str]]]:
    implied = np.array([float(leg.implied_prob) for leg in legs], dtype=np.float64)
    activation = implied.copy()
    co_activation = np.outer(activation, activation)
    pricing_details = {
        idx: {
            "pricing_source": "market_fallback",
            "pricing_label": "Market fallback",
        }
        for idx in range(len(legs))
    }

    game_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    mlb_prop_cache: dict[str, dict[str, float]] = {}
    nba_prop_cache: dict[str, dict[str, float]] = {}
    for idx, leg in enumerate(legs):
        if leg.category not in {"ml", "total"}:
            if leg.sport == "mlb" and leg.category == "prop":
                if leg.game not in mlb_prop_cache:
                    game_prop_legs = [candidate for candidate in legs if candidate.game == leg.game and candidate.sport == "mlb" and candidate.category == "prop"]
                    mlb_prop_cache[leg.game] = simulate_live_mlb_leg_probabilities(leg.game, date_str, game_prop_legs)
                sim_prob = mlb_prop_cache.get(leg.game, {}).get(leg.label)
                if sim_prob is not None:
                    activation[idx] = sim_prob
                    pricing_details[idx] = {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}
            if leg.sport == "nba" and leg.category == "prop":
                if leg.game not in nba_prop_cache:
                    game_prop_legs = [candidate for candidate in legs if candidate.game == leg.game and candidate.sport == "nba" and candidate.category == "prop"]
                    nba_prop_cache[leg.game] = simulate_live_nba_leg_probabilities(leg.game, date_str, game_prop_legs)
                sim_prob = nba_prop_cache.get(leg.game, {}).get(leg.label)
                if sim_prob is not None:
                    activation[idx] = sim_prob
                    pricing_details[idx] = {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}
            continue
        cache_key = (leg.sport, leg.game)
        if cache_key not in game_cache:
            away_code, home_code = leg.game.split("@")
            game_cache[cache_key] = simulate_game_distributions(
                sport=leg.sport,
                away_code=away_code,
                home_code=home_code,
                date_str=date_str,
            )
        away_scores, home_scores = game_cache[cache_key]
        sim_prob = simulated_leg_probability(leg, away_scores, home_scores)
        if sim_prob is not None:
            activation[idx] = sim_prob
            pricing_details[idx] = {"pricing_source": "simulation", "pricing_label": "Monte Carlo"}

    for i, first in enumerate(legs):
        for j in range(i + 1, len(legs)):
            second = legs[j]
            if first.sport == second.sport and first.game == second.game:
                cache_key = (first.sport, first.game)
                away_scores, home_scores = game_cache.get(cache_key, (None, None))
                if away_scores is not None and home_scores is not None:
                    first_mask = simulated_leg_probability(first, away_scores, home_scores)
                    second_mask = simulated_leg_probability(second, away_scores, home_scores)
                    if first_mask is not None and second_mask is not None:
                        first_hit = leg_hit_mask(first, away_scores, home_scores)
                        second_hit = leg_hit_mask(second, away_scores, home_scores)
                        if first_hit is not None and second_hit is not None:
                            joint = float(np.mean(first_hit & second_hit))
                            co_activation[i, j] = joint
                            co_activation[j, i] = joint
                            continue
            bonus = direct_pair_bonus(first, second, "sim")
            if bonus:
                adjusted = float(np.clip(co_activation[i, j] + bonus, 0.0, 1.0))
                co_activation[i, j] = adjusted
                co_activation[j, i] = adjusted

    activation = np.clip(activation, 0.02, 0.98)
    np.fill_diagonal(co_activation, activation)
    return activation, co_activation, pricing_details


def leg_hit_mask(leg: Leg, away_scores: np.ndarray, home_scores: np.ndarray) -> np.ndarray | None:
    total_scores = away_scores + home_scores
    away_code, home_code = leg.game.split("@")
    if leg.category == "ml":
        side = leg.label.split()[0]
        if side == away_code:
            return away_scores > home_scores
        if side == home_code:
            return home_scores > away_scores
        return None
    if leg.category == "total":
        line = direct_total_line(leg)
        if line is None:
            return None
        if is_over_leg(leg):
            return total_scores > line
        if is_under_leg(leg):
            return total_scores < line
    return None


def parse_mlb_prop_label(label: str) -> tuple[str, str, float] | None:
    match = re.match(r"^(?P<player>.+?)\s+O\s+(?P<threshold>\d+)\s+(?P<metric>HR|H|K)$", label)
    if not match:
        return None
    player = match.group("player").strip()
    threshold = float(match.group("threshold"))
    metric = match.group("metric")
    stat = {"HR": "home_runs", "H": "hits", "K": "strikeouts"}[metric]
    line = threshold - 0.5
    return player, stat, line


def parse_nba_prop_label(label: str) -> tuple[str, str, float] | None:
    match = re.match(r"^(?P<player>.+?)\s+O\s+(?P<threshold>\d+)\s+(?P<metric>PTS|REB|AST)$", label)
    if not match:
        return None
    player = match.group("player").strip()
    threshold = float(match.group("threshold"))
    metric = match.group("metric")
    stat = {"PTS": "points", "REB": "rebounds", "AST": "assists"}[metric]
    return player, stat, threshold - 0.5


def calibrate_mlb_offense_factor(target_mean: float, baseline_mean: float) -> float:
    safe_target = max(target_mean, 1.5)
    safe_baseline = max(baseline_mean, 1.5)
    raw_factor = safe_target / safe_baseline
    damped_factor = raw_factor ** 0.72
    return float(np.clip(damped_factor, 0.50, 1.35))


def sample_mlb_score_means(
    away,
    home,
    date_str: str,
    game: str,
    *,
    tag: str,
    n_simulations: int = 90,
) -> tuple[float, float]:
    calibrator = MLBGameSimulator(
        MLBGameConfig(
            n_simulations=n_simulations,
            random_seed=abs(hash((tag, date_str, game))) % (2**32),
        )
    )
    result = calibrator.simulate_game(away=away, home=home)
    return float(np.mean(result.away_scores)), float(np.mean(result.home_scores))


def project_nba_player_means(
    profiles: list[dict],
    team_total: float,
    availability_entries: list[dict] | None = None,
) -> dict[str, dict[str, float]]:
    availability_entries = availability_entries or []
    availability_by_name = {
        str(entry.get("player_name", "")).strip(): str(entry.get("status", "")).strip().lower()
        for entry in availability_entries
        if entry.get("player_name")
    }
    status_minute_multiplier = {
        "out": 0.0,
        "doubtful": 0.20,
        "questionable": 0.85,
        "probable": 0.96,
    }
    rotation = []
    for profile in profiles:
        name = str(profile.get("name", "")).strip()
        if not name:
            continue
        status = availability_by_name.get(name, str(profile.get("status", "active")).strip().lower())
        minute_multiplier = status_minute_multiplier.get(status, 1.0)
        base_minutes = max(float(profile.get("minutes", 0.0)), 8.0)
        base_points = max(float(profile.get("points", 0.0)), 0.4)
        base_rebounds = max(float(profile.get("rebounds", 0.0)), 0.15)
        base_assists = max(float(profile.get("assists", 0.0)), 0.10)
        games_sample = max(float(profile.get("games_sample", 0.0)), 1.0)
        projected_minutes = base_minutes * minute_multiplier
        if projected_minutes < 1.0:
            continue
        rotation.append(
            {
                "name": name,
                "status": status,
                "games_sample": games_sample,
                "base_minutes": base_minutes,
                "projected_minutes": projected_minutes,
                "base_points": base_points,
                "base_rebounds": base_rebounds,
                "base_assists": base_assists,
            }
        )

    if not rotation:
        return {}

    total_minutes = sum(player["projected_minutes"] for player in rotation)
    minute_scale = 240.0 / max(total_minutes, 1.0)
    for player in rotation:
        player["projected_minutes"] = min(40.0, player["projected_minutes"] * minute_scale)

    baseline_points = sum(
        player["base_points"] * max(player["projected_minutes"], 1.0) / max(player["base_minutes"], 1.0)
        for player in rotation
    ) or 1.0
    usage_scale = team_total / baseline_points
    results: dict[str, dict[str, float]] = {}
    for player in rotation:
        minutes_ratio = max(player["projected_minutes"], 1.0) / max(player["base_minutes"], 1.0)
        projected_points = max(0.5, player["base_points"] * minutes_ratio * usage_scale)
        pace_scale = (team_total / max(NBA_BASELINE_POINTS, 1.0)) ** 0.5
        projected_rebounds = max(0.2, player["base_rebounds"] * minutes_ratio * pace_scale)
        projected_assists = max(0.2, player["base_assists"] * minutes_ratio * usage_scale ** 0.65)
        results[player["name"]] = {
            "minutes": player["projected_minutes"],
            "points": projected_points,
            "rebounds": projected_rebounds,
            "assists": projected_assists,
            "status": player["status"],
            "games_sample": player["games_sample"],
        }
    return results


def nba_stat_dispersion(
    *,
    stat: str,
    mean: float,
    minutes: float,
    games_sample: float,
    status: str,
) -> float:
    base = {"points": 0.22, "rebounds": 0.18, "assists": 0.26}.get(stat, 0.20)
    minute_penalty = max(0.0, (26.0 - min(minutes, 26.0)) / 40.0)
    sample_penalty = max(0.0, (6.0 - min(games_sample, 6.0)) / 10.0)
    status_penalty = {
        "questionable": 0.18,
        "doubtful": 0.28,
        "probable": 0.06,
    }.get(status, 0.0)
    star_penalty = 0.05 if mean >= 20.0 and stat == "points" else 0.0
    return max(0.08, base + minute_penalty + sample_penalty + status_penalty + star_penalty)


def sample_nba_stat_over_probability(
    *,
    mean: float,
    line: float,
    stat: str,
    minutes: float,
    games_sample: float,
    status: str,
    rng: np.random.Generator,
    n_samples: int = 700,
) -> float:
    dispersion = nba_stat_dispersion(
        stat=stat,
        mean=mean,
        minutes=minutes,
        games_sample=games_sample,
        status=status,
    )
    variance = mean * (1.0 + dispersion * max(mean, 1.0) / 3.0)
    if variance <= mean + 0.05:
        samples = rng.poisson(lam=max(mean, 0.05), size=n_samples)
    else:
        shape = max((mean**2) / max(variance - mean, 0.05), 0.75)
        scale = max(mean / shape, 0.02)
        latent_rate = rng.gamma(shape=shape, scale=scale, size=n_samples)
        samples = rng.poisson(lam=latent_rate)
    return float(np.mean(samples > line))


def simulate_live_mlb_leg_probabilities(
    game: str,
    date_str: str,
    game_legs: list[Leg],
) -> dict[str, float]:
    if team_context_from_cached_payload is None or MLBGameSimulator is None or MLBGameConfig is None:
        return {}
    payload = load_matchup_profile_snapshot(date_str, game)
    if payload is None:
        return {}
    away, home = build_calibrated_live_mlb_contexts(date_str, game, payload)
    simulator = MLBGameSimulator(MLBGameConfig(n_simulations=750, random_seed=abs(hash((date_str, game))) % (2**32)))
    result = simulator.simulate_game(away=away, home=home)
    probabilities: dict[str, float] = {}
    for leg in game_legs:
        parsed = parse_mlb_prop_label(leg.label)
        if parsed is None:
            continue
        player_name, stat, line = parsed
        distribution = result.player_props.get((player_name, stat))
        if distribution is None:
            continue
        sim_prob = distribution.over_probabilities.get(line)
        if sim_prob is not None:
            probabilities[leg.label] = sim_prob
    return probabilities


def simulate_live_nba_leg_probabilities(
    game: str,
    date_str: str,
    game_legs: list[Leg],
) -> dict[str, float]:
    payload = load_nba_matchup_profile_snapshot(date_str, game)
    game_context = load_game_context_snapshot(date_str, "nba", game) or {}
    if payload is None:
        return {}
    away_code, home_code = game.split("@")
    away_mean, home_mean = expected_nba_points(away_code, home_code, load_team_form_snapshot(date_str, "nba"), game_context)
    rng = np.random.default_rng(abs(hash(("nba-props", date_str, game))) % (2**32))

    def team_probabilities(profiles: list[dict], team_total: float, side: str) -> dict[tuple[str, str], float]:
        availability = (game_context.get("availability") or {}).get(side, [])
        projected_means = project_nba_player_means(profiles, team_total, availability)
        results: dict[tuple[str, str], float] = {}
        parsed_legs = [parse_nba_prop_label(leg.label) for leg in game_legs]
        for player_name, means in projected_means.items():
            for parsed in parsed_legs:
                if parsed is None:
                    continue
                leg_player, stat, line = parsed
                if leg_player != player_name:
                    continue
                results[(player_name, stat)] = sample_nba_stat_over_probability(
                    mean=max(float(means[stat]), 0.05),
                    line=line,
                    stat=stat,
                    minutes=float(means.get("minutes", 24.0)),
                    games_sample=float(means.get("games_sample", 4.0)),
                    status=str(means.get("status", "active")),
                    rng=rng,
                )
        return results

    away_probs = team_probabilities(payload.get("away_profiles", []), away_mean, "away")
    home_probs = team_probabilities(payload.get("home_profiles", []), home_mean, "home")
    combined = {**away_probs, **home_probs}
    probabilities: dict[str, float] = {}
    for leg in game_legs:
        parsed = parse_nba_prop_label(leg.label)
        if parsed is None:
            continue
        player_name, stat, _ = parsed
        sim_prob = combined.get((player_name, stat))
        if sim_prob is not None:
            probabilities[leg.label] = sim_prob
    return probabilities


def build_calibrated_live_mlb_contexts(
    date_str: str,
    game: str,
    payload: dict | None = None,
):
    if team_context_from_cached_payload is None or MLBGameSimulator is None or MLBGameConfig is None:
        raise RuntimeError("Live MLB team-context calibration dependencies are unavailable")
    payload = payload or load_matchup_profile_snapshot(date_str, game)
    if payload is None:
        raise RuntimeError(f"No cached MLB matchup profile for {game} on {date_str}")
    away_code, home_code = game.split("@")
    away = team_context_from_cached_payload(away_code, payload.get("away_lineup", []), payload.get("away_pitcher", {}))
    home = team_context_from_cached_payload(home_code, payload.get("home_lineup", []), payload.get("home_pitcher", {}))
    game_context = load_game_context_snapshot(date_str, "mlb", game)
    target_away, target_home = expected_mlb_runs(away_code, home_code, load_team_form_snapshot(date_str, "mlb"), game_context)
    baseline_away, baseline_home = sample_mlb_score_means(away, home, date_str, game, tag="cal-base", n_simulations=120)
    total_target = target_away + target_home
    total_baseline = baseline_away + baseline_home
    total_factor = calibrate_mlb_offense_factor(total_target, total_baseline)
    away_split_factor = calibrate_mlb_offense_factor(target_away, baseline_away)
    home_split_factor = calibrate_mlb_offense_factor(target_home, baseline_home)
    away_factor = float(np.clip((away_split_factor * 0.65) + (total_factor * 0.35), 0.50, 1.25))
    home_factor = float(np.clip((home_split_factor * 0.65) + (total_factor * 0.35), 0.50, 1.25))
    away = replace(away, offense_factor=away_factor)
    home = replace(home, offense_factor=home_factor)
    pass1_away, pass1_home = sample_mlb_score_means(away, home, date_str, game, tag="cal-pass1")
    pass1_total_factor = calibrate_mlb_offense_factor(total_target, pass1_away + pass1_home)
    pass1_away_factor = calibrate_mlb_offense_factor(target_away, pass1_away)
    pass1_home_factor = calibrate_mlb_offense_factor(target_home, pass1_home)
    away = replace(
        away,
        offense_factor=float(
            np.clip(
                away.offense_factor * ((pass1_away_factor * 0.55) + (pass1_total_factor * 0.45)) ** 0.55,
                0.50,
                1.30,
            )
        ),
    )
    home = replace(
        home,
        offense_factor=float(
            np.clip(
                home.offense_factor * ((pass1_home_factor * 0.55) + (pass1_total_factor * 0.45)) ** 0.55,
                0.50,
                1.30,
            )
        ),
    )
    return away, home


def direct_activation_and_coactivation(
    legs: list[Leg],
    score_source: str,
    date_str: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, str]]]:
    if score_source == "sim":
        if not date_str:
            raise ValueError("Simulation scoring requires a target date")
        return simulation_activation_and_coactivation(legs, date_str)

    implied = np.array([float(leg.implied_prob) for leg in legs], dtype=np.float64)
    activation = implied.copy()
    if score_source == "residual":
        pricing_details = {idx: {"pricing_source": "residual", "pricing_label": "Residual model"} for idx in range(len(legs))}
    elif score_source == "heuristic":
        pricing_details = {idx: {"pricing_source": "heuristic", "pricing_label": "Heuristic model"} for idx in range(len(legs))}
    else:
        pricing_details = {idx: {"pricing_source": "market", "pricing_label": "Market implied"} for idx in range(len(legs))}

    if score_source == "heuristic":
        bias_bonus = -compute_biases(legs) * 0.12
        activation = activation + bias_bonus
        for idx, leg in enumerate(legs):
            if is_under_leg(leg):
                activation[idx] += 0.01
            elif is_over_leg(leg):
                activation[idx] += 0.003
            elif leg.category == "ml" and leg.implied_prob < 0.5:
                activation[idx] += 0.005
    elif score_source == "residual":
        team_form = fetch_live_mlb_team_form(date_str) if date_str else {}
        for idx, leg in enumerate(legs):
            activation[idx] += live_mlb_residual_adjustment(leg, team_form)

    activation = np.clip(activation, 0.02, 0.98)
    co_activation = np.outer(activation, activation)

    for i, first in enumerate(legs):
        for j in range(i + 1, len(legs)):
            second = legs[j]
            bonus = direct_pair_bonus(first, second, score_source)
            if bonus:
                adjusted = float(np.clip(co_activation[i, j] + bonus, 0.0, 1.0))
                co_activation[i, j] = adjusted
                co_activation[j, i] = adjusted

    np.fill_diagonal(co_activation, activation)
    return activation, co_activation, pricing_details


def build_filtered_parlay(
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    size: int,
    min_prob: float,
    max_prob: float,
) -> list[int]:
    eligible = [
        idx
        for idx, leg in enumerate(legs)
        if min_prob <= float(leg.implied_prob) <= max_prob
    ]
    if len(eligible) < size:
        ranked = sorted(
            range(len(legs)),
            key=lambda idx: abs(float(legs[idx].implied_prob) - ((min_prob + max_prob) / 2.0)),
        )
        for idx in ranked:
            if idx not in eligible:
                eligible.append(idx)
            if len(eligible) >= size:
                break

    if not eligible:
        return []

    sub_legs = [replace(legs[idx], id=sub_idx) for sub_idx, idx in enumerate(eligible)]
    sub_activation = activation[eligible]
    sub_co = co_activation[np.ix_(eligible, eligible)]
    sub_parlay = build_greedy_parlay(sub_legs, sub_activation, sub_co, size)
    return [eligible[idx] for idx in sub_parlay]


def build_tiered_parlays(
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    pricing_details: dict[int, dict[str, str]] | None = None,
    cash_activation: np.ndarray | None = None,
    cash_co_activation: np.ndarray | None = None,
) -> list[dict]:
    pricing_details = pricing_details or {}
    tiers = []
    for tier in TIER_DEFINITIONS:
        tier_activation = activation
        tier_co_activation = co_activation
        tier_score_mode = "implied"
        residual_requested = tier["key"] == "cash" and cash_activation is not None and cash_co_activation is not None
        if tier["key"] == "cash" and cash_activation is not None and cash_co_activation is not None:
            tier_activation = cash_activation
            tier_co_activation = cash_co_activation
            tier_score_mode = "residual"
        parlay = build_filtered_parlay(
            legs=legs,
            activation=tier_activation,
            co_activation=tier_co_activation,
            size=tier["size"],
            min_prob=tier["min_prob"],
            max_prob=tier["max_prob"],
        )
        if (
            tier["key"] == "cash"
            and len(parlay) < tier["size"]
            and cash_activation is not None
            and cash_co_activation is not None
        ):
            tier_activation = activation
            tier_co_activation = co_activation
            tier_score_mode = "implied_fallback"
            parlay = build_filtered_parlay(
                legs=legs,
                activation=tier_activation,
                co_activation=tier_co_activation,
                size=tier["size"],
                min_prob=tier["min_prob"],
                max_prob=tier["max_prob"],
            )
        combo_prob = float(np.prod([tier_activation[idx] for idx in parlay])) if parlay else 0.0
        tiers.append(
            {
                **tier,
                "actual_size": len(parlay),
                "payout_estimate": (1.0 / combo_prob) if combo_prob > 0 else None,
                "score_mode": tier_score_mode,
                "residual_requested": residual_requested,
                "legs": [
                    {
                        "label": legs[idx].label,
                        "category": legs[idx].category,
                        "game": legs[idx].game,
                        "activation": float(tier_activation[idx]),
                        "implied_prob": float(legs[idx].implied_prob),
                        "score_delta": float(tier_activation[idx]) - float(legs[idx].implied_prob),
                        "pricing_source": pricing_details.get(idx, {}).get("pricing_source", tier_score_mode),
                        "pricing_label": pricing_details.get(idx, {}).get("pricing_label", tier_score_mode.title()),
                        "notes": legs[idx].notes,
                    }
                    for idx in parlay
                ],
            }
        )
    return tiers


def summarize_from_scores(
    legs: list[Leg],
    activation: np.ndarray,
    co_activation: np.ndarray,
    entropy_source,
    slate_mode: str,
    loader_meta: dict,
    pricing_details: dict[int, dict[str, str]] | None = None,
    cash_activation: np.ndarray | None = None,
    cash_co_activation: np.ndarray | None = None,
) -> dict:
    pricing_details = pricing_details or {}
    n_samples = int(loader_meta.get("games", 0))
    ranked = sorted(range(len(legs)), key=lambda idx: activation[idx], reverse=True)
    pricing_summary: dict[str, int] = {}
    for idx in range(len(legs)):
        source = pricing_details.get(idx, {}).get("pricing_source", "market")
        pricing_summary[source] = pricing_summary.get(source, 0) + 1

    top_legs = []
    for rank, idx in enumerate(ranked[:12], start=1):
        leg = legs[idx]
        top_legs.append(
            {
                "rank": rank,
                "label": leg.label,
                "category": leg.category,
                "game": leg.game,
                "activation": float(activation[idx]),
                "implied_prob": float(leg.implied_prob),
                "score_delta": float(activation[idx]) - float(leg.implied_prob),
                "pricing_source": pricing_details.get(idx, {}).get("pricing_source", "market"),
                "pricing_label": pricing_details.get(idx, {}).get("pricing_label", "Market implied"),
                "notes": leg.notes,
            }
        )

    parlays = []
    for requested_size in (3, 4, 5):
        parlay = build_greedy_parlay(legs, activation, co_activation, requested_size)
        if not parlay:
            parlays.append(
                {"requested_size": requested_size, "actual_size": 0, "payout_estimate": None, "legs": []}
            )
            continue
        combo_prob = float(np.prod([activation[idx] for idx in parlay]))
        parlays.append(
            {
                "requested_size": requested_size,
                "actual_size": len(parlay),
                "payout_estimate": (1.0 / combo_prob) if combo_prob > 0 else None,
                "legs": [
                    {
                        "label": legs[idx].label,
                        "category": legs[idx].category,
                        "game": legs[idx].game,
                        "activation": float(activation[idx]),
                        "implied_prob": float(legs[idx].implied_prob),
                        "score_delta": float(activation[idx]) - float(legs[idx].implied_prob),
                        "pricing_source": pricing_details.get(idx, {}).get("pricing_source", "market"),
                        "pricing_label": pricing_details.get(idx, {}).get("pricing_label", "Market implied"),
                        "notes": legs[idx].notes,
                    }
                    for idx in parlay
                ],
            }
        )

    tier_parlays = build_tiered_parlays(
        legs,
        activation,
        co_activation,
        pricing_details=pricing_details,
        cash_activation=cash_activation,
        cash_co_activation=cash_co_activation,
    )

    hr_candidates = [idx for idx in ranked if is_hr_prop(legs[idx])]
    moonshot = None
    if hr_candidates:
        best_idx = hr_candidates[0]
        moonshot = {
            "label": legs[best_idx].label,
            "category": legs[best_idx].category,
            "game": legs[best_idx].game,
            "activation": float(activation[best_idx]),
            "implied_prob": float(legs[best_idx].implied_prob),
            "score_delta": float(activation[best_idx]) - float(legs[best_idx].implied_prob),
            "pricing_source": pricing_details.get(best_idx, {}).get("pricing_source", "market"),
            "pricing_label": pricing_details.get(best_idx, {}).get("pricing_label", "Market implied"),
            "notes": legs[best_idx].notes,
        }

    fades = []
    for idx in ranked[-6:]:
        fades.append(
            {
                "label": legs[idx].label,
                "category": legs[idx].category,
                "game": legs[idx].game,
                "activation": float(activation[idx]),
                "implied_prob": float(legs[idx].implied_prob),
                "score_delta": float(activation[idx]) - float(legs[idx].implied_prob),
                "pricing_source": pricing_details.get(idx, {}).get("pricing_source", "market"),
                "pricing_label": pricing_details.get(idx, {}).get("pricing_label", "Market implied"),
                "notes": legs[idx].notes,
            }
        )

    return {
        "meta": {
            "entropy_source": entropy_source.source,
            "random_bytes_consumed": entropy_source.total_consumed,
            "samples_collected": n_samples,
            "slate_mode": slate_mode,
            "pricing_summary": pricing_summary,
            **loader_meta,
        },
        "top_legs": top_legs,
        "parlays": parlays,
        "tier_parlays": tier_parlays,
        "moonshot": moonshot,
        "fades": fades,
    }


def summarize_results(
    legs: list[Leg],
    samples: np.ndarray,
    qrng: QuantumEntropySource,
    slate_mode: str,
    loader_meta: dict,
) -> dict:
    n_samples = samples.shape[0]
    binary = (samples + 1.0) / 2.0
    activation = np.mean(binary, axis=0)
    co_activation = (binary.T @ binary) / n_samples
    ranked = sorted(range(len(legs)), key=lambda idx: activation[idx], reverse=True)

    top_legs = []
    for rank, idx in enumerate(ranked[:12], start=1):
        leg = legs[idx]
        top_legs.append(
            {
                "rank": rank,
                "label": leg.label,
                "category": leg.category,
                "game": leg.game,
                "activation": float(activation[idx]),
                "notes": leg.notes,
            }
        )

    parlays = []
    for requested_size in (3, 4, 5):
        parlay = build_greedy_parlay(legs, activation, co_activation, requested_size)
        if not parlay:
            parlays.append(
                {"requested_size": requested_size, "actual_size": 0, "payout_estimate": None, "legs": []}
            )
            continue
        combo_prob = float(np.prod([activation[idx] for idx in parlay]))
        parlays.append(
            {
                "requested_size": requested_size,
                "actual_size": len(parlay),
                "payout_estimate": (1.0 / combo_prob) if combo_prob > 0 else None,
                "legs": [
                    {
                        "label": legs[idx].label,
                        "category": legs[idx].category,
                        "game": legs[idx].game,
                        "activation": float(activation[idx]),
                        "notes": legs[idx].notes,
                    }
                    for idx in parlay
                ],
            }
        )

    tier_parlays = build_tiered_parlays(legs, activation, co_activation)

    hr_candidates = [idx for idx in ranked if is_hr_prop(legs[idx])]
    moonshot = None
    if hr_candidates:
        best_idx = hr_candidates[0]
        moonshot = {
            "label": legs[best_idx].label,
            "category": legs[best_idx].category,
            "game": legs[best_idx].game,
            "activation": float(activation[best_idx]),
            "notes": legs[best_idx].notes,
        }

    fades = []
    for idx in ranked[-6:]:
        fades.append(
            {
                "label": legs[idx].label,
                "category": legs[idx].category,
                "game": legs[idx].game,
                "activation": float(activation[idx]),
                "notes": legs[idx].notes,
            }
        )

    return {
        "meta": {
            "entropy_source": qrng.source,
            "random_bytes_consumed": qrng.total_consumed,
            "samples_collected": n_samples,
            "slate_mode": slate_mode,
            **loader_meta,
        },
        "top_legs": top_legs,
        "parlays": parlays,
        "tier_parlays": tier_parlays,
        "moonshot": moonshot,
        "fades": fades,
    }


def analyze(
    legs: list[Leg],
    samples: np.ndarray,
    qrng: QuantumEntropySource,
    slate_mode: str,
    loader_meta: dict,
) -> None:
    summary = summarize_results(legs, samples, qrng, slate_mode, loader_meta)
    meta = summary["meta"]

    print()
    print("=" * 68)
    print("QUANTUM PARLAY ORACLE RESULTS")
    print("=" * 68)
    print(f"Entropy source: {meta.get('entropy_source')}")
    print(f"Slate mode: {meta.get('slate_mode')}")
    print(f"Target date: {meta.get('date')}")
    print(f"Sports: {meta.get('sports', 'mlb')}")
    print(f"Games loaded: {meta.get('games')}")
    print(f"Kalshi markets scanned: {meta.get('kalshi_markets')}")
    print(f"Recognized legs: {meta.get('recognized_legs')}")
    print(f"Random bytes consumed: {meta.get('random_bytes_consumed', 0):,}")
    print(f"Samples collected: {meta.get('samples_collected', 0):,}")
    print()
    print("Top 12 legs by activation:")
    for item in summary["top_legs"]:
        bar = "#" * int(item["activation"] * 30)
        print(f"{item['rank']:2d}. {item['label']:<24} {item['activation']:.3f} {bar}")

    print()
    print("Recommended parlays:")
    for parlay in summary["parlays"]:
        if not parlay["legs"]:
            print()
            print(f"{parlay['requested_size']}-leg parlay unavailable for this slate")
            continue
        print()
        print(
            f"{parlay['actual_size']}-leg parlay "
            f"(requested {parlay['requested_size']}, naive fair-odds estimate: "
            f"{parlay['payout_estimate']:.1f}x)"
        )
        for item in parlay["legs"]:
            print(f"  - {item['label']:<24} act={item['activation']:.3f} | {item['notes']}")

    if summary["moonshot"]:
        print()
        print("Moonshot add-on:")
        print(
            f"  - {summary['moonshot']['label']} act={summary['moonshot']['activation']:.3f} | "
            f"{summary['moonshot']['notes']}"
        )

    print()
    print("Fades:")
    for item in summary["fades"]:
        print(f"  - {item['label']:<24} act={item['activation']:.3f} | {item['notes']}")


def run_oracle(
    *,
    date_str: str,
    sport: str = "mlb",
    slate_mode: str = "auto",
    score_source: str = "implied",
    kalshi_pages: int = 25,
    fallback: bool = False,
    n_bytes: int = 65536,
    samples_per_beta: int = 1500,
    warmup: int = 300,
    thin: int = 3,
) -> dict:
    sports = ["mlb", "nba"] if sport == "both" else [sport]
    legs, resolved_slate_mode, loader_meta = load_legs(
        date_str=date_str,
        mode=slate_mode,
        sports=sports,
        kalshi_pages=kalshi_pages,
    )
    if score_source == "ising":
        entropy = QuantumEntropySource(n_bytes=n_bytes, fallback=fallback)
        samples = run_ensemble(
            legs=legs,
            qrng=entropy,
            betas=DEFAULT_BETAS,
            samples_per_beta=samples_per_beta,
            warmup=warmup,
            thin=thin,
        )
        summary = summarize_results(legs, samples, entropy, resolved_slate_mode, loader_meta)
    else:
        entropy = StaticEntropySource(
            "Direct market scoring"
            if score_source == "implied"
            else (
                "Live matchup simulation"
                if score_source == "sim"
                else (
                    "MLB residual market scoring (Cash tier only)"
                    if score_source == "residual"
                    else "Direct heuristic market scoring"
                )
            )
        )
        cash_activation = None
        cash_co_activation = None
        pricing_details = None
        if score_source == "residual":
            cash_activation, cash_co_activation, _ = direct_activation_and_coactivation(
                legs,
                "residual",
                date_str=date_str,
            )
            activation, co_activation, pricing_details = direct_activation_and_coactivation(
                legs,
                "implied",
                date_str=date_str,
            )
        else:
            activation, co_activation, pricing_details = direct_activation_and_coactivation(
                legs,
                score_source,
                date_str=date_str,
            )
        summary = summarize_from_scores(
            legs=legs,
            activation=activation,
            co_activation=co_activation,
            entropy_source=entropy,
            slate_mode=resolved_slate_mode,
            loader_meta=loader_meta,
            pricing_details=pricing_details,
            cash_activation=cash_activation,
            cash_co_activation=cash_co_activation,
        )
    summary["config"] = {
        "date": date_str,
        "sport": sport,
        "slate_mode": slate_mode,
        "score_source": score_source,
        "kalshi_pages": kalshi_pages,
        "fallback": fallback,
        "bytes": n_bytes,
        "samples_per_beta": samples_per_beta,
        "warmup": warmup,
        "thin": thin,
    }
    return summary


def run_ensemble(
    legs: list[Leg],
    qrng: QuantumEntropySource,
    betas: list[float],
    samples_per_beta: int,
    warmup: int,
    thin: int,
) -> np.ndarray:
    biases = compute_biases(legs)
    coupling = build_coupling_matrix(legs)
    all_samples = []

    for beta in betas:
        print(f"  Sampling beta={beta:.2f} with {samples_per_beta} retained samples")
        samples = gibbs_sample(
            biases=biases,
            coupling=coupling,
            beta=beta,
            qrng=qrng,
            n_warmup=warmup,
            n_samples=samples_per_beta,
            thin=thin,
        )
        all_samples.append(samples)

    return np.concatenate(all_samples, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parlay State Search")
    parser.add_argument(
        "--sport",
        choices=["mlb", "nba", "both"],
        default="mlb",
        help="Target sport feed",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Target slate date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--slate-mode",
        choices=["auto", "live"],
        default="auto",
        help="Use live API ingestion",
    )
    parser.add_argument(
        "--score-source",
        choices=["heuristic", "implied", "residual", "sim", "ising"],
        default="implied",
        help="Scoring engine for state search",
    )
    parser.add_argument(
        "--kalshi-pages",
        type=int,
        default=25,
        help="Maximum number of Kalshi market pages to scan",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use OS entropy immediately instead of ANU QRNG",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=65536,
        help="Number of bytes to prefetch from the entropy source",
    )
    parser.add_argument(
        "--samples-per-beta",
        type=int,
        default=1500,
        help="Retained samples per temperature",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=300,
        help="Warmup sweeps per temperature",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=3,
        help="Thinning factor for retained samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sports = ["mlb", "nba"] if args.sport == "both" else [args.sport]

    print()
    print("=" * 68)
    print("PARLAY STATE SEARCH")
    print("Multi-Sport Prototype v0.3")
    print("=" * 68)
    if args.score_source == "ising":
        print("Experimental mode: Ising energy landscape with Gibbs sampling")
    elif args.score_source == "implied":
        print("Default market mode: state search over direct implied probabilities")
    elif args.score_source == "sim":
        print("Simulation mode: live matchup model for MLB/NBA moneylines and totals, implied fallback for props")
    elif args.score_source == "residual":
        print("Hybrid market mode: MLB residuals for Cash tier, implied prices elsewhere")
    else:
        print("Default market mode: state search over implied probabilities plus heuristics")
    print()
    summary = run_oracle(
        date_str=args.date,
        sport=args.sport,
        slate_mode=args.slate_mode,
        score_source=args.score_source,
        kalshi_pages=args.kalshi_pages,
        fallback=args.fallback,
        n_bytes=args.bytes,
        samples_per_beta=args.samples_per_beta,
        warmup=args.warmup,
        thin=args.thin,
    )

    meta = summary["meta"]
    print("Results")
    print("=" * 68)
    print(f"Scoring mode: {summary['config']['score_source']}")
    print(f"Entropy source: {meta.get('entropy_source')}")
    print(f"Slate mode: {meta.get('slate_mode')}")
    print(f"Target date: {meta.get('date')}")
    print(f"Sports: {meta.get('sports', 'mlb')}")
    print(f"Games loaded: {meta.get('games')}")
    print(f"Kalshi markets scanned: {meta.get('kalshi_markets')}")
    print(f"Recognized legs: {meta.get('recognized_legs')}")
    print(f"Score rows collected: {meta.get('samples_collected', 0):,}")
    print()
    print("Top 12 legs by score:")
    for item in summary["top_legs"]:
        bar = "#" * int(item["activation"] * 30)
        print(f"{item['rank']:2d}. {item['label']:<24} {item['activation']:.3f} {bar}")
    print()
    print("Recommended parlays:")
    for parlay in summary["parlays"]:
        if not parlay["legs"]:
            print(f"{parlay['requested_size']}-leg parlay unavailable for this slate")
            continue
        print(
            f"{parlay['actual_size']}-leg parlay "
            f"(requested {parlay['requested_size']}, naive fair-odds estimate: "
            f"{parlay['payout_estimate']:.1f}x)"
        )
        for item in parlay["legs"]:
            print(f"  - {item['label']:<24} score={item['activation']:.3f} | {item['notes']}")
        print()


if __name__ == "__main__":
    main()
