import argparse
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


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
DEFAULT_BETAS = [0.8, 1.0, 1.2, 1.5, 2.0]
DATE_CODE_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})(\d{4})?([A-Z0-9]+)?$")
TEAM_WORD_RE = re.compile(r"[^a-z0-9]+")
EASTERN_TZ = ZoneInfo("America/New_York")

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
        return []

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

    target_code = target_date.strftime("%d%b%y").upper()
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
    dt = datetime.fromisoformat(game.game_time.replace("Z", "+00:00")).astimezone(EASTERN_TZ)
    return f"{dt.strftime('%d%b%y').upper()}{dt.strftime('%H%M')}{game.away_code}{game.home_code}"


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
    if game.sport != "mlb":
        return []

    token = kalshi_game_token(game)
    event_prefixes = [
        "KXMLBGAME",
        "KXMLBTOTAL",
        "KXMLBSPREAD",
        "KXMLBHR",
        "KXMLBHIT",
        "KXMLBTB",
        "KXMLBHRR",
    ]

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
        return Leg(-1, f"{team} ML", "ml", game.matchup, implied_prob, note)

    if game.sport == "mlb":
        if event_ticker.startswith("KXMLBHR-") or "home runs?" in title_lower:
            if market.get("floor_strike") not in (None, 0.5):
                return None
            player = title.split(":")[0].strip()
            return Leg(-1, f"{player} HR", "prop", game.matchup, implied_prob, note)

        if "strikeout" in title_lower or "strikeouts" in title_lower or event_ticker.startswith("KXMLBSO"):
            player = title.split(":")[0].strip()
            return Leg(-1, f"{player} O K's", "prop", game.matchup, implied_prob, note)

        if event_ticker.startswith("KXMLBTOTAL") or "total runs?" in title_lower:
            strike = market.get("floor_strike")
            if strike is None:
                raw = yes_subtitle or title
                cleaned = raw.replace("Over", "O").replace("over", "O").strip()
                return Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note)
            return Leg(-1, f"{game.matchup} O{strike + 0.0:g}", "total", game.matchup, implied_prob, note)

        if event_ticker.startswith("KXMLBSPREAD") or "wins by over" in title_lower:
            return None

    if game.sport == "nba":
        player = title.split(":")[0].strip()
        if event_ticker.startswith("KXNBAPTS") or " points?" in title_lower:
            return Leg(-1, f"{player} O PTS", "prop", game.matchup, implied_prob, note)
        if event_ticker.startswith("KXNBAREB") or " rebounds?" in title_lower:
            return Leg(-1, f"{player} O REB", "prop", game.matchup, implied_prob, note)
        if event_ticker.startswith("KXNBAAST") or " assists?" in title_lower:
            return Leg(-1, f"{player} O AST", "prop", game.matchup, implied_prob, note)
        if "total points?" in title_lower or event_ticker.startswith("KXNBAOU"):
            raw = yes_subtitle or title
            if "over" in raw.lower():
                cleaned = raw.replace("Over", "O").replace("over", "O").strip()
                return Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note)
            if "under" in raw.lower():
                cleaned = raw.replace("Under", "U").replace("under", "U").strip()
                return Leg(-1, f"{game.matchup} {cleaned}", "total", game.matchup, implied_prob, note)

    return None


def dedupe_legs(legs: list[Leg]) -> list[Leg]:
    deduped = {}
    for leg in legs:
        if leg.category == "total":
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


def fallback_static_legs(sports: list[str]) -> list[Leg]:
    if sports == ["mlb"]:
        return [replace(leg, id=index) for index, leg in enumerate(STATIC_LEGS)]
    return []


def load_live_legs(date_str: str, sports: list[str], kalshi_pages: int) -> tuple[list[Leg], dict]:
    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    all_games = []
    all_markets = []
    legs = []

    for sport in sports:
        games = fetch_schedule_for_sport(date_str, sport)
        token_lookup, game_list = build_game_lookup(games)
        if sport == "mlb":
            markets = []
            for game in games:
                markets.extend(fetch_kalshi_markets_for_game(game))
        else:
            markets = fetch_kalshi_sport_markets(target_date, sport=sport, max_pages=kalshi_pages)
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


def load_legs(date_str: str, mode: str, sports: list[str], kalshi_pages: int) -> tuple[list[Leg], str, dict]:
    if mode == "static":
        legs = fallback_static_legs(sports)
        if not legs:
            raise RuntimeError("No static fallback slate is defined for the requested sport selection")
        return legs, "static", {"date": date_str, "sports": ",".join(sports), "games": 0, "kalshi_markets": 0, "recognized_legs": len(legs)}

    try:
        live_legs, meta = load_live_legs(date_str, sports=sports, kalshi_pages=kalshi_pages)
        if live_legs:
            return live_legs, "live", meta
        if mode == "live":
            raise RuntimeError("No recognizable Kalshi sports legs found for the target date")
    except Exception as exc:
        if mode == "live":
            raise
        print(f"  Live loader unavailable: {exc}")

    legs = fallback_static_legs(sports)
    if not legs:
        raise RuntimeError("Live loader unavailable and no static fallback slate exists for the requested sport selection")
    return legs, "static-fallback", {"date": date_str, "sports": ",".join(sports), "games": 0, "kalshi_markets": 0, "recognized_legs": len(legs)}


def is_over_leg(leg: Leg) -> bool:
    return leg.category == "total" and " O" in f" {leg.label}"


def is_under_leg(leg: Leg) -> bool:
    return leg.category == "total" and " U" in f" {leg.label}"


def is_k_prop(leg: Leg) -> bool:
    return leg.category == "prop" and "O K" in leg.label


def is_hr_prop(leg: Leg) -> bool:
    return leg.category == "prop" and "HR" in leg.label


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


def build_greedy_parlay(
    legs: list[Leg], activation: np.ndarray, co_activation: np.ndarray, size: int
) -> list[int]:
    ranked = sorted(range(len(legs)), key=lambda idx: activation[idx], reverse=True)
    starters = [idx for idx in ranked if not is_hr_prop(legs[idx])]
    if starters:
        parlay = [starters[0]]
    elif ranked:
        parlay = [ranked[0]]
    else:
        return []

    target_size = min(size, len(legs))

    while len(parlay) < target_size:
        best_idx = None
        best_score = -np.inf

        for candidate in ranked:
            if candidate in parlay:
                continue
            if any(incompatible_pair(legs[candidate], legs[pick]) for pick in parlay):
                continue

            score = float(sum(co_activation[candidate, pick] for pick in parlay))
            score += activation[candidate] * 0.5

            if score > best_score:
                best_score = score
                best_idx = candidate

        if best_idx is None:
            break
        parlay.append(best_idx)

    return parlay


def analyze(
    legs: list[Leg],
    samples: np.ndarray,
    qrng: QuantumEntropySource,
    slate_mode: str,
    loader_meta: dict,
) -> None:
    n_samples = samples.shape[0]
    binary = (samples + 1.0) / 2.0
    activation = np.mean(binary, axis=0)
    co_activation = (binary.T @ binary) / n_samples
    ranked = sorted(range(len(legs)), key=lambda idx: activation[idx], reverse=True)

    print()
    print("=" * 68)
    print("QUANTUM PARLAY ORACLE RESULTS")
    print("=" * 68)
    print(f"Entropy source: {qrng.source}")
    print(f"Slate mode: {slate_mode}")
    print(f"Target date: {loader_meta.get('date')}")
    print(f"Sports: {loader_meta.get('sports', 'mlb')}")
    print(f"Games loaded: {loader_meta.get('games')}")
    print(f"Kalshi markets scanned: {loader_meta.get('kalshi_markets')}")
    print(f"Recognized legs: {loader_meta.get('recognized_legs')}")
    print(f"Random bytes consumed: {qrng.total_consumed:,}")
    print(f"Samples collected: {n_samples:,}")
    print()
    print("Top 12 legs by activation:")
    for rank, idx in enumerate(ranked[:12], start=1):
        leg = legs[idx]
        freq = activation[idx]
        bar = "#" * int(freq * 30)
        print(f"{rank:2d}. {leg.label:<24} {freq:.3f} {bar}")

    print()
    print("Recommended parlays:")
    for size in (3, 4, 5):
        parlay = build_greedy_parlay(legs, activation, co_activation, size)
        if not parlay:
            print()
            print(f"{size}-leg parlay unavailable for this slate")
            continue
        combo_prob = float(np.prod([activation[idx] for idx in parlay])) if parlay else 0.0
        payout = (1.0 / combo_prob) if combo_prob > 0 else float("inf")
        print()
        print(
            f"{len(parlay)}-leg parlay "
            f"(requested {size}, naive fair-odds estimate: {payout:.1f}x)"
        )
        for idx in parlay:
            leg = legs[idx]
            print(f"  - {leg.label:<24} act={activation[idx]:.3f} | {leg.notes}")

    hr_candidates = [idx for idx in ranked if is_hr_prop(legs[idx])]
    if hr_candidates:
        best_hr = hr_candidates[0]
        print()
        print("Moonshot add-on:")
        print(
            f"  - {legs[best_hr].label} act={activation[best_hr]:.3f} | "
            f"{legs[best_hr].notes}"
        )

    print()
    print("Fades:")
    for idx in ranked[-6:]:
        leg = legs[idx]
        print(f"  - {leg.label:<24} act={activation[idx]:.3f} | {leg.notes}")


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
    parser = argparse.ArgumentParser(description="Quantum Parlay Oracle")
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
        choices=["auto", "live", "static"],
        default="auto",
        help="Use live API ingestion, static fallback, or auto",
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
    print("QUANTUM PARLAY ORACLE")
    print("Multi-Sport Prototype v0.3")
    print("=" * 68)
    print("Sampling an Ising energy landscape with quantum-resolved Gibbs updates")
    print()

    legs, slate_mode, loader_meta = load_legs(
        date_str=args.date,
        mode=args.slate_mode,
        sports=sports,
        kalshi_pages=args.kalshi_pages,
    )

    qrng = QuantumEntropySource(n_bytes=args.bytes, fallback=args.fallback)

    print()
    print("Running multi-temperature ensemble...")
    samples = run_ensemble(
        legs=legs,
        qrng=qrng,
        betas=DEFAULT_BETAS,
        samples_per_beta=args.samples_per_beta,
        warmup=args.warmup,
        thin=args.thin,
    )
    analyze(legs, samples, qrng, slate_mode=slate_mode, loader_meta=loader_meta)


if __name__ == "__main__":
    main()
