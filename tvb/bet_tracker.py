"""Bet tracker: log algorithm proposals, resolve outcomes, compute performance.

Persistence backend is selected automatically:
- DATABASE_URL env var / Streamlit secret → PostgreSQL via SQLAlchemy (persistent).
- Otherwise → local SQLite file via SQLAlchemy.
- If SQLAlchemy is not installed → raw sqlite3 fallback (SQLite only).
"""
from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .config import DB_PATH, SACKMANN_REPOS

_SCORES_BASE = "https://api.the-odds-api.com/v4"
_SACK_CACHE: dict = {}   # {tour: (fetched_at, DataFrame)}

# ── optional SQLAlchemy ────────────────────────────────────────────────────────
try:
    from sqlalchemy import create_engine, text as _sa_text
    _SA = True
except ImportError:
    _SA = False

_DATABASE_URL = os.environ.get("DATABASE_URL", "")
_ENGINE = None


def _engine():
    global _ENGINE
    if not _SA:
        return None
    if _ENGINE is None:
        if _DATABASE_URL:
            url = _DATABASE_URL.strip().strip('"').strip("'")
            url = url.replace("postgres://", "postgresql://", 1)
            _ENGINE = create_engine(url, pool_pre_ping=True)
        else:
            _ENGINE = create_engine(
                f"sqlite:///{DB_PATH}",
                connect_args={"check_same_thread": False})
    return _ENGINE


def _is_pg() -> bool:
    return _SA and bool(_DATABASE_URL)


# ── unified read helper ────────────────────────────────────────────────────────
def _read(sql: str, params: dict | None = None) -> pd.DataFrame:
    if _SA:
        with _engine().connect() as conn:
            result = conn.execute(_sa_text(sql), params or {})
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    # sqlite3 fallback: convert :name → ? and extract positional values
    keys = re.findall(r":(\w+)", sql)
    sql_q = re.sub(r":\w+", "?", sql)
    vals = tuple((params or {}).get(k) for k in keys)
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql_q, conn, params=vals if vals else None)


# ── schema ─────────────────────────────────────────────────────────────────────
_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at     TEXT             NOT NULL,
    match_id      TEXT             NOT NULL,
    player1       TEXT             NOT NULL,
    player2       TEXT             NOT NULL,
    commence_time TEXT             NOT NULL,
    sport_key     TEXT             NOT NULL DEFAULT '',
    tour          TEXT             NOT NULL DEFAULT '',
    market        TEXT             NOT NULL,
    selection     TEXT             NOT NULL,
    odds          REAL             NOT NULL,
    model_prob    REAL,
    edge          REAL,
    ev            REAL,
    kelly         REAL,
    stake         REAL             NOT NULL DEFAULT 10.0,
    result        TEXT             NOT NULL DEFAULT 'pending',
    profit        REAL,
    resolved_at   TEXT,
    UNIQUE(match_id, market, selection)
)
"""

_CREATE_PG = """
CREATE TABLE IF NOT EXISTS bets (
    id            BIGSERIAL        PRIMARY KEY,
    logged_at     TEXT             NOT NULL,
    match_id      TEXT             NOT NULL,
    player1       TEXT             NOT NULL,
    player2       TEXT             NOT NULL,
    commence_time TEXT             NOT NULL,
    sport_key     TEXT             NOT NULL DEFAULT '',
    tour          TEXT             NOT NULL DEFAULT '',
    market        TEXT             NOT NULL,
    selection     TEXT             NOT NULL,
    odds          DOUBLE PRECISION NOT NULL,
    model_prob    DOUBLE PRECISION,
    edge          DOUBLE PRECISION,
    ev            DOUBLE PRECISION,
    kelly         DOUBLE PRECISION,
    stake         DOUBLE PRECISION NOT NULL DEFAULT 10.0,
    result        TEXT             NOT NULL DEFAULT 'pending',
    profit        DOUBLE PRECISION,
    resolved_at   TEXT,
    UNIQUE(match_id, market, selection)
)
"""

_DB_READY = False


def init_tracker_db() -> None:
    global _DB_READY
    if _DB_READY:
        return
    _DB_READY = True
    if _SA:
        with _engine().begin() as conn:
            conn.execute(_sa_text(_CREATE_PG if _is_pg() else _CREATE_SQLITE))
            if _is_pg():
                conn.execute(_sa_text(
                    "ALTER TABLE bets ADD COLUMN IF NOT EXISTS "
                    "tour TEXT NOT NULL DEFAULT ''"))
            else:
                existing = {r[1] for r in
                            conn.execute(_sa_text(
                                "PRAGMA table_info(bets)")).fetchall()}
                if "tour" not in existing:
                    conn.execute(_sa_text(
                        "ALTER TABLE bets ADD COLUMN "
                        "tour TEXT NOT NULL DEFAULT ''"))
            conn.execute(_sa_text(
                "UPDATE bets SET tour = "
                "CASE WHEN LOWER(sport_key) LIKE '%wta%' THEN 'wta' ELSE 'atp' END "
                "WHERE tour = ''"))
    else:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(_CREATE_SQLITE)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(bets)")}
            if "tour" not in cols:
                conn.execute(
                    "ALTER TABLE bets ADD COLUMN tour TEXT NOT NULL DEFAULT ''")
                conn.execute(
                    "UPDATE bets SET tour = "
                    "CASE WHEN LOWER(sport_key) LIKE '%wta%' THEN 'wta' ELSE 'atp' END "
                    "WHERE tour = ''")


# ── insert ─────────────────────────────────────────────────────────────────────
_INSERT_PG = """
INSERT INTO bets
    (logged_at, match_id, player1, player2, commence_time,
     sport_key, tour, market, selection, odds, model_prob, edge, ev, kelly, stake)
VALUES
    (:logged_at, :match_id, :player1, :player2, :commence_time,
     :sport_key, :tour, :market, :selection, :odds, :model_prob, :edge, :ev,
     :kelly, :stake)
ON CONFLICT (match_id, market, selection) DO NOTHING
"""

_INSERT_SQLITE_SA = """
INSERT OR IGNORE INTO bets
    (logged_at, match_id, player1, player2, commence_time,
     sport_key, tour, market, selection, odds, model_prob, edge, ev, kelly, stake)
VALUES
    (:logged_at, :match_id, :player1, :player2, :commence_time,
     :sport_key, :tour, :market, :selection, :odds, :model_prob, :edge, :ev,
     :kelly, :stake)
"""

_INSERT_SQLITE_RAW = """
INSERT OR IGNORE INTO bets
    (logged_at, match_id, player1, player2, commence_time,
     sport_key, tour, market, selection, odds, model_prob, edge, ev, kelly, stake)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def log_bet(player1: str, player2: str, commence_time: str, sport_key: str,
            market: str, selection: str, odds: float, model_prob: float,
            edge: float, ev: float, kelly: float, stake: float = 10.0) -> bool:
    """Insert a bet if not already logged. Returns True if newly inserted."""
    init_tracker_db()
    # Normalize commence_time to YYYY-MM-DD (date only) to prevent duplicates
    # when the Odds API changes a match's scheduled hour between fetches.
    try:
        _ct_norm = datetime.fromisoformat(
            commence_time.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        _ct_norm = commence_time[:10]
    match_id = f"{player1}|{player2}|{_ct_norm}"
    tour = "wta" if "wta" in sport_key.lower() else "atp"
    p = {"logged_at": datetime.now(timezone.utc).isoformat(),
         "match_id": match_id, "player1": player1, "player2": player2,
         "commence_time": commence_time, "sport_key": sport_key,
         "tour": tour, "market": market, "selection": selection,
         "odds": float(odds), "model_prob": float(model_prob),
         "edge": float(edge), "ev": float(ev), "kelly": float(kelly),
         "stake": float(stake)}
    try:
        if _SA:
            sql = _INSERT_PG if _is_pg() else _INSERT_SQLITE_SA
            with _engine().begin() as conn:
                return conn.execute(_sa_text(sql), p).rowcount > 0
        else:
            vals = (p["logged_at"], p["match_id"], p["player1"], p["player2"],
                    p["commence_time"], p["sport_key"], p["tour"], p["market"],
                    p["selection"], p["odds"], p["model_prob"], p["edge"],
                    p["ev"], p["kelly"], p["stake"])
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(_INSERT_SQLITE_RAW, vals)
                return conn.execute("SELECT changes()").fetchone()[0] > 0
    except Exception:
        return False


# ── scores / result resolution ─────────────────────────────────────────────────
def _fetch_scores(sport_key: str, api_key: str, days_from: int = 3) -> list:
    try:
        resp = requests.get(
            f"{_SCORES_BASE}/sports/{sport_key}/scores",
            params={"apiKey": api_key, "daysFrom": days_from},
            timeout=20)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


def _winner_from_scores(scores: list) -> str | None:
    if not scores or len(scores) < 2:
        return None
    try:
        s0 = int(scores[0].get("score", -1))
        s1 = int(scores[1].get("score", -1))
        if s0 > s1:
            return scores[0]["name"]
        if s1 > s0:
            return scores[1]["name"]
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _ct_key(ct: str) -> str:
    """Normalize a commence_time string for loose comparison."""
    return ct.replace("Z", "").replace("+00:00", "").strip()


def _names_match(a: str, b: str) -> bool:
    """True if names are identical or one contains the other (handles abbreviations)."""
    a, b = a.strip().lower(), b.strip().lower()
    return a == b or a in b or b in a


# ── Sackmann-based result resolution ──────────────────────────────────────────
def _norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    return " ".join(
        "".join(c for c in s if not unicodedata.combining(c))
        .lower().replace("-", " ").split())


def _surname(name: str) -> str:
    parts = _norm_name(name).split()
    return parts[-1] if parts else ""


def _players_match(sw: str, sl: str, p1: str, p2: str) -> bool:
    """True when Sackmann winner/loser names match both pending players."""
    def _match_one(sack: str, odds: str) -> bool:
        sn = _norm_name(sack)
        on = _norm_name(odds)
        if sn == on or _surname(sack) == _surname(odds):
            return True
        sack_tok = {t for t in sn.split() if len(t) > 2}
        odds_tok = {t for t in on.split() if len(t) > 2}
        common = sack_tok & odds_tok
        if not common:
            return False
        sur_s = sn.split()[-1] if sn else ""
        sur_o = on.split()[-1] if on else ""
        return sur_s in common or sur_o in common

    return ((_match_one(sw, p1) and _match_one(sl, p2)) or
            (_match_one(sw, p2) and _match_one(sl, p1)))


def _total_from_score(score: str) -> int | None:
    """Sum all games from Sackmann score, e.g. '6-3 7-5' → 21."""
    try:
        total = 0
        for s in str(score).split():
            nums = re.split(r"[-/]", s.split("(")[0])
            if len(nums) == 2:
                total += int(nums[0]) + int(nums[1])
        return total if total > 0 else None
    except Exception:
        return None


def _sackmann_recent(tour: str) -> pd.DataFrame:
    """Download Sackmann CSVs (main tour + challenger/qual); cached 1 h."""
    cached = _SACK_CACHE.get(tour)
    if cached:
        ts, df = cached
        if (datetime.now(timezone.utc) - ts).total_seconds() < 3600:
            return df

    now = datetime.now(timezone.utc)
    year = now.year
    prev = year - 1
    cutoff = int((now - timedelta(days=90)).strftime("%Y%m%d"))

    base = SACKMANN_REPOS[tour]
    if tour == "atp":
        urls = [
            f"{base}/atp_matches_{year}.csv",
            f"{base}/atp_matches_qual_chall_{year}.csv",
            f"{base}/atp_matches_{prev}.csv",
            f"{base}/atp_matches_qual_chall_{prev}.csv",
        ]
    else:
        urls = [
            f"{base}/wta_matches_{year}.csv",
            f"{base}/wta_matches_qual_itf_{year}.csv",
            f"{base}/wta_matches_{prev}.csv",
            f"{base}/wta_matches_qual_itf_{prev}.csv",
        ]

    frames = []
    for url in urls:
        try:
            part = pd.read_csv(url, usecols=lambda c: c in
                               {"tourney_date", "winner_name", "loser_name", "score"})
            part = part[pd.to_numeric(part["tourney_date"], errors="coerce") >= cutoff]
            part = part.dropna(subset=["winner_name", "loser_name"])
            if not part.empty:
                frames.append(part)
        except Exception:
            continue

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _SACK_CACHE[tour] = (now, df)
    return df


def _update_from_sackmann() -> int:
    """Resolve pending bets using Sackmann GitHub results (free, no API quota)."""
    pending = _read(
        "SELECT DISTINCT tour, player1, player2, commence_time, match_id "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return 0
    resolved = 0
    done: set = set()
    for tour in ("atp", "wta"):
        tp = pending[pending["tour"] == tour]
        if tp.empty:
            continue
        recent = _sackmann_recent(tour)
        if recent.empty:
            continue
        for srow in recent.itertuples(index=False):
            sw = str(srow.winner_name)
            sl = str(srow.loser_name)
            tg = _total_from_score(getattr(srow, "score", ""))
            for prow in tp.itertuples(index=False):
                if prow.match_id in done:
                    continue
                if not _players_match(sw, sl, prow.player1, prow.player2):
                    continue
                # Identify which DB player is the winner using the same
                # fuzzy logic as _players_match (not just surname equality).
                def _match_one_name(sack: str, odds: str) -> bool:
                    sn = _norm_name(sack); on = _norm_name(odds)
                    if sn == on or _surname(sack) == _surname(odds):
                        return True
                    tok_s = {t for t in sn.split() if len(t) > 2}
                    tok_o = {t for t in on.split() if len(t) > 2}
                    common = tok_s & tok_o
                    if not common:
                        return False
                    sur_s = sn.split()[-1] if sn else ""
                    sur_o = on.split()[-1] if on else ""
                    return sur_s in common or sur_o in common
                db_winner = (prow.player1
                             if _match_one_name(sw, prow.player1)
                             else prow.player2)
                n = _resolve_match(prow.match_id, db_winner, total_games=tg)
                if n:
                    resolved += n
                    done.add(prow.match_id)
                break
    return resolved


def _match_one_name(a: str, b: str) -> bool:
    """Fuzzy single-name match.

    A shared token (>2 chars) qualifies only if it is the surname (last token)
    of at least one of the two names. This prevents common first names like
    'carlos', 'juan', 'maria' from causing false positives between two
    completely different players who share a first name.
    """
    sn = _norm_name(a)
    on = _norm_name(b)
    if sn == on or _surname(a) == _surname(b):
        return True
    tok_a = {t for t in sn.split() if len(t) > 2}
    tok_b = {t for t in on.split() if len(t) > 2}
    common = tok_a & tok_b
    if not common:
        return False
    # Only accept if the shared token is the surname of at least one player
    sur_a = sn.split()[-1] if sn else ""
    sur_b = on.split()[-1] if on else ""
    return sur_a in common or sur_b in common


def update_results(api_key: str, days_from: int = 3) -> int:
    """Resolve pending bets via Odds API scores (max daysFrom=3 on free tier)."""
    init_tracker_db()
    total = _update_from_odds_api(api_key, days_from)
    total += _update_from_sackmann()
    return total


# ── ESPN hidden API (free, no auth, real-time results) ────────────────────────
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"


def _fetch_espn_day(date_str: str) -> list[tuple[str, str, int | None]]:
    """
    Return list of (winner_name, loser_name, total_games) for all finished
    tennis matches on date_str (YYYY-MM-DD) from ESPN, covering ATP + WTA.
    total_games is the sum of all games played across all sets (or None if
    linescores are unavailable).
    """
    date_nodash = date_str.replace("-", "")
    results = []
    seen: set[tuple[str, str]] = set()
    for tour in ("atp", "wta"):
        try:
            resp = requests.get(
                f"{_ESPN_BASE}/{tour}/scoreboard",
                params={"dates": date_nodash},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for event in data.get("events", []):
                # Support both groupings→competitions and direct competitions
                competitions: list = []
                for grp in event.get("groupings", []):
                    competitions.extend(grp.get("competitions", []))
                if not competitions:
                    competitions = event.get("competitions", [])
                for comp in competitions:
                    status = comp.get("status", {}).get("type", {}).get("name", "")
                    if status != "STATUS_FINAL":
                        continue
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue
                    winner = next((c for c in competitors if c.get("winner")), None)
                    loser  = next((c for c in competitors if not c.get("winner")), None)
                    if not winner or not loser:
                        continue
                    w_name = (winner.get("athlete") or {}).get("displayName", "")
                    l_name = (loser.get("athlete") or {}).get("displayName", "")
                    if not w_name or not l_name:
                        continue
                    key = (w_name, l_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Extract total_games from per-set linescores
                    total_games: int | None = None
                    try:
                        w_ls = [ls.get("value", 0) for ls in winner.get("linescores", [])]
                        l_ls = [ls.get("value", 0) for ls in loser.get("linescores", [])]
                        if w_ls and l_ls and len(w_ls) == len(l_ls):
                            total_games = int(sum(w_ls) + sum(l_ls))
                    except Exception:
                        pass
                    results.append((w_name, l_name, total_games))
        except Exception:
            continue
    return results


def _update_from_espn() -> int:
    """Resolve pending bets using ESPN hidden API (free, real-time, ATP + WTA)."""
    pending = _read(
        "SELECT DISTINCT player1, player2, commence_time, match_id "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return 0

    now = datetime.now(timezone.utc)
    dates: set[str] = set()
    for ct in pending["commence_time"]:
        try:
            dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            if (now - dt).days <= 14:
                dates.add(dt.strftime("%Y-%m-%d"))
        except Exception:
            pass
    dates.add(now.strftime("%Y-%m-%d"))

    # Build lookup: (norm_p1, norm_p2) → (espn_winner_name, total_games)
    espn_results: dict[tuple, tuple[str, int | None]] = {}
    for date_str in sorted(dates):
        for winner_espn, loser_espn, total_games in _fetch_espn_day(date_str):
            key1 = (_norm_name(winner_espn), _norm_name(loser_espn))
            key2 = (_norm_name(loser_espn), _norm_name(winner_espn))
            # Keep first occurrence (earliest date wins)
            if key1 not in espn_results:
                espn_results[key1] = (winner_espn, total_games)
            if key2 not in espn_results:
                espn_results[key2] = (winner_espn, total_games)

    resolved = 0
    seen_pairs: set[tuple] = set()
    for prow in pending.itertuples(index=False):
        pair_key = (_norm_name(prow.player1), _norm_name(prow.player2))
        if pair_key in seen_pairs:
            continue

        winner_espn: str | None = None
        tg: int | None = None
        for (wn, ln), (w, tg_val) in espn_results.items():
            if (_match_one_name(wn, prow.player1) and _match_one_name(ln, prow.player2)) or \
               (_match_one_name(wn, prow.player2) and _match_one_name(ln, prow.player1)):
                winner_espn = w
                tg = tg_val
                break

        if not winner_espn:
            continue

        seen_pairs.add(pair_key)
        db_winner = (prow.player1
                     if _match_one_name(winner_espn, prow.player1)
                     else prow.player2)

        # Resolve ALL pending bets for this player pair (handles duplicate match_ids)
        all_ids = _read(
            "SELECT DISTINCT match_id FROM bets "
            "WHERE result='pending' AND player1=:p1 AND player2=:p2",
            {"p1": prow.player1, "p2": prow.player2})
        for mid in all_ids["match_id"]:
            resolved += _resolve_match(mid, db_winner, total_games=tg)

    return resolved


def update_from_espn() -> int:
    """Resolve pending bets using ESPN (free, no auth, real-time ATP + WTA)."""
    init_tracker_db()
    return _update_from_espn()


# ── RapidAPI "Tennis API - ATP WTA ITF" ───────────────────────────────────────
_RAPIDAPI_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"


def _fetch_rapidapi_day(date_str: str, rapidapi_key: str) -> list:
    """Fetch ATP + WTA fixtures for date_str via Tennis API - ATP WTA ITF on RapidAPI."""
    headers = {"X-RapidAPI-Key": rapidapi_key, "X-RapidAPI-Host": _RAPIDAPI_HOST}
    matches = []
    for tour in ("atp", "wta"):
        try:
            resp = requests.get(
                f"https://{_RAPIDAPI_HOST}/tennis/v2/{tour}/fixtures/{date_str}",
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            if isinstance(data, list):
                matches.extend(data)
            else:
                for key in ("result", "results", "data", "fixtures", "response"):
                    if key in data and isinstance(data[key], list):
                        matches.extend(data[key])
                        break
        except Exception:
            continue
    return matches


def _rapidapi_winner(match: dict) -> tuple[str | None, str | None, int | None]:
    """
    Parse a RapidAPI match dict.
    Returns (home_name, away_name, winner_name) or (None, None, None) if not finished.
    """
    # Status check — different APIs use different field names
    status = ""
    for path in [("status", "long"), ("status",), ("match_status",), ("state",)]:
        val = match
        try:
            for p in path:
                val = val[p]
            status = str(val).lower()
            break
        except (KeyError, TypeError):
            continue

    finished_terms = {"finished", "ended", "complete", "ft", "atp", "final", "closed"}
    if not any(t in status for t in finished_terms):
        return None, None, None

    # Extract player names from common field patterns
    home = away = ""
    for h_path, a_path in [
        (("home_team",), ("away_team",)),
        (("home_competitor", "name"), ("away_competitor", "name")),
        (("players", "home", "name"), ("players", "away", "name")),
        (("player1",), ("player2",)),
        (("home", "name"), ("away", "name")),
    ]:
        try:
            h = match
            for p in h_path:
                h = h[p]
            a = match
            for p in a_path:
                a = a[p]
            if h and a:
                home, away = str(h), str(a)
                break
        except (KeyError, TypeError):
            continue

    if not home or not away:
        return None, None, None

    # Extract scores
    h_score = a_score = -1
    for h_path, a_path in [
        (("scores", "home", "score"), ("scores", "away", "score")),
        (("home_score",), ("away_score",)),
        (("score", "home"), ("score", "away")),
        (("scores", "home"), ("scores", "away")),
    ]:
        try:
            hs = match
            for p in h_path:
                hs = hs[p]
            as_ = match
            for p in a_path:
                as_ = as_[p]
            h_score, a_score = int(hs), int(as_)
            break
        except (KeyError, TypeError, ValueError):
            continue

    if h_score < 0 or a_score < 0 or h_score == a_score:
        return home, away, None

    winner = home if h_score > a_score else away
    return home, away, winner


def _update_from_rapidapi(rapidapi_key: str) -> int:
    """Resolve pending bets using RapidAPI Tennis Live Data (real-time, free tier)."""
    pending = _read(
        "SELECT DISTINCT player1, player2, commence_time, match_id "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return 0

    now = datetime.now(timezone.utc)
    dates: set[str] = set()
    for ct in pending["commence_time"]:
        try:
            dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
            if (now - dt).days <= 7:
                dates.add(dt.strftime("%Y-%m-%d"))
        except Exception:
            pass
    dates.add(now.strftime("%Y-%m-%d"))

    resolved = 0
    done: set = set()

    for date_str in sorted(dates):
        for match in _fetch_rapidapi_day(date_str, rapidapi_key):
            home, away, winner_ra = _rapidapi_winner(match)
            if not winner_ra or not home or not away:
                continue
            for prow in pending.itertuples(index=False):
                if prow.match_id in done:
                    continue
                if not _players_match(home, away, prow.player1, prow.player2):
                    continue
                db_winner = (prow.player1
                             if _match_one_name(winner_ra, prow.player1)
                             else prow.player2)
                n = _resolve_match(prow.match_id, db_winner)
                if n:
                    resolved += n
                    done.add(prow.match_id)
                break

    return resolved


def update_from_rapidapi(rapidapi_key: str) -> int:
    """Resolve pending bets using RapidAPI Tennis Live Data."""
    init_tracker_db()
    return _update_from_rapidapi(rapidapi_key)


def update_from_sackmann() -> int:
    """Resolve pending bets using Sackmann GitHub data only (free, no API quota)."""
    init_tracker_db()
    return _update_from_sackmann()


def clear_sackmann_cache() -> None:
    """Force re-download of Sackmann CSVs on next call."""
    _SACK_CACHE.clear()


def _update_from_odds_api(api_key: str, days_from: int) -> int:
    """Resolve via The Odds API scores endpoint (Match winner only)."""
    pending = _read(
        "SELECT DISTINCT sport_key, player1, player2, commence_time, match_id "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return 0
    resolved = 0
    for sport_key in pending["sport_key"].unique():
        if not sport_key:
            continue
        for event in _fetch_scores(sport_key, api_key, days_from=days_from):
            if not event.get("completed"):
                continue
            p1 = event.get("home_team", "")
            p2 = event.get("away_team", "")
            ct = event.get("commence_time", "")
            winner = _winner_from_scores(event.get("scores") or [])
            if not winner:
                continue
            mask = ((pending["commence_time"] == ct) &
                    (pending["player1"] == p1) &
                    (pending["player2"] == p2))
            if not mask.any():
                ct_k = _ct_key(ct)
                mask = pending.apply(
                    lambda r: (
                        _ct_key(r["commence_time"]) == ct_k and
                        _names_match(r["player1"], p1) and
                        _names_match(r["player2"], p2)
                    ), axis=1)
            if not mask.any():
                continue
            match_id = pending.loc[mask, "match_id"].iloc[0]
            resolved += _resolve_match(match_id, winner, total_games=None)
    return resolved


def _resolve_match(match_id: str, winner: str,
                   total_games: int | None = None) -> int:
    count = 0
    bets = _read(
        "SELECT id, market, selection, odds, stake FROM bets "
        "WHERE match_id = :mid AND result = 'pending'",
        {"mid": match_id})
    for row in bets.itertuples(index=False):
        won = _evaluate_bet(row.market, row.selection, winner, total_games)
        if won is None:
            continue
        profit = row.stake * (row.odds - 1.0) if won else -row.stake
        now = datetime.now(timezone.utc).isoformat()
        if _SA:
            with _engine().begin() as conn:
                conn.execute(
                    _sa_text("UPDATE bets SET result=:r, profit=:p, "
                             "resolved_at=:ra WHERE id=:id"),
                    {"r": "won" if won else "lost", "p": profit,
                     "ra": now, "id": int(row.id)})
        else:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE bets SET result=?, profit=?, resolved_at=? WHERE id=?",
                    ("won" if won else "lost", profit, now, int(row.id)))
        count += 1
    return count


# Markets the system can auto-resolve. Others are voided when match finishes.
_RESOLVABLE_MARKETS = {"Match winner", "Total games"}


def _evaluate_bet(market: str, selection: str, winner: str,
                  total_games: int | None = None) -> bool | None:
    if market == "Match winner":
        return selection == winner
    if market == "Total games" and total_games is not None:
        parts = selection.split(None, 1)
        if len(parts) == 2:
            direction, line_str = parts
            try:
                line = float(line_str)
                if direction == "Over":
                    return total_games > line
                if direction == "Under":
                    return total_games < line
            except ValueError:
                pass
    return None


def void_unresolvable_bets() -> int:
    """
    Mark pending bets as 'void' in two cases:
    1. The Match winner bet for the same match_id is already resolved (won/lost)
       → void ALL remaining pending bets for that match (Total games, Handicap,
       Vincente 1° set, Tie-break, and any Total games ESPN couldn't resolve).
    2. ANY bet is still pending >72 h after commence_time (match is definitely
       over — catches matches where name-matching failed entirely).
    Returns count of voided bets.
    """
    init_tracker_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    # Use strftime with Z suffix to match Odds API's commence_time format exactly.
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    # Case 1: Match winner resolved → void ALL other pending bets for same match_id
    sql1 = """
        SELECT b.id
        FROM bets b
        WHERE b.result = 'pending'
          AND EXISTS (
              SELECT 1 FROM bets b2
              WHERE b2.match_id = b.match_id
                AND b2.market = 'Match winner'
                AND b2.result IN ('won', 'lost')
          )
    """
    # Case 2: any market pending >72h after commence_time
    sql2 = """
        SELECT b.id
        FROM bets b
        WHERE b.result = 'pending'
          AND b.commence_time < :cutoff
    """
    ids_to_void: set[int] = set()
    for sql, params in [(sql1, None), (sql2, {"cutoff": cutoff_iso})]:
        df = _read(sql, params)
        if not df.empty:
            ids_to_void.update(int(i) for i in df["id"])

    count = 0
    for bid in ids_to_void:
        try:
            if _SA:
                with _engine().begin() as conn:
                    conn.execute(
                        _sa_text("UPDATE bets SET result='void', profit=0, "
                                 "resolved_at=:ra WHERE id=:id"),
                        {"ra": now_iso, "id": bid})
            else:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "UPDATE bets SET result='void', profit=0, resolved_at=? WHERE id=?",
                        (now_iso, bid))
            count += 1
        except Exception:
            pass
    return count


def get_pending_matches(tour: str = "") -> pd.DataFrame:
    """Unique pending matches with bet counts, ordered by commence_time."""
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    return _read(
        f"SELECT match_id, player1, player2, commence_time, tour, sport_key, "
        f"COUNT(*) as n_bets "
        f"FROM bets WHERE result = 'pending' {tc} "
        f"GROUP BY match_id, player1, player2, commence_time, tour, sport_key "
        f"ORDER BY commence_time", p)


def resolve_match_manual(match_id: str, winner: str,
                          total_games: int | None = None) -> int:
    """Bulk-resolve all pending bets for a match. Returns count resolved."""
    init_tracker_db()
    return _resolve_match(match_id, winner, total_games=total_games)


# ── analytics ──────────────────────────────────────────────────────────────────
def get_pending_bets(tour: str = "") -> pd.DataFrame:
    """Pending bets ordered by Kelly descending — today's live proposals."""
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    return _read(
        f"SELECT * FROM bets WHERE result = 'pending' {tc} "
        f"ORDER BY kelly DESC", p)


def get_bets_df(tour: str = "") -> pd.DataFrame:
    init_tracker_db()
    tc = "WHERE tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    return _read(f"SELECT * FROM bets {tc} ORDER BY logged_at DESC", p)


def performance_stats(tour: str = "") -> dict:
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    resolved = _read(
        f"SELECT result, stake, profit FROM bets "
        f"WHERE result IN ('won','lost') {tc}", p)
    n_pending = int(_read(
        f"SELECT COUNT(*) AS c FROM bets WHERE result='pending' {tc}", p
    ).iloc[0, 0])
    if resolved.empty:
        return {"n_pending": n_pending, "n_resolved": 0,
                "n_won": 0, "n_lost": 0, "win_rate": 0.0,
                "total_staked": 0.0, "total_profit": 0.0, "roi": 0.0}
    n_won = int((resolved["result"] == "won").sum())
    n_lost = int((resolved["result"] == "lost").sum())
    n_resolved = n_won + n_lost
    total_staked = float(resolved["stake"].sum())
    total_profit = float(resolved["profit"].fillna(0).sum())
    return {
        "n_pending": n_pending, "n_resolved": n_resolved,
        "n_won": n_won, "n_lost": n_lost,
        "win_rate": n_won / n_resolved if n_resolved else 0.0,
        "total_staked": total_staked, "total_profit": total_profit,
        "roi": total_profit / total_staked if total_staked else 0.0,
    }


def accuracy_by_player(tour: str = "") -> pd.DataFrame:
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    df = _read(
        f"SELECT selection, market, result, profit FROM bets "
        f"WHERE result != 'pending' {tc}", p)
    if df.empty:
        return pd.DataFrame(columns=["Giocatore/Esito", "Mercato", "Bet",
                                     "Vinte", "% Accuratezza", "Profitto netto (€)"])
    g = df.groupby(["selection", "market"]).agg(
        n=("result", "count"),
        wins=("result", lambda x: (x == "won").sum()),
        profit=("profit", "sum"),
    ).reset_index()
    g["win_rate"] = g["wins"] / g["n"]
    g = g.rename(columns={
        "selection": "Giocatore/Esito", "market": "Mercato", "n": "Bet",
        "wins": "Vinte", "win_rate": "% Accuratezza",
        "profit": "Profitto netto (€)"})
    g["% Accuratezza"] = (g["% Accuratezza"] * 100).round(1)
    g["Profitto netto (€)"] = g["Profitto netto (€)"].round(2)
    return g.sort_values("% Accuratezza", ascending=False).reset_index(drop=True)


def manual_resolve_bet(bet_id: int, result: str) -> bool:
    """Manually mark a pending bet as won or lost. Returns True on success."""
    if result not in ("won", "lost"):
        raise ValueError(f"result must be 'won' or 'lost', got {result!r}")
    init_tracker_db()
    row_df = _read("SELECT odds, stake FROM bets WHERE id = :id AND result = 'pending'",
                   {"id": bet_id})
    if row_df.empty:
        return False
    row = row_df.iloc[0]
    profit = float(row["stake"]) * (float(row["odds"]) - 1.0) if result == "won" else -float(row["stake"])
    now = datetime.now(timezone.utc).isoformat()
    try:
        if _SA:
            with _engine().begin() as conn:
                conn.execute(
                    _sa_text("UPDATE bets SET result=:r, profit=:p, resolved_at=:ra WHERE id=:id"),
                    {"r": result, "p": profit, "ra": now, "id": bet_id})
        else:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE bets SET result=?, profit=?, resolved_at=? WHERE id=?",
                    (result, profit, now, bet_id))
        return True
    except Exception:
        return False


def scores_debug(api_key: str) -> list[dict]:
    """Return raw debug info from the scores API for pending sport keys."""
    init_tracker_db()
    pending = _read(
        "SELECT DISTINCT sport_key, player1, player2, commence_time "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return []
    out = []
    for sport_key in pending["sport_key"].unique():
        if not sport_key:
            continue
        events = _fetch_scores(sport_key, api_key, days_from=3)
        completed = [e for e in events if e.get("completed")]
        matched = 0
        for ev in completed:
            p1 = ev.get("home_team", "")
            p2 = ev.get("away_team", "")
            ct = ev.get("commence_time", "")
            mask = ((pending["commence_time"] == ct) &
                    (pending["player1"] == p1) &
                    (pending["player2"] == p2))
            if mask.any():
                matched += 1
        out.append({
            "sport_key": sport_key,
            "total_events": len(events),
            "completed": len(completed),
            "matched_pending": matched,
            "sample_completed": [
                {"p1": e.get("home_team"), "p2": e.get("away_team"),
                 "ct": e.get("commence_time"),
                 "scores": e.get("scores"),
                 "winner": _winner_from_scores(e.get("scores") or [])}
                for e in completed[:5]
            ],
        })
    return out


def equity_curve(tour: str = "") -> pd.DataFrame:
    """Cumulative profit over time for resolved bets, ordered by log time."""
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    df = _read(
        f"SELECT logged_at, profit FROM bets "
        f"WHERE result != 'pending' {tc} ORDER BY logged_at", p)
    if df.empty:
        return pd.DataFrame(columns=["Data", "Profitto cumulato (€)"])
    df["Profitto cumulato (€)"] = df["profit"].fillna(0).cumsum().round(2)
    df["Data"] = pd.to_datetime(df["logged_at"]).dt.strftime("%d/%m %H:%M")
    return df[["Data", "Profitto cumulato (€)"]].reset_index(drop=True)
