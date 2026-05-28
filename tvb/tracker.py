"""Bet tracker: log algorithm proposals, resolve outcomes, compute performance.

Persistence backend is selected automatically:
- Set the DATABASE_URL secret (Streamlit Cloud) or env var for PostgreSQL.
- Without it, falls back to a local SQLite file (good for local dev).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from .config import DB_PATH

_SCORES_BASE = "https://api.the-odds-api.com/v4"

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        url = os.environ.get("DATABASE_URL", "")
        if url:
            _ENGINE = create_engine(url, pool_pre_ping=True)
        else:
            _ENGINE = create_engine(
                f"sqlite:///{DB_PATH}",
                connect_args={"check_same_thread": False})
    return _ENGINE


def _is_pg() -> bool:
    return bool(os.environ.get("DATABASE_URL", ""))


def _read(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


_CREATE_SQLITE = """
CREATE TABLE IF NOT EXISTS bets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at  TEXT    NOT NULL,
    match_id   TEXT    NOT NULL,
    player1    TEXT    NOT NULL,
    player2    TEXT    NOT NULL,
    commence_time TEXT NOT NULL,
    sport_key  TEXT    NOT NULL DEFAULT '',
    tour       TEXT    NOT NULL DEFAULT '',
    market     TEXT    NOT NULL,
    selection  TEXT    NOT NULL,
    odds       REAL    NOT NULL,
    model_prob REAL,
    edge       REAL,
    ev         REAL,
    kelly      REAL,
    stake      REAL    NOT NULL DEFAULT 10.0,
    result     TEXT    NOT NULL DEFAULT 'pending',
    profit     REAL,
    resolved_at TEXT,
    UNIQUE(match_id, market, selection)
)
"""

_CREATE_PG = """
CREATE TABLE IF NOT EXISTS bets (
    id            BIGSERIAL PRIMARY KEY,
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
    with _engine().begin() as conn:
        conn.execute(text(_CREATE_PG if _is_pg() else _CREATE_SQLITE))
        # Add tour column to tables created before this column existed
        if _is_pg():
            conn.execute(text(
                "ALTER TABLE bets ADD COLUMN IF NOT EXISTS "
                "tour TEXT NOT NULL DEFAULT ''"))
        else:
            existing = {r[1] for r in
                        conn.execute(text("PRAGMA table_info(bets)")).fetchall()}
            if "tour" not in existing:
                conn.execute(text(
                    "ALTER TABLE bets ADD COLUMN tour TEXT NOT NULL DEFAULT ''"))
        # Backfill tour from sport_key for rows that have it empty
        conn.execute(text(
            "UPDATE bets SET tour = "
            "CASE WHEN LOWER(sport_key) LIKE '%wta%' THEN 'wta' ELSE 'atp' END "
            "WHERE tour = ''"))


_INSERT_SQLITE = """
INSERT OR IGNORE INTO bets
    (logged_at, match_id, player1, player2, commence_time,
     sport_key, tour, market, selection, odds, model_prob, edge, ev, kelly, stake)
VALUES
    (:logged_at, :match_id, :player1, :player2, :commence_time,
     :sport_key, :tour, :market, :selection, :odds, :model_prob, :edge, :ev,
     :kelly, :stake)
"""

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


def log_bet(player1: str, player2: str, commence_time: str, sport_key: str,
            market: str, selection: str, odds: float, model_prob: float,
            edge: float, ev: float, kelly: float, stake: float = 10.0) -> bool:
    """Insert a bet if not already logged. Returns True if newly inserted."""
    init_tracker_db()
    match_id = f"{player1}|{player2}|{commence_time}"
    tour = "wta" if "wta" in sport_key.lower() else "atp"
    try:
        with _engine().begin() as conn:
            result = conn.execute(
                text(_INSERT_PG if _is_pg() else _INSERT_SQLITE),
                {"logged_at": datetime.now(timezone.utc).isoformat(),
                 "match_id": match_id, "player1": player1, "player2": player2,
                 "commence_time": commence_time, "sport_key": sport_key,
                 "tour": tour, "market": market, "selection": selection,
                 "odds": float(odds), "model_prob": float(model_prob),
                 "edge": float(edge), "ev": float(ev), "kelly": float(kelly),
                 "stake": float(stake)})
            return result.rowcount > 0
    except Exception:
        return False


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
    except (ValueError, KeyError):
        pass
    return None


def update_results(api_key: str) -> int:
    """Fetch completed scores and resolve pending Match winner bets."""
    init_tracker_db()
    pending = _read(
        "SELECT DISTINCT sport_key, player1, player2, commence_time, match_id "
        "FROM bets WHERE result = 'pending'")
    if pending.empty:
        return 0

    resolved = 0
    for sport_key in pending["sport_key"].unique():
        if not sport_key:
            continue
        for event in _fetch_scores(sport_key, api_key):
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
                continue
            match_id = pending.loc[mask, "match_id"].iloc[0]
            resolved += _resolve_match(match_id, winner)
    return resolved


def _resolve_match(match_id: str, winner: str) -> int:
    count = 0
    bets = _read(
        "SELECT id, market, selection, odds, stake FROM bets "
        "WHERE match_id = :mid AND result = 'pending'",
        {"mid": match_id})
    for row in bets.itertuples(index=False):
        won = _evaluate_bet(row.market, row.selection, winner)
        if won is None:
            continue
        profit = row.stake * (row.odds - 1.0) if won else -row.stake
        with _engine().begin() as conn:
            conn.execute(
                text("UPDATE bets SET result=:r, profit=:p, resolved_at=:ra "
                     "WHERE id=:id"),
                {"r": "won" if won else "lost", "p": profit,
                 "ra": datetime.now(timezone.utc).isoformat(), "id": int(row.id)})
        count += 1
    return count


def _evaluate_bet(market: str, selection: str, winner: str) -> bool | None:
    if market == "Match winner":
        return selection == winner
    return None


def get_bets_df(tour: str = "") -> pd.DataFrame:
    init_tracker_db()
    where = "WHERE tour = :tour" if tour else ""
    params = {"tour": tour} if tour else {}
    return _read(f"SELECT * FROM bets {where} ORDER BY logged_at DESC", params)


def performance_stats(tour: str = "") -> dict:
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    resolved = _read(f"SELECT result, stake, profit FROM bets "
                     f"WHERE result != 'pending' {tc}", p)
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
    df = _read(f"SELECT selection, market, result, profit FROM bets "
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
        "wins": "Vinte", "win_rate": "% Accuratezza", "profit": "Profitto netto (€)"})
    g["% Accuratezza"] = (g["% Accuratezza"] * 100).round(1)
    g["Profitto netto (€)"] = g["Profitto netto (€)"].round(2)
    return g.sort_values("% Accuratezza", ascending=False).reset_index(drop=True)


def equity_curve(tour: str = "") -> pd.DataFrame:
    """Cumulative profit over time for resolved bets, ordered by log time."""
    init_tracker_db()
    tc = "AND tour = :tour" if tour else ""
    p = {"tour": tour} if tour else {}
    df = _read(f"SELECT logged_at, profit FROM bets "
               f"WHERE result != 'pending' {tc} ORDER BY logged_at", p)
    if df.empty:
        return pd.DataFrame(columns=["Data", "Profitto cumulato (€)"])
    df["Profitto cumulato (€)"] = df["profit"].fillna(0).cumsum().round(2)
    df["Data"] = pd.to_datetime(df["logged_at"]).dt.strftime("%d/%m %H:%M")
    return df[["Data", "Profitto cumulato (€)"]].reset_index(drop=True)
