"""Bet tracker: log algorithm proposals, resolve outcomes, compute performance."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DB_PATH

_SCORES_BASE = "https://api.the-odds-api.com/v4"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_tracker_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                match_id TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                sport_key TEXT NOT NULL DEFAULT '',
                market TEXT NOT NULL,
                selection TEXT NOT NULL,
                odds REAL NOT NULL,
                model_prob REAL,
                edge REAL,
                ev REAL,
                kelly REAL,
                stake REAL NOT NULL DEFAULT 10.0,
                result TEXT NOT NULL DEFAULT 'pending',
                profit REAL,
                resolved_at TEXT,
                UNIQUE(match_id, market, selection)
            )
        """)


def log_bet(player1: str, player2: str, commence_time: str, sport_key: str,
            market: str, selection: str, odds: float, model_prob: float,
            edge: float, ev: float, kelly: float, stake: float = 10.0) -> bool:
    """Insert a bet if not already logged. Returns True if newly inserted."""
    match_id = f"{player1}|{player2}|{commence_time}"
    init_tracker_db()
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO bets
                    (logged_at, match_id, player1, player2, commence_time,
                     sport_key, market, selection, odds, model_prob, edge,
                     ev, kelly, stake)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now(timezone.utc).isoformat(), match_id,
                  player1, player2, commence_time, sport_key,
                  market, selection, float(odds), float(model_prob),
                  float(edge), float(ev), float(kelly), float(stake)))
            return conn.execute("SELECT changes()").fetchone()[0] > 0
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
    """Fetch completed scores and resolve pending Match winner bets. Returns count."""
    init_tracker_db()
    with _conn() as conn:
        pending = pd.read_sql(
            "SELECT DISTINCT sport_key, player1, player2, commence_time, match_id "
            "FROM bets WHERE result = 'pending'", conn)
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
    """Resolve all resolvable pending bets for a match. Returns count resolved."""
    count = 0
    with _conn() as conn:
        bets = pd.read_sql(
            "SELECT id, market, selection, odds, stake FROM bets "
            "WHERE match_id = ? AND result = 'pending'",
            conn, params=(match_id,))
        for row in bets.itertuples(index=False):
            won = _evaluate_bet(row.market, row.selection, winner)
            if won is None:
                continue
            profit = row.stake * (row.odds - 1.0) if won else -row.stake
            conn.execute(
                "UPDATE bets SET result=?, profit=?, resolved_at=? WHERE id=?",
                ("won" if won else "lost", profit,
                 datetime.now(timezone.utc).isoformat(), row.id))
            count += 1
    return count


def _evaluate_bet(market: str, selection: str, winner: str) -> bool | None:
    if market == "Match winner":
        return selection == winner
    return None  # other markets need game-level scores, not yet available


def get_bets_df() -> pd.DataFrame:
    init_tracker_db()
    with _conn() as conn:
        return pd.read_sql("SELECT * FROM bets ORDER BY logged_at DESC", conn)


def performance_stats() -> dict:
    init_tracker_db()
    with _conn() as conn:
        resolved = pd.read_sql(
            "SELECT * FROM bets WHERE result != 'pending'", conn)
        n_pending = pd.read_sql(
            "SELECT COUNT(*) as c FROM bets WHERE result='pending'",
            conn).iloc[0, 0]
    if resolved.empty:
        return {"n_pending": int(n_pending), "n_resolved": 0,
                "n_won": 0, "n_lost": 0, "win_rate": 0.0,
                "total_staked": 0.0, "total_profit": 0.0, "roi": 0.0}
    n_won = int((resolved["result"] == "won").sum())
    n_lost = int((resolved["result"] == "lost").sum())
    n_resolved = n_won + n_lost
    total_staked = float(resolved["stake"].sum())
    total_profit = float(resolved["profit"].fillna(0).sum())
    return {
        "n_pending": int(n_pending),
        "n_resolved": n_resolved,
        "n_won": n_won,
        "n_lost": n_lost,
        "win_rate": n_won / n_resolved if n_resolved else 0.0,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": total_profit / total_staked if total_staked else 0.0,
    }


def accuracy_by_player() -> pd.DataFrame:
    init_tracker_db()
    with _conn() as conn:
        df = pd.read_sql(
            "SELECT selection, market, result, profit FROM bets "
            "WHERE result != 'pending'", conn)
    if df.empty:
        return pd.DataFrame(
            columns=["Giocatore/Esito", "Mercato", "Bet", "Vinte",
                     "% Accuratezza", "Profitto netto (€)"])
    g = df.groupby(["selection", "market"]).agg(
        n=("result", "count"),
        wins=("result", lambda x: (x == "won").sum()),
        profit=("profit", "sum")
    ).reset_index()
    g["win_rate"] = g["wins"] / g["n"]
    g = g.rename(columns={
        "selection": "Giocatore/Esito",
        "market": "Mercato",
        "n": "Bet",
        "wins": "Vinte",
        "win_rate": "% Accuratezza",
        "profit": "Profitto netto (€)"
    })
    g["% Accuratezza"] = (g["% Accuratezza"] * 100).round(1)
    g["Profitto netto (€)"] = g["Profitto netto (€)"].round(2)
    return g.sort_values("% Accuratezza", ascending=False).reset_index(drop=True)


def equity_curve() -> pd.DataFrame:
    """Cumulative profit over time for all resolved bets, ordered by log time."""
    init_tracker_db()
    with _conn() as conn:
        df = pd.read_sql(
            "SELECT logged_at, profit FROM bets "
            "WHERE result != 'pending' ORDER BY logged_at",
            conn)
    if df.empty:
        return pd.DataFrame(columns=["Data", "Profitto cumulato (€)"])
    df["Profitto cumulato (€)"] = df["profit"].fillna(0).cumsum().round(2)
    df["Data"] = pd.to_datetime(df["logged_at"]).dt.strftime("%d/%m %H:%M")
    return df[["Data", "Profitto cumulato (€)"]].reset_index(drop=True)
