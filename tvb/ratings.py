"""Bridge between the raw matches table and the simulator's inputs:
player name lookup and matchup -> (p0, p1) conversion."""
from __future__ import annotations

import unicodedata

import pandas as pd

from .serve_return import matchup_serve_probs


def player_names(matches: pd.DataFrame) -> pd.DataFrame:
    """Unique [player_id, name] mapping extracted from a matches table."""
    a = matches[["winner_id", "winner_name"]].rename(
        columns={"winner_id": "player_id", "winner_name": "name"})
    b = matches[["loser_id", "loser_name"]].rename(
        columns={"loser_id": "player_id", "loser_name": "name"})
    return (pd.concat([a, b])
            .dropna()
            .drop_duplicates("player_id")
            .sort_values("name")
            .reset_index(drop=True))


def matchup_probs(serve_return: pd.DataFrame, pid_a: int, pid_b: int,
                  tour: str = "atp") -> tuple[float, float]:
    """Look up two players in a serve/return table and return (p_a, p_b):
    the probability each wins a point on his own serve."""
    row_a = serve_return.loc[serve_return["player_id"] == pid_a]
    row_b = serve_return.loc[serve_return["player_id"] == pid_b]
    if row_a.empty or row_b.empty:
        raise KeyError("player not found in the serve/return table")
    a, b = row_a.iloc[0], row_b.iloc[0]
    return matchup_serve_probs(a["spw"], a["rpw"], b["spw"], b["rpw"], tour)


def _normalize(name: str) -> str:
    """Lowercase, accent-stripped, hyphen/whitespace-collapsed name."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(c for c in decomposed
                         if not unicodedata.combining(c))
    return " ".join(ascii_name.lower().replace("-", " ").split())


def find_player_id(names: pd.DataFrame, query: str):
    """Best-effort match of a free-text player name to a player_id, or None.
    Tries an exact normalized match, then a unique surname match — handles
    accent and capitalisation differences between data sources."""
    norm_to_id = {_normalize(n): pid
                  for pid, n in zip(names["player_id"], names["name"])}
    q = _normalize(query)
    if q in norm_to_id:
        return norm_to_id[q]
    parts = q.split()
    if parts:
        hits = [pid for nn, pid in norm_to_id.items()
                if nn.split() and nn.split()[-1] == parts[-1]]
        if len(hits) == 1:
            return hits[0]
    return None
