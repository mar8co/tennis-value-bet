"""Surface-aware Elo ratings computed from historical match results."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .config import ELO_BASE, ELO_K


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def compute_elo(matches: pd.DataFrame, by_surface: bool = True) -> pd.DataFrame:
    """Walk matches chronologically and return final Elo per player.

    `matches` needs columns: tourney_date, winner_id, loser_id, surface.
    Returns DataFrame [player_id, surface, elo, matches]. With
    by_surface=False every match falls under surface "all".
    """
    matches = matches.sort_values("tourney_date")
    ratings: dict = defaultdict(lambda: ELO_BASE)
    counts: dict = defaultdict(int)

    for row in matches.itertuples(index=False):
        surf = row.surface if (by_surface and isinstance(row.surface, str)) else "all"
        wkey, lkey = (row.winner_id, surf), (row.loser_id, surf)
        rw, rl = ratings[wkey], ratings[lkey]
        exp_w = _expected(rw, rl)
        ratings[wkey] = rw + ELO_K * (1 - exp_w)
        ratings[lkey] = rl - ELO_K * (1 - exp_w)
        counts[wkey] += 1
        counts[lkey] += 1

    rows = [{"player_id": pid, "surface": surf, "elo": elo,
             "matches": counts[(pid, surf)]}
            for (pid, surf), elo in ratings.items()]
    return pd.DataFrame(rows)


def win_probability(elo_a: float, elo_b: float) -> float:
    """P(player A beats player B) from their Elo ratings."""
    return _expected(elo_a, elo_b)
