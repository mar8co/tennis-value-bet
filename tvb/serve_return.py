"""Estimate serve / return point-win rates per player, and combine two
players into the (p0, p1) inputs the simulator needs."""
from __future__ import annotations

import pandas as pd

from .config import LEAGUE_SPW

_STAT_COLS = ["w_svpt", "w_1stWon", "w_2ndWon",
              "l_svpt", "l_1stWon", "l_2ndWon"]


def player_serve_return(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate serve-points-won (spw) and return-points-won (rpw) per player.

    Uses Sackmann match columns. Returns DataFrame [player_id, spw, rpw,
    matches]. For a real model these should be weighted by recency and
    surface — kept as a flat mean here for the scaffold.
    """
    m = matches.dropna(subset=_STAT_COLS)
    m = m[(m["w_svpt"] > 0) & (m["l_svpt"] > 0)]

    records = []
    for row in m.itertuples(index=False):
        w_spw = (row.w_1stWon + row.w_2ndWon) / row.w_svpt
        l_spw = (row.l_1stWon + row.l_2ndWon) / row.l_svpt
        records.append((row.winner_id, w_spw, 1 - l_spw))   # winner's serve / return
        records.append((row.loser_id, l_spw, 1 - w_spw))    # loser's serve / return

    df = pd.DataFrame(records, columns=["player_id", "spw", "rpw"])
    agg = df.groupby("player_id").agg(spw=("spw", "mean"),
                                      rpw=("rpw", "mean"),
                                      matches=("spw", "size"))
    return agg.reset_index()


def matchup_serve_probs(spw_a: float, rpw_a: float,
                        spw_b: float, rpw_b: float,
                        tour: str = "atp") -> tuple[float, float]:
    """Combine two players into (p_a, p_b): the probability each wins a point
    on his own serve, adjusting for the opponent's return strength.

        p_a = league + (spw_a - league) - (rpw_b - league_rpw)

    This is the standard Klaassen-Magnus style additive combination.
    """
    league = LEAGUE_SPW.get(tour, 0.62)
    league_rpw = 1.0 - league
    p_a = league + (spw_a - league) - (rpw_b - league_rpw)
    p_b = league + (spw_b - league) - (rpw_a - league_rpw)
    def _clamp(x: float) -> float:
        return min(max(x, 0.40), 0.85)
    return _clamp(p_a), _clamp(p_b)
