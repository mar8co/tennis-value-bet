"""Incremental, leak-free serve / return ratings.

Walk matches in chronological order and fold each result into both players'
ratings. A player's serve and return point-win rates are a recency-weighted
(exponentially decayed) average of past performances, each one adjusted for
the strength of the opponent faced.

Two improvements over the flat mean in serve_return.py:
  * recency  — a per-match decay factor fades older matches;
  * opponent — a serve % achieved against a strong returner is worth more
               than the same % against a weak one.

Because every adjustment uses only the *pre-match* ratings, a prediction for
match N depends solely on matches 1..N-1 — no look-ahead leakage.
"""
from __future__ import annotations

from collections import defaultdict

from .config import LEAGUE_SPW, SR_DECAY, SR_PRIOR_WEIGHT


def _clip(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return lo if x < lo else hi if x > hi else x


class SRRatings:
    """Recency-weighted, opponent-adjusted serve/return ratings."""

    def __init__(self, tour: str = "atp", decay: float = SR_DECAY,
                 prior_weight: float = SR_PRIOR_WEIGHT):
        self.league_spw = LEAGUE_SPW.get(tour, 0.62)
        self.league_rpw = 1.0 - self.league_spw
        self.decay = decay
        self.prior_weight = prior_weight
        self._spw_num: dict = defaultdict(float)
        self._spw_den: dict = defaultdict(float)
        self._rpw_num: dict = defaultdict(float)
        self._rpw_den: dict = defaultdict(float)
        self._matches: dict = defaultdict(int)

    def spw(self, pid) -> float:
        """Serve-points-won rating, shrunk toward the league average."""
        return ((self._spw_num[pid] + self.prior_weight * self.league_spw)
                / (self._spw_den[pid] + self.prior_weight))

    def rpw(self, pid) -> float:
        """Return-points-won rating, shrunk toward the league average."""
        return ((self._rpw_num[pid] + self.prior_weight * self.league_rpw)
                / (self._rpw_den[pid] + self.prior_weight))

    def count(self, pid) -> int:
        return self._matches[pid]

    def update(self, w, l, w_svpt: float, w_won: float,
               l_svpt: float, l_won: float) -> None:
        """Fold one completed match into both players' ratings.

        `w`/`l` are the winner/loser ids; `*_svpt` the serve points played
        and `*_won` the serve points won. Opponent adjustment uses the
        pre-match ratings, so calling this in chronological order keeps the
        whole system leak-free.
        """
        s_w = w_won / w_svpt              # winner raw serve-points-won rate
        s_l = l_won / l_svpt             # loser raw serve-points-won rate
        r_w = 1.0 - s_l                  # winner raw return rate (on loser serve)
        r_l = 1.0 - s_w                  # loser raw return rate

        # opponent adjustment — observed rate corrected to a "vs average" basis
        s_w_adj = _clip(s_w + (self.rpw(l) - self.league_rpw))
        s_l_adj = _clip(s_l + (self.rpw(w) - self.league_rpw))
        r_w_adj = _clip(r_w + (self.spw(l) - self.league_spw))
        r_l_adj = _clip(r_l + (self.spw(w) - self.league_spw))

        d = self.decay
        for pid, sa, sp, ra, rp in (
                (w, s_w_adj, w_svpt, r_w_adj, l_svpt),
                (l, s_l_adj, l_svpt, r_l_adj, w_svpt)):
            self._spw_num[pid] = self._spw_num[pid] * d + sa * sp
            self._spw_den[pid] = self._spw_den[pid] * d + sp
            self._rpw_num[pid] = self._rpw_num[pid] * d + ra * rp
            self._rpw_den[pid] = self._rpw_den[pid] * d + rp
            self._matches[pid] += 1
