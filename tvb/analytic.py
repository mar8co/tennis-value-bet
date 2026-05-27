"""Closed-form / DP match-winner probability from per-player serve point-win
probabilities.

Matches the Monte Carlo simulator in expectation but is fast enough to call
inside a backtest loop (the simulator is too slow to run per match). Sets are
treated as i.i.d. and the first serve is averaged out — a standard, accurate
simplification for the *match-winner* market.
"""
from __future__ import annotations

from math import comb


def game_win_prob(p: float) -> float:
    """P(server wins a game) given per-point serve win probability p."""
    q = 1.0 - p
    denom = p * p + q * q
    deuce = (p * p) / denom if denom > 0 else 0.5
    return (p**4 + 4.0 * p**4 * q + 10.0 * p**4 * q * q
            + 20.0 * p**3 * q**3 * deuce)


def tiebreak_win_prob(pa: float, pb: float, a_serves_first: bool = True,
                      target: int = 7) -> float:
    """P(A wins a tie-break). pa/pb = each player's point-win prob on own serve.
    Serve order: 1 point, then 2 points each, alternating."""
    states = {(0, 0): 1.0}
    p_a = 0.0
    for i in range(60):
        a_serves = (a_serves_first if ((i + 1) // 2) % 2 == 0
                    else not a_serves_first)
        pa_point = pa if a_serves else (1.0 - pb)
        nxt: dict = {}
        for (a, b), m in states.items():
            for (na, nb), pr in (((a + 1, b), pa_point),
                                 ((a, b + 1), 1.0 - pa_point)):
                mass = m * pr
                if na >= target and na - nb >= 2:
                    p_a += mass
                elif nb >= target and nb - na >= 2:
                    pass                       # B wins this branch
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + mass
        states = nxt
        if not states:
            break
    return p_a + 0.5 * sum(states.values())    # residual mass is negligible


def set_win_prob(pa: float, pb: float, a_serves_first: bool = True) -> float:
    """P(A wins a set), including a tie-break at 6-6."""
    hold_a = game_win_prob(pa)                 # A wins a game on A's serve
    hold_b = game_win_prob(pb)                 # B wins a game on B's serve
    tb = tiebreak_win_prob(pa, pb, a_serves_first)
    states = {(0, 0): 1.0}
    p_a = 0.0
    while states:
        nxt: dict = {}
        for (a, b), m in states.items():
            a_serves = (a_serves_first if (a + b) % 2 == 0
                        else not a_serves_first)
            pa_game = hold_a if a_serves else (1.0 - hold_b)
            for (na, nb), pr in (((a + 1, b), pa_game),
                                 ((a, b + 1), 1.0 - pa_game)):
                mass = m * pr
                if na == 6 and nb <= 4:
                    p_a += mass
                elif nb == 6 and na <= 4:
                    pass
                elif na == 7 and nb == 5:
                    p_a += mass
                elif nb == 7 and na == 5:
                    pass
                elif na == 6 and nb == 6:
                    p_a += mass * tb
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + mass
        states = nxt
    return p_a


def set_winner_prob(pa: float, pb: float) -> float:
    """P(A wins a single set), with the first serve averaged over the coin
    toss. This is also the first-set-winner probability."""
    return 0.5 * set_win_prob(pa, pb, True) + 0.5 * set_win_prob(pa, pb, False)


def match_win_prob(pa: float, pb: float, best_of: int = 3) -> float:
    """P(A wins the match). Sets are treated as i.i.d.; who serves first is
    averaged over the coin toss, which also makes the result exactly
    symmetric (match_win_prob(p, p) == 0.5)."""
    s = set_winner_prob(pa, pb)
    if best_of >= 5:
        return s**3 * (1.0 + 3.0 * (1.0 - s) + 6.0 * (1.0 - s) ** 2)
    return s * s * (3.0 - 2.0 * s)


# ------------------ total-games & game-margin distributions ----------------

def _convolve(dists: list) -> dict:
    """Convolve a list of {games: prob} distributions (sum of independent RVs)."""
    result = {0: 1.0}
    for d in dists:
        nxt: dict = {}
        for g1, p1 in result.items():
            for g2, p2 in d.items():
                nxt[g1 + g2] = nxt.get(g1 + g2, 0.0) + p1 * p2
        result = nxt
    return result


def set_score_dist(pa: float, pb: float, a_serves_first: bool = True) -> dict:
    """Distribution of a set's final score: {(games_a, games_b): probability}.
    A tie-break set ends 7-6 or 6-7."""
    hold_a = game_win_prob(pa)
    hold_b = game_win_prob(pb)
    tb = tiebreak_win_prob(pa, pb, a_serves_first)
    states = {(0, 0): 1.0}
    out: dict = {}
    while states:
        nxt: dict = {}
        for (a, b), m in states.items():
            a_serves = (a_serves_first if (a + b) % 2 == 0
                        else not a_serves_first)
            pa_game = hold_a if a_serves else (1.0 - hold_b)
            for (na, nb), pr in (((a + 1, b), pa_game),
                                 ((a, b + 1), 1.0 - pa_game)):
                mass = m * pr
                if ((na == 6 and nb <= 4) or (na == 7 and nb == 5)
                        or (nb == 6 and na <= 4) or (nb == 7 and na == 5)):
                    out[(na, nb)] = out.get((na, nb), 0.0) + mass
                elif na == 6 and nb == 6:
                    out[(7, 6)] = out.get((7, 6), 0.0) + mass * tb
                    out[(6, 7)] = out.get((6, 7), 0.0) + mass * (1.0 - tb)
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + mass
        states = nxt
    return out


def _averaged_set_dist(pa: float, pb: float) -> dict:
    """Set-score distribution averaged over who serves first (coin toss)."""
    sd: dict = {}
    for first in (True, False):
        for score, v in set_score_dist(pa, pb, first).items():
            sd[score] = sd.get(score, 0.0) + 0.5 * v
    return sd


def _combine_sets(s0: float, f0: dict, s1: float, f1: dict,
                  best_of: int) -> dict:
    """Combine per-set quantity distributions — f0 when A wins a set, f1 when
    B wins — into the match-total distribution, summing over set counts.
    s0 / s1 are the set-win probabilities of A / B."""
    need = best_of // 2 + 1
    total: dict = {}
    for win_s, win_f, los_s, los_f in ((s0, f0, s1, f1), (s1, f1, s0, f0)):
        for k in range(need, best_of + 1):
            losses = k - need
            weight = comb(k - 1, losses) * win_s**need * los_s**losses
            if weight <= 0.0:
                continue
            for g, p in _convolve([win_f] * need + [los_f] * losses).items():
                total[g] = total.get(g, 0.0) + weight * p
    return total


def total_games_dist(pa: float, pb: float, best_of: int = 3) -> dict:
    """Distribution of the match's total games: {total_games: probability}.
    Sets are treated as i.i.d. and the first serve is averaged out."""
    sd = _averaged_set_dist(pa, pb)
    s0 = sum(v for (ga, gb), v in sd.items() if ga > gb)
    s1 = 1.0 - s0
    f0: dict = {}
    f1: dict = {}
    for (ga, gb), v in sd.items():
        tgt = f0 if ga > gb else f1
        tgt[ga + gb] = tgt.get(ga + gb, 0.0) + v
    f0 = {k: v / s0 for k, v in f0.items()} if s0 > 1e-12 else {0: 1.0}
    f1 = {k: v / s1 for k, v in f1.items()} if s1 > 1e-12 else {0: 1.0}
    return _combine_sets(s0, f0, s1, f1, best_of)


def total_margin_dist(pa: float, pb: float, best_of: int = 3) -> dict:
    """Distribution of the match game margin (games_a - games_b):
    {margin: probability}. Positive = A won more games."""
    sd = _averaged_set_dist(pa, pb)
    s0 = sum(v for (ga, gb), v in sd.items() if ga > gb)
    s1 = 1.0 - s0
    f0: dict = {}
    f1: dict = {}
    for (ga, gb), v in sd.items():
        tgt = f0 if ga > gb else f1
        tgt[ga - gb] = tgt.get(ga - gb, 0.0) + v
    f0 = {k: v / s0 for k, v in f0.items()} if s0 > 1e-12 else {0: 1.0}
    f1 = {k: v / s1 for k, v in f1.items()} if s1 > 1e-12 else {0: 1.0}
    return _combine_sets(s0, f0, s1, f1, best_of)


def total_tiebreaks_dist(pa: float, pb: float, best_of: int = 3) -> dict:
    """Distribution of the number of tie-breaks in the match:
    {tiebreak_count: probability}. A set has a tie-break iff it ends 7-6."""
    sd = _averaged_set_dist(pa, pb)
    s0 = sum(v for (ga, gb), v in sd.items() if ga > gb)
    s1 = 1.0 - s0
    f0: dict = {}
    f1: dict = {}
    for (ga, gb), v in sd.items():
        tb = 1 if {ga, gb} == {6, 7} else 0
        tgt = f0 if ga > gb else f1
        tgt[tb] = tgt.get(tb, 0.0) + v
    f0 = {k: v / s0 for k, v in f0.items()} if s0 > 1e-12 else {0: 1.0}
    f1 = {k: v / s1 for k, v in f1.items()} if s1 > 1e-12 else {0: 1.0}
    return _combine_sets(s0, f0, s1, f1, best_of)


def prob_tiebreak(pa: float, pb: float, best_of: int = 3) -> float:
    """P(the match contains at least one tie-break)."""
    return 1.0 - total_tiebreaks_dist(pa, pb, best_of).get(0, 0.0)


def set_breaks_dist(pa: float, pb: float, a_serves_first: bool = True) -> dict:
    """Distribution of (set winner, number of breaks of serve) for one set.
    The tie-break game itself is not counted as a break."""
    hold_a = game_win_prob(pa)
    hold_b = game_win_prob(pb)
    tb = tiebreak_win_prob(pa, pb, a_serves_first)
    states = {(0, 0, 0): 1.0}                  # (games_a, games_b, breaks)
    out: dict = {}
    while states:
        nxt: dict = {}
        for (ga, gb, bk), m in states.items():
            a_serves = (a_serves_first if (ga + gb) % 2 == 0
                        else not a_serves_first)
            hold = hold_a if a_serves else hold_b
            for server_won, pr in ((True, hold), (False, 1.0 - hold)):
                if a_serves:
                    nga, ngb = (ga + 1, gb) if server_won else (ga, gb + 1)
                else:
                    nga, ngb = (ga, gb + 1) if server_won else (ga + 1, gb)
                nbk = bk if server_won else bk + 1
                mass = m * pr
                if (nga == 6 and ngb <= 4) or (nga == 7 and ngb == 5):
                    out[(0, nbk)] = out.get((0, nbk), 0.0) + mass
                elif (ngb == 6 and nga <= 4) or (ngb == 7 and nga == 5):
                    out[(1, nbk)] = out.get((1, nbk), 0.0) + mass
                elif nga == 6 and ngb == 6:
                    out[(0, nbk)] = out.get((0, nbk), 0.0) + mass * tb
                    out[(1, nbk)] = out.get((1, nbk), 0.0) + mass * (1.0 - tb)
                else:
                    nxt[(nga, ngb, nbk)] = nxt.get((nga, ngb, nbk), 0.0) + mass
        states = nxt
    return out


def total_breaks_dist(pa: float, pb: float, best_of: int = 3) -> dict:
    """Distribution of the total number of service breaks in the match."""
    sd: dict = {}
    for first in (True, False):
        for k, v in set_breaks_dist(pa, pb, first).items():
            sd[k] = sd.get(k, 0.0) + 0.5 * v
    s0 = sum(v for (w, _), v in sd.items() if w == 0)
    s1 = 1.0 - s0
    f0: dict = {}
    f1: dict = {}
    for (w, bk), v in sd.items():
        tgt = f0 if w == 0 else f1
        tgt[bk] = tgt.get(bk, 0.0) + v
    f0 = {k: v / s0 for k, v in f0.items()} if s0 > 1e-12 else {0: 1.0}
    f1 = {k: v / s1 for k, v in f1.items()} if s1 > 1e-12 else {0: 1.0}
    return _combine_sets(s0, f0, s1, f1, best_of)


def prob_over(dist: dict, line: float) -> float:
    """P(total > line) from a {games: prob} distribution."""
    return sum(p for g, p in dist.items() if g > line)


def dist_mean(dist: dict) -> float:
    """Expected value of a {games: prob} distribution."""
    return sum(g * p for g, p in dist.items())
