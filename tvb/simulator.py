"""Point-by-point Monte Carlo match simulator — the core of the project.

The whole engine is driven by two numbers:
    p0 = probability player 0 wins a point on his/her own serve
    p1 = probability player 1 wins a point on his/her own serve

Simulating every point reproduces the *joint* distribution of all markets
(match winner, set winner, total games, handicap, set score, tie-break,
breaks) coherently — no separate model per market, no self-arbitrage.

The Monte Carlo loop is plain Python. ~20k best-of-3 sims run in a few
seconds; vectorising with numpy is a future optimisation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    winner: int               # 0 or 1
    set_scores: list          # [(games_p0, games_p1), ...] per set
    sets_won: tuple           # (sets_p0, sets_p1)
    total_games: int
    tiebreaks: int            # number of tie-breaks played
    breaks: tuple             # (service breaks by p0, by p1)


def _play_game(p_server: float, rng: random.Random) -> bool:
    """Simulate one service game. Return True if the server held."""
    s = r = 0
    while True:
        if rng.random() < p_server:
            s += 1
        else:
            r += 1
        if s >= 4 and s - r >= 2:
            return True
        if r >= 4 and r - s >= 2:
            return False


def _play_tiebreak(p0: float, p1: float, first_server: int,
                   rng: random.Random, target: int = 7) -> int:
    """Simulate a tie-break. Serve order: 1 point, then 2 points each."""
    pts = [0, 0]
    i = 0
    while True:
        server = first_server if ((i + 1) // 2) % 2 == 0 else 1 - first_server
        p = p0 if server == 0 else p1
        winner = server if rng.random() < p else 1 - server
        pts[winner] += 1
        i += 1
        if pts[0] >= target and pts[0] - pts[1] >= 2:
            return 0
        if pts[1] >= target and pts[1] - pts[0] >= 2:
            return 1


def simulate_match(p0: float, p1: float, best_of: int = 3,
                   first_server: int | None = None,
                   final_set_tb_target: int = 7,
                   rng: random.Random | None = None) -> MatchResult:
    """Simulate a full match point by point."""
    rng = rng or random
    if first_server is None:
        first_server = rng.randint(0, 1)
    server = first_server                     # toggles every game (and tie-break)
    sets_to_win = best_of // 2 + 1
    sets_won = [0, 0]
    set_scores: list = []
    tiebreaks = 0
    breaks = [0, 0]

    while max(sets_won) < sets_to_win:
        games = [0, 0]
        is_final = (sets_won[0] == sets_to_win - 1
                    and sets_won[1] == sets_to_win - 1)
        while True:
            if max(games) >= 6 and abs(games[0] - games[1]) >= 2:
                break
            if games[0] == 6 and games[1] == 6:
                target = final_set_tb_target if is_final else 7
                tb_winner = _play_tiebreak(p0, p1, server, rng, target)
                games[tb_winner] += 1
                tiebreaks += 1
                server = 1 - server
                break
            p_server = p0 if server == 0 else p1
            server_won = _play_game(p_server, rng)
            game_winner = server if server_won else 1 - server
            games[game_winner] += 1
            if not server_won:                # returner broke serve
                breaks[game_winner] += 1
            server = 1 - server
        set_winner = 0 if games[0] > games[1] else 1
        sets_won[set_winner] += 1
        set_scores.append((games[0], games[1]))

    winner = 0 if sets_won[0] > sets_won[1] else 1
    total_games = sum(g0 + g1 for g0, g1 in set_scores)
    return MatchResult(winner, set_scores, tuple(sets_won),
                       total_games, tiebreaks, tuple(breaks))


@dataclass
class MarketBook:
    """Per-simulation outcomes plus market-probability query methods."""
    n: int
    winner: np.ndarray
    set1_winner: np.ndarray
    total_games: np.ndarray
    margin: np.ndarray            # games(p0) - games(p1)
    had_tiebreak: np.ndarray
    total_breaks: np.ndarray
    sets_won_p0: np.ndarray
    sets_won_p1: np.ndarray

    def p_match_winner(self, player: int) -> float:
        return float((self.winner == player).mean())

    def p_set1_winner(self, player: int) -> float:
        return float((self.set1_winner == player).mean())

    def p_total_over(self, line: float) -> float:
        return float((self.total_games > line).mean())

    def p_total_under(self, line: float) -> float:
        return float((self.total_games < line).mean())

    def p_handicap(self, player: int, line: float) -> float:
        """P(player covers the game handicap). line < 0 = giving games away."""
        m = self.margin if player == 0 else -self.margin
        return float((m + line > 0).mean())

    def p_tiebreak_yes(self) -> float:
        return float(self.had_tiebreak.mean())

    def p_tiebreak_no(self) -> float:
        return 1.0 - self.p_tiebreak_yes()

    def p_set_score(self, player: int, dropped_sets: int) -> float:
        """P(player wins having lost `dropped_sets` sets). 2-0 -> 0, 2-1 -> 1."""
        opp = self.sets_won_p1 if player == 0 else self.sets_won_p0
        return float(((self.winner == player) & (opp == dropped_sets)).mean())

    def p_total_breaks_over(self, line: float) -> float:
        return float((self.total_breaks > line).mean())


def monte_carlo(p0: float, p1: float, best_of: int = 3,
                n_sims: int = 20_000, seed: int | None = None,
                final_set_tb_target: int = 7) -> MarketBook:
    """Run `n_sims` simulated matches and collect every market outcome."""
    rng = random.Random(seed)
    winner = np.empty(n_sims, dtype=np.int8)
    set1 = np.empty(n_sims, dtype=np.int8)
    tgames = np.empty(n_sims, dtype=np.int16)
    margin = np.empty(n_sims, dtype=np.int16)
    had_tb = np.empty(n_sims, dtype=bool)
    breaks = np.empty(n_sims, dtype=np.int16)
    sw0 = np.empty(n_sims, dtype=np.int8)
    sw1 = np.empty(n_sims, dtype=np.int8)

    for i in range(n_sims):
        r = simulate_match(p0, p1, best_of, None, final_set_tb_target, rng)
        s1 = r.set_scores[0]
        winner[i] = r.winner
        set1[i] = 0 if s1[0] > s1[1] else 1
        tgames[i] = r.total_games
        margin[i] = (sum(s[0] for s in r.set_scores)
                     - sum(s[1] for s in r.set_scores))
        had_tb[i] = r.tiebreaks > 0
        breaks[i] = r.breaks[0] + r.breaks[1]
        sw0[i] = r.sets_won[0]
        sw1[i] = r.sets_won[1]

    return MarketBook(n_sims, winner, set1, tgames, margin,
                      had_tb, breaks, sw0, sw1)
