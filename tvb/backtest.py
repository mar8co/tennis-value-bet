"""Leak-free historical backtest of the match-winner model.

Walks Tennis-Data.co.uk matches in chronological order, maintaining Elo
ratings updated only with *past* results. At each match the model's
pre-match probability is recorded and later compared with the actual
outcome and with the bookmaker's de-margined implied probability.

The Elo here is overall (not surface-specific) to limit cold-start noise;
surface-specific Elo and the full simulation model are future extensions.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .analytic import (dist_mean, match_win_prob, prob_over, prob_tiebreak,
                       set_winner_prob, total_breaks_dist, total_games_dist,
                       total_margin_dist)
from .config import ELO_BASE, ELO_K
from .elo import win_probability
from .serve_return import matchup_serve_probs
from .sr_ratings import SRRatings
from .value import expected_value, fair_probabilities

# bookmaker odds column pairs, sharpest first
_BOOK_PAIRS = [("psw", "psl"), ("avgw", "avgl"),
               ("b365w", "b365l"), ("maxw", "maxl")]


def _pick_book(df: pd.DataFrame, bookmaker: str = "auto") -> tuple:
    """Choose the (winner_odds, loser_odds) column pair to evaluate against."""
    if bookmaker != "auto":
        return f"{bookmaker}w", f"{bookmaker}l"
    for w, l in _BOOK_PAIRS:
        if w in df.columns and df[w].notna().any():
            return w, l
    return None, None


def run_elo_backtest(odds_hist: pd.DataFrame, min_prior: int = 10,
                     bookmaker: str = "auto") -> pd.DataFrame:
    """Replay matches chronologically; return per-match prediction records.

    `odds_hist` needs columns: date, winner, loser, comment + bookmaker
    odds columns (e.g. psw/psl). Matches where either player has fewer
    than `min_prior` earlier matches still update Elo but are excluded
    from the returned records (cold-start filtering).

    Returned columns: date, p_model, p_book, y, odds_a, odds_b — where
    'A' is the alphabetically-first player (a result-independent label)
    and y = 1 if A won.
    """
    df = odds_hist.dropna(subset=["date", "winner", "loser"]).copy()
    df = df[df["comment"].astype(str).str.strip() == "Completed"]
    df = df.sort_values("date")
    bw, bl = _pick_book(df, bookmaker)

    elo: dict = defaultdict(lambda: ELO_BASE)
    seen: dict = defaultdict(int)
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner, r.loser
        rw, rl = elo[w], elo[l]
        p_w = win_probability(rw, rl)            # P(winner wins), pre-match

        a_is_winner = str(w) <= str(l)           # neutral A/B label
        pa_model = p_w if a_is_winner else 1.0 - p_w
        ya = 1 if a_is_winner else 0

        odds_w = getattr(r, bw) if bw else float("nan")
        odds_l = getattr(r, bl) if bl else float("nan")
        odds_a = odds_w if a_is_winner else odds_l
        odds_b = odds_l if a_is_winner else odds_w
        if (np.isfinite(odds_a) and np.isfinite(odds_b)
                and odds_a > 1 and odds_b > 1):
            pa_book = fair_probabilities([odds_a, odds_b])[0]
        else:
            pa_book, odds_a, odds_b = float("nan"), float("nan"), float("nan")

        if seen[w] >= min_prior and seen[l] >= min_prior:
            rows.append((r.date, pa_model, pa_book, ya, odds_a, odds_b))

        # update Elo *after* the prediction has been recorded (no leakage)
        elo[w] = rw + ELO_K * (1.0 - p_w)
        elo[l] = rl - ELO_K * (1.0 - p_w)
        seen[w] += 1
        seen[l] += 1

    return pd.DataFrame(
        rows, columns=["date", "p_model", "p_book", "y", "odds_a", "odds_b"])


# ----------------------------------------------------------------- metrics
def brier(p, y) -> float:
    """Mean squared error of probability vs 0/1 outcome. Lower is better."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-15) -> float:
    """Logarithmic loss — punishes confident wrong calls. Lower is better."""
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(p, y) -> float:
    """Share of matches where the more likely side actually won."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p > 0.5) == (y == 1)))


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability vs observed win rate per bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                     "n": int(mask.sum()),
                     "pred_mean": round(float(p[mask].mean()), 3),
                     "obs_rate": round(float(y[mask].mean()), 3)})
    return pd.DataFrame(rows)


def betting_simulation(records: pd.DataFrame, min_ev: float = 0.0,
                       stake: float = 1.0) -> dict:
    """Flat-stake bet on the best side of each market that exceeds `min_ev`.

    At most one side per match is bet: if both sides show positive EV (a sign
    of large model disagreement with the book), only the higher-EV side is
    taken.  Betting both sides of the same binary market guarantees a loss
    when odds are margined.
    """
    n, staked, profit = 0, 0.0, 0.0
    for r in records.itertuples(index=False):
        if not (np.isfinite(r.odds_a) and np.isfinite(r.odds_b)):
            continue
        sides = (
            (r.p_model,       r.odds_a, r.y == 1),
            (1.0 - r.p_model, r.odds_b, r.y == 0),
        )
        best = max(sides, key=lambda t: expected_value(t[0], t[1]))
        p_model, odds, won = best
        if expected_value(p_model, odds) > min_ev:
            n += 1
            staked += stake
            profit += stake * (odds - 1.0) if won else -stake
    return {"n_bets": n, "staked": staked, "profit": profit,
            "roi": profit / staked if staked else 0.0}


def summarize(records: pd.DataFrame) -> dict:
    """Headline metrics; model vs bookmaker compared on the same subset."""
    rb = records[records["p_book"].notna()]
    has = len(rb) > 0
    return {
        "n_eval": len(records),
        "n_with_odds": len(rb),
        "date_min": records["date"].min(),
        "date_max": records["date"].max(),
        "brier_model": brier(rb["p_model"], rb["y"]) if has else float("nan"),
        "logloss_model": log_loss(rb["p_model"], rb["y"]) if has else float("nan"),
        "acc_model": accuracy(rb["p_model"], rb["y"]) if has else float("nan"),
        "brier_book": brier(rb["p_book"], rb["y"]) if has else float("nan"),
        "logloss_book": log_loss(rb["p_book"], rb["y"]) if has else float("nan"),
        "acc_book": accuracy(rb["p_book"], rb["y"]) if has else float("nan"),
    }


# ----------- Sackmann backtest: serve/return model vs the Elo baseline -----

_SR_STAT_COLS = ["w_svpt", "w_1stWon", "w_2ndWon",
                 "l_svpt", "l_1stWon", "l_2ndWon"]


def run_sackmann_backtest(matches: pd.DataFrame, tour: str = "atp",
                          min_prior: int = 10, decay: float = 0.97,
                          prior_weight: float = 250.0) -> pd.DataFrame:
    """Replay Sackmann matches chronologically, scoring two predictors on
    every match: plain overall Elo and the recency-weighted, opponent-
    adjusted serve/return model (via the analytic match-winner formula).

    Both predictors see exactly the same matches, so their metrics are
    directly comparable. Returns columns date, y, p_elo, p_sr — where 'A'
    is the player with the smaller id (a result-independent label) and
    y = 1 if A won.
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df.sort_values("tourney_date")

    elo: dict = defaultdict(lambda: ELO_BASE)
    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        best_of = int(r.best_of) if r.best_of in (3, 5) else 3

        # predictions, made before either rating is updated
        p_w_elo = win_probability(elo[w], elo[l])
        pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                     sr.spw(l), sr.rpw(l), tour)
        p_w_sr = match_win_prob(pw, pl, best_of)

        a_is_winner = w < l                       # neutral A/B label by id
        ya = 1 if a_is_winner else 0
        pa_elo = p_w_elo if a_is_winner else 1.0 - p_w_elo
        pa_sr = p_w_sr if a_is_winner else 1.0 - p_w_sr
        if seen[w] >= min_prior and seen[l] >= min_prior:
            rows.append((r.tourney_date, ya, pa_elo, pa_sr))

        # updates
        elo[w] += ELO_K * (1.0 - p_w_elo)
        elo[l] -= ELO_K * (1.0 - p_w_elo)
        seen[w] += 1
        seen[l] += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "y", "p_elo", "p_sr"])


# ----------------- over/under total-games backtest (best-of-3) --------------

def parse_score(score) -> tuple | None:
    """(match-winner games, loser games) from a Sackmann score string, or
    None if the match did not finish (retirement / walkover) or the score
    cannot be parsed."""
    if not isinstance(score, str) or not score.strip():
        return None
    wg = lg = 0
    for token in score.split():
        core = token.split("(")[0]            # drop tie-break annotation
        a, sep, b = core.partition("-")
        if not sep or not (a.isdigit() and b.isdigit()):
            return None                       # RET / W/O / DEF / junk token
        wg += int(a)
        lg += int(b)
    return (wg, lg) if (wg + lg) > 0 else None


def parse_total_games(score) -> int | None:
    """Total games of a completed match, or None."""
    parsed = parse_score(score)
    return None if parsed is None else parsed[0] + parsed[1]


def parse_first_set(score) -> int | None:
    """1 if the match winner won the first set, 0 if the match loser did,
    None if the match did not finish or the score cannot be parsed."""
    if parse_score(score) is None:
        return None
    a, sep, b = score.split()[0].split("(")[0].partition("-")
    if not sep or not (a.isdigit() and b.isdigit()):
        return None
    return 1 if int(a) > int(b) else 0


def parse_tiebreak(score) -> int | None:
    """1 if the match had at least one tie-break (a 7-6 / 6-7 set), 0 if not,
    None if the match did not finish or the score cannot be parsed."""
    if parse_score(score) is None:
        return None
    for token in score.split():
        a, _, b = token.split("(")[0].partition("-")
        if {a, b} == {"7", "6"}:
            return 1
    return 0


def run_total_games_backtest(matches: pd.DataFrame, tour: str = "atp",
                             line: float = 22.5, min_prior: int = 10,
                             decay: float = 0.97,
                             prior_weight: float = 250.0) -> pd.DataFrame:
    """Backtest the over/under total-games market on completed best-of-3
    Sackmann matches.

    For each match the recency-weighted serve/return model predicts the
    total-games distribution; the actual total is parsed from the score.
    A naive baseline uses the empirical distribution of *past* matches'
    totals. Returns columns date, actual, pred_mean, p_model, p_base, y,
    dist (y = 1 if the actual total was over `line`; dist = the raw
    predicted total-games distribution, kept for fitting a bias correction).
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df[df["best_of"] == 3].sort_values("tourney_date")

    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    past_totals: dict = defaultdict(int)      # total games -> count (baseline)
    past_n = 0
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        actual = parse_total_games(r.score)

        if (actual is not None and past_n > 0
                and seen[w] >= min_prior and seen[l] >= min_prior):
            pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                         sr.spw(l), sr.rpw(l), tour)
            dist = total_games_dist(pw, pl, best_of=3)
            p_base = sum(c for t, c in past_totals.items()
                         if t > line) / past_n
            rows.append((r.tourney_date, actual, dist_mean(dist),
                         prob_over(dist, line), p_base,
                         1 if actual > line else 0, dist))

        # updates
        seen[w] += 1
        seen[l] += 1
        if actual is not None:
            past_totals[actual] += 1
            past_n += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "actual", "pred_mean",
                                       "p_model", "p_base", "y", "dist"])


# ------------------- game-handicap backtest (best-of-3) --------------------

_HANDICAP_LINES = (-7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5)


def run_handicap_backtest(matches: pd.DataFrame, tour: str = "atp",
                          min_prior: int = 10, decay: float = 0.97,
                          prior_weight: float = 250.0,
                          lines: tuple = _HANDICAP_LINES) -> pd.DataFrame:
    """Backtest the game-handicap market on completed best-of-3 Sackmann
    matches.

    The recency-weighted serve/return model predicts the game-margin
    distribution; the actual margin is parsed from the score. A naive
    baseline uses the empirical distribution of *past* margins. The market
    is evaluated over a grid of handicap lines.

    Returns long records — one row per (match, line): date, actual,
    pred_mean, line, p_model, p_base, y. 'A' is the player with the smaller
    id; margin = games(A) - games(B); A covers a line L when margin + L > 0,
    so y = 1 when actual > -L.
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df[df["best_of"] == 3].sort_values("tourney_date")

    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    past_margins: dict = defaultdict(int)     # margin -> count (baseline)
    past_n = 0
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        parsed = parse_score(r.score)
        actual = None
        if parsed is not None:
            wg, lg = parsed
            actual = (wg - lg) if w < l else (lg - wg)   # margin, A's view

        if (actual is not None and past_n > 0
                and seen[w] >= min_prior and seen[l] >= min_prior):
            pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                         sr.spw(l), sr.rpw(l), tour)
            # margin distribution from A's perspective (A = smaller id)
            dist = (total_margin_dist(pw, pl, best_of=3) if w < l
                    else total_margin_dist(pl, pw, best_of=3))
            mean_m = dist_mean(dist)
            for line in lines:
                p_base = sum(c for mg, c in past_margins.items()
                             if mg > -line) / past_n
                rows.append((r.tourney_date, actual, mean_m, line,
                             prob_over(dist, -line), p_base,
                             1 if actual > -line else 0))

        # updates
        seen[w] += 1
        seen[l] += 1
        if actual is not None:
            past_margins[actual] += 1
            past_n += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "actual", "pred_mean", "line",
                                       "p_model", "p_base", "y"])


# ---------------------- first-set-winner backtest --------------------------

def run_set1_backtest(matches: pd.DataFrame, tour: str = "atp",
                      min_prior: int = 10, decay: float = 0.97,
                      prior_weight: float = 250.0) -> pd.DataFrame:
    """Backtest the first-set-winner market on completed Sackmann matches,
    scoring the recency-weighted serve/return set-winner model and a plain
    Elo baseline on identical matches.

    Returns columns date, y, p_elo, p_sr — 'A' is the player with the
    smaller id and y = 1 if A won the first set.
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df.sort_values("tourney_date")

    elo: dict = defaultdict(lambda: ELO_BASE)
    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        winner_won_set1 = parse_first_set(r.score)        # 1 / 0 / None
        p_w_elo = win_probability(elo[w], elo[l])

        if (winner_won_set1 is not None
                and seen[w] >= min_prior and seen[l] >= min_prior):
            pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                         sr.spw(l), sr.rpw(l), tour)
            p_w_set1 = set_winner_prob(pw, pl)            # P(winner takes set 1)
            a_is_winner = w < l
            ya = winner_won_set1 if a_is_winner else 1 - winner_won_set1
            pa_elo = p_w_elo if a_is_winner else 1.0 - p_w_elo
            pa_sr = p_w_set1 if a_is_winner else 1.0 - p_w_set1
            rows.append((r.tourney_date, ya, pa_elo, pa_sr))

        # updates
        elo[w] += ELO_K * (1.0 - p_w_elo)
        elo[l] -= ELO_K * (1.0 - p_w_elo)
        seen[w] += 1
        seen[l] += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "y", "p_elo", "p_sr"])


# ---------------------- tie-break yes/no backtest (best-of-3) --------------

def run_tiebreak_backtest(matches: pd.DataFrame, tour: str = "atp",
                          min_prior: int = 10, decay: float = 0.97,
                          prior_weight: float = 250.0) -> pd.DataFrame:
    """Backtest the tie-break yes/no market on completed best-of-3 Sackmann
    matches. The serve/return model predicts P(at least one tie-break); the
    actual outcome is parsed from the score. A naive baseline uses the
    empirical tie-break rate of past matches.

    Returns columns date, y, p_model, p_base — y = 1 if the match had a
    tie-break. The market is match-level, so there is no A/B labelling.
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df[df["best_of"] == 3].sort_values("tourney_date")

    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    past_yes = 0
    past_n = 0
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        actual = parse_tiebreak(r.score)          # 1 / 0 / None

        if (actual is not None and past_n > 0
                and seen[w] >= min_prior and seen[l] >= min_prior):
            pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                         sr.spw(l), sr.rpw(l), tour)
            rows.append((r.tourney_date, actual,
                         prob_tiebreak(pw, pl, best_of=3),
                         past_yes / past_n))

        seen[w] += 1
        seen[l] += 1
        if actual is not None:
            past_yes += actual
            past_n += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "y", "p_model", "p_base"])


# ------------------- total-breaks backtest (best-of-3) ---------------------

_BREAK_COLS = ["w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]


def run_total_breaks_backtest(matches: pd.DataFrame, tour: str = "atp",
                              line: float = 6.5, min_prior: int = 10,
                              decay: float = 0.97,
                              prior_weight: float = 250.0) -> pd.DataFrame:
    """Backtest the over/under total-breaks market on completed best-of-3
    Sackmann matches.

    The serve/return model predicts the total-breaks distribution; the actual
    break count is recovered from the Sackmann break-point columns
    (bpFaced - bpSaved, summed over both players). A naive baseline uses the
    empirical distribution of past matches' break counts. Returns columns
    date, actual, pred_mean, p_model, p_base, y, dist.
    """
    df = matches.dropna(subset=["winner_id", "loser_id", "tourney_date"]).copy()
    df = df[df["best_of"] == 3].sort_values("tourney_date")

    seen: dict = defaultdict(int)
    sr = SRRatings(tour=tour, decay=decay, prior_weight=prior_weight)
    past: dict = defaultdict(int)              # break count -> count (baseline)
    past_n = 0
    rows = []

    for r in df.itertuples(index=False):
        w, l = r.winner_id, r.loser_id
        actual = None
        if all(pd.notna(getattr(r, c)) for c in _BREAK_COLS):
            actual = int((r.w_bpFaced - r.w_bpSaved)
                         + (r.l_bpFaced - r.l_bpSaved))

        if (actual is not None and actual >= 0 and past_n > 0
                and seen[w] >= min_prior and seen[l] >= min_prior):
            pw, pl = matchup_serve_probs(sr.spw(w), sr.rpw(w),
                                         sr.spw(l), sr.rpw(l), tour)
            dist = total_breaks_dist(pw, pl, best_of=3)
            p_base = sum(c for t, c in past.items() if t > line) / past_n
            rows.append((r.tourney_date, actual, dist_mean(dist),
                         prob_over(dist, line), p_base,
                         1 if actual > line else 0, dist))

        seen[w] += 1
        seen[l] += 1
        if actual is not None and actual >= 0:
            past[actual] += 1
            past_n += 1
        if (all(pd.notna(getattr(r, c)) for c in _SR_STAT_COLS)
                and r.w_svpt > 0 and r.l_svpt > 0):
            sr.update(w, l, r.w_svpt, r.w_1stWon + r.w_2ndWon,
                      r.l_svpt, r.l_1stWon + r.l_2ndWon)

    return pd.DataFrame(rows, columns=["date", "actual", "pred_mean",
                                       "p_model", "p_base", "y", "dist"])
