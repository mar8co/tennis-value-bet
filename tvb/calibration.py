"""Output recalibration for the match-winner model.

The point / serve-return model is systematically overconfident — its raw
probabilities sit too far from 50%. Temperature scaling corrects this with a
single fitted parameter T:

    p_cal = sigmoid( logit(p_raw) / T )

T > 1 pulls probabilities toward 50% (less confident); T < 1 sharpens them.

Because the match-winner market is symmetric — swapping the two players maps
p -> 1 - p — temperature scaling with no intercept is the principled choice:
it preserves that symmetry exactly, whereas a 2-parameter logistic fit would
not. The transform is monotone, so it never changes which side is favoured
(accuracy is untouched); it only makes the probabilities honest.
"""
from __future__ import annotations

import numpy as np

from .analytic import prob_over


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def apply_temperature(p, temperature: float):
    """Recalibrate probabilities with a fitted temperature.

    Works on a scalar or an array. Returns a numpy value — wrap a scalar
    result in float() if a plain float is needed.
    """
    z = _logit(np.asarray(p, dtype=float))
    return 1.0 / (1.0 + np.exp(-z / temperature))


def fit_temperature(p, y) -> float:
    """Find the temperature that minimises log loss of p against outcomes y.

    Log loss is convex in 1/T, so a coarse-then-fine grid search finds the
    optimum exactly enough (and keeps the module dependency-free).
    """
    z = _logit(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)

    def nll(t: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    best_t, best = 1.0, nll(1.0)
    for t in np.arange(0.50, 4.001, 0.05):
        v = nll(float(t))
        if v < best:
            best, best_t = v, float(t)
    for t in np.arange(best_t - 0.05, best_t + 0.05, 0.005):
        if t > 0:
            v = nll(float(t))
            if v < best:
                best, best_t = v, float(t)
    return round(best_t, 3)


def apply_logistic(p, a: float, b: float):
    """Recalibrate probabilities with a fitted 2-parameter logistic:
    p_cal = sigmoid(a * logit(p) + b). The intercept b lets it correct a
    directional bias, not just miscalibrated spread."""
    z = _logit(np.asarray(p, dtype=float))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def fit_logistic(p, y) -> tuple:
    """Fit the 2-parameter logistic recalibration (Platt scaling) that
    minimises log loss. Unlike temperature scaling this has an intercept, so
    it suits a non-symmetric market (e.g. tie-break yes/no) where one side is
    systematically over- or under-predicted. Coarse-then-fine grid search."""
    z = _logit(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)

    def nll(a: float, b: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    best = (1.0, 0.0)
    best_nll = nll(1.0, 0.0)
    for a in np.arange(0.30, 2.001, 0.10):
        for b in np.arange(-2.0, 1.001, 0.10):
            v = nll(float(a), float(b))
            if v < best_nll:
                best_nll, best = v, (float(a), float(b))
    a0, b0 = best
    for a in np.arange(a0 - 0.10, a0 + 0.101, 0.01):
        for b in np.arange(b0 - 0.10, b0 + 0.101, 0.01):
            v = nll(float(a), float(b))
            if v < best_nll:
                best_nll, best = v, (float(a), float(b))
    return round(best[0], 3), round(best[1], 3)


# ----------------- total-games bias correction (location shift) ------------
#
# The over/under model is not over/under-confident but *biased*: the i.i.d.-
# points assumption inflates set length, so the predicted total-games
# distribution sits too high. The fix is a location shift, not a temperature.


def apply_total_shift(dist: dict, line: float, delta: float) -> float:
    """P(total over `line`) after shifting the predicted total-games
    distribution down by `delta` games. Shifting the distribution down by δ
    is equivalent to reading the raw distribution at line + δ."""
    return prob_over(dist, line + delta)


def fit_total_shift(dists, actuals, line: float) -> float:
    """Find the location shift δ (games) that minimises log loss of the
    over/under bet at `line`. One parameter — a coarse-then-fine grid search.
    The model overpredicts totals, so δ > 0 is expected."""
    y = (np.asarray(actuals, dtype=float) > line).astype(float)
    dists = list(dists)

    def nll(delta: float) -> float:
        p = np.array([prob_over(d, line + delta) for d in dists])
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    best_d, best = 0.0, nll(0.0)
    for delta in np.arange(-2.0, 5.001, 0.25):
        v = nll(float(delta))
        if v < best:
            best, best_d = v, float(delta)
    for delta in np.arange(best_d - 0.25, best_d + 0.25, 0.02):
        v = nll(float(delta))
        if v < best:
            best, best_d = v, float(delta)
    return round(best_d, 3)
