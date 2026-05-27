"""Convert bookmaker odds into probabilities and value metrics."""
from __future__ import annotations


def implied_probability(odds: float) -> float:
    """Raw implied probability — still contains the bookmaker margin."""
    return 1.0 / odds


def book_margin(odds_list: list[float]) -> float:
    """Overround: how far the book's implied probabilities exceed 1.0."""
    return sum(1.0 / o for o in odds_list) - 1.0


def fair_probabilities(odds_list: list[float],
                       method: str = "proportional") -> list[float]:
    """Remove the margin so probabilities sum to 1.

    'proportional' is the simple normalisation. It mis-prices under the
    favourite-longshot bias; the Shin method is the planned upgrade.
    """
    inv = [1.0 / o for o in odds_list]
    total = sum(inv)
    if method == "proportional":
        return [x / total for x in inv]
    raise ValueError(f"unknown method: {method}")


def expected_value(model_prob: float, odds: float) -> float:
    """EV per unit staked:  p_model * odds - 1."""
    return model_prob * odds - 1.0


def edge(model_prob: float, fair_prob: float) -> float:
    """How much higher the model's probability is than the fair market one."""
    return model_prob - fair_prob


def kelly_fraction(model_prob: float, odds: float,
                   fraction: float = 0.25) -> float:
    """Fractional-Kelly stake as a share of bankroll. 0 when there is no edge."""
    b = odds - 1.0
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, f * fraction)


def confidence(edge_value: float, model_prob: float) -> str:
    """Rough confidence label.

    A very large edge is usually a sign of model error, not free money,
    so it is deliberately downgraded.
    """
    if edge_value < 0.03:
        return "bassa"
    if edge_value > 0.15:
        return "bassa"          # too good to be true -> suspect the model
    if edge_value >= 0.07:
        return "alta"
    return "media"
