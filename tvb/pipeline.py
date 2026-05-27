"""Glue: turn a simulated match + bookmaker odds into a ranked value-bet list."""
from __future__ import annotations

from dataclasses import dataclass

from .config import MIN_EDGE, MIN_EV
from .value import (confidence, edge, expected_value, fair_probabilities,
                    kelly_fraction)


@dataclass
class ValueBet:
    match: str
    market: str
    selection: str
    odds: float
    market_prob: float       # fair (de-margined) probability
    model_prob: float
    edge: float
    ev: float
    kelly: float
    confidence: str


def evaluate_match(match_name: str, book, markets: dict) -> list[ValueBet]:
    """Compute model-vs-market metrics for every selection.

    `markets` maps a market name to:
        {"selections": [
            {"label": str,
             "odds": float,
             "model": callable(MarketBook) -> probability}, ...]}
    Selections inside one market are de-margined together.
    """
    bets: list[ValueBet] = []
    for market_name, spec in markets.items():
        sels = spec["selections"]
        fair = fair_probabilities([s["odds"] for s in sels])
        for s, fp in zip(sels, fair):
            mp = float(s["model"](book))
            eg = edge(mp, fp)
            ev = expected_value(mp, s["odds"])
            bets.append(ValueBet(
                match=match_name, market=market_name, selection=s["label"],
                odds=s["odds"], market_prob=fp, model_prob=mp,
                edge=eg, ev=ev, kelly=kelly_fraction(mp, s["odds"]),
                confidence=confidence(eg, mp),
            ))
    return bets


def rank_value_bets(bets: list[ValueBet], min_edge: float = MIN_EDGE,
                    min_ev: float = MIN_EV,
                    min_prob: float = 0.0) -> list[ValueBet]:
    """Keep only selections above the edge / EV / win-probability thresholds.

    Sorted by fractional Kelly (descending). Kelly jointly rewards a higher
    win probability and a meaningful edge, so the top picks are the most
    *feasible* +EV bets — short-odds favourites and balanced edges naturally
    rise above long-shot lottery tickets with the same nominal EV.
    """
    keep = [b for b in bets
            if b.edge >= min_edge and b.ev > min_ev
            and b.model_prob >= min_prob]
    return sorted(keep, key=lambda b: b.kelly, reverse=True)
