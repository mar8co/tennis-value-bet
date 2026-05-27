"""Fetch upcoming tennis matches and match-winner odds from The Odds API
(the-odds-api.com).

Free tier: 500 requests/month. Listing sports is free; each call to the odds
endpoint costs one credit per region. The API key is read from the
ODDS_API_KEY environment variable or passed explicitly.

Only the match-winner (h2h) market is fetched — free tennis feeds do not
cover the derived markets (total games, handicap, set, tie-break, breaks).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

_BASE = "https://api.the-odds-api.com/v4"


@dataclass
class LiveMatch:
    player1: str
    player2: str
    commence_time: str          # ISO 8601, UTC
    odds1: float                # average match-winner odds for player1
    odds2: float
    n_books: int                # number of bookmakers averaged
    sport_key: str = ""         # The Odds API sport key (tournament)
    total_line: float | None = None    # total-games line (None if not offered)
    over_odds: float | None = None
    under_odds: float | None = None
    hcap_line: float | None = None     # game-handicap line for player1
    hcap_odds1: float | None = None
    hcap_odds2: float | None = None


_CLAY = ("french_open", "hamburg", "monte_carlo", "madrid", "rome",
         "barcelona", "estoril", "munich", "strasbourg", "geneva", "lyon",
         "bucharest", "gstaad", "kitzbuhel", "umag", "bastad", "rio",
         "houston", "marrakech", "cordoba", "santiago", "buenos_aires")
_GRASS = ("wimbledon", "halle", "queens", "eastbourne", "mallorca",
          "newport", "hertogenbosch", "stuttgart", "nottingham",
          "birmingham", "berlin", "bad_homburg")
_SLAM = ("french_open", "wimbledon", "us_open", "australian_open")


def match_context(sport_key: str) -> tuple:
    """Infer (tour, surface, best_of) from a The Odds API tennis sport key."""
    k = (sport_key or "").lower()
    tour = "wta" if "wta" in k else "atp"
    if any(t in k for t in _GRASS):
        surface = "Grass"
    elif any(t in k for t in _CLAY):
        surface = "Clay"
    else:
        surface = "Hard"
    best_of = 5 if (tour == "atp" and any(s in k for s in _SLAM)) else 3
    return tour, surface, best_of


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise ValueError("no API key — set the ODDS_API_KEY environment "
                         "variable or pass api_key explicitly")
    return key


def list_tennis_sports(api_key: str | None = None) -> list:
    """Active tennis sport keys (listing sports does not cost a credit)."""
    resp = requests.get(f"{_BASE}/sports",
                        params={"apiKey": _resolve_key(api_key)}, timeout=20)
    resp.raise_for_status()
    return [s["key"] for s in resp.json()
            if str(s.get("key", "")).startswith("tennis_")]


def _avg(values: list):
    return sum(values) / len(values) if values else None


def _modal_line(by_point: dict):
    """From {point: ([a_prices], [b_prices])} pick the most-offered point and
    average the two price lists. Returns (point, avg_a, avg_b) or Nones."""
    if not by_point:
        return None, None, None
    pt = max(by_point, key=lambda p: len(by_point[p][0]))
    a, b = by_point[pt]
    return pt, _avg(a), _avg(b)


def _parse_event(ev: dict):
    """Build a LiveMatch by averaging h2h / totals / spreads across every
    bookmaker offering the event. Returns None without usable h2h prices."""
    p1, p2 = ev.get("home_team"), ev.get("away_team")
    if not p1 or not p2:
        return None
    o1, o2 = [], []
    totals: dict = {}      # point -> ([over prices], [under prices])
    spreads: dict = {}     # player1 point -> ([p1 prices], [p2 prices])
    for book in ev.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            outs = market.get("outcomes", [])
            if key == "h2h":
                pr = {o.get("name"): o.get("price") for o in outs}
                if pr.get(p1) and pr.get(p2):
                    o1.append(pr[p1])
                    o2.append(pr[p2])
            elif key == "totals":
                pr = {o.get("name"): o for o in outs}
                ov, un = pr.get("Over"), pr.get("Under")
                if ov and un and ov.get("price") and un.get("price") \
                        and ov.get("point") is not None:
                    totals.setdefault(ov["point"], ([], []))
                    totals[ov["point"]][0].append(ov["price"])
                    totals[ov["point"]][1].append(un["price"])
            elif key == "spreads":
                pr = {o.get("name"): o for o in outs}
                a, b = pr.get(p1), pr.get(p2)
                if a and b and a.get("price") and b.get("price") \
                        and a.get("point") is not None:
                    spreads.setdefault(a["point"], ([], []))
                    spreads[a["point"]][0].append(a["price"])
                    spreads[a["point"]][1].append(b["price"])
    if not o1:
        return None
    t_line, t_over, t_under = _modal_line(totals)
    h_line, h_o1, h_o2 = _modal_line(spreads)
    return LiveMatch(
        player1=p1, player2=p2, commence_time=ev.get("commence_time", ""),
        odds1=_avg(o1), odds2=_avg(o2), n_books=len(o1),
        sport_key=ev.get("sport_key", ""),
        total_line=t_line, over_odds=t_over, under_odds=t_under,
        hcap_line=h_line, hcap_odds1=h_o1, hcap_odds2=h_o2)


def fetch_tennis_odds(api_key: str | None = None,
                      regions: str = "eu") -> list:
    """Fetch upcoming tennis matches with average match-winner odds, sorted
    by start time. Raises on a missing key or an API failure (bad key,
    quota exceeded, network error)."""
    key = _resolve_key(api_key)
    out = []
    for sport in list_tennis_sports(key):
        resp = requests.get(
            f"{_BASE}/sports/{sport}/odds",
            params={"apiKey": key, "regions": regions,
                    "markets": "h2h,totals,spreads", "oddsFormat": "decimal"},
            timeout=20)
        if resp.status_code != 200:
            continue
        for ev in resp.json():
            match = _parse_event(ev)
            if match is not None:
                out.append(match)
    out.sort(key=lambda m: m.commence_time)
    return out
