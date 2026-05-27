"""Download match data (Jeff Sackmann) and historical odds (Tennis-Data.co.uk)
and load them into a local SQLite database."""
from __future__ import annotations

import sqlite3

import pandas as pd
import requests

from .config import (DB_PATH, PROCESSED_DIR, RAW_DIR, SACKMANN_REPOS,
                     TENNIS_DATA_URL)


def download_matches(tour: str = "atp", years=range(2020, 2025)) -> list:
    """Download {tour}_matches_YYYY.csv files into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base = SACKMANN_REPOS[tour]
    saved = []
    for year in years:
        fname = f"{tour}_matches_{year}.csv"
        dest = RAW_DIR / fname
        try:
            df = pd.read_csv(f"{base}/{fname}")
        except Exception as exc:               # network / missing file
            print(f"  skip {fname}: {exc}")
            continue
        df.to_csv(dest, index=False)
        saved.append(dest)
        print(f"  saved {fname} ({len(df)} rows)")
    return saved


def load_to_db(tour: str = "atp") -> int:
    """Concatenate downloaded CSVs for a tour into a SQLite `{tour}_matches` table."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob(f"{tour}_matches_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no raw files for {tour} — run download_matches() first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(f"{tour}_matches", conn, if_exists="replace", index=False)
    return len(df)


def read_matches(tour: str = "atp") -> pd.DataFrame:
    """Read the stored matches table for a tour."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {tour}_matches", conn)


# ----------------------- Tennis-Data.co.uk historical odds ------------------

_ODDS_COLUMNS = ["Date", "Surface", "Round", "Winner", "Loser", "Comment",
                 "WRank", "LRank", "B365W", "B365L", "PSW", "PSL",
                 "AvgW", "AvgL", "MaxW", "MaxL"]


def download_tennis_data(tour: str = "atp", years=range(2021, 2027)) -> list:
    """Download Tennis-Data.co.uk season files (results + closing odds)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if tour == "atp" else "w"
    saved = []
    for year in years:
        url = f"{TENNIS_DATA_URL}/{year}{suffix}/{year}.xlsx"
        dest = RAW_DIR / f"{tour}_odds_{year}.xlsx"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as exc:               # network / missing file
            print(f"  skip {year}: {exc}")
            continue
        dest.write_bytes(resp.content)
        saved.append(dest)
        print(f"  saved {dest.name} ({len(resp.content) // 1024} KB)")
    return saved


def load_oddshist_to_db(tour: str = "atp") -> int:
    """Concatenate downloaded Tennis-Data files into a `{tour}_oddshist` table."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob(f"{tour}_odds_*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"no Tennis-Data files for {tour} — run download_tennis_data() first")
    frames = []
    for f in files:
        df = pd.read_excel(f)
        frames.append(df[[c for c in _ODDS_COLUMNS if c in df.columns]])
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower() for c in out.columns]
    with sqlite3.connect(DB_PATH) as conn:
        out.to_sql(f"{tour}_oddshist", conn, if_exists="replace", index=False)
    return len(out)


def read_oddshist(tour: str = "atp") -> pd.DataFrame:
    """Read the stored Tennis-Data odds/results table, with parsed dates."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {tour}_oddshist", conn)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
