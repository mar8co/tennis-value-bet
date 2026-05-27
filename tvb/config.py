"""Central configuration: paths, data sources, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "tennis.db"

# Jeff Sackmann GitHub raw data (atp_matches_YYYY.csv / wta_matches_YYYY.csv)
SACKMANN_REPOS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master",
}

# Monte Carlo
N_SIMS = 20_000

# Value-bet thresholds
MIN_EDGE = 0.03        # minimum edge over the fair (de-margined) probability
MIN_EV = 0.0           # only keep positive expected value
KELLY_FRACTION = 0.25  # quarter-Kelly staking — never bet full Kelly

# League-average share of serve points won. Used to combine two players
# into the simulator's (p0, p1) inputs. ATP servers win ~64%, WTA ~56%.
LEAGUE_SPW = {"atp": 0.642, "wta": 0.562}

# Elo
ELO_K = 32
ELO_BASE = 1500

# Tennis-Data.co.uk — historical results + closing odds, used for backtesting
TENNIS_DATA_URL = "http://www.tennis-data.co.uk"

# Backtest: ignore matches where a player has fewer than this many priors
BACKTEST_MIN_PRIOR = 10

# Serve/return rating model (recency-weighted, opponent-adjusted)
SR_DECAY = 0.97          # per-match decay applied to a player's history
SR_PRIOR_WEIGHT = 250.0  # pseudo-points of shrinkage toward league average

# Output recalibration — temperature fitted & validated on the match-winner
# backtest (scripts/calibrate.py, ATP). T > 1 corrects model overconfidence.
SR_TEMPERATURE = 1.335

# Total-games bias correction — location shift δ (games), fitted & validated
# on the over/under backtest (scripts/calibrate_totals.py, ATP).
SR_TOTAL_SHIFT = 2.5

# Game-handicap overconfidence correction — temperature, fitted & validated
# on the handicap backtest (scripts/calibrate_handicap.py, ATP).
SR_HANDICAP_TEMPERATURE = 1.265

# Tie-break yes/no bias correction — 2-param logistic (a, b), fitted &
# validated on the tie-break backtest (scripts/calibrate_tiebreak.py, ATP).
SR_TIEBREAK_LOGISTIC = (0.98, -0.27)
