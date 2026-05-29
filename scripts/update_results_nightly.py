"""Nightly result updater — run as a scheduled task (Windows Task Scheduler).

Usage:
    python scripts/update_results_nightly.py

Schedule suggestion: every night at 23:59 (or later, e.g. 01:00 to let
Sackmann CSV propagate — Sackmann typically lags 12-24 h after match end).
"""
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from the project root regardless of working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_FILE = ROOT / "data" / "processed" / "nightly_update.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("=== Nightly result update started ===")

    try:
        from tvb.bet_tracker import (
            clear_sackmann_cache,
            init_tracker_db,
            update_from_sackmann,
            update_results,
        )
    except Exception as exc:
        log.error("Failed to import bet_tracker: %s", exc)
        sys.exit(1)

    init_tracker_db()

    # Force fresh download from Sackmann GitHub (ignore 1-hour cache).
    clear_sackmann_cache()

    # 1. Try Sackmann GitHub first (free, no quota).
    log.info("Fetching results from Sackmann GitHub...")
    try:
        n_sack = update_from_sackmann()
        log.info("Sackmann: %d bet(s) resolved.", n_sack)
    except Exception as exc:
        log.warning("Sackmann update failed: %s", exc)
        n_sack = 0

    # 2. Try The Odds API scores endpoint if an API key is configured.
    api_key = os.environ.get("ODDS_API_KEY", "")
    n_api = 0
    if api_key:
        log.info("Fetching results from The Odds API...")
        try:
            n_api = update_results(api_key)
            log.info("Odds API: %d bet(s) resolved.", n_api)
        except Exception as exc:
            log.warning("Odds API update failed: %s", exc)
    else:
        log.info("ODDS_API_KEY not set — skipping Odds API step.")

    total = n_sack + n_api
    log.info("=== Done: %d total bet(s) resolved ===", total)


if __name__ == "__main__":
    main()
