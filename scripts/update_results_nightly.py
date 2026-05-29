"""Nightly result updater — runs via GitHub Actions every night at 01:00 UTC.

Uses only Sackmann GitHub data (free, no API quota).
Odds API updates are triggered manually from the dashboard.
"""
import logging
import os
import sys
from pathlib import Path

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
            update_from_espn,
        )
    except Exception as exc:
        log.error("Failed to import bet_tracker: %s", exc)
        sys.exit(1)

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        scheme = db_url.split("://")[0] if "://" in db_url else f"NO SCHEME — starts with: {db_url[:30]!r}"
        log.info("DATABASE_URL scheme: %s", scheme)
    else:
        log.info("DATABASE_URL not set — using local SQLite")

    init_tracker_db()
    clear_sackmann_cache()

    # ESPN: free, no auth, real-time ATP + WTA results
    log.info("Fetching results from ESPN...")
    try:
        n_rapi = update_from_espn()
        log.info("ESPN: %d bet(s) resolved.", n_rapi)
    except Exception as exc:
        log.warning("ESPN update failed: %s", exc)
        n_rapi = 0

    # Sackmann: fonte storica affidabile per partite più vecchie
    log.info("Fetching results from Sackmann GitHub...")
    try:
        n_sack = update_from_sackmann()
        log.info("Sackmann: %d bet(s) resolved.", n_sack)
    except Exception as exc:
        log.warning("Sackmann update failed: %s", exc)
        n_sack = 0

    log.info("=== Done: %d total bet(s) resolved ===", n_rapi + n_sack)


if __name__ == "__main__":
    main()
