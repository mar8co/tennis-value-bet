"""CLI: backtest the first-set-winner market.

    python scripts/backtest_set1.py --tour atp

Scores the serve/return set-winner model against a plain Elo baseline on
identical Sackmann matches. Run scripts/download_data.py first.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_set1_backtest)
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest the first-set-winner market.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_set1_backtest(matches, tour=args.tour, min_prior=args.min_prior,
                            decay=args.decay, prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    y = rec["y"]
    print(f"=== BACKTEST vincente 1o set - {args.tour.upper()} ===")
    print(f"Periodo        : {_fmt(rec['date'].min())} -> "
          f"{_fmt(rec['date'].max())}")
    print(f"Match valutati : {len(rec)}\n")

    print(f"{'Metrica':<14}{'Elo (baseline)':>16}{'SR set-winner':>16}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(rec['p_elo'], y):>16.4f}"
              f"{fn(rec['p_sr'], y):>16.4f}")
    print(f"{'Accuratezza':<14}{accuracy(rec['p_elo'], y):>16.3f}"
          f"{accuracy(rec['p_sr'], y):>16.3f}")
    print("(Brier/Log loss: piu' basso e' meglio. L'Elo e' tarato sugli "
          "esiti match, quindi troppo netto per un singolo set.)\n")

    print("Calibrazione SR set-winner (probabilita' prevista vs osservata):")
    print(calibration_table(rec["p_sr"], y).to_string(index=False))


if __name__ == "__main__":
    main()
