"""CLI: backtest the tie-break yes/no market (best-of-3 matches).

    python scripts/backtest_tiebreak.py --tour atp

Run scripts/download_data.py first. The serve/return model's P(at least one
tie-break) is compared with the actual outcome and with a naive baseline
(the empirical tie-break rate of past matches).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_tiebreak_backtest)
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest the tie-break yes/no market.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_tiebreak_backtest(matches, tour=args.tour,
                                min_prior=args.min_prior, decay=args.decay,
                                prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    y = rec["y"]
    print(f"=== BACKTEST tie-break si/no (best-of-3) - {args.tour.upper()} ===")
    print(f"Periodo        : {_fmt(rec['date'].min())} -> "
          f"{_fmt(rec['date'].max())}")
    print(f"Match valutati : {len(rec)}\n")

    print("MERCATO tie-break si/no:")
    print(f"  tie-break osservato      : {y.mean() * 100:.1f}%")
    print(f"  tie-break previsto (mod.): {rec['p_model'].mean() * 100:.1f}%\n")

    print(f"{'Metrica':<14}{'Modello':>12}{'Baseline':>12}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(rec['p_model'], y):>12.4f}"
              f"{fn(rec['p_base'], y):>12.4f}")
    print(f"{'Accuratezza':<14}{accuracy(rec['p_model'], y):>12.3f}"
          f"{accuracy(rec['p_base'], y):>12.3f}")
    print("(Baseline = tasso di tie-break empirico dei match passati.)\n")

    print("Calibrazione modello (P(tie-break) prevista vs osservata):")
    print(calibration_table(rec["p_model"], y).to_string(index=False))


if __name__ == "__main__":
    main()
