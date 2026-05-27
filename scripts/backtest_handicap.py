"""CLI: backtest the game-handicap market (best-of-3 matches).

    python scripts/backtest_handicap.py --tour atp

Run scripts/download_data.py first. The serve/return model's predicted game
margin is compared with the actual margin and with a naive baseline (the
empirical distribution of past margins), pooled over a grid of handicap lines.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_handicap_backtest)
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backtest the game-handicap market.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_handicap_backtest(matches, tour=args.tour,
                                min_prior=args.min_prior, decay=args.decay,
                                prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    n_lines = rec["line"].nunique()
    n_matches = len(rec) // n_lines
    actual, pred, y = rec["actual"], rec["pred_mean"], rec["y"]

    print(f"=== BACKTEST handicap game (best-of-3) - {args.tour.upper()} ===")
    print(f"Periodo        : {_fmt(rec['date'].min())} -> "
          f"{_fmt(rec['date'].max())}")
    print(f"Match valutati : {n_matches}  ({len(rec)} valutazioni su "
          f"{n_lines} linee)\n")

    print("MARGINE GAME (games A - games B, A = etichetta neutra):")
    print(f"  media reale       : {actual.mean():+.2f}   "
          f"(attesa ~0 per simmetria)")
    print(f"  media modello     : {pred.mean():+.2f}")
    print(f"  dev.std reale     : {actual.std():.2f}")
    print(f"  MAE (media modello vs reale): {(pred - actual).abs().mean():.2f}\n")

    print(f"Mercato handicap, aggregato su {n_lines} linee (out-of-sample "
          f"per il modello):")
    print(f"{'Metrica':<14}{'Modello':>12}{'Baseline':>12}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(rec['p_model'], y):>12.4f}"
              f"{fn(rec['p_base'], y):>12.4f}")
    print(f"{'Accuratezza':<14}{accuracy(rec['p_model'], y):>12.3f}"
          f"{accuracy(rec['p_base'], y):>12.3f}")
    print("(Baseline = distribuzione empirica dei margini passati, "
          "senza info sul matchup.)\n")

    print("Calibrazione modello (P(copre) prevista vs osservata):")
    print(calibration_table(rec["p_model"], y).to_string(index=False))


if __name__ == "__main__":
    main()
