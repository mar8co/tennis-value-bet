"""CLI: fit and validate a bias correction for the over/under total-games model.

    python scripts/calibrate_totals.py --tour atp --line 22.5

The model systematically overpredicts total games (the i.i.d.-points
assumption inflates set length). A single location shift δ is fitted on an
earlier slice of matches and evaluated on a held-out later slice, so the
reported improvement is genuine — and compared against the naive baseline.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_total_games_backtest)
from tvb.calibration import apply_total_shift, fit_total_shift
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit/validate the totals bias correction.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--line", type=float, default=22.5)
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_total_games_backtest(matches, tour=args.tour, line=args.line,
                                   min_prior=args.min_prior, decay=args.decay,
                                   prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    cut = int(len(rec) * args.train_frac)
    train, test = rec.iloc[:cut], rec.iloc[cut:].copy()
    delta = fit_total_shift(train["dist"], train["actual"], args.line)
    test["p_cal"] = [apply_total_shift(d, args.line, delta)
                     for d in test["dist"]]
    y = test["y"]

    print(f"=== CORREZIONE BIAS over/under game - {args.tour.upper()} ===")
    print(f"Train : {len(train):>5} match  {_fmt(train['date'].min())} -> "
          f"{_fmt(train['date'].max())}")
    print(f"Test  : {len(test):>5} match  {_fmt(test['date'].min())} -> "
          f"{_fmt(test['date'].max())}  (held-out)")
    print(f"Shift stimato sul train : delta = {delta} game")
    print("(il modello prevedeva troppi game; la correzione abbassa "
          "la distribuzione)\n")

    print(f"Media game sul test:  reale {test['actual'].mean():.2f}  |  "
          f"modello grezzo {test['pred_mean'].mean():.2f}  |  "
          f"corretto {test['pred_mean'].mean() - delta:.2f}\n")

    print(f"Mercato over/under {args.line} sul test (out-of-sample):")
    print(f"{'Metrica':<14}{'Baseline':>12}{'Mod. grezzo':>13}"
          f"{'Mod. corretto':>15}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(test['p_base'], y):>12.4f}"
              f"{fn(test['p_model'], y):>13.4f}{fn(test['p_cal'], y):>15.4f}")
    print(f"{'Accuratezza':<14}{accuracy(test['p_base'], y):>12.3f}"
          f"{accuracy(test['p_model'], y):>13.3f}"
          f"{accuracy(test['p_cal'], y):>15.3f}")
    print("(Brier/Log loss: piu' basso e' meglio.)\n")

    print("Calibrazione sul test - PRIMA (modello grezzo):")
    print(calibration_table(test["p_model"], y).to_string(index=False))
    print("\nCalibrazione sul test - DOPO (modello corretto):")
    print(calibration_table(test["p_cal"], y).to_string(index=False))


if __name__ == "__main__":
    main()
