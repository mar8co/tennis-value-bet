"""CLI: fit and validate the bias correction for the tie-break yes/no model.

    python scripts/calibrate_tiebreak.py --tour atp

The model overpredicts tie-breaks. A 2-parameter logistic recalibration —
with an intercept, since tie-break yes/no is not a symmetric two-player
market — is fitted on an earlier slice of matches and evaluated on a
held-out later slice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_tiebreak_backtest)
from tvb.calibration import apply_logistic, fit_logistic
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit/validate the tie-break bias correction.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_tiebreak_backtest(matches, tour=args.tour,
                                min_prior=args.min_prior, decay=args.decay,
                                prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    cut = int(len(rec) * args.train_frac)
    train, test = rec.iloc[:cut], rec.iloc[cut:].copy()
    a, b = fit_logistic(train["p_model"], train["y"])
    test["p_cal"] = apply_logistic(test["p_model"], a, b)
    y = test["y"]

    print(f"=== RICALIBRAZIONE tie-break si/no - {args.tour.upper()} ===")
    print(f"Train : {len(train):>5} match  {_fmt(train['date'].min())} -> "
          f"{_fmt(train['date'].max())}")
    print(f"Test  : {len(test):>5} match  {_fmt(test['date'].min())} -> "
          f"{_fmt(test['date'].max())}  (held-out)")
    print(f"Logistica stimata sul train : a = {a}, b = {b}")
    print("(b < 0 = abbassa la probabilita'; il modello sovrastimava "
          "i tie-break)\n")

    print(f"Tasso tie-break sul test:  reale {y.mean() * 100:.1f}%  |  "
          f"modello grezzo {test['p_model'].mean() * 100:.1f}%  |  "
          f"corretto {test['p_cal'].mean() * 100:.1f}%\n")

    print("Metriche sul test set (out-of-sample):")
    print(f"{'Metrica':<14}{'Baseline':>12}{'Mod. grezzo':>13}"
          f"{'Mod. corretto':>15}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(test['p_base'], y):>12.4f}"
              f"{fn(test['p_model'], y):>13.4f}{fn(test['p_cal'], y):>15.4f}")
    print(f"{'Accuratezza':<14}{accuracy(test['p_base'], y):>12.3f}"
          f"{accuracy(test['p_model'], y):>13.3f}"
          f"{accuracy(test['p_cal'], y):>15.3f}\n")

    print("Calibrazione sul test - PRIMA (modello grezzo):")
    print(calibration_table(test["p_model"], y).to_string(index=False))
    print("\nCalibrazione sul test - DOPO (modello corretto):")
    print(calibration_table(test["p_cal"], y).to_string(index=False))


if __name__ == "__main__":
    main()
