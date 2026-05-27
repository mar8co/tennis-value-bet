"""CLI: fit and validate the overconfidence correction (temperature) for the
game-handicap model.

    python scripts/calibrate_handicap.py --tour atp

The handicap model beats the naive baseline but is overconfident (predictions
pushed too far from 50%). A temperature is fitted on an earlier slice of
matches and evaluated on a held-out later slice, so the gain is genuine.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_handicap_backtest)
from tvb.calibration import apply_temperature, fit_temperature
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit/validate the handicap overconfidence correction.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_handicap_backtest(matches, tour=args.tour,
                                min_prior=args.min_prior, decay=args.decay,
                                prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    # split chronologically, keeping each match's lines together
    n_lines = rec["line"].nunique()
    cut = (int(len(rec) * args.train_frac) // n_lines) * n_lines
    train, test = rec.iloc[:cut], rec.iloc[cut:].copy()
    temperature = fit_temperature(train["p_model"], train["y"])
    test["p_cal"] = apply_temperature(test["p_model"], temperature)
    y = test["y"]

    print(f"=== RICALIBRAZIONE handicap game - {args.tour.upper()} ===")
    print(f"Train : {len(train) // n_lines:>5} match  "
          f"{_fmt(train['date'].min())} -> {_fmt(train['date'].max())}")
    print(f"Test  : {len(test) // n_lines:>5} match  "
          f"{_fmt(test['date'].min())} -> {_fmt(test['date'].max())}  "
          f"(held-out)")
    print(f"Temperatura stimata sul train : T = {temperature}")
    print("(T > 1 = il modello era troppo sicuro; la ricalibrazione "
          "avvicina le probabilita' al 50%)\n")

    print(f"Metriche sul test set (out-of-sample, aggregate su {n_lines} "
          f"linee):")
    print(f"{'Metrica':<14}{'Baseline':>12}{'SR grezzo':>13}"
          f"{'SR ricalibr.':>15}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        print(f"{label:<14}{fn(test['p_base'], y):>12.4f}"
              f"{fn(test['p_model'], y):>13.4f}{fn(test['p_cal'], y):>15.4f}")
    print(f"{'Accuratezza':<14}{accuracy(test['p_base'], y):>12.3f}"
          f"{accuracy(test['p_model'], y):>13.3f}"
          f"{accuracy(test['p_cal'], y):>15.3f}")
    print("(la temperatura non cambia l'accuratezza: trasformazione "
          "monotona)\n")

    print("Calibrazione sul test - PRIMA (SR grezzo):")
    print(calibration_table(test["p_model"], y).to_string(index=False))
    print("\nCalibrazione sul test - DOPO (SR ricalibrato):")
    print(calibration_table(test["p_cal"], y).to_string(index=False))


if __name__ == "__main__":
    main()
