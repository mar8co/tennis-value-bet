"""CLI: leak-free historical backtest of the match-winner model.

    python scripts/backtest.py --download --tour atp --from 2021 --to 2026

--download fetches Tennis-Data.co.uk files first; omit it to reuse the
data already in the local database.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (betting_simulation, calibration_table,
                          run_elo_backtest, summarize)
from tvb.config import BACKTEST_MIN_PRIOR
from tvb.ingest import download_tennis_data, load_oddshist_to_db, read_oddshist


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the match-winner model.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--from", dest="y0", type=int, default=2021)
    ap.add_argument("--to", dest="y1", type=int, default=2026)
    ap.add_argument("--download", action="store_true",
                    help="fetch Tennis-Data files before running")
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    args = ap.parse_args()

    if args.download:
        print(f"Downloading Tennis-Data {args.tour} {args.y0}-{args.y1} ...")
        download_tennis_data(args.tour, range(args.y0, args.y1 + 1))
        n = load_oddshist_to_db(args.tour)
        print(f"Loaded {n} rows into '{args.tour}_oddshist'.\n")

    hist = read_oddshist(args.tour)
    hist = hist[(hist["date"].dt.year >= args.y0)
                & (hist["date"].dt.year <= args.y1)]

    records = run_elo_backtest(hist, min_prior=args.min_prior)
    if records.empty:
        print("No matches to evaluate — check the data range / database.")
        return

    s = summarize(records)
    print(f"=== BACKTEST match winner (Elo) - {args.tour.upper()} ===")
    print(f"Periodo        : {s['date_min'].date()} -> {s['date_max'].date()}")
    print(f"Match valutati : {s['n_eval']}  (con quote: {s['n_with_odds']})")
    print(f"Burn-in        : esclusi match con < {args.min_prior} precedenti\n")

    print(f"{'Metrica':<14}{'Modello (Elo)':>16}{'Bookmaker':>14}")
    print(f"{'Brier':<14}{s['brier_model']:>16.4f}{s['brier_book']:>14.4f}")
    print(f"{'Log loss':<14}{s['logloss_model']:>16.4f}{s['logloss_book']:>14.4f}")
    print(f"{'Accuratezza':<14}{s['acc_model']:>16.3f}{s['acc_book']:>14.3f}")
    print("(Brier/Log loss: piu' basso e' meglio. Confronto sulle stesse "
          "partite con quote.)\n")

    print("Calibrazione modello (probabilita' prevista vs osservata):")
    print(calibration_table(records["p_model"], records["y"]).to_string(index=False))

    print("\nSimulazione scommesse (puntata fissa 1u su ogni selezione +EV):")
    for thr in (0.0, 0.05, 0.10):
        b = betting_simulation(records, min_ev=thr)
        print(f"  EV > {thr * 100:>4.0f}% : {b['n_bets']:>6} bet  "
              f"staked {b['staked']:>9.0f}  profit {b['profit']:>+10.1f}  "
              f"ROI {b['roi'] * 100:>+7.2f}%")


if __name__ == "__main__":
    main()
