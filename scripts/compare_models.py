"""CLI: compare the recency-weighted, opponent-adjusted serve/return model
against the plain Elo baseline on identical Sackmann matches (match winner).

    python scripts/compare_models.py --tour atp --min-prior 15

Run scripts/download_data.py first to populate the database. Both models are
scored on the same matches, so the metrics are directly comparable.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.backtest import (accuracy, brier, calibration_table, log_loss,
                          run_sackmann_backtest)
from tvb.config import BACKTEST_MIN_PRIOR, SR_DECAY, SR_PRIOR_WEIGHT
from tvb.ingest import read_matches


def _fmt_date(d) -> str:
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare SR model vs Elo baseline.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--min-prior", type=int, default=BACKTEST_MIN_PRIOR)
    ap.add_argument("--decay", type=float, default=SR_DECAY)
    ap.add_argument("--prior-weight", type=float, default=SR_PRIOR_WEIGHT)
    args = ap.parse_args()

    matches = read_matches(args.tour)
    rec = run_sackmann_backtest(matches, tour=args.tour,
                                min_prior=args.min_prior,
                                decay=args.decay,
                                prior_weight=args.prior_weight)
    if rec.empty:
        print("No matches to evaluate — check the database.")
        return

    y = rec["y"]
    print(f"=== CONFRONTO MODELLI match winner - {args.tour.upper()} ===")
    print(f"Periodo        : {_fmt_date(rec['date'].min())} -> "
          f"{_fmt_date(rec['date'].max())}")
    print(f"Match valutati : {len(rec)}")
    print(f"Burn-in        : esclusi match con < {args.min_prior} precedenti")
    print(f"SR model       : decay={args.decay}  "
          f"prior_weight={args.prior_weight}\n")

    print(f"{'Metrica':<14}{'Elo (baseline)':>16}{'SR recency+opp':>17}"
          f"{'Delta':>10}")
    for label, fn in (("Brier", brier), ("Log loss", log_loss)):
        e, s = fn(rec["p_elo"], y), fn(rec["p_sr"], y)
        print(f"{label:<14}{e:>16.4f}{s:>17.4f}{s - e:>+10.4f}")
    ae, as_ = accuracy(rec["p_elo"], y), accuracy(rec["p_sr"], y)
    print(f"{'Accuratezza':<14}{ae:>16.3f}{as_:>17.3f}{as_ - ae:>+10.3f}")
    print("(Brier/Log loss: piu' basso e' meglio -> Delta negativo = "
          "SR migliore)\n")

    print("Calibrazione SR model (probabilita' prevista vs osservata):")
    print(calibration_table(rec["p_sr"], y).to_string(index=False))


if __name__ == "__main__":
    main()
