"""CLI: download Sackmann data and build the local database.

    python scripts/download_data.py --tour atp --from 2020 --to 2024
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.ingest import download_matches, load_to_db


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Jeff Sackmann match data.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--from", dest="y0", type=int, default=2020)
    ap.add_argument("--to", dest="y1", type=int, default=2024)
    args = ap.parse_args()

    print(f"Downloading {args.tour} matches {args.y0}-{args.y1} ...")
    download_matches(args.tour, range(args.y0, args.y1 + 1))
    n = load_to_db(args.tour)
    print(f"Loaded {n} rows into table '{args.tour}_matches'.")


if __name__ == "__main__":
    main()
