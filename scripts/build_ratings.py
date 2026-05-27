"""CLI: compute Elo + serve/return ratings from the local database.

    python scripts/build_ratings.py --tour atp

Run scripts/download_data.py first to populate the database.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvb.elo import compute_elo
from tvb.ingest import read_matches
from tvb.ratings import player_names
from tvb.serve_return import player_serve_return


def main() -> None:
    ap = argparse.ArgumentParser(description="Build player ratings.")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    args = ap.parse_args()

    matches = read_matches(args.tour)
    print(f"{len(matches)} matches loaded.\n")
    names = player_names(matches)

    elo = compute_elo(matches, by_surface=True).merge(names, on="player_id", how="left")
    hard = elo[elo["surface"] == "Hard"].sort_values("elo", ascending=False)
    print("Top 10 Elo (Hard):")
    print(hard[["name", "elo", "matches"]].head(10).to_string(index=False))

    sr = player_serve_return(matches).merge(names, on="player_id", how="left")
    print(f"\nServe/return computed for {len(sr)} players. Best servers:")
    top = sr[sr["matches"] >= 20].sort_values("spw", ascending=False)
    print(top[["name", "spw", "rpw", "matches"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
