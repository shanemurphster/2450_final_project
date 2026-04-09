"""
merge_dataset.py  —  Billboard Boxing
======================================
Combines the positive (Billboard Top 100) and negative (non-hit) Spotify
matches into a single labeled CSV ready for EDA and modeling.

Input files
-----------
  data/raw/spotify_billboard_matches.csv      — positives (label=1)
  data/processed/spotify_negatives_by_year.csv — negatives (label=0)

Output file
-----------
  data/processed/billboard_modeling_dataset.csv

Label column
------------
  label = 1  →  Billboard Year-End Top 100 song
  label = 0  →  Non-hit song from the same era

Usage
-----
    python3 src/cleaning/merge_dataset.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POSITIVES_CSV = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_billboard_matches.csv")
NEGATIVES_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "spotify_negatives_by_year.csv")
OUTPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "billboard_modeling_dataset.csv")

# ---------------------------------------------------------------------------
# Columns to keep in the final dataset
# (drops internal pipeline cols that aren't useful for modeling)
# ---------------------------------------------------------------------------

KEEP_COLS = [
    # --- Identity ---
    "label",
    "year",
    "rank",
    "title",
    "artist",
    "title_clean",
    "artist_clean",
    "primary_artist",
    # --- Spotify metadata ---
    "spotify_id",
    "spotify_title",
    "spotify_artist",
    "spotify_album",
    "album_type",
    "release_date",
    "release_year",
    "popularity",
    "explicit",
    "duration_ms",
    "isrc",
    # --- Audio features (NaN for now — filled in after Kaggle merge) ---
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "audio_features_available",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load ---
    print("Loading positives...")
    pos = pd.read_csv(POSITIVES_CSV)
    print(f"  {len(pos)} rows")

    print("Loading negatives...")
    neg = pd.read_csv(NEGATIVES_CSV)
    print(f"  {len(neg)} rows")

    # --- Label positives ---
    pos["label"] = 1

    # --- Derive release_year for positives from release_date ---
    # Negatives already have release_year; positives only have release_date
    if "release_year" not in pos.columns:
        pos["release_year"] = (
            pd.to_datetime(pos["release_date"], errors="coerce").dt.year
        )

    # --- Drop pipeline-only columns ---
    pos = pos.drop(columns=["second_pass_label", "billboard_year"], errors="ignore")
    neg = neg.drop(columns=["billboard_year"], errors="ignore")

    # --- Align to shared column set ---
    all_cols = set(pos.columns) | set(neg.columns)
    for col in all_cols:
        if col not in pos.columns:
            pos[col] = None
        if col not in neg.columns:
            neg[col] = None

    # --- Combine ---
    combined = pd.concat([pos, neg], ignore_index=True)

    # --- Keep only the columns we want, in a clean order ---
    final_cols = [c for c in KEEP_COLS if c in combined.columns]
    combined = combined[final_cols]

    # --- Sort ---
    combined = combined.sort_values(["year", "label"], ascending=[True, False]).reset_index(drop=True)

    # --- Deduplication check ---
    before = len(combined)
    combined = combined.drop_duplicates(subset=["spotify_id"]).reset_index(drop=True)
    dropped = before - len(combined)
    if dropped:
        print(f"  Dropped {dropped} duplicate spotify_id rows.")

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    # --- Summary ---
    print("\n" + "=" * 56)
    print("  MERGE SUMMARY")
    print("=" * 56)
    print(f"  Total rows         : {len(combined)}")
    print(f"  Positives (label=1): {(combined['label'] == 1).sum()}")
    print(f"  Negatives (label=0): {(combined['label'] == 0).sum()}")
    print(f"  Years covered      : {sorted(combined['year'].dropna().unique().astype(int).tolist())}")
    print(f"  Columns            : {len(combined.columns)}")
    print(f"\n  Saved -> {OUTPUT_CSV}")
    print("=" * 56)


if __name__ == "__main__":
    main()
