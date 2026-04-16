"""
merge_audio_features.py  —  Billboard Boxing
=============================================
Enriches the merged modeling dataset with audio features from the
Yamaerenay Spotify Dataset (170k tracks, 1921-2020, local CSV).

Match strategy (in order)
--------------------------
1. Exact spotify_id match  (id == spotify_id)
2. Fuzzy title + artist match (lowercased, stripped)

Audio feature columns added
----------------------------
danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo

Any song that already has audio features filled in (audio_features_available
== True) is skipped — its existing values are preserved.

Output
------
  data/processed/billboard_modeling_dataset.csv  (overwritten in place)

Usage
-----
    python3 src/cleaning/merge_audio_features.py
"""

import os
import re
import ast
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELING_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "billboard_modeling_dataset.csv")
KAGGLE_CSV    = os.path.join(PROJECT_ROOT, "data", "archive", "data.csv")

# ---------------------------------------------------------------------------
# Audio feature columns to pull from the Kaggle dataset
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# ---------------------------------------------------------------------------
# Text normalisation for matching
# ---------------------------------------------------------------------------

_FEAT_RE    = re.compile(r'\s*(?:featuring|feat\.?|ft\.?|with)\s+.*$', re.IGNORECASE)
_BRACKET_RE = re.compile(r'\s*[\(\[].*?[\)\]]')
_DASH_RE    = re.compile(r'\s*[-–—]\s+.*$')
_VERSION_RE = re.compile(
    r'\s+(?:remix|mix|edit|version|instrumental|karaoke|live|reprise|'
    r'remaster|acoustic|radio|extended|club|single|unplugged)\b.*$',
    re.IGNORECASE,
)
_CLEAN_RE   = re.compile(r'[^a-z0-9\s]')


def _normalise(text: str) -> str:
    """Lowercase, strip featuring clause, remove non-alphanumeric chars."""
    t = str(text).lower().strip()
    t = _FEAT_RE.sub('', t)
    t = _CLEAN_RE.sub('', t)
    return t.strip()


def _normalise_title(text: str) -> str:
    """Aggressive title normalisation: strip brackets, dashes, and version/remix keywords."""
    t = str(text).lower().strip()
    t = _FEAT_RE.sub('', t)
    t = _BRACKET_RE.sub('', t)
    t = _DASH_RE.sub('', t)
    t = _VERSION_RE.sub('', t)
    t = _CLEAN_RE.sub('', t)
    return t.strip()

# ---------------------------------------------------------------------------
# Load local Kaggle dataset
# ---------------------------------------------------------------------------

def _parse_artists(raw: str) -> str:
    """Extract the first artist from a stringified list like \"['Artist A', 'Artist B']\"."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return str(raw)


def load_kaggle_features() -> pd.DataFrame:
    log.info(f"Loading Kaggle Spotify dataset from {KAGGLE_CSV} ...")
    df = pd.read_csv(KAGGLE_CSV)
    log.info(f"  Loaded {len(df)} rows.")

    # Normalise column names to match what the rest of the script expects
    df = df.rename(columns={"id": "track_id", "name": "track_name"})

    # artists column is a stringified Python list — extract primary artist
    df["primary_artist"] = df["artists"].apply(_parse_artists)

    # Normalise for matching — both standard and aggressive (base title)
    df["match_title"]      = df["track_name"].apply(_normalise)
    df["match_title_base"] = df["track_name"].apply(_normalise_title)
    df["match_artist"]     = df["primary_artist"].apply(_normalise)

    # Keep the most popular entry per (title, artist) pair
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["match_title", "match_artist"], keep="first")

    return df

# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------

def _build_id_lookup(kaggle_df: pd.DataFrame) -> dict[str, dict]:
    """spotify_id -> feature dict"""
    return {
        row["track_id"]: row.to_dict()
        for _, row in kaggle_df.iterrows()
    }


def _build_fuzzy_lookup(kaggle_df: pd.DataFrame) -> dict[tuple, dict]:
    """(norm_title, norm_artist) -> feature dict"""
    return {
        (row["match_title"], row["match_artist"]): row.to_dict()
        for _, row in kaggle_df.iterrows()
    }


def _build_base_title_lookup(kaggle_df: pd.DataFrame) -> dict[tuple, dict]:
    """(norm_title_base, norm_artist) -> feature dict — for matching stripped remix/version titles."""
    lookup = {}
    for _, row in kaggle_df.iterrows():
        key = (row["match_title_base"], row["match_artist"])
        if key not in lookup:  # already sorted by popularity desc, keep first
            lookup[key] = row.to_dict()
    return lookup


def _extract_features(kaggle_row: dict) -> dict:
    """Pull just the audio feature columns from a kaggle row."""
    features = {col: kaggle_row.get(col) for col in FEATURE_COLS}
    features["audio_features_available"] = True
    return features

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Load modeling dataset ---
    log.info(f"Loading modeling dataset from {MODELING_CSV}")
    df = pd.read_csv(MODELING_CSV)
    log.info(f"  {len(df)} rows loaded.")

    # --- Load Kaggle features ---
    kaggle_df        = load_kaggle_features()
    id_lookup        = _build_id_lookup(kaggle_df)
    fuzzy_lookup     = _build_fuzzy_lookup(kaggle_df)
    base_ttl_lookup  = _build_base_title_lookup(kaggle_df)

    # --- Match loop ---
    matched_id        = 0
    matched_fuzzy     = 0
    matched_base_ttl  = 0
    already_had       = 0
    unmatched         = 0

    for idx, row in df.iterrows():
        # Skip rows that already have features
        if row.get("audio_features_available") is True:
            already_had += 1
            continue

        features = None

        # Pass 1: exact spotify_id match
        sid = str(row.get("spotify_id", "")).strip()
        if sid and sid in id_lookup:
            features = _extract_features(id_lookup[sid])
            matched_id += 1

        # Pass 2: normalised title + primary artist
        if features is None:
            norm_title  = _normalise(str(row.get("spotify_title") or row.get("title", "")))
            norm_artist = _normalise(str(row.get("primary_artist") or row.get("artist", "")))
            key = (norm_title, norm_artist)
            if key in fuzzy_lookup:
                features = _extract_features(fuzzy_lookup[key])
                matched_fuzzy += 1

        # Pass 3: base title (brackets/dashes/version keywords stripped) + primary artist
        if features is None:
            base_title  = _normalise_title(str(row.get("spotify_title") or row.get("title", "")))
            norm_artist = _normalise(str(row.get("primary_artist") or row.get("artist", "")))
            key = (base_title, norm_artist)
            if key in base_ttl_lookup:
                features = _extract_features(base_ttl_lookup[key])
                matched_base_ttl += 1

        if features is None:
            unmatched += 1
            continue

        # Write features back into the dataframe
        for col, val in features.items():
            if col not in df.columns:
                df[col] = None
            df.at[idx, col] = val

    # --- Save ---
    df.to_csv(MODELING_CSV, index=False)

    # --- Summary ---
    total_with_features = int(df["audio_features_available"].sum())
    print("\n" + "=" * 56)
    print("  AUDIO FEATURES MERGE SUMMARY")
    print("=" * 56)
    print(f"  Total rows                  : {len(df)}")
    print(f"  Already had features        : {already_had}")
    print(f"  Matched via spotify_id      : {matched_id}")
    print(f"  Matched via title+artist    : {matched_fuzzy}")
    print(f"  Matched via base title      : {matched_base_ttl}")
    print(f"  Unmatched (no features)     : {unmatched}")
    print(f"  Total rows with features    : {total_with_features}")
    print(f"  Feature fill rate           : {total_with_features/len(df)*100:.1f}%")
    print(f"\n  Saved -> {MODELING_CSV}")
    print("=" * 56)


if __name__ == "__main__":
    main()
