"""
build_expanded_dataset.py  —  Billboard Boxing
===============================================
Builds an expanded ~50/50 labeled dataset from two local files:

  Positives (label=1):
    data/hot-100-current.csv  — Billboard Hot 100 weekly, 1958-2024
    Matched against the Kaggle audio features dataset to get features.

  Negatives (label=0):
    data/archive/data.csv  — Kaggle 170k Spotify tracks
    Songs NOT in the Billboard dataset, sampled to match the positive count.

Both classes end up with audio features already filled — no Spotify API needed.

Match strategy for positives (in order)
-----------------------------------------
1. Exact title + artist  (normalised)
2. Base title + artist   (brackets, dashes, remix/version keywords stripped)

Output
------
  data/processed/billboard_expanded_dataset.csv

Usage
-----
    python3 src/cleaning/build_expanded_dataset.py
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
BILLBOARD_CSV = os.path.join(PROJECT_ROOT, "data", "hot-100-current.csv")
KAGGLE_CSV    = os.path.join(PROJECT_ROOT, "data", "archive", "data.csv")
OUTPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "billboard_expanded_dataset.csv")

# ---------------------------------------------------------------------------
# Audio feature columns to carry over
# ---------------------------------------------------------------------------

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# ---------------------------------------------------------------------------
# Text normalisation
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


def _norm(text: str) -> str:
    """Lowercase, strip featuring clause, remove non-alphanumeric chars."""
    t = str(text).lower().strip()
    t = _FEAT_RE.sub('', t)
    t = _CLEAN_RE.sub('', t)
    return t.strip()


def _norm_base(text: str) -> str:
    """Aggressive: also strips brackets, dashes, and remix/version keywords."""
    t = str(text).lower().strip()
    t = _FEAT_RE.sub('', t)
    t = _BRACKET_RE.sub('', t)
    t = _DASH_RE.sub('', t)
    t = _VERSION_RE.sub('', t)
    t = _CLEAN_RE.sub('', t)
    return t.strip()


def _parse_first_artist(raw: str) -> str:
    """Extract the first artist from a stringified list like \"['Artist A', 'Artist B']\"."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return str(raw)

# ---------------------------------------------------------------------------
# Load & prepare Kaggle dataset
# ---------------------------------------------------------------------------

def load_kaggle(path: str) -> pd.DataFrame:
    log.info(f"Loading Kaggle dataset from {path} ...")
    df = pd.read_csv(path)
    log.info(f"  {len(df):,} rows loaded.")

    df = df.rename(columns={"id": "spotify_id", "name": "title"})
    df["primary_artist"] = df["artists"].apply(_parse_first_artist)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["_norm_title"]      = df["title"].apply(_norm)
    df["_norm_title_base"] = df["title"].apply(_norm_base)
    df["_norm_artist"]     = df["primary_artist"].apply(_norm)

    # Keep most popular version per (title, artist)
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["_norm_title", "_norm_artist"], keep="first")

    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load & prepare Billboard dataset
# ---------------------------------------------------------------------------

def load_billboard(path: str) -> pd.DataFrame:
    log.info(f"Loading Billboard Hot 100 from {path} ...")
    df = pd.read_csv(path)
    df["year"] = pd.to_datetime(df["chart_week"], errors="coerce").dt.year

    # Deduplicate — keep the entry where the song peaked highest
    df = df.sort_values("peak_pos", ascending=True)
    df = df.drop_duplicates(subset=["title", "performer"], keep="first")
    df = df.rename(columns={"performer": "artist", "current_week": "peak_rank"})

    df["_norm_title"]      = df["title"].apply(_norm)
    df["_norm_title_base"] = df["title"].apply(_norm_base)
    df["_norm_artist"]     = df["artist"].apply(_norm)

    log.info(f"  {len(df):,} unique songs after deduplication.")
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Match Billboard songs to Kaggle audio features
# ---------------------------------------------------------------------------

def match_features(billboard: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    log.info("Building Kaggle lookup tables ...")
    exact_lookup = {
        (row["_norm_title"], row["_norm_artist"]): row
        for _, row in kaggle.iterrows()
    }
    base_lookup = {}
    for _, row in kaggle.iterrows():
        key = (row["_norm_title_base"], row["_norm_artist"])
        if key not in base_lookup:
            base_lookup[key] = row

    matched_exact = 0
    matched_base  = 0
    unmatched     = 0
    records       = []

    log.info("Matching Billboard songs to audio features ...")
    for _, brow in billboard.iterrows():
        krow = exact_lookup.get((brow["_norm_title"], brow["_norm_artist"]))
        if krow is not None:
            matched_exact += 1
        else:
            krow = base_lookup.get((brow["_norm_title_base"], brow["_norm_artist"]))
            if krow is not None:
                matched_base += 1
            else:
                unmatched += 1

        rec = {
            "label":          1,
            "year":           brow["year"],
            "title":          brow["title"],
            "artist":         brow["artist"],
            "peak_rank":      brow.get("peak_rank"),
            "wks_on_chart":   brow.get("wks_on_chart"),
        }
        if krow is not None:
            rec["spotify_id"]              = krow["spotify_id"]
            rec["popularity"]              = krow["popularity"]
            rec["explicit"]                = krow.get("explicit")
            rec["duration_ms"]             = krow.get("duration_ms")
            rec["release_date"]            = krow.get("release_date")
            rec["audio_features_available"] = True
            for col in AUDIO_COLS:
                rec[col] = krow.get(col)
        else:
            rec["audio_features_available"] = False

        records.append(rec)

    log.info(f"  Matched exact  : {matched_exact:,}")
    log.info(f"  Matched base   : {matched_base:,}")
    log.info(f"  Unmatched      : {unmatched:,}")
    log.info(f"  Match rate     : {(matched_exact + matched_base) / len(billboard) * 100:.1f}%")

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Sample negatives from Kaggle (songs not in Billboard)
# ---------------------------------------------------------------------------

def sample_negatives(kaggle: pd.DataFrame, billboard: pd.DataFrame, n: int,
                     seed: int = 42) -> pd.DataFrame:
    log.info(f"Sampling {n:,} negatives from Kaggle ...")

    # Build a set of all Billboard (norm_title, norm_artist) pairs to exclude
    hit_keys = set(zip(billboard["_norm_title"], billboard["_norm_artist"]))
    hit_keys |= set(zip(billboard["_norm_title_base"], billboard["_norm_artist"]))

    non_hits = kaggle[
        ~kaggle.apply(lambda r: (r["_norm_title"], r["_norm_artist"]) in hit_keys, axis=1)
    ].copy()
    log.info(f"  {len(non_hits):,} candidate non-hits before sampling.")

    sampled = non_hits.sample(n=min(n, len(non_hits)), random_state=seed)

    records = []
    for _, row in sampled.iterrows():
        rec = {
            "label":                    0,
            "year":                     row["year"],
            "title":                    row["title"],
            "artist":                   row["primary_artist"],
            "spotify_id":               row["spotify_id"],
            "popularity":               row["popularity"],
            "explicit":                 row.get("explicit"),
            "duration_ms":              row.get("duration_ms"),
            "release_date":             row.get("release_date"),
            "audio_features_available": True,
        }
        for col in AUDIO_COLS:
            rec[col] = row.get(col)
        records.append(rec)

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Impute missing audio features (median by year, fallback to global median)
# ---------------------------------------------------------------------------

def impute_audio_features(dataset: pd.DataFrame) -> pd.DataFrame:
    log.info("Imputing missing audio features ...")
    n_missing_before = int((dataset["audio_features_available"] == False).sum())

    # Compute per-year medians from rows that already have features
    year_medians = (
        dataset[dataset["audio_features_available"] == True]
        .groupby("year")[AUDIO_COLS]
        .median()
    )
    global_medians = dataset[dataset["audio_features_available"] == True][AUDIO_COLS].median()

    for idx, row in dataset[dataset["audio_features_available"] == False].iterrows():
        year = row["year"]
        for col in AUDIO_COLS:
            if year in year_medians.index:
                dataset.at[idx, col] = year_medians.loc[year, col]
            else:
                dataset.at[idx, col] = global_medians[col]
        dataset.at[idx, "audio_features_available"] = True
        dataset.at[idx, "features_imputed"] = True

    log.info(f"  Imputed features for {n_missing_before:,} rows.")
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    kaggle    = load_kaggle(KAGGLE_CSV)
    billboard = load_billboard(BILLBOARD_CSV)

    positives = match_features(billboard, kaggle)
    n_hits    = len(positives)

    negatives = sample_negatives(kaggle, billboard, n=n_hits)

    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.sort_values(["year", "label"], ascending=[True, False]).reset_index(drop=True)

    dataset = impute_audio_features(dataset)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    dataset.to_csv(OUTPUT_CSV, index=False)

    total    = len(dataset)
    n_pos    = int((dataset["label"] == 1).sum())
    n_neg    = int((dataset["label"] == 0).sum())
    n_real    = int((dataset["audio_features_available"] == True).sum())
    n_imputed = int(dataset.get("features_imputed", pd.Series(False, index=dataset.index)).sum())

    print("\n" + "=" * 60)
    print("  EXPANDED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows                  : {total:,}")
    print(f"  Positives  (label=1)        : {n_pos:,}  ({n_pos/total*100:.1f}%)")
    print(f"  Negatives  (label=0)        : {n_neg:,}  ({n_neg/total*100:.1f}%)")
    print(f"  Real audio features         : {n_real:,}  ({n_real/total*100:.1f}%)")
    print(f"  Imputed audio features      : {n_imputed:,}  ({n_imputed/total*100:.1f}%)")
    print(f"  Year range                  : {int(dataset['year'].min())} – {int(dataset['year'].max())}")
    print(f"\n  Saved -> {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
