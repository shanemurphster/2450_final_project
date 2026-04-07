"""
spotify_negative_sampler.py  —  Billboard Boxing
=================================================
Collects negative (non-hit) song samples from Spotify for each Billboard
year-end Hot 100 year.  These form the label=0 class for binary classification.

Strategy
--------
For each Billboard year Y:
  1. Search Spotify for tracks released in years Y, Y-1, Y-2 using a small
     set of broad genre/mood search terms.  This gives a varied pool without
     overcomplicated logic.
  2. Flatten and clean every candidate track.
  3. Remove any song that appears in the Billboard top-100 for year Y
     (overlap check uses cleaned title + primary artist; scoped to year Y only).
  4. Deduplicate by spotify_id, then by cleaned title + primary_artist.
  5. Sample up to TARGET_PER_YEAR (400) rows; warn if fewer than MIN_PER_YEAR (350).

Output files
------------
  data/raw/spotify_negative_candidates.csv    — full candidate pool (pre-sample)
  data/processed/spotify_negatives_by_year.csv — final sampled negatives, label=0

Schema is kept as close as possible to the positive matches file so the two
datasets can be concatenated for modelling without extra wrangling.

Credentials
-----------
Set environment variables (or add to billboard-boxing/.env):

    SPOTIFY_CLIENT_ID=<your client id>
    SPOTIFY_CLIENT_SECRET=<your client secret>

Usage
-----
    python src/scraping/spotify_negative_sampler.py
    python src/scraping/spotify_negative_sampler.py --start-year 2015 --end-year 2019
"""

import os
import re
import time
import random
import logging
import argparse

import requests
import pandas as pd
from dotenv import load_dotenv

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

BILLBOARD_CSV      = os.path.join(PROJECT_ROOT, "data", "processed", "billboard_clean.csv")
CANDIDATES_CSV     = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_negative_candidates.csv")
NEGATIVES_CSV      = os.path.join(PROJECT_ROOT, "data", "processed", "spotify_negatives_by_year.csv")

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------

TARGET_PER_YEAR = 400   # ideal number of negatives per Billboard year
MIN_PER_YEAR    = 350   # warn if we fall below this

# Spotify returns max 50 results per search call; we page through offsets.
# SEARCHES_PER_YEAR * RESULTS_PER_SEARCH is the raw candidate pool size
# before filtering.  Adjust upward if too many are removed as overlaps.
RESULTS_PER_SEARCH = 50   # Spotify max per call
OFFSETS_PER_QUERY  = 6    # pages per query  → 50 * 6 = 300 results per query
SEARCH_DELAY       = 0.4  # seconds between API calls
MAX_RETRIES        = 3
CHECKPOINT_EVERY   = 50   # save to disk every N candidates collected

# Broad search terms used to generate candidate pools.
# These are intentionally generic — we want variety, not precision.
SEARCH_TERMS = [
    "pop",
    "hip hop",
    "rock",
    "r&b",
    "dance",
    "indie",
    "country",
    "electronic",
    "soul",
    "alternative",
]

# ---------------------------------------------------------------------------
# Spotify API constants
# ---------------------------------------------------------------------------

TOKEN_URL  = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_credentials() -> tuple[str, str]:
    """Load SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET from env."""
    cid    = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise EnvironmentError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set."
        )
    return cid, secret


def get_access_token(client_id: str, client_secret: str) -> str:
    """Obtain a Client Credentials bearer token."""
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    log.info("Spotify access token obtained.")
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# HTTP session with retry / rate-limit handling
# ---------------------------------------------------------------------------

class SpotifySession:
    """Requests wrapper: bearer auth, auto token-refresh, retry on 429/5xx."""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id     = client_id
        self.client_secret = client_secret
        self._token        = get_access_token(client_id, client_secret)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url, headers=self._headers(), params=params, timeout=15
                )
            except requests.RequestException as exc:
                log.warning(f"Network error (attempt {attempt}/{MAX_RETRIES}): {exc}")
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                log.warning("Token expired — refreshing.")
                self._token = get_access_token(self.client_id, self.client_secret)
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                log.warning(f"Rate limited — waiting {wait}s.")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                log.warning(f"Server error {resp.status_code} (attempt {attempt}).")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 403:
                log.warning(f"HTTP 403 (app-tier limit): {url}")
                resp.raise_for_status()
            log.error(f"HTTP {resp.status_code}: {url}")
            resp.raise_for_status()

        raise requests.HTTPError(f"All {MAX_RETRIES} attempts failed for {url}")


# ---------------------------------------------------------------------------
# Text cleaning helpers (mirrors clean_billboard_data.py)
# ---------------------------------------------------------------------------

_FEAT_RE = re.compile(
    r"\s*(?:featuring|feat\.?|ft\.?|with|&)\s+.*$", flags=re.IGNORECASE
)

def _clean_text(text: str) -> str:
    """Lowercase, strip whitespace and Wikipedia-style footnote markers."""
    t = re.sub(r"\[\w[\w\s]*\]", "", str(text))
    return t.strip().lower()

def _primary_artist(artist: str) -> str:
    """Strip 'featuring …' and return the lead artist, lowercased."""
    return _FEAT_RE.sub("", _clean_text(artist)).strip()


# ---------------------------------------------------------------------------
# 1. Load Billboard data
# ---------------------------------------------------------------------------

def load_billboard_data(path: str = BILLBOARD_CSV) -> pd.DataFrame:
    """
    Load the cleaned Billboard CSV.

    Returns a DataFrame with at minimum:
        year, rank, title, artist, title_clean, primary_artist
    """
    df = pd.read_csv(path)
    log.info(f"Billboard data loaded: {len(df)} rows, years {sorted(df['year'].unique())}")

    # Ensure helper columns exist even if loading a simpler CSV
    if "title_clean" not in df.columns:
        df["title_clean"] = df["title"].apply(_clean_text)
    if "primary_artist" not in df.columns:
        df["primary_artist"] = df["artist"].apply(_primary_artist)

    return df


# ---------------------------------------------------------------------------
# 2. Generate search queries for a target year
# ---------------------------------------------------------------------------

def generate_year_pool(target_year: int) -> list[tuple[str, int]]:
    """
    Return a list of (query_string, release_year) pairs to search.

    Each (term, release_year) combination becomes one Spotify search.
    Release years covered: target_year, target_year-1, target_year-2.

    Parameters
    ----------
    target_year : int
        The Billboard chart year we are collecting negatives for.

    Returns
    -------
    list of (query, release_year) tuples
    """
    pairs = []
    for release_year in [target_year, target_year - 1, target_year - 2]:
        for term in SEARCH_TERMS:
            pairs.append((f"{term} year:{release_year}", release_year))
    return pairs


# ---------------------------------------------------------------------------
# 3. Flatten a single Spotify track object
# ---------------------------------------------------------------------------

def _flatten_track(track: dict, billboard_year: int, release_year: int) -> dict:
    """
    Extract fields from a Spotify track object into a flat dict.

    Mirrors the positive schema from spotify_api.py so the two datasets
    can be concatenated for modelling without extra wrangling.
    Added columns vs. positives:
        billboard_year  — the Billboard year this negative is assigned to
        release_year    — the actual Spotify release year
        label           — always 0
    """
    album   = track.get("album", {})
    artists = track.get("artists", [{}])

    raw_title  = track.get("name", "")
    raw_artist = ", ".join(a.get("name", "") for a in artists)

    return {
        # ---- Billboard-side columns (null for negatives) ----
        "year":            billboard_year,   # maps to Billboard year in positives
        "rank":            None,
        "title":           raw_title,
        "artist":          raw_artist,
        "title_clean":     _clean_text(raw_title),
        "artist_clean":    _clean_text(raw_artist),
        "primary_artist":  _primary_artist(raw_artist),

        # ---- Spotify metadata (mirrors positive schema) ----
        "spotify_id":               track.get("id"),
        "spotify_title":            raw_title,
        "spotify_artist":           raw_artist,
        "spotify_album":            album.get("name"),
        "album_type":               album.get("album_type"),
        "release_date":             album.get("release_date"),
        "release_date_precision":   album.get("release_date_precision"),
        "popularity":               track.get("popularity"),
        "explicit":                 track.get("explicit"),
        "duration_ms":              track.get("duration_ms"),
        "disc_number":              track.get("disc_number"),
        "track_number":             track.get("track_number"),
        "isrc":                     track.get("external_ids", {}).get("isrc"),
        "spotify_url":              track.get("external_urls", {}).get("spotify"),
        "preview_url":              track.get("preview_url"),

        # ---- Audio features: filled as null (not fetched here) ----
        # These match the positive schema; set to None to allow concatenation.
        "danceability":             None,
        "energy":                   None,
        "key":                      None,
        "loudness":                 None,
        "mode":                     None,
        "speechiness":              None,
        "acousticness":             None,
        "instrumentalness":         None,
        "liveness":                 None,
        "valence":                  None,
        "tempo":                    None,
        "time_signature":           None,
        "audio_features_available": False,

        # ---- Negative-specific columns ----
        "billboard_year": billboard_year,
        "release_year":   release_year,
        "label":          0,
    }


# ---------------------------------------------------------------------------
# 4. Collect raw candidates for one Billboard year
# ---------------------------------------------------------------------------

def collect_candidates_for_year(
    session: SpotifySession,
    target_year: int,
    existing_ids: set[str],
) -> list[dict]:
    """
    Search Spotify and return a flat list of candidate track dicts for
    Billboard year *target_year*.

    Parameters
    ----------
    session : SpotifySession
    target_year : int
    existing_ids : set[str]
        Spotify track IDs already collected in previous years / resume load.
        Used to skip tracks we already have so the candidate pool stays fresh.

    Returns
    -------
    list[dict]  — unsorted, unfiltered flat track records
    """
    queries = generate_year_pool(target_year)
    candidates: list[dict] = []
    seen_ids: set[str] = set(existing_ids)  # local copy, don't mutate caller's set

    log.info(f"  [{target_year}] Collecting candidates via {len(queries)} queries …")

    for query, release_year in queries:
        for offset in range(0, OFFSETS_PER_QUERY * RESULTS_PER_SEARCH, RESULTS_PER_SEARCH):
            try:
                data = session.get(
                    SEARCH_URL,
                    params={
                        "q":      query,
                        "type":   "track",
                        "limit":  RESULTS_PER_SEARCH,
                        "offset": offset,
                    },
                )
            except requests.HTTPError as exc:
                log.warning(f"    Search error (year={target_year}, q='{query}', offset={offset}): {exc}")
                time.sleep(SEARCH_DELAY)
                continue

            items = data.get("tracks", {}).get("items") or []
            if not items:
                break  # no more results for this query

            for track in items:
                tid = track.get("id")
                if not tid or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                candidates.append(_flatten_track(track, target_year, release_year))

            time.sleep(SEARCH_DELAY)

    log.info(f"  [{target_year}] Raw candidates collected: {len(candidates)}")
    return candidates


# ---------------------------------------------------------------------------
# 5. Clean candidate pool
# ---------------------------------------------------------------------------

def clean_candidate_pool(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Apply basic quality filters to the candidate pool for *target_year*:

    - Drop rows with missing spotify_id, title_clean, or primary_artist.
    - Parse release year from release_date and keep only rows from
      target_year, target_year-1, target_year-2.
    - Deduplicate: first by spotify_id, then by (title_clean, primary_artist).

    Parameters
    ----------
    df : pd.DataFrame
        Raw candidate rows from collect_candidates_for_year().
    target_year : int

    Returns
    -------
    pd.DataFrame  — cleaned, deduplicated
    """
    before = len(df)

    # Drop rows with no usable identity
    df = df[df["spotify_id"].notna() & df["spotify_id"].str.strip().astype(bool)]
    df = df[df["title_clean"].notna()  & df["title_clean"].str.strip().astype(bool)]
    df = df[df["primary_artist"].notna() & df["primary_artist"].str.strip().astype(bool)]

    # Parse release year from release_date (format: YYYY, YYYY-MM, or YYYY-MM-DD)
    df = df.copy()
    df["release_year"] = (
        df["release_date"]
        .astype(str)
        .str[:4]
        .pipe(pd.to_numeric, errors="coerce")
    )

    valid_years = {target_year, target_year - 1, target_year - 2}
    df = df[df["release_year"].isin(valid_years)]

    # Deduplicate
    df = df.drop_duplicates(subset=["spotify_id"])
    df = df.drop_duplicates(subset=["title_clean", "primary_artist"])

    log.info(f"  [{target_year}] After cleaning: {len(df)} rows (dropped {before - len(df)})")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Remove Billboard overlaps (scoped to year Y only)
# ---------------------------------------------------------------------------

def remove_billboard_overlaps(
    candidates: pd.DataFrame,
    billboard_year_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove any candidate that appears in the Billboard top-100 for this year.

    Matching is done on (title_clean, primary_artist) pairs to be robust to
    minor formatting differences.  The check is scoped to the specific
    Billboard year — a song from 2016 is NOT removed just because it appeared
    on a different year's chart.

    Parameters
    ----------
    candidates : pd.DataFrame
        Cleaned candidate pool for one Billboard year.
    billboard_year_df : pd.DataFrame
        Rows from billboard_clean.csv for that same year only.

    Returns
    -------
    pd.DataFrame  — candidates with Billboard hits removed
    """
    # Build a set of (title_clean, primary_artist) tuples from Billboard
    bb_pairs = set(
        zip(
            billboard_year_df["title_clean"].str.strip().str.lower(),
            billboard_year_df["primary_artist"].str.strip().str.lower(),
        )
    )

    before = len(candidates)
    mask = candidates.apply(
        lambda r: (
            str(r["title_clean"]).strip().lower(),
            str(r["primary_artist"]).strip().lower(),
        )
        not in bb_pairs,
        axis=1,
    )
    filtered = candidates[mask].reset_index(drop=True)
    removed = before - len(filtered)
    if removed:
        log.info(f"    Removed {removed} Billboard overlaps.")
    return filtered


# ---------------------------------------------------------------------------
# 7. Sample negatives
# ---------------------------------------------------------------------------

def sample_negatives(df: pd.DataFrame, n: int = TARGET_PER_YEAR) -> pd.DataFrame:
    """
    Sample up to *n* rows from *df*.

    - If len(df) >= n: random sample of exactly n rows.
    - If MIN_PER_YEAR <= len(df) < n: keep all rows, log info.
    - If len(df) < MIN_PER_YEAR: keep all rows, log a warning.

    Parameters
    ----------
    df : pd.DataFrame
    n : int

    Returns
    -------
    pd.DataFrame
    """
    if len(df) >= n:
        return df.sample(n=n, random_state=42).reset_index(drop=True)
    if len(df) >= MIN_PER_YEAR:
        log.info(f"    Sample: only {len(df)} rows available (< {n}), keeping all.")
    else:
        log.warning(
            f"    Sample: only {len(df)} rows available — below minimum of {MIN_PER_YEAR}."
        )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def _save_atomic(df: pd.DataFrame, path: str) -> None:
    """Write *df* to *path* atomically via a .tmp file + os.replace()."""
    if df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_existing(path: str) -> pd.DataFrame:
    """
    Load an existing CSV if it exists, else return an empty DataFrame.
    Logs how many rows were loaded.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str)
        log.info(f"  Resume: loaded {len(df)} existing rows from {path}")
        return df
    except Exception as exc:
        log.warning(f"  Could not read {path}: {exc} — starting fresh.")
        return pd.DataFrame()


def save_outputs(
    all_candidates: list[dict],
    all_negatives: list[dict],
) -> None:
    """
    Write the final candidate pool and sampled negatives to disk.

    Parameters
    ----------
    all_candidates : list[dict]
        Every cleaned candidate row across all years (pre-sample).
    all_negatives : list[dict]
        The final sampled negative rows (post-sample, label=0).
    """
    cand_df = pd.DataFrame(all_candidates) if all_candidates else pd.DataFrame()
    neg_df  = pd.DataFrame(all_negatives)  if all_negatives  else pd.DataFrame()

    _save_atomic(cand_df, CANDIDATES_CSV)
    _save_atomic(neg_df,  NEGATIVES_CSV)

    log.info(f"Candidates saved -> {CANDIDATES_CSV}  ({len(cand_df)} rows)")
    log.info(f"Negatives saved  -> {NEGATIVES_CSV}  ({len(neg_df)} rows)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(start_year: int | None = None, end_year: int | None = None) -> None:
    """
    Full pipeline: for each Billboard year, collect candidates, filter,
    sample, checkpoint, and save.

    Parameters
    ----------
    start_year, end_year : int | None
        Override the year range derived from the Billboard CSV.
    """
    # --- Auth ---
    client_id, client_secret = load_credentials()
    session = SpotifySession(client_id, client_secret)

    # --- Load Billboard data ---
    bb_df = load_billboard_data()
    years = sorted(bb_df["year"].unique())
    if start_year:
        years = [y for y in years if y >= start_year]
    if end_year:
        years = [y for y in years if y <= end_year]
    log.info(f"Processing years: {years}\n")

    # --- Resume: load partial outputs ---
    existing_cand_df = _load_existing(CANDIDATES_CSV)
    existing_neg_df  = _load_existing(NEGATIVES_CSV)

    # Years already fully processed (present in the negatives file)
    done_years: set[int] = set()
    if not existing_neg_df.empty and "billboard_year" in existing_neg_df.columns:
        done_years = set(existing_neg_df["billboard_year"].astype(float).astype(int))
        if done_years:
            log.info(f"  Resume: skipping already-processed years: {sorted(done_years)}")

    # Seed seen_ids from existing candidates to avoid duplicates across runs
    seen_ids_global: set[str] = set()
    if not existing_cand_df.empty and "spotify_id" in existing_cand_df.columns:
        seen_ids_global = set(existing_cand_df["spotify_id"].dropna().unique())

    # Accumulate rows in memory (start from existing data)
    all_candidates: list[dict] = (
        existing_cand_df.to_dict(orient="records") if not existing_cand_df.empty else []
    )
    all_negatives: list[dict] = (
        existing_neg_df.to_dict(orient="records") if not existing_neg_df.empty else []
    )

    year_stats: list[dict] = []
    candidates_since_checkpoint = 0

    # --- Year loop ---
    for year in years:
        if year in done_years:
            log.info(f"[{year}] Already processed — skipping.")
            continue

        log.info(f"\n{'='*52}")
        log.info(f"[{year}] Collecting negatives …")

        # Billboard rows for this year only (used for overlap removal)
        bb_year_df = bb_df[bb_df["year"] == year].copy()

        # 3+4: collect raw candidates
        raw = collect_candidates_for_year(session, year, seen_ids_global)
        if not raw:
            log.warning(f"[{year}] No candidates collected — skipping year.")
            year_stats.append({"year": year, "candidates": 0, "negatives": 0})
            continue

        # Update global seen_ids so future years don't repeat these tracks
        for r in raw:
            if r.get("spotify_id"):
                seen_ids_global.add(r["spotify_id"])

        # 5: clean
        raw_df = pd.DataFrame(raw)
        cleaned = clean_candidate_pool(raw_df, year)

        # 6: remove Billboard overlaps (year Y only)
        filtered = remove_billboard_overlaps(cleaned, bb_year_df)

        # 7: sample
        sampled = sample_negatives(filtered, n=TARGET_PER_YEAR)
        sampled["billboard_year"] = year  # ensure column is set

        # Accumulate
        all_candidates.extend(cleaned.to_dict(orient="records"))
        all_negatives.extend(sampled.to_dict(orient="records"))
        candidates_since_checkpoint += len(cleaned)

        log.info(
            f"[{year}] candidates={len(cleaned)}  "
            f"after_overlap_removal={len(filtered)}  "
            f"sampled={len(sampled)}"
        )
        year_stats.append({
            "year":       year,
            "candidates": len(filtered),
            "negatives":  len(sampled),
        })

        # Checkpoint every CHECKPOINT_EVERY candidates
        if candidates_since_checkpoint >= CHECKPOINT_EVERY:
            _save_atomic(pd.DataFrame(all_candidates), CANDIDATES_CSV)
            _save_atomic(pd.DataFrame(all_negatives),  NEGATIVES_CSV)
            log.info("  Checkpoint saved.")
            candidates_since_checkpoint = 0

    # --- Final save ---
    save_outputs(all_candidates, all_negatives)

    # --- Summary ---
    total_neg = sum(s["negatives"] for s in year_stats)
    print("\n" + "=" * 56)
    print("  NEGATIVE SAMPLING SUMMARY")
    print("=" * 56)
    print(f"  {'Year':<8}  {'Candidates':>12}  {'Sampled':>9}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*9}")
    for s in year_stats:
        print(f"  {s['year']:<8}  {s['candidates']:>12}  {s['negatives']:>9}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*9}")
    print(f"  {'TOTAL':<8}  {'':>12}  {total_neg:>9}")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Spotify negative samples for Billboard Boxing")
    p.add_argument("--start-year", type=int, default=None, help="First Billboard year to process")
    p.add_argument("--end-year",   type=int, default=None, help="Last Billboard year to process")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(start_year=args.start_year, end_year=args.end_year)
