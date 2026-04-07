"""
spotify_api.py  —  Billboard Boxing
====================================
Searches Spotify for every song in the cleaned Billboard Year-End Hot 100
dataset, collects track metadata and audio features, and writes two CSVs:

    data/raw/spotify_billboard_matches.csv   — matched rows (Billboard + Spotify)
    data/raw/spotify_unmatched.csv           — songs with no Spotify match

Credentials
-----------
Export these before running:

    export SPOTIFY_CLIENT_ID=<your client id>
    export SPOTIFY_CLIENT_SECRET=<your client secret>

Get credentials at https://developer.spotify.com/dashboard

Usage
-----
    python src/scraping/spotify_api.py
    python src/scraping/spotify_api.py --input data/processed/billboard_clean.csv
"""

import os
import re
import time
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

# Load billboard-boxing/.env into os.environ before credentials are read
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_INPUT_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "billboard_clean.csv")
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "spotify_billboard_matches.csv")
DEFAULT_UNMATCHED_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "spotify_unmatched.csv")

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

TOKEN_URL    = "https://accounts.spotify.com/api/token"
SEARCH_URL   = "https://api.spotify.com/v1/search"
TRACKS_URL   = "https://api.spotify.com/v1/tracks"
FEATURES_URL = "https://api.spotify.com/v1/audio-features"

# Seconds to sleep between individual search requests (~150 req/min, limit is ~180)
SEARCH_DELAY = 0.4

# Max retries for a single API call before giving up
MAX_RETRIES = 3

# Minimum ratio (0–1) of overlapping words for a fuzzy artist match
ARTIST_MATCH_THRESHOLD = 0.5

# Save progress to disk every N newly-processed songs
CHECKPOINT_EVERY = 25

# ---------------------------------------------------------------------------
# Credentials & authentication
# ---------------------------------------------------------------------------

def load_credentials() -> tuple[str, str]:
    """
    Read Spotify credentials from environment variables.

    Required env vars:
        SPOTIFY_CLIENT_ID
        SPOTIFY_CLIENT_SECRET

    Raises
    ------
    EnvironmentError if either variable is missing or empty.
    """
    cid = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise EnvironmentError(
            "Both SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set "
            "as environment variables."
        )
    return cid, secret


def get_access_token(client_id: str, client_secret: str) -> str:
    """
    Obtain a Client Credentials bearer token from Spotify.

    This flow grants read-only access to public data (search, track info,
    audio features) without a user login.

    Parameters
    ----------
    client_id, client_secret : str

    Returns
    -------
    str  — bearer token valid for ~3600 seconds
    """
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    log.info("Spotify access token obtained.")
    return token


# ---------------------------------------------------------------------------
# HTTP helper with retry / rate-limit handling
# ---------------------------------------------------------------------------

class SpotifySession:
    """
    Wraps requests with:
    - Bearer token auth
    - Auto-refresh on 401 (token expired)
    - Back-off + retry on 429 (rate limited) and transient 5xx errors
    """

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = get_access_token(client_id, client_secret)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, url: str, params: dict | None = None) -> dict:
        """
        GET *url* with *params*, returning parsed JSON.

        Retries up to MAX_RETRIES times on rate-limit or server errors.
        Raises requests.HTTPError if all retries fail.
        """
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
                log.warning("Token expired — refreshing...")
                self._token = get_access_token(self.client_id, self.client_secret)
                continue  # retry immediately with new token

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                log.warning(f"Rate limited — waiting {wait}s (attempt {attempt}/{MAX_RETRIES}).")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                log.warning(f"Server error {resp.status_code} (attempt {attempt}/{MAX_RETRIES}).")
                time.sleep(2 ** attempt)
                continue

            # 4xx that isn't 401/429 — not worth retrying.
            # Log 403 as warning (common for audio-features on free-tier apps).
            if resp.status_code == 403:
                log.warning(f"HTTP 403 (access denied — may be app tier limit): {url}")
            else:
                log.error(f"HTTP {resp.status_code}: {url}")
            resp.raise_for_status()

        raise requests.HTTPError(f"All {MAX_RETRIES} attempts failed for {url}")


# ---------------------------------------------------------------------------
# Fuzzy match helper
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """Return the set of lowercase alphabetic words in *text*."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _artist_similarity(expected: str, candidate: str) -> float:
    """
    Jaccard-like word overlap between two artist strings (0.0 – 1.0).

    Ignores stopwords and punctuation so 'Lady Gaga' matches 'Lady Gaga feat. …'.
    """
    STOPWORDS = {"the", "a", "an", "and", "or", "of", "feat", "ft", "featuring", "with"}
    a = _word_set(expected) - STOPWORDS
    b = _word_set(candidate) - STOPWORDS
    if not a:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------

def search_track(
    session: SpotifySession,
    song_title: str,
    artist: str,
    year: int | None = None,
) -> list[dict]:
    """
    Search Spotify for a track and return up to 5 candidate results.

    Tries queries from most to least specific:
    1. Structured:  track:"<title>" artist:"<artist>"  year:<year>
    2. Structured:  track:"<title>" artist:"<artist>"
    3. Loose:       <title> <artist>

    Parameters
    ----------
    session : SpotifySession
    song_title : str   — cleaned (lowercase, no outer quotes)
    artist : str       — primary artist, lowercase, no "featuring …"
    year : int | None  — chart year; used as a release-year hint

    Returns
    -------
    list[dict]
        Up to 5 raw Spotify track objects (may be empty).
    """
    base_structured = f'track:"{song_title}" artist:"{artist}"'
    queries = []

    if year:
        queries.append(f"{base_structured} year:{year}")   # most specific
    queries.append(base_structured)                         # without year
    queries.append(f"{song_title} {artist}")                # loose fallback

    for query in queries:
        try:
            data = session.get(SEARCH_URL, params={"q": query, "type": "track", "limit": 5})
            items = data.get("tracks", {}).get("items", [])
            if items:
                return items
        except requests.HTTPError as exc:
            log.warning(f"Search failed for '{song_title}' / '{artist}': {exc}")
            return []

    return []


def choose_best_match(
    candidates: list[dict],
    expected_artist: str,
    expected_title: str,
) -> dict | None:
    """
    Choose the most appropriate track from Spotify search candidates.

    Selection logic (conservative — prefers precision over recall):
    1. Score each candidate by artist word-overlap with *expected_artist*.
    2. Require the score to exceed ARTIST_MATCH_THRESHOLD.
    3. Among qualifying candidates, pick the one with the highest Spotify
       popularity (a reasonable proxy for the well-known version).
    4. Return None if no candidate clears the threshold.

    Parameters
    ----------
    candidates : list[dict]
        Raw Spotify track objects from search_track().
    expected_artist : str
        Primary artist name (lowercase, no featuring).
    expected_title : str
        Cleaned song title (lowercase, no outer quotes).

    Returns
    -------
    dict or None  — best-matching Spotify track object, or None.
    """
    if not candidates:
        return None

    scored = []
    for track in candidates:
        spotify_artists = " ".join(a.get("name", "") for a in track.get("artists", []))
        score = _artist_similarity(expected_artist, spotify_artists)
        scored.append((score, track.get("popularity", 0), track))

    # Filter by artist similarity threshold
    qualified = [(s, pop, t) for s, pop, t in scored if s >= ARTIST_MATCH_THRESHOLD]

    if not qualified:
        return None

    # Best = highest artist score, then highest popularity as tiebreaker
    qualified.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return qualified[0][2]


def get_track_metadata(session: SpotifySession, track_id: str) -> dict:
    """
    Fetch full track metadata from the /tracks/{id} endpoint.

    This is richer than the search result object (includes more album detail).
    Returns an empty dict on failure rather than raising, so one bad ID
    doesn't halt the pipeline.

    Parameters
    ----------
    session : SpotifySession
    track_id : str

    Returns
    -------
    dict  — flat dict of selected metadata fields.
    """
    try:
        track = session.get(f"{TRACKS_URL}/{track_id}")
    except requests.HTTPError as exc:
        log.warning(f"Track metadata fetch failed for {track_id}: {exc}")
        return {}

    album   = track.get("album", {})
    artists = track.get("artists", [{}])

    return {
        "spotify_id":         track.get("id"),
        "spotify_title":      track.get("name"),
        "spotify_artist":     ", ".join(a.get("name", "") for a in artists),
        "spotify_album":      album.get("name"),
        "album_type":         album.get("album_type"),
        "release_date":       album.get("release_date"),
        "release_date_precision": album.get("release_date_precision"),
        "popularity":         track.get("popularity"),
        "explicit":           track.get("explicit"),
        "duration_ms":        track.get("duration_ms"),
        "disc_number":        track.get("disc_number"),
        "track_number":       track.get("track_number"),
        "isrc":               track.get("external_ids", {}).get("isrc"),
        "spotify_url":        track.get("external_urls", {}).get("spotify"),
        "preview_url":        track.get("preview_url"),
    }


def get_audio_features(session: SpotifySession, track_id: str) -> dict:
    """
    Fetch audio features for a single Spotify track.

    Returns a flat dict of feature values, or a dict of Nones if the
    track has no audio analysis on Spotify (common for older or regional
    tracks). Logs the outcome either way — nothing is silently dropped.

    Parameters
    ----------
    session : SpotifySession
    track_id : str

    Returns
    -------
    dict
        Keys: danceability, energy, key, loudness, mode, speechiness,
              acousticness, instrumentalness, liveness, valence, tempo,
              time_signature, audio_features_available.
    """
    FEATURE_KEYS = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature",
    ]
    null_result = {k: None for k in FEATURE_KEYS}
    null_result["audio_features_available"] = False

    try:
        data = session.get(f"{FEATURES_URL}/{track_id}")
    except requests.HTTPError as exc:
        log.warning(f"Audio features unavailable for {track_id}: {exc}")
        return null_result

    if not data or data.get("id") is None:
        log.debug(f"Audio features: no data for {track_id}")
        return null_result

    result = {k: data.get(k) for k in FEATURE_KEYS}
    result["audio_features_available"] = True
    return result


# ---------------------------------------------------------------------------
# Resume / checkpoint helpers
# ---------------------------------------------------------------------------

def _row_key(row: dict) -> tuple:
    """
    Return a stable identity tuple for a Billboard row.

    Uses year + rank + cleaned title + primary artist so that the key is
    consistent whether the row came from the input CSV or a saved checkpoint.
    """
    return (
        str(row.get("year", "")).strip(),
        str(row.get("rank", "")).strip(),
        str(row.get("title_clean", row.get("title", ""))).strip().lower(),
        str(row.get("primary_artist", row.get("artist", ""))).strip().lower(),
    )


def _load_existing_keys(csv_path: str) -> tuple[list[dict], set[tuple]]:
    """
    Load an existing output CSV and return (rows_as_dicts, set_of_row_keys).

    Returns ([], set()) if the file does not exist or cannot be read.
    The key set is used to skip already-processed rows on resume.
    """
    if not os.path.exists(csv_path):
        return [], set()
    try:
        df = pd.read_csv(csv_path, dtype=str)
        rows = df.to_dict(orient="records")
        keys = {_row_key(r) for r in rows}
        log.info(f"  Resume: loaded {len(rows)} existing rows from {csv_path}")
        return rows, keys
    except Exception as exc:
        log.warning(f"Could not read {csv_path}: {exc} — starting fresh for this file.")
        return [], set()


def _save_checkpoint(rows: list[dict], path: str) -> None:
    """
    Write *rows* to *path* atomically using a .tmp file + os.replace().

    os.replace() is atomic on the same filesystem (POSIX and Windows), so a
    crash mid-write leaves the previous checkpoint intact rather than a
    half-written CSV.

    Does nothing if *rows* is empty.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)  # atomic swap


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_billboard_csv(
    input_csv: str = DEFAULT_INPUT_CSV,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    unmatched_csv: str = DEFAULT_UNMATCHED_CSV,
) -> None:
    """
    End-to-end pipeline: load Billboard CSV → resume from existing output →
    search Spotify for new rows → enrich with audio features → checkpoint
    every 25 songs → final save + summary.

    No rows from *input_csv* are silently dropped:
    - Matched rows appear in *output_csv*.
    - Unmatched rows appear in *unmatched_csv*.

    Resume behaviour
    ----------------
    If *output_csv* or *unmatched_csv* already exist (from a prior interrupted
    run), the rows they contain are loaded and their keys are used to skip
    already-processed input rows. No duplicates are written on resume.

    Checkpoint behaviour
    --------------------
    Every CHECKPOINT_EVERY newly-processed songs, both output files are
    written atomically (tmp file + os.replace) so a crash loses at most
    CHECKPOINT_EVERY songs worth of work.
    """
    # --- Credentials & session ---
    client_id, client_secret = load_credentials()
    session = SpotifySession(client_id, client_secret)

    # --- Load Billboard input ---
    log.info(f"Loading Billboard data from {input_csv}")
    df = pd.read_csv(input_csv)
    total_input = len(df)
    log.info(f"  {total_input} rows loaded.")

    # Normalise column names — handle both raw and cleaned CSVs
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("title", "song", "song_title", "single"):
            col_map[col] = "title"
        elif lower == "title_clean":
            col_map[col] = "title_clean"
        elif lower in ("artist", "artist_name"):
            col_map[col] = "artist"
        elif lower == "primary_artist":
            col_map[col] = "primary_artist"
        elif lower == "year":
            col_map[col] = "year"
        elif lower == "rank":
            col_map[col] = "rank"
    df = df.rename(columns=col_map)

    # Derive cleaned search fields if not already present
    if "title_clean" not in df.columns:
        df["title_clean"] = df["title"].astype(str).str.strip().str.lower()
    if "primary_artist" not in df.columns:
        df["primary_artist"] = (
            df["artist"].astype(str)
            .str.lower()
            .str.replace(
                r"\s*(?:featuring|feat\.?|ft\.?|with|&)\s+.*$", "", regex=True
            )
            .str.strip()
        )

    # --- Resume: load any existing output files ---
    log.info("Checking for existing output files to resume from...")
    matched_rows,   matched_keys   = _load_existing_keys(output_csv)
    unmatched_rows, unmatched_keys = _load_existing_keys(unmatched_csv)

    # A row is already processed if its key appears in either output file
    already_processed_keys = matched_keys | unmatched_keys
    skipped = 0

    # --- Search loop ---
    newly_processed = 0  # count since last checkpoint
    log.info(f"Starting Spotify search. {total_input} input rows, "
             f"{len(already_processed_keys)} already processed.\n")

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        key = _row_key(row_dict)

        # Skip rows already in a previous run's output
        if key in already_processed_keys:
            skipped += 1
            continue

        title  = row.get("title_clean", row.get("title", ""))
        artist = row.get("primary_artist", row.get("artist", ""))
        year   = row.get("year")

        # --- Search Spotify ---
        candidates = search_track(session, title, artist, year)
        best = choose_best_match(candidates, artist, title)

        if best is None:
            log.debug(f"  No match: '{title}' — '{artist}' ({year})")
            unmatched_rows.append(row_dict)
        else:
            # Full track metadata
            track_id = best.get("id")
            metadata = get_track_metadata(session, track_id)
            time.sleep(SEARCH_DELAY)

            # Audio features — 403 is logged as warning and never fatal
            features = get_audio_features(session, track_id)
            time.sleep(SEARCH_DELAY)

            status = "features OK" if features.get("audio_features_available") else "no features"
            log.debug(f"  Matched ({status}): '{title}' — '{artist}'")
            matched_rows.append({**row_dict, **metadata, **features})

        newly_processed += 1
        time.sleep(SEARCH_DELAY)

        # --- Progress report + checkpoint every CHECKPOINT_EVERY songs ---
        if newly_processed % CHECKPOINT_EVERY == 0:
            total_done = skipped + newly_processed
            print(
                f"  [{total_done}/{total_input}]  "
                f"new: {newly_processed}  matched: {len(matched_rows)}  "
                f"unmatched: {len(unmatched_rows)}"
            )
            _save_checkpoint(matched_rows,   output_csv)
            _save_checkpoint(unmatched_rows, unmatched_csv)
            log.info("  Checkpoint saved.")

    # --- Final save (catches any remainder after the last checkpoint) ---
    _save_checkpoint(matched_rows,   output_csv)
    _save_checkpoint(unmatched_rows, unmatched_csv)

    if matched_rows:
        log.info(f"Matches saved   -> {output_csv}")
    else:
        log.warning("No matches found — matches CSV not written.")

    if unmatched_rows:
        log.info(f"Unmatched saved -> {unmatched_csv}")
    else:
        log.info("All songs matched — no unmatched CSV written.")

    # --- Summary ---
    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
    with_features    = int(matched_df["audio_features_available"].sum()) if not matched_df.empty else 0
    without_features = len(matched_rows) - with_features

    print("\n" + "=" * 56)
    print("  SUMMARY")
    print("=" * 56)
    print(f"  Total input songs         : {total_input}")
    print(f"  Skipped (already done)    : {skipped}")
    print(f"  Newly processed this run  : {newly_processed}")
    print(f"  Matched (all-time)        : {len(matched_rows)}")
    print(f"  Unmatched (all-time)      : {len(unmatched_rows)}")
    print(f"  With audio features       : {with_features}")
    print(f"  Without audio features    : {without_features}")
    print("=" * 56 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spotify enrichment for Billboard Boxing")
    p.add_argument("--input",     default=DEFAULT_INPUT_CSV,   help="Path to cleaned Billboard CSV")
    p.add_argument("--output",    default=DEFAULT_OUTPUT_CSV,  help="Destination for matched rows CSV")
    p.add_argument("--unmatched", default=DEFAULT_UNMATCHED_CSV, help="Destination for unmatched rows CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_billboard_csv(
        input_csv=args.input,
        output_csv=args.output,
        unmatched_csv=args.unmatched,
    )
