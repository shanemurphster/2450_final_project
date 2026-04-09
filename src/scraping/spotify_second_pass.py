"""
spotify_second_pass.py  —  Billboard Boxing
============================================
Second-pass matcher for songs that failed the first Spotify search round.

Why songs failed the first pass (diagnosed from spotify_unmatched.csv)
-----------------------------------------------------------------------
1. TRIPLE-QUOTED TITLES  — Wikipedia curly quotes stacked with CSV quoting
   produce titles like  \"\"\"Empire State of Mind\"\"\"  which corrupt the
   Spotify search query.

2. TRUNCATED primary_artist — a regex edge-case cuts some names mid-word
   e.g. "taylor swi", "da" (Daft Punk).  Searches then fail the artist
   similarity threshold.

3. COMPLEX MULTI-ARTIST CREDITS — entries like
   "Nio García, Darell and Casper Mágico featuring Bad Bunny, Nicky Jam…"
   generate noisy primary_artist values that nothing on Spotify matches.

4. GENUINELY UNAVAILABLE — a small tail of very new or obscure tracks
   (e.g. 2025 Huntrix songs) may simply not be on Spotify yet.

Strategy (applied in order, stopping at first hit)
---------------------------------------------------
Pass A — Clean title + full artist string (fixes issue 1 + 2)
Pass B — Clean title + first word(s) of artist  (fixes issue 3)
Pass C — Title-only search, pick most popular result  (broad fallback)
Pass D — Title + year search, no artist filter       (catches remixes etc.)

Each pass uses a LOWER artist-similarity threshold than the original
(0.3 vs 0.5) because we already know the first-pass conservative threshold
was too strict for these edge cases.

Output files
------------
  data/raw/spotify_second_pass_matched.csv   — newly matched rows
  data/raw/spotify_second_pass_unmatched.csv — still unmatched after all passes
  data/raw/spotify_billboard_matches.csv     — UPDATED: original matches + new ones

Usage
-----
    python src/scraping/spotify_second_pass.py
    python src/scraping/spotify_second_pass.py --unmatched data/raw/spotify_unmatched.csv
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
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_UNMATCHED_CSV   = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_unmatched.csv")
SECOND_MATCHED_CSV      = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_second_pass_matched.csv")
SECOND_UNMATCHED_CSV    = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_second_pass_unmatched.csv")
ORIGINAL_MATCHES_CSV    = os.path.join(PROJECT_ROOT, "data", "raw",       "spotify_billboard_matches.csv")

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

TOKEN_URL  = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
TRACKS_URL = "https://api.spotify.com/v1/tracks"

SEARCH_DELAY   = 0.4
MAX_RETRIES    = 3

# Looser than the original 0.5 — these songs already failed the strict pass
ARTIST_THRESHOLD_NORMAL = 0.3
# For pass C (title-only) we skip artist filtering entirely
TITLE_ONLY_POPULARITY_MIN = 10   # ignore near-zero-popularity results

CHECKPOINT_EVERY = 20

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_credentials() -> tuple[str, str]:
    cid    = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        raise EnvironmentError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set."
        )
    return cid, secret


def get_access_token(client_id: str, client_secret: str) -> str:
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
# HTTP session
# ---------------------------------------------------------------------------

class SpotifySession:
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
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()

        raise requests.HTTPError(f"All {MAX_RETRIES} attempts failed for {url}")


# ---------------------------------------------------------------------------
# Text cleaning — fixes the issues identified in the unmatched file
# ---------------------------------------------------------------------------

# Matches any leading/trailing combination of straight and curly quotes
_OUTER_QUOTES = re.compile(
    r'^[\"\u201c\u2018\u0022\u2019\u201d]+|[\"\u201c\u2018\u0022\u2019\u201d]+$'
)

# Strips "featuring …" / "feat." / "ft." / "with …" and everything after
_FEAT_RE = re.compile(
    r'\s*(?:featuring|feat\.?|ft\.?|with)\s+.*$', flags=re.IGNORECASE
)

# Also strip " and <name>" only when it follows a known-artist pattern
# (used for "Artist A and Artist B" style credits)
_AND_RE = re.compile(r'\s+and\s+.*$', flags=re.IGNORECASE)

# Ampersand collaborations like "Macklemore & Ryan Lewis"
_AMP_RE = re.compile(r'\s*&\s+.*$')


def clean_title(raw: str) -> str:
    """
    Strip triple/double/curly quotes, Wikipedia footnote refs, and
    extra whitespace from a title string.

    This directly fixes failure pattern 1 (stacked quoting).
    """
    t = str(raw).strip()
    # Remove Wikipedia footnote refs like [1], [a], [note 2]
    t = re.sub(r'\[\w[\w\s]*\]', '', t)
    # Strip outer quote characters (handles triple-quoting from CSV)
    t = _OUTER_QUOTES.sub('', t).strip()
    return t.lower()


def clean_artist_full(raw: str) -> str:
    """Return the full artist credit, lowercased and trimmed."""
    return str(raw).strip().lower()


def primary_from_full(full_artist: str) -> str:
    """
    Extract the primary (lead) artist from a full artist credit string.

    Strips: featuring …, feat. …, ft. …, with …, & <name>, and <name>
    This re-derives primary_artist cleanly, fixing failure pattern 2
    (truncated primary_artist from the original cleaning bug).
    """
    a = str(full_artist).strip()
    a = _FEAT_RE.sub('', a)
    a = _AMP_RE.sub('', a)
    # Only strip " and <name>" if there's content before it (avoid "the band")
    # We check for a comma or known separator before "and"
    a = re.sub(r',.*$', '', a)   # strip everything after first comma
    return a.strip().lower()


def first_word_artist(primary: str) -> str:
    """
    Return only the first 'word token' of the primary artist name.

    Useful for very long or unusual artist credits where even primary_artist
    is noisy (failure pattern 3).

    'dj khaled' -> 'dj khaled' (keep 2-word stage names intact)
    'macklemore' -> 'macklemore'
    'mike will made-it' -> 'mike'  (last-resort only)
    """
    words = primary.split()
    # Preserve common two-word stage-name prefixes
    TWO_WORD_PREFIXES = {"dj", "lil", "young", "big", "rich", "mac", "jack",
                         "dr", "mr", "ms", "post", "bad", "pop", "slim"}
    if len(words) >= 2 and words[0].lower() in TWO_WORD_PREFIXES:
        return " ".join(words[:2])
    return words[0] if words else primary


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    STOPWORDS = {"the", "a", "an", "and", "or", "of", "feat", "ft",
                 "featuring", "with"}
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def artist_similarity(expected: str, candidate: str) -> float:
    a = _word_set(expected)
    b = _word_set(candidate)
    if not a:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Spotify search helpers
# ---------------------------------------------------------------------------

def _search(session: SpotifySession, query: str, limit: int = 5) -> list[dict]:
    """Raw search — returns list of track objects or []."""
    try:
        data = session.get(SEARCH_URL, params={"q": query, "type": "track", "limit": limit})
        return data.get("tracks", {}).get("items", []) or []
    except requests.HTTPError as exc:
        log.warning(f"Search error: {exc}")
        return []


def _best_by_artist(
    candidates: list[dict],
    expected_artist: str,
    threshold: float = ARTIST_THRESHOLD_NORMAL,
) -> dict | None:
    """Pick the highest-popularity candidate that clears the artist threshold."""
    scored = []
    for t in candidates:
        spotify_artists = " ".join(a.get("name", "") for a in t.get("artists", []))
        sim = artist_similarity(expected_artist, spotify_artists)
        scored.append((sim, t.get("popularity", 0), t))

    qualified = [(s, p, t) for s, p, t in scored if s >= threshold]
    if not qualified:
        return None
    qualified.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return qualified[0][2]


def _best_by_popularity(candidates: list[dict]) -> dict | None:
    """Pick the most popular candidate with no artist filtering."""
    filtered = [t for t in candidates if t.get("popularity", 0) >= TITLE_ONLY_POPULARITY_MIN]
    if not filtered:
        return None
    return max(filtered, key=lambda t: t.get("popularity", 0))


# ---------------------------------------------------------------------------
# Multi-pass search — the core of this script
# ---------------------------------------------------------------------------

def search_second_pass(
    session: SpotifySession,
    title_raw: str,
    artist_raw: str,
    year: int,
) -> tuple[dict | None, str]:
    """
    Try four progressively looser strategies to find a Spotify match.

    Returns
    -------
    (track_dict_or_None, pass_label)
        pass_label is one of: "A", "B", "C", "D", "none"
    """
    # --- Re-derive clean fields from scratch (fixes issues 1 & 2) ---
    title   = clean_title(title_raw)
    primary = primary_from_full(artist_raw)
    first   = first_word_artist(primary)

    time.sleep(SEARCH_DELAY)

    # Pass A: clean title + re-derived primary artist (no year)
    # Fixes stacked-quote and truncated-artist bugs
    candidates = _search(session, f'track:"{title}" artist:"{primary}"')
    hit = _best_by_artist(candidates, primary)
    if hit:
        return hit, "A"
    time.sleep(SEARCH_DELAY)

    # Pass B: clean title + first-word artist
    # Helps with complex multi-artist credits like "Macklemore & Ryan Lewis"
    if first != primary:
        candidates = _search(session, f'track:"{title}" artist:"{first}"')
        hit = _best_by_artist(candidates, first)
        if hit:
            return hit, "B"
        time.sleep(SEARCH_DELAY)

    # Pass C: title + year, no artist filter — pick most popular result
    # Catches cases where artist name is completely unparseable
    candidates = _search(session, f'track:"{title}" year:{year}', limit=10)
    hit = _best_by_popularity(candidates)
    if hit:
        return hit, "C"
    time.sleep(SEARCH_DELAY)

    # Pass D: bare title only (last resort — most likely to mis-match)
    candidates = _search(session, f'track:"{title}"', limit=10)
    hit = _best_by_popularity(candidates)
    if hit:
        return hit, "D"

    return None, "none"


# ---------------------------------------------------------------------------
# Track metadata (same as spotify_api.py)
# ---------------------------------------------------------------------------

def get_track_metadata(session: SpotifySession, track_id: str) -> dict:
    try:
        track = session.get(f"{TRACKS_URL}/{track_id}")
    except requests.HTTPError as exc:
        log.warning(f"Metadata fetch failed for {track_id}: {exc}")
        return {}

    album   = track.get("album", {})
    artists = track.get("artists", [{}])
    return {
        "spotify_id":              track.get("id"),
        "spotify_title":           track.get("name"),
        "spotify_artist":          ", ".join(a.get("name", "") for a in artists),
        "spotify_album":           album.get("name"),
        "album_type":              album.get("album_type"),
        "release_date":            album.get("release_date"),
        "release_date_precision":  album.get("release_date_precision"),
        "popularity":              track.get("popularity"),
        "explicit":                track.get("explicit"),
        "duration_ms":             track.get("duration_ms"),
        "disc_number":             track.get("disc_number"),
        "track_number":            track.get("track_number"),
        "isrc":                    track.get("external_ids", {}).get("isrc"),
        "spotify_url":             track.get("external_urls", {}).get("spotify"),
        "preview_url":             track.get("preview_url"),
        # audio features will be None — we're getting those from external dataset
        "audio_features_available": False,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_atomic(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as exc:
        log.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(unmatched_csv: str = DEFAULT_UNMATCHED_CSV) -> None:
    """
    Load the unmatched CSV, run four-pass search for each song,
    save newly matched rows, and append them to the main matches file.
    """
    client_id, client_secret = load_credentials()
    session = SpotifySession(client_id, client_secret)

    # --- Load unmatched songs ---
    log.info(f"Loading unmatched songs from: {unmatched_csv}")
    df = pd.read_csv(unmatched_csv)
    total = len(df)
    log.info(f"  {total} songs to retry.\n")

    # --- Resume: skip anything already in second_pass_matched ---
    existing_matched   = _load_existing(SECOND_MATCHED_CSV)
    existing_unmatched = _load_existing(SECOND_UNMATCHED_CSV)

    done_keys: set[tuple] = set()
    for edf in [existing_matched, existing_unmatched]:
        if not edf.empty:
            for _, r in edf.iterrows():
                done_keys.add((str(r.get("year","")), str(r.get("rank",""))))

    matched_rows:   list[dict] = (
        existing_matched.to_dict("records") if not existing_matched.empty else []
    )
    still_unmatched: list[dict] = (
        existing_unmatched.to_dict("records") if not existing_unmatched.empty else []
    )

    pass_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "none": 0}
    newly_processed = 0

    # --- Main loop ---
    for _, row in df.iterrows():
        key = (str(row.get("year", "")), str(row.get("rank", "")))
        if key in done_keys:
            continue

        year       = int(row.get("year", 0))
        title_raw  = str(row.get("title", ""))
        artist_raw = str(row.get("artist", ""))
        title_clean = clean_title(title_raw)
        primary    = primary_from_full(artist_raw)

        log.info(f"  [{year} #{row.get('rank')}] '{title_clean}' — '{primary}'")

        track, pass_label = search_second_pass(session, title_raw, artist_raw, year)
        pass_counts[pass_label] += 1

        row_dict = row.to_dict()

        if track is None:
            log.info(f"    → still unmatched")
            still_unmatched.append(row_dict)
        else:
            metadata = get_track_metadata(session, track.get("id"))
            log.info(
                f"    → matched via pass {pass_label}: "
                f"'{metadata.get('spotify_title')}' by '{metadata.get('spotify_artist')}'"
            )
            matched_rows.append({
                **row_dict,
                **metadata,
                "second_pass_label": pass_label,
            })

        newly_processed += 1
        done_keys.add(key)

        # Checkpoint
        if newly_processed % CHECKPOINT_EVERY == 0:
            _save_atomic(pd.DataFrame(matched_rows),    SECOND_MATCHED_CSV)
            _save_atomic(pd.DataFrame(still_unmatched), SECOND_UNMATCHED_CSV)
            log.info(f"  [checkpoint] {newly_processed}/{total} processed")

    # --- Final save ---
    matched_df   = pd.DataFrame(matched_rows)   if matched_rows   else pd.DataFrame()
    unmatched_df = pd.DataFrame(still_unmatched) if still_unmatched else pd.DataFrame()

    _save_atomic(matched_df,   SECOND_MATCHED_CSV)
    _save_atomic(unmatched_df, SECOND_UNMATCHED_CSV)

    # --- Append newly matched rows to the main matches file ---
    if not matched_df.empty:
        original = _load_existing(ORIGINAL_MATCHES_CSV)
        if not original.empty:
            combined = pd.concat([original, matched_df], ignore_index=True)
        else:
            combined = matched_df
        _save_atomic(combined, ORIGINAL_MATCHES_CSV)
        log.info(f"Appended {len(matched_df)} rows to {ORIGINAL_MATCHES_CSV}")

    # --- Summary ---
    newly_matched = len(matched_rows) - (
        len(existing_matched) if not existing_matched.empty else 0
    )
    print("\n" + "=" * 56)
    print("  SECOND-PASS MATCHING SUMMARY")
    print("=" * 56)
    print(f"  Total songs retried          : {total}")
    print(f"  Newly matched this run       : {newly_matched}")
    print(f"  Still unmatched              : {len(still_unmatched)}")
    print(f"  --- Matched via pass ---")
    print(f"    Pass A (clean title+artist): {pass_counts['A']}")
    print(f"    Pass B (first-word artist) : {pass_counts['B']}")
    print(f"    Pass C (title + year)      : {pass_counts['C']}")
    print(f"    Pass D (title only)        : {pass_counts['D']}")
    print("=" * 56 + "\n")

    if not unmatched_df.empty:
        print("Songs still unmatched after all passes:")
        for _, r in unmatched_df.iterrows():
            print(f"  [{r.get('year')} #{r.get('rank')}] {r.get('title')} — {r.get('artist')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second-pass Spotify matcher for Billboard Boxing")
    p.add_argument(
        "--unmatched",
        default=DEFAULT_UNMATCHED_CSV,
        help="Path to the unmatched CSV from the first pass",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(unmatched_csv=args.unmatched)
