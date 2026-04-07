"""
wiki_scraper.py

Scrapes Billboard Year-End Hot 100 chart tables from Wikipedia for every
year from 2010 through the current year, combines them into one DataFrame,
and saves the result to data/raw/billboard_year_end_hot_100_2010_present.csv.

Strategy
--------
1. Try pandas.read_html first (fast, no extra parsing needed).
2. If read_html can't find a usable table, fall back to BeautifulSoup to
   locate the <table> manually and hand the HTML back to pd.read_html.

Usage
-----
    python src/scraping/wiki_scraper.py
"""

import os
import datetime
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root is two levels above this file (src/scraping → project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_FILENAME = "billboard_year_end_hot_100_2010_present.csv"

START_YEAR = 2010
END_YEAR = datetime.date.today().year  # scrape up to the current calendar year

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY = 1.0

# Wikipedia URL pattern for Billboard Year-End Hot 100
WIKI_URL_TEMPLATE = (
    "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
)

# Browser-like headers so Wikipedia doesn't reject the request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; billboard-boxing-scraper/1.0; "
        "+https://github.com/example/billboard-boxing)"
    )
}

# ---------------------------------------------------------------------------
# Column-name normalisation
# ---------------------------------------------------------------------------

# Maps the many column-name variants found across Wikipedia years → canonical name.
# Keys are lowercase fragments that appear in raw column names.
COLUMN_MAP = {
    # rank
    "no.": "rank",
    "no":  "rank",
    "#":   "rank",
    "rank": "rank",
    "pos": "rank",
    # title / song
    "title": "title",
    "song":  "title",
    "single": "title",
    # artist
    "artist": "artist",
    "act":    "artist",
    "performer": "artist",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw Wikipedia column names to canonical names
    (rank, title, artist) using COLUMN_MAP.

    Unrecognised columns are dropped so every year ends up with
    exactly the same schema.
    """
    rename = {}
    for raw_col in df.columns:
        key = str(raw_col).strip().lower()
        for fragment, canonical in COLUMN_MAP.items():
            if fragment in key:
                rename[raw_col] = canonical
                break

    df = df.rename(columns=rename)

    # Keep only the columns we care about (some years have extras)
    keep = [c for c in ("rank", "title", "artist") if c in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# URL generation
# ---------------------------------------------------------------------------

def build_url(year: int) -> str:
    """Return the Wikipedia URL for the given year's Billboard Year-End Hot 100."""
    return WIKI_URL_TEMPLATE.format(year=year)


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def _pick_table_from_list(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """
    Given a list of DataFrames returned by pd.read_html, pick the one that
    looks most like a Billboard chart:
      - at least 50 rows (a full Hot 100 has exactly 100)
      - columns that mention rank / title / artist

    Falls back to the largest table if none pass the heuristic.
    """
    for tbl in tables:
        if len(tbl) < 50:
            continue
        cols_str = " ".join(str(c).lower() for c in tbl.columns)
        if any(kw in cols_str for kw in ("title", "song", "single", "artist", "no.", "rank")):
            return tbl

    # Fallback: return the largest table on the page
    return max(tables, key=len) if tables else None


def _fetch_with_bs4(url: str) -> pd.DataFrame | None:
    """
    Fallback path: download the raw HTML, use BeautifulSoup to locate the
    main wikitable, then hand just that <table> tag back to pd.read_html.

    Returns None if no suitable table is found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("    [bs4 fallback] beautifulsoup4 is not installed — skipping.")
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [bs4 fallback] HTTP error: {exc}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Wikipedia chart tables always carry the 'wikitable' CSS class
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        print("    [bs4 fallback] No wikitable found on page.")
        return None

    # Pick the largest wikitable (most rows)
    best = max(tables, key=lambda t: len(t.find_all("tr")))
    dfs = pd.read_html(str(best))
    return dfs[0] if dfs else None


def extract_table(year: int, url: str) -> pd.DataFrame | None:
    """
    Download the Wikipedia page for *year* and return the raw Hot 100 table
    as a DataFrame, or None on failure.

    Tries pd.read_html first; falls back to BeautifulSoup if needed.
    """
    print(f"  [{year}] Fetching {url}")

    # --- Primary path: pandas.read_html ---
    try:
        tables = pd.read_html(url, attrs={"class": "wikitable"})
    except Exception:
        # read_html with attrs may fail on some years; retry without the filter
        try:
            tables = pd.read_html(url)
        except Exception as exc:
            print(f"    [read_html] Failed: {exc}")
            tables = []

    if tables:
        df = _pick_table_from_list(tables)
        if df is not None and len(df) >= 50:
            print(f"    [read_html] OK — {len(df)} rows")
            return df

    # --- Fallback path: BeautifulSoup ---
    print("    [read_html] No suitable table — trying BeautifulSoup fallback...")
    df = _fetch_with_bs4(url)
    if df is not None:
        print(f"    [bs4] OK — {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Per-year scraping
# ---------------------------------------------------------------------------

def scrape_year(year: int) -> pd.DataFrame | None:
    """
    Scrape, clean, and return a normalised DataFrame for one chart year.

    Columns returned: year, rank, title, artist
    Returns None if the page or table could not be retrieved.
    """
    url = build_url(year)
    raw_df = extract_table(year, url)

    if raw_df is None:
        print(f"    WARNING: could not retrieve table for {year} — skipping.")
        return None

    df = _normalise_columns(raw_df)

    # Validate that we got the columns we need
    missing = [c for c in ("rank", "title", "artist") if c not in df.columns]
    if missing:
        print(f"    WARNING: missing columns {missing} for {year} — skipping.")
        return None

    # Clean rank: coerce to int, drop any non-numeric rows (header repeats, etc.)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"])
    df["rank"] = df["rank"].astype(int)

    # Strip extra whitespace from text columns
    df["title"] = df["title"].astype(str).str.strip()
    df["artist"] = df["artist"].astype(str).str.strip()

    # Prepend year so it's the first column
    df.insert(0, "year", year)
    df = df.reset_index(drop=True)

    print(f"    DONE — {len(df)} songs kept")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def scrape_all_years(
    start: int = START_YEAR,
    end: int = END_YEAR,
    delay: float = REQUEST_DELAY,
) -> pd.DataFrame:
    """
    Scrape every year from *start* to *end* (inclusive), combine into one
    DataFrame, and return it.

    Parameters
    ----------
    start, end : int
        Inclusive year range.
    delay : float
        Seconds to sleep between requests (be polite to Wikipedia).
    """
    years = list(range(start, end + 1))
    print(f"Billboard Year-End Hot 100 scraper — {start} to {end} ({len(years)} years)\n")

    frames = []
    for i, year in enumerate(years):
        df = scrape_year(year)
        if df is not None:
            frames.append(df)

        # Don't sleep after the last request
        if i < len(years) - 1:
            time.sleep(delay)
        print()

    if not frames:
        raise RuntimeError("No data was collected — check your internet connection or the URLs.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined)} total rows across {len(frames)} year(s).")
    return combined


def save_combined(df: pd.DataFrame, out_dir: str = RAW_DATA_DIR) -> str:
    """
    Save the combined DataFrame to data/raw/billboard_year_end_hot_100_2010_present.csv.

    Returns the absolute path of the written file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, OUTPUT_FILENAME)
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    combined_df = scrape_all_years()
    save_combined(combined_df)
