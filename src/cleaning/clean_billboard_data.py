"""
clean_billboard_data.py

Loads the raw Billboard Year-End Hot 100 CSV produced by wiki_scraper.py,
applies a cleaning pipeline, and saves the result to data/processed/.

Cleaning steps
--------------
1. Standardise column names (already clean from scraper, but enforced here).
2. Strip surrounding quotation marks from title strings.
3. Strip extra whitespace from all text columns.
4. Remove obvious punctuation noise (repeated punctuation, stray symbols).
5. Create helper columns (title_clean, artist_clean, primary_artist) that are
   lowercased and have the "featuring ..." portion removed — useful for
   fuzzy-matching against Spotify later.
6. Drop exact duplicate rows.
7. Save to data/processed/billboard_clean.csv.

Original columns are preserved; helper columns are additive.

Usage
-----
    python src/cleaning/clean_billboard_data.py
"""

import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "billboard_year_end_hot_100_2010_present.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "billboard_clean.csv")

# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and underscore-join all column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def strip_outer_quotes(text: str) -> str:
    """
    Remove surrounding straight or curly quotation marks from a string.

    Wikipedia wraps song titles in typographic quotes like "Tik Tok",
    which read_html preserves. This removes them.

    Examples
    --------
    '"Tik Tok"'   -> 'Tik Tok'
    '\u201cBlinding Lights\u201d' -> 'Blinding Lights'
    """
    # Match opening quote at start and closing quote at end
    # Covers: "..." (straight), \u201c...\u201d (curly), '...' (single)
    return re.sub(r'^[\"\u201c\u2018\u0022]+|[\"\u201d\u2019\u0022]+$', '', text).strip()


def remove_punctuation_noise(text: str) -> str:
    """
    Remove stray/repeated punctuation that adds no meaning.

    Keeps apostrophes (contractions), hyphens, commas, and periods that are
    part of normal text. Targets:
    - Runs of the same punctuation character (e.g. '...' -> '')
    - Wikipedia footnote markers like [1], [note 1]
    - Stray brackets or pipe characters
    """
    # Remove Wikipedia footnote refs: [1], [a], [note 2], etc.
    text = re.sub(r'\[\w[\w\s]*\]', '', text)
    # Remove stray pipe characters (table artifacts)
    text = re.sub(r'\|', '', text)
    # Collapse runs of 2+ identical punctuation chars (except letters/digits/space)
    text = re.sub(r'([^\w\s])\1+', r'\1', text)
    return text.strip()


def make_clean_title(title: str) -> str:
    """
    Return a normalised, lowercase version of a song title for matching.

    Steps: strip outer quotes → remove noise → lowercase → trim.
    """
    t = strip_outer_quotes(str(title))
    t = remove_punctuation_noise(t)
    return t.lower().strip()


def make_clean_artist(artist: str) -> str:
    """
    Return a normalised, lowercase version of the full artist credit
    (including any featured artists) for matching.

    Steps: remove noise → lowercase → trim.
    """
    a = remove_punctuation_noise(str(artist))
    return a.lower().strip()


# Regex that matches common featuring separators in the artist field.
# Covers: "featuring", "feat.", "ft.", "with", "&" — case-insensitive.
_FEAT_PATTERN = re.compile(
    r'\s*(?:featuring|feat\.?|ft\.?|with|&)\s+.*$',
    flags=re.IGNORECASE,
)


def extract_primary_artist(artist_clean: str) -> str:
    """
    Strip the featured-artist portion from an already-lowercased artist string
    and return only the primary (lead) artist credit.

    Examples
    --------
    'katy perry featuring snoop dogg'  -> 'katy perry'
    'eminem featuring rihanna'         -> 'eminem'
    'the weeknd'                       -> 'the weeknd'
    """
    return _FEAT_PATTERN.sub('', artist_clean).strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning pipeline and return the cleaned DataFrame.

    New columns added
    -----------------
    title_clean     : stripped, lowercased title (no outer quotes, no noise)
    artist_clean    : stripped, lowercased full artist credit
    primary_artist  : lead artist only (featuring portion removed)
    """
    # 1. Standardise column names
    df = standardise_columns(df)

    # 2. Enforce expected dtypes
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["title"] = df["title"].astype(str)
    df["artist"] = df["artist"].astype(str)

    # 3. Strip whitespace from original text columns (keeps originals intact)
    df["title"] = df["title"].str.strip()
    df["artist"] = df["artist"].str.strip()

    # 4. Build cleaned / normalised helper columns
    df["title_clean"] = df["title"].apply(make_clean_title)
    df["artist_clean"] = df["artist"].apply(make_clean_artist)
    df["primary_artist"] = df["artist_clean"].apply(extract_primary_artist)

    # 5. Drop rows where title or artist ended up empty after cleaning
    before = len(df)
    df = df[df["title_clean"].str.len() > 0]
    df = df[df["artist_clean"].str.len() > 0]
    dropped_empty = before - len(df)
    if dropped_empty:
        print(f"  Dropped {dropped_empty} rows with empty title/artist after cleaning.")

    # 6. Remove exact duplicate rows (same year + rank + title_clean + artist_clean)
    before = len(df)
    df = df.drop_duplicates(subset=["year", "rank", "title_clean", "artist_clean"])
    dropped_dupes = before - len(df)
    if dropped_dupes:
        print(f"  Dropped {dropped_dupes} exact duplicate rows.")

    # 7. Sort for readability
    df = df.sort_values(["year", "rank"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading raw data from:\n  {RAW_PATH}\n")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df_raw.shape}")

    print("\nCleaning...")
    df_clean = clean(df_raw)

    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"Years covered: {sorted(df_clean['year'].dropna().unique().tolist())}")
    print(f"\nSample output:")
    print(df_clean[["year", "rank", "title", "title_clean", "artist", "primary_artist"]].head(10).to_string(index=False))

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
