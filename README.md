# Billboard Boxing

A data science project that analyses Billboard Hot 100 Year-End charts.

## Project layout

```
billboard-boxing/
├── data/
│   ├── raw/          # CSVs scraped straight from Wikipedia (do not edit)
│   └── processed/    # Cleaned / merged data ready for analysis
├── notebooks/        # Exploratory Jupyter notebooks
├── outputs/
│   └── figures/      # Saved plots and visualisations
├── src/
│   ├── scraping/
│   │   └── wiki_scraper.py   # Wikipedia → data/raw
│   ├── cleaning/             # (coming soon) raw → processed
│   └── modeling/             # (coming soon) analysis & modelling
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape Billboard Year-End Hot 100 tables from Wikipedia
python src/scraping/wiki_scraper.py
```

Raw CSVs for each year land in `data/raw/hot100_<year>.csv`.

## Data sources

| Source | What | Status |
|--------|------|--------|
| Wikipedia | Billboard Year-End Hot 100 (2015-2023) | Scraping ready |
| Spotify API | Audio features, popularity | Coming soon |

## Roadmap

- [x] Wikipedia scraper
- [ ] Data cleaning pipeline (`src/cleaning/`)
- [ ] Spotify API enrichment
- [ ] Exploratory analysis notebooks
- [ ] Modelling / visualisations
