# film-reco

Film recommendations from a Letterboxd export, enriched with TMDB metadata.

## How it works

1. **Ingestion**: reads the export ZIP (ratings, diary, favorites, watchlist) and finds each film on TMDB (fuzzy title + year, miniseries left out).
2. **Catalog**: candidate films = TMDB recommendations of every rated film + the 1,000 most voted films (~3,700 films).
3. **Evaluation**: 5-fold cross-validation on the ratings, repeated 10 times. What counts is *liking* a film, not watching it:
   - *taste AUC*: chance that a hidden film rated ≥ 4 scores above one rated ≤ 3;
   - *recall@100*: share of hidden films rated ≥ 4 that reach the top 100 of the whole catalog.

## Results

| Scorer | Taste AUC | Recall@100 |
|---|---:|---:|
| Random | 0.51 | 5 % |
| Popularity (TMDB) | 0.57 | 30 % |
| Quality (TMDB rating, Bayesian average) | 0.71 | 40 % |

## Usage

```bash
uv sync
cp .env.example .env               # add your TMDB API key
# put your Letterboxd export (letterboxd-*.zip) in data/
uv run film-reco ingest            # -> data/films.parquet, ratings.parquet, watchlist.parquet
uv run film-reco catalog           # -> data/catalog.parquet
uv run film-reco evaluate          # results table
```

Development: `uv run pytest`, `uv run ruff check`, `uv run mypy src`.

This product uses the TMDB API but is not endorsed or certified by TMDB.
