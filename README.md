# film-reco

Film recommendations from a Letterboxd export, enriched with TMDB metadata.

## How it works

1. **Ingestion**: reads the export ZIP (ratings, diary, favorites, watchlist) and finds each film on TMDB (fuzzy title + year, miniseries left out).
2. **Catalog**: candidate films = TMDB recommendations of every rated film + the 1,000 most voted films (~3,700 films).
3. **Evaluation**: 5-fold cross-validation on the ratings, repeated 10 times. What counts is *liking* a film, not watching it:
   - *taste AUC*: chance that a hidden film rated ≥ 4 scores above one rated ≤ 3;
   - *recall@100*: share of hidden films rated ≥ 4 that reach the top 100 of the whole catalog. Secondary: watched films are chosen, often for being popular or acclaimed, so this metric partly rewards guessing what gets watched.
4. **Content**: each film is embedded from its TMDB description (genres, keywords, director, cast, overview) with `multilingual-e5-small`. Every rated film votes for similar films, weighted by its rating minus the user's mean rating.
5. **Collaborative**: a truncated SVD of the [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) ratings gives each film a 128-dimension vector (close when liked by the same people); same voting as above. MovieLens knows 76 % of the catalog (it stops in 2023).
6. **Hybrid**: equal-weight sum of the quality, content and collaborative scores, each standardized over the catalog. Learning the weights with a ridge regression did worse (AUC 0.68): ~160 ratings per fold are too few to fit them reliably.

## Results

| Scorer | Taste AUC | Recall@100 |
|---|---:|---:|
| Random | 0.51 | 5 % |
| Popularity (TMDB) | 0.57 | 29 % |
| Quality (TMDB rating, Bayesian average) | 0.71 | 40 % |
| Content (embedding similarity) | 0.71 | 4 % |
| Collaborative (MovieLens SVD) | 0.68 | 17 % |
| **Hybrid** | **0.74** | 36 % |

## Usage

```bash
uv sync
cp .env.example .env               # add your TMDB API key
# put your Letterboxd export (letterboxd-*.zip) in data/
uv run film-reco ingest            # -> data/films.parquet, ratings.parquet, watchlist.parquet
uv run film-reco catalog           # -> data/catalog.parquet
# download https://files.grouplens.org/datasets/movielens/ml-32m.zip into data/movielens/
uv run film-reco evaluate          # results table
uv run film-reco recommend         # top 20 unrated films, with the share of each signal
```

Development: `uv run pytest`, `uv run ruff check`, `uv run mypy src`.

This product uses the TMDB API but is not endorsed or certified by TMDB.
