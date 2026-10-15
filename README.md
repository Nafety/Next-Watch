<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/nextwatch-wordmark.svg">
  <img alt="Next Watch" src="web/public/nextwatch-wordmark-light.svg" height="48">
</picture>

Film recommendations from a Letterboxd export, enriched with TMDB metadata: a research pipeline in Python, and a web app anyone can use, running entirely in the browser.

## How it works

1. **Ingestion**: reads the export ZIP (ratings, diary, favorites, watchlist) and finds each film on TMDB (fuzzy title + year, miniseries left out).
2. **Catalog**: candidate films = the 10,000 most voted films on TMDB (the most the API lets one browse), the same for every user.
3. **Evaluation**: 5-fold cross-validation on the ratings, repeated 10 times. What counts is *liking* a film, not watching it:
   - *taste AUC*: chance that a hidden film rated ≥ 4 scores above one rated ≤ 3;
   - *recall@100*: share of hidden films rated ≥ 4 that reach the top 100 of the whole catalog. Secondary: watched films are chosen, often for being popular or acclaimed, so this metric partly rewards guessing what gets watched.
4. **Content**: each film is embedded from its TMDB description (genres, keywords, director, cast, overview) with `multilingual-e5-small`. Every rated film votes for similar films, weighted by its rating minus the user's mean rating.
5. **Collaborative**: a truncated SVD of the [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) ratings gives each film a 128-dimension vector (close when liked by the same people); same voting as above. MovieLens knows 87 % of the catalog (it stops in 2023).
6. **Hybrid**: weighted sum of the quality, content and collaborative scores, each standardized over the catalog. Three modes: *mix* (equal weights), *taste* (content + collaborative only: close to one's ratings) and *public* (TMDB rating only). Learning the weights with a ridge regression did worse (AUC 0.68): ~160 ratings per fold are too few to fit them reliably.
7. **Web app** (`web/`, React + TypeScript, on a Cloudflare Worker): the same scoring, ported to TypeScript and checked against Python by a parity test, runs **in the visitor's browser**, so ratings never leave it. The catalog's vectors are shipped as int8 (8.7 MB; the evaluation does not change). The Worker serves the static site and relays TMDB requests (`web/worker/`), keeping the API key secret. Rated films outside the catalog get their content vector in the browser (Transformers.js, same e5 model; cosine 0.995 with Python's), loaded from a CDN only when needed: its 25 MB WebAssembly runtime is over Cloudflare's 25 MiB file size limit. Multilingual e5 was kept over lighter English models (bge-small, MiniLM: content AUC 0.63 vs 0.71). Visitors pick a mode, filter by decade and genre, page through the ranking, and heart films: favourites are kept in the browser per Letterboxd account and export as a CSV that Letterboxd imports into the watchlist in one go. The last analysis is kept too, so a reload goes straight back to the recommendations.

## Results

On one user's 199 rated films:

| Scorer | Taste AUC | Recall@100 |
|---|---:|---:|
| Random | 0.50 | 1 % |
| Popularity (TMDB) | 0.57 | 22 % |
| Quality (TMDB rating, Bayesian average) | 0.70 | 37 % |
| Content (embedding similarity) | 0.71 | 2 % |
| Collaborative (MovieLens SVD) | 0.68 | 8 % |
| Hybrid, *taste* mode | 0.73 | 9 % |
| **Hybrid, *mix* mode** | **0.74** | 26 % |

## Usage

```bash
uv sync
cp .env.example .env               # add your TMDB API key
# put your Letterboxd export (letterboxd-*.zip) in data/
# and https://files.grouplens.org/datasets/movielens/ml-32m.zip in data/movielens/
uv run film-reco ingest            # -> data/films.parquet, ratings.parquet, watchlist.parquet
uv run film-reco catalog           # -> data/catalog.parquet
uv run film-reco evaluate          # results table
uv run film-reco recommend         # top 20 unrated films, with the share of each signal
uv run film-reco export-app        # -> web/public/data/, the web app's data (committed, for deployment)

cd web && npm install
npm test                           # TypeScript scoring = Python scoring
npm run preview:worker             # site + TMDB relay on localhost:8787, as deployed (TMDB_API_KEY in web/.dev.vars)
```

**Deploy the web app** as a Cloudflare Worker (`web/wrangler.jsonc`): import the repository in Workers & Pages, root directory `web`, build command `npm run build`, deploy command `npx wrangler deploy`, and add the secret `TMDB_API_KEY` in the Worker's settings (Node 22 is pinned in `web/.node-version`).

Development: `uv run pytest`, `uv run ruff check`, `uv run mypy src`.

This product uses the TMDB API but is not endorsed or certified by TMDB.
