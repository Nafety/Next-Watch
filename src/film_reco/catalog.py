"""Step 2: the candidate films to recommend from.

TMDB recommendations for each rated film, plus the most voted films on TMDB, plus the user's own films.
Output: data/catalog.parquet, same columns as films.parquet.
"""

from concurrent.futures import ThreadPoolExecutor

import polars as pl

from film_reco.config import Settings
from film_reco.tmdb import TMDBClient

MOST_VOTED_PAGES = 50  # 1,000 films


def build_catalog(settings: Settings) -> None:
    own = pl.read_parquet(settings.data_dir / "films.parquet")
    # Requests are independent: a few threads make the first (uncached) run several times faster
    with (
        TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.data_dir / "cache" / "tmdb") as tmdb,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        neighbours = [r.id for page in pool.map(tmdb.recommendations, own["tmdb_id"]) for r in page]
        popular = [r.id for page in pool.map(tmdb.most_voted, range(1, MOST_VOTED_PAGES + 1)) for r in page]
        ids = sorted((set(neighbours) | set(popular)) - set(own["tmdb_id"]))
        rows = [movie.to_row() for movie in pool.map(tmdb.movie, ids)]

    catalog = pl.concat([own, pl.DataFrame(rows, schema=own.schema)])
    catalog.write_parquet(settings.data_dir / "catalog.parquet")
    print(
        f"Catalog: {len(catalog)} films ({len(own)} rated or in the watchlist, "
        f"{len(set(neighbours) - set(own['tmdb_id']))} from TMDB recommendations, "
        f"{len(set(popular) - set(own['tmdb_id']))} among the most voted)"
    )
