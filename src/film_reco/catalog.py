"""Step 2: the candidate films to recommend from.

The 10,000 most voted films on TMDB (the most the API lets one browse), plus the user's own films.
The same catalog is shipped with the app, so it does not depend on any one user.
Output: data/catalog.parquet, same columns as films.parquet.
"""

from concurrent.futures import ThreadPoolExecutor

import polars as pl

from film_reco.config import Settings
from film_reco.tmdb import TMDBClient

MOST_VOTED_PAGES = 500  # 20 films per page


def build_catalog(settings: Settings) -> None:
    own = pl.read_parquet(settings.data_dir / "films.parquet")
    # Requests are independent: a few threads make the first (uncached) run several times faster
    with (
        TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.cache_dir / "tmdb") as tmdb,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        popular = {r.id for page in pool.map(tmdb.most_voted, range(1, MOST_VOTED_PAGES + 1)) for r in page}
        ids = sorted(popular - set(own["tmdb_id"]))
        rows = [movie.to_row() for movie in pool.map(tmdb.movie, ids)]

    catalog = pl.concat([own, pl.DataFrame(rows, schema=own.schema)])
    catalog.write_parquet(settings.data_dir / "catalog.parquet")
    print(f"Catalog: {len(catalog)} films ({len(own)} rated or in the watchlist, {len(ids)} others)")
