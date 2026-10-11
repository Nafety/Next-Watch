"""Files shipped with the web app (data/app/), computed once from the catalog and MovieLens.

catalog.parquet: per film, only what scoring and filters need (no TMDB texts).
vectors.npz: content and collaborative vectors, in the catalog's order.
movielens.npz: collaborative vectors of every film MovieLens knows, for users' films outside the catalog.
"""

import numpy as np
import polars as pl

from film_reco.collaborative import collaborative_embeddings, movielens_vectors
from film_reco.config import Settings
from film_reco.content import content_embeddings

CATALOG_COLUMNS = ["tmdb_id", "year", "genres", "vote_average", "vote_count", "popularity"]


def export_app_data(settings: Settings) -> None:
    catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
    out = settings.data_dir / "app"
    out.mkdir(exist_ok=True)
    catalog.select(CATALOG_COLUMNS).write_parquet(out / "catalog.parquet")
    np.savez_compressed(
        out / "vectors.npz",
        ids=catalog["tmdb_id"].to_numpy(),
        content=content_embeddings(settings, catalog).astype(np.float16),
        collaborative=collaborative_embeddings(settings, catalog).astype(np.float16),
    )
    ids, vectors = movielens_vectors(settings)
    np.savez_compressed(out / "movielens.npz", ids=ids, vectors=vectors.astype(np.float16))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"App data: {len(catalog)} catalog films, {len(ids)} MovieLens films, {size:.0f} MB in {out}")
