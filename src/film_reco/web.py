"""Web app logic, independent of the interface: score the shipped catalog for one user's ratings."""

import io
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from film_reco.app_data import CATALOG_COLUMNS
from film_reco.hybrid import HybridScorer
from film_reco.ingest import link_films
from film_reco.letterboxd import read_ratings, read_watchlist
from film_reco.tmdb import TMDBClient

Embed = Callable[[list[dict[str, object]]], np.ndarray]


class AppModel:
    def __init__(self, folder: Path) -> None:
        self.catalog = pl.read_parquet(folder / "catalog.parquet")
        vectors = np.load(folder / "vectors.npz")
        self.content = vectors["content"].astype(np.float32)
        self.collaborative = vectors["collaborative"].astype(np.float32)
        movielens = np.load(folder / "movielens.npz")
        self._movielens_row = {int(i): n for n, i in enumerate(movielens["ids"])}
        self._movielens = movielens["vectors"].astype(np.float32)
        self.known = set(self.catalog["tmdb_id"].to_list())

    def _movielens_vector(self, tmdb_id: int) -> np.ndarray:
        row = self._movielens_row.get(tmdb_id)
        return self._movielens[row] if row is not None else np.zeros(self._movielens.shape[1], np.float32)

    def score(
        self,
        rated: pl.DataFrame,
        extra_films: list[dict[str, object]],
        embed: Embed,
    ) -> pl.DataFrame:
        """Every catalog film the user has not rated, best first, with the share of each signal.

        rated: tmdb_id, rating. extra_films: TMDB rows (Movie.to_row) of rated films outside the catalog;
        their vectors are computed here (embed: list of films -> content vectors).
        """
        catalog, content, collaborative = self.catalog, self.content, self.collaborative
        if extra_films:
            extra = pl.DataFrame(extra_films).select(CATALOG_COLUMNS).cast(catalog.schema)
            catalog = pl.concat([catalog, extra])
            content = np.vstack([content, embed(extra_films)])
            extra_vectors = np.array([self._movielens_vector(i) for i in extra["tmdb_id"]])
            collaborative = np.vstack([collaborative, extra_vectors])

        scorer = HybridScorer(catalog, content, collaborative)
        scorer.fit(rated)
        candidates = self.catalog.filter(~pl.col("tmdb_id").is_in(rated["tmdb_id"].implode()))
        ids = candidates["tmdb_id"].to_numpy()
        return candidates.with_columns(score=scorer.score(ids), **scorer.explain(ids)).sort(
            "score", descending=True
        )


def analyse_export(
    export: bytes, model: AppModel, tmdb: TMDBClient, embed: Embed
) -> tuple[pl.DataFrame, dict[str, int], set[int]]:
    """One Letterboxd export -> scored catalog, matching counts, and the watchlist's TMDB ids."""
    ratings, watchlist = read_ratings(io.BytesIO(export)), read_watchlist(io.BytesIO(export))
    link = link_films(tmdb, pl.concat([ratings.select("letterboxd_uri", "title", "year"), watchlist]))
    rated = (
        ratings.join(link, on="letterboxd_uri")
        .filter(pl.col("tmdb_id").is_not_null())
        .unique("tmdb_id", maintain_order=True)
        .select("tmdb_id", "rating")
    )
    outside = [int(i) for i in rated["tmdb_id"] if int(i) not in model.known]
    with ThreadPoolExecutor(max_workers=8) as pool:
        extra = [movie.to_row() for movie in pool.map(tmdb.movie, outside)]

    counts = {
        "rated": len(ratings),
        "found": len(rated),
        "miniseries": int(ratings.join(link, on="letterboxd_uri")["series"].sum()),
    }
    in_watchlist = set(watchlist.join(link, on="letterboxd_uri")["tmdb_id"].drop_nulls().to_list())
    scored = model.score(rated, extra, embed) if len(rated) else model.catalog.head(0)
    return scored, counts, in_watchlist
