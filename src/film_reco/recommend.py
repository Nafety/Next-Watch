"""Recommendations: the hybrid scorer learns from every rating, then ranks the unrated catalog films."""

import polars as pl

from film_reco.collaborative import collaborative_embeddings
from film_reco.config import Settings
from film_reco.content import content_embeddings
from film_reco.evaluate import load_rated
from film_reco.hybrid import HybridScorer


def recommend(settings: Settings, rated: pl.DataFrame, top: int = 20) -> pl.DataFrame:
    """Best unrated catalog films, with the standardized contribution of each signal."""
    catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
    scorer = HybridScorer(
        catalog, content_embeddings(settings, catalog), collaborative_embeddings(settings, catalog)
    )
    scorer.fit(rated)
    candidates = catalog.filter(~pl.col("tmdb_id").is_in(rated["tmdb_id"].implode()))
    ids = candidates["tmdb_id"].to_numpy()
    parts = scorer.explain(ids)
    return (
        candidates.select("tmdb_id", "title", "year", "directors", "poster_path")
        .with_columns(score=scorer.score(ids), **{name: values for name, values in parts.items()})
        .sort("score", descending=True)
        .head(top)
    )


def print_recommendations(settings: Settings, top: int) -> None:
    watchlist = set(pl.read_parquet(settings.data_dir / "watchlist.parquet")["tmdb_id"].drop_nulls())
    table = recommend(settings, load_rated(settings), top)
    for n, film in enumerate(table.iter_rows(named=True), 1):
        flag = "  (in your watchlist)" if film["tmdb_id"] in watchlist else ""
        directors = ", ".join(film["directors"][:2])
        print(
            f"{n:2}. {film['title']} ({film['year']}), {directors}{flag}\n"
            f"    quality {film['quality']:+.1f} | content {film['content']:+.1f} | "
            f"collaborative {film['collaborative']:+.1f}"
        )
