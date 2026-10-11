"""Step 1: Letterboxd export -> TMDB ids and metadata -> Parquet files in data/.

Outputs: films.parquet (one row per TMDB film), ratings.parquet and watchlist.parquet (with tmdb_id).
Miniseries, which Letterboxd lists among films, get no tmdb_id.
"""

from concurrent.futures import ThreadPoolExecutor

import polars as pl

from film_reco.config import Settings
from film_reco.letterboxd import find_export, read_ratings, read_watchlist
from film_reco.matching import best_match
from film_reco.tmdb import TMDBClient

LINK_SCHEMA = {
    "letterboxd_uri": pl.String,
    "tmdb_id": pl.Int64,
    "match_score": pl.Float64,
    "series": pl.Boolean,
}


def match_film(tmdb: TMDBClient, uri: str, title: str, year: int | None) -> dict[str, object]:
    """TMDB id of one Letterboxd film, or none (not found, or a miniseries)."""
    # The year filter is strict (release dates differ between countries): retry without it
    match = best_match(title, year, tmdb.search_movie(title, year)) or best_match(
        title, year, tmdb.search_movie(title)
    )
    series = False
    # No exact film: it may be a miniseries ("Chernobyl", 2019)
    if match is None or match[1] < 100:
        show = best_match(title, year, tmdb.search_tv(title, year))
        if show and show[1] == 100:
            series, match = True, None
    return {
        "letterboxd_uri": uri,
        "tmdb_id": match[0].id if match else None,
        "match_score": match[1] if match else None,
        "series": series,
    }


def link_films(tmdb: TMDBClient, films: pl.DataFrame, workers: int = 8) -> pl.DataFrame:
    """films: letterboxd_uri, title, year -> one row per film with its tmdb_id (searches run in parallel)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda film: match_film(tmdb, *film), films.iter_rows()))
    return pl.DataFrame(rows, schema=LINK_SCHEMA)


def ingest(settings: Settings) -> None:
    export = find_export(settings.data_dir)
    ratings, watchlist = read_ratings(export), read_watchlist(export)
    films = pl.concat([ratings.select("letterboxd_uri", "title", "year"), watchlist]).unique("letterboxd_uri")
    print(f"{export.name}: {len(ratings)} rated films, {len(watchlist)} in the watchlist")

    with TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.cache_dir / "tmdb") as tmdb:
        link = link_films(tmdb, films)
        ids = link["tmdb_id"].drop_nulls().unique().sort()
        with ThreadPoolExecutor(max_workers=8) as pool:
            catalog = pl.DataFrame([movie.to_row() for movie in pool.map(tmdb.movie, ids)])

    catalog.write_parquet(settings.data_dir / "films.parquet")
    columns = ["letterboxd_uri", "tmdb_id", "match_score"]
    ratings.join(link.select(columns), on="letterboxd_uri", how="left").write_parquet(
        settings.data_dir / "ratings.parquet"
    )
    watchlist.join(link.select(columns), on="letterboxd_uri", how="left").write_parquet(
        settings.data_dir / "watchlist.parquet"
    )

    report = films.join(link, on="letterboxd_uri")
    found = report.filter(pl.col("tmdb_id").is_not_null())
    print(f"Matched films: {len(found)}/{len(report)}, miniseries left out: {report['series'].sum()}")
    for title, year, series in (
        report.filter(pl.col("tmdb_id").is_null()).select("title", "year", "series").iter_rows()
    ):
        print(f"  {'miniseries' if series else 'not found'}: {title} ({year})")
    doubtful = found.filter(pl.col("match_score") < 100).join(
        catalog.select("tmdb_id", pl.col("title").alias("tmdb_title"), pl.col("year").alias("tmdb_year")),
        on="tmdb_id",
    )
    for title, year, tmdb_title, tmdb_year in doubtful.select(
        "title", "year", "tmdb_title", "tmdb_year"
    ).iter_rows():
        print(f"  to check: {title} ({year}) -> {tmdb_title} ({tmdb_year})")
