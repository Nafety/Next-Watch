"""Step 1: Letterboxd export -> TMDB ids and metadata -> Parquet files in data/.

Outputs: films.parquet (one row per TMDB film), ratings.parquet and watchlist.parquet (with tmdb_id).
Miniseries, which Letterboxd lists among films, get no tmdb_id.
"""

import polars as pl

from film_reco.config import Settings
from film_reco.letterboxd import find_export, read_ratings, read_watchlist
from film_reco.matching import best_match
from film_reco.tmdb import TMDBClient


def ingest(settings: Settings) -> None:
    export = find_export(settings.data_dir)
    ratings, watchlist = read_ratings(export), read_watchlist(export)
    films = pl.concat([ratings.select("letterboxd_uri", "title", "year"), watchlist]).unique("letterboxd_uri")
    print(f"{export.name}: {len(ratings)} rated films, {len(watchlist)} in the watchlist")

    links: list[dict[str, object]] = []
    details: dict[int, dict[str, object]] = {}
    series: list[str] = []
    with TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.data_dir / "cache" / "tmdb") as tmdb:
        for uri, title, year in films.iter_rows():
            # The year filter is strict (release dates differ between countries): retry without it
            match = best_match(title, year, tmdb.search_movie(title, year)) or best_match(
                title, year, tmdb.search_movie(title)
            )
            # No exact film: it may be a miniseries ("Chernobyl", 2019)
            if match is None or match[1] < 100:
                show = best_match(title, year, tmdb.search_tv(title, year))
                if show and show[1] == 100:
                    series.append(f"{title} ({year})")
                    match = None
            links.append(
                {
                    "letterboxd_uri": uri,
                    "tmdb_id": match[0].id if match else None,
                    "match_score": match[1] if match else None,
                }
            )
            if match and match[0].id not in details:
                details[match[0].id] = tmdb.movie(match[0].id).to_row()

    link = pl.DataFrame(
        links, schema={"letterboxd_uri": pl.String, "tmdb_id": pl.Int64, "match_score": pl.Float64}
    )
    catalog = pl.DataFrame(list(details.values()))
    catalog.write_parquet(settings.data_dir / "films.parquet")
    ratings.join(link, on="letterboxd_uri", how="left").write_parquet(settings.data_dir / "ratings.parquet")
    watchlist.join(link, on="letterboxd_uri", how="left").write_parquet(
        settings.data_dir / "watchlist.parquet"
    )

    report = films.join(link, on="letterboxd_uri")
    found = report.filter(pl.col("tmdb_id").is_not_null())
    print(f"Matched films: {len(found)}/{len(report)}, miniseries left out: {len(series)}")
    for name in series:
        print(f"  miniseries: {name}")
    for title, year in report.filter(pl.col("tmdb_id").is_null()).select("title", "year").iter_rows():
        if f"{title} ({year})" not in series:
            print(f"  not found: {title} ({year})")
    doubtful = found.filter(pl.col("match_score") < 100).join(
        catalog.select("tmdb_id", pl.col("title").alias("tmdb_title"), pl.col("year").alias("tmdb_year")),
        on="tmdb_id",
    )
    for title, year, tmdb_title, tmdb_year in doubtful.select(
        "title", "year", "tmdb_title", "tmdb_year"
    ).iter_rows():
        print(f"  to check: {title} ({year}) -> {tmdb_title} ({tmdb_year})")
