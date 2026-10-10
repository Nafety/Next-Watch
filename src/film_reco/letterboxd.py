"""Reading a Letterboxd data export (the ZIP from Settings > Data > Export your data)."""

import csv
import io
import zipfile
from pathlib import Path

import polars as pl


def find_export(data_dir: Path) -> Path:
    """Most recent export in the data folder."""
    exports = sorted(data_dir.glob("letterboxd-*.zip"))
    if not exports:
        raise FileNotFoundError(f"No Letterboxd export (letterboxd-*.zip) in {data_dir}")
    return exports[-1]


def _read_csv(export: zipfile.ZipFile, name: str) -> pl.DataFrame:
    return pl.read_csv(export.read(name), infer_schema=False)


def _favorite_uris(export: zipfile.ZipFile) -> set[str]:
    """Films in the "Favorites" list and the four profile favorites."""
    uris: set[str] = set()
    if "lists/favorites.csv" in export.namelist():
        # A list export has a header block, then a blank line, then one row per film
        text = export.read("lists/favorites.csv").decode("utf-8")
        _, _, films = text.partition("\n\n")
        uris |= {row["URL"] for row in csv.DictReader(io.StringIO(films)) if row.get("URL")}
    profile = list(csv.DictReader(io.StringIO(export.read("profile.csv").decode("utf-8"))))
    if profile and profile[0].get("Favorite Films"):
        uris |= {u.strip() for u in profile[0]["Favorite Films"].split(",")}
    return uris


def read_ratings(path: Path) -> pl.DataFrame:
    """One row per rated film: uri, title, year, rating (0.5-5), rated_on, watched_on (diary), favorite."""
    with zipfile.ZipFile(path) as export:
        ratings = _read_csv(export, "ratings.csv")
        diary = _read_csv(export, "diary.csv")
        favorites = _favorite_uris(export)

    # Diary entries have their own URIs: join on title and year, keeping the last viewing
    watched = (
        diary.filter(pl.col("Watched Date") != "")
        .group_by("Name", "Year")
        .agg(pl.col("Watched Date").str.to_date().max().alias("watched_on"))
    )
    return ratings.join(watched, on=["Name", "Year"], how="left").select(
        pl.col("Letterboxd URI").alias("letterboxd_uri"),
        pl.col("Name").alias("title"),
        pl.col("Year").cast(pl.Int32, strict=False).alias("year"),
        pl.col("Rating").cast(pl.Float32).alias("rating"),
        pl.col("Date").str.to_date().alias("rated_on"),
        "watched_on",
        pl.col("Letterboxd URI").is_in(favorites).alias("favorite"),
    )


def read_watchlist(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as export:
        return _read_csv(export, "watchlist.csv").select(
            pl.col("Letterboxd URI").alias("letterboxd_uri"),
            pl.col("Name").alias("title"),
            pl.col("Year").cast(pl.Int32, strict=False).alias("year"),
        )
