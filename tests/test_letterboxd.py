import zipfile
from datetime import date
from pathlib import Path

from film_reco.letterboxd import read_ratings, read_watchlist


def make_export(path: Path) -> Path:
    files = {
        "profile.csv": 'Date Joined,Username,Favorite Films\n2025-01-01,me,"https://boxd.it/b, https://boxd.it/x"\n',
        "ratings.csv": "Date,Name,Year,Letterboxd URI,Rating\n"
        "2025-02-01,Parasite,2019,https://boxd.it/a,3.5\n2025-03-01,Fight Club,1999,https://boxd.it/b,4.5\n",
        "diary.csv": "Date,Name,Year,Letterboxd URI,Rating,Rewatch,Tags,Watched Date\n"
        "2025-03-01,Fight Club,1999,https://boxd.it/e1,4,,,2025-02-28\n"
        "2025-03-05,Fight Club,1999,https://boxd.it/e2,4.5,Yes,,2025-03-04\n",
        "watchlist.csv": "Date,Name,Year,Letterboxd URI\n2025-04-01,Heat,1995,https://boxd.it/h\n",
        "lists/favorites.csv": "Letterboxd list export v7\nDate,Name,Tags,URL,Description\n"
        "2025-01-01,Favorites,,https://boxd.it/l,\n\nPosition,Name,Year,URL,Description\n"
        "1,Parasite,2019,https://boxd.it/a,\n",
    }
    export = path / "letterboxd-me-2025.zip"
    with zipfile.ZipFile(export, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return export


def test_ratings(tmp_path: Path) -> None:
    ratings = read_ratings(make_export(tmp_path)).sort("title")
    assert ratings["title"].to_list() == ["Fight Club", "Parasite"]
    assert ratings["rating"].to_list() == [4.5, 3.5]
    assert ratings["watched_on"].to_list() == [date(2025, 3, 4), None]  # last viewing from the diary
    assert ratings["favorite"].to_list() == [True, True]  # profile favorite, then favorites list


def test_watchlist(tmp_path: Path) -> None:
    watchlist = read_watchlist(make_export(tmp_path))
    assert watchlist.row(0) == ("https://boxd.it/h", "Heat", 1995)
