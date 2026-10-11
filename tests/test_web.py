from pathlib import Path

import numpy as np
import polars as pl

from film_reco.web import AppModel


def make_app_data(folder: Path) -> None:
    ids = np.array([1, 2, 3, 4, 5])
    pl.DataFrame(
        {
            "tmdb_id": ids,
            "year": [2000, 2001, 2002, 2003, 2004],
            "genres": [["Drama"], ["Drama"], ["Comedy"], ["Comedy"], ["Drama"]],
            "vote_average": [7.0, 7.0, 7.0, 7.0, 7.0],
            "vote_count": [1000, 1000, 1000, 1000, 1000],
            "popularity": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    ).write_parquet(folder / "catalog.parquet")
    # Films 1, 2, 5 point one way (dramas), 3 and 4 the other way (comedies)
    content = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float16)
    np.savez(folder / "vectors.npz", ids=ids, content=content, collaborative=content)
    np.savez(folder / "movielens.npz", ids=np.array([99]), vectors=np.array([[1, 0]], dtype=np.float16))


def test_score_ranks_unrated_films_by_taste(tmp_path: Path) -> None:
    make_app_data(tmp_path)
    model = AppModel(tmp_path)
    rated = pl.DataFrame({"tmdb_id": [1, 3], "rating": [5.0, 1.0]})  # loves a drama, hates a comedy
    scored = model.score(rated, [], embed=lambda films: np.zeros((len(films), 2)))
    assert set(scored["tmdb_id"]) == {2, 4, 5}  # rated films are left out
    assert scored["tmdb_id"][-1] == 4  # the comedy comes last


def test_rated_film_outside_the_catalog_is_used(tmp_path: Path) -> None:
    make_app_data(tmp_path)
    model = AppModel(tmp_path)
    extra = {
        "tmdb_id": 99,
        "year": 2010,
        "genres": ["Drama"],
        "vote_average": 7.0,
        "vote_count": 1000,
        "popularity": 10.0,
        "title": "Outside",
        "overview": "",
    }
    rated = pl.DataFrame({"tmdb_id": [99, 3], "rating": [5.0, 1.0]})
    scored = model.score(rated, [extra], embed=lambda films: np.array([[1.0, 0.0]] * len(films)))
    assert scored["tmdb_id"][-1] == 4  # film 99 (a drama, known by MovieLens) pulls dramas up
