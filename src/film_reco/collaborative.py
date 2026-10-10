"""Step 4: collaborative vectors. Two films are close when the same people like them (MovieLens 32M).

The MovieLens ratings form a users x films table; a truncated SVD compresses it into one vector per film.
Download: https://files.grouplens.org/datasets/movielens/ml-32m.zip, into data/movielens/.
"""

import zipfile

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from film_reco.config import Settings

MIN_RATINGS = 20  # films with fewer MovieLens ratings get no vector
DIMENSIONS = 128


def _movielens(settings: Settings, name: str) -> pl.DataFrame:
    folder = settings.data_dir / "movielens"
    path = folder / "ml-32m" / name
    if not path.exists():
        with zipfile.ZipFile(folder / "ml-32m.zip") as archive:
            archive.extract(f"ml-32m/{name}", folder)
    return pl.read_csv(path)


def movielens_vectors(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(tmdb ids, normalized film vectors) learned from MovieLens."""
    ratings = _movielens(settings, "ratings.csv").select("userId", "movieId", "rating")
    counts = ratings.group_by("movieId").len()
    ratings = ratings.join(counts.filter(pl.col("len") >= MIN_RATINGS), on="movieId", how="semi")
    # Centering on each user's mean: what counts is liking a film more or less than one usually does
    ratings = ratings.with_columns(pl.col("rating") - pl.col("rating").mean().over("userId"))

    users = ratings["userId"].rank("dense").cast(pl.Int64).to_numpy() - 1
    movie_ids = np.unique(ratings["movieId"].to_numpy())
    columns = np.searchsorted(movie_ids, ratings["movieId"].to_numpy())
    table = csr_matrix(
        (ratings["rating"].to_numpy().astype(np.float32), (users, columns)),
        shape=(int(users.max()) + 1, len(movie_ids)),
    )
    _, strengths, film_factors = randomized_svd(table, n_components=DIMENSIONS, n_iter=5, random_state=0)
    vectors = film_factors.T * strengths
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    links = _movielens(settings, "links.csv").drop_nulls("tmdbId").unique("movieId")
    tmdb = dict(zip(links["movieId"], links["tmdbId"], strict=True))
    keep = [n for n, m in enumerate(movie_ids) if m in tmdb]
    return np.array([tmdb[movie_ids[n]] for n in keep]), vectors[keep]


def collaborative_embeddings(settings: Settings, catalog: pl.DataFrame) -> np.ndarray:
    """One vector per catalog film (zeros if MovieLens does not know it), cached until the catalog changes."""
    cache = settings.data_dir / "collaborative_embeddings.npz"
    ids = catalog["tmdb_id"].to_numpy()
    if cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["ids"], ids):
            return np.asarray(stored["vectors"])

    known_ids, known_vectors = movielens_vectors(settings)
    row = {int(i): n for n, i in enumerate(known_ids)}
    vectors = np.zeros((len(ids), known_vectors.shape[1]), dtype=np.float32)
    for n, i in enumerate(ids):
        if int(i) in row:
            vectors[n] = known_vectors[row[int(i)]]
    np.savez(cache, ids=ids, vectors=vectors)
    print(f"MovieLens knows {int((vectors != 0).any(axis=1).sum())} of {len(ids)} catalog films")
    return vectors
