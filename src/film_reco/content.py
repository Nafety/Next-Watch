"""Step 3: content-based scoring. Films are compared through an embedding of their TMDB description."""

import numpy as np
import polars as pl

from film_reco.config import Settings

EMBEDDER = "intfloat/multilingual-e5-small"


def film_text(film: dict[str, object]) -> str:
    """What the embedding model reads for one film."""

    def names(key: str, n: int) -> str:
        values = film.get(key) or []
        assert isinstance(values, list)
        return ", ".join(values[:n])

    return (
        f"{film['title']}. {names('genres', 5)}. Keywords: {names('keywords', 15)}. "
        f"Directed by {names('directors', 2)}. Starring {names('cast', 5)}. {film['overview']}"
    )


def content_embeddings(settings: Settings, catalog: pl.DataFrame) -> np.ndarray:
    """One normalized vector per catalog film (same order), cached in data/ until the catalog changes."""
    cache = settings.data_dir / "content_embeddings.npz"
    ids = catalog["tmdb_id"].to_numpy()
    if cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["ids"], ids):
            vectors: np.ndarray = np.asarray(stored["vectors"])
            return vectors

    from sentence_transformers import SentenceTransformer

    texts = [f"passage: {film_text(film)}" for film in catalog.iter_rows(named=True)]
    vectors = SentenceTransformer(EMBEDDER, device="cpu").encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    np.savez(cache, ids=ids, vectors=vectors)
    return vectors


class ContentScorer:
    """Each rated film votes for similar films, weighted by (rating - the user's mean rating).

    neighbours=None: every rated film votes (a single taste profile).
    neighbours=k: only the k rated films most similar to the candidate vote (copes with varied tastes).
    """

    def __init__(self, ids: np.ndarray, vectors: np.ndarray, neighbours: int | None = None) -> None:
        self._row = {int(i): n for n, i in enumerate(ids)}
        self._vectors = vectors
        self._k = neighbours
        self.name = "Content (profile)" if neighbours is None else f"Content ({neighbours} nearest)"

    def fit(self, rated: pl.DataFrame) -> None:
        ratings = rated["rating"].to_numpy()
        self._rated = self._vectors[[self._row[int(i)] for i in rated["tmdb_id"]]]
        self._weights = ratings - ratings.mean()

    def score(self, ids: np.ndarray) -> np.ndarray:
        similarity = self._vectors[[self._row[int(i)] for i in ids]] @ self._rated.T  # candidates x rated
        if self._k is None:
            return np.asarray(similarity @ self._weights)
        nearest = np.argpartition(-similarity, self._k, axis=1)[:, : self._k]
        weights = self._weights[nearest]
        return (np.take_along_axis(similarity, nearest, axis=1) * weights).sum(axis=1)
