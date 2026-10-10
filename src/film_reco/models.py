"""Scorers: each learns from the user's ratings, then scores any catalog film (higher = better)."""

from typing import Protocol

import numpy as np
import polars as pl


class Scorer(Protocol):
    name: str

    def fit(self, rated: pl.DataFrame) -> None:
        """rated: the user's known ratings (columns tmdb_id, rating)."""

    def score(self, ids: np.ndarray) -> np.ndarray: ...


class RandomScorer:
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def fit(self, rated: pl.DataFrame) -> None:
        pass

    def score(self, ids: np.ndarray) -> np.ndarray:
        return self._rng.random(len(ids))


class PopularityScorer:
    name = "Popularity (TMDB)"

    def __init__(self, catalog: pl.DataFrame) -> None:
        self._values = dict(zip(catalog["tmdb_id"], catalog["popularity"], strict=True))

    def fit(self, rated: pl.DataFrame) -> None:
        pass

    def score(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self._values[i] for i in ids])


class QualityScorer:
    """TMDB average rating, shrunk towards the catalog mean when there are few votes (Bayesian average)."""

    name = "Quality (TMDB rating)"

    def __init__(self, catalog: pl.DataFrame, prior_votes: int = 1000) -> None:
        mean = float(catalog["vote_average"].mean())  # type: ignore[arg-type]
        weighted = (catalog["vote_average"] * catalog["vote_count"] + prior_votes * mean) / (
            catalog["vote_count"] + prior_votes
        )
        self._values = dict(zip(catalog["tmdb_id"], weighted, strict=True))

    def fit(self, rated: pl.DataFrame) -> None:
        pass

    def score(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self._values[i] for i in ids])


class SimilarityScorer:
    """Each rated film votes for similar films, weighted by (rating - the user's mean rating).

    Films are compared through vectors (content embeddings, or collaborative ones); a film without
    a vector (all zeros) gets a neutral score of 0.
    """

    def __init__(self, name: str, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.name = name
        self._row = {int(i): n for n, i in enumerate(ids)}
        self._vectors = vectors

    def fit(self, rated: pl.DataFrame) -> None:
        ratings = rated["rating"].to_numpy()
        self._rated = self._vectors[[self._row[int(i)] for i in rated["tmdb_id"]]]
        self._weights = ratings - ratings.mean()

    def score(self, ids: np.ndarray) -> np.ndarray:
        similarity = self._vectors[[self._row[int(i)] for i in ids]] @ self._rated.T  # candidates x rated
        return np.asarray(similarity @ self._weights) / len(self._weights)  # same scale for any profile size
