"""Step 5: the hybrid score, an equal-weight sum of the quality, content and collaborative scores.

Each score is standardized over the whole catalog (mean 0, standard deviation 1) before the sum.
Learning the weights (ridge regression) did worse: with ~160 ratings per fold they are too noisy to fit.
"""

import numpy as np
import polars as pl

from film_reco.models import QualityScorer, Scorer, SimilarityScorer


class HybridScorer:
    name = "Hybrid (quality + content + collaborative)"

    def __init__(self, catalog: pl.DataFrame, content: np.ndarray, collaborative: np.ndarray) -> None:
        self._ids = catalog["tmdb_id"].to_numpy()
        self.parts: dict[str, Scorer] = {
            "quality": QualityScorer(catalog),
            "content": SimilarityScorer("content", self._ids, content),
            "collaborative": SimilarityScorer("collaborative", self._ids, collaborative),
        }

    def fit(self, rated: pl.DataFrame) -> None:
        self._scale = {}
        for name, scorer in self.parts.items():
            scorer.fit(rated)
            everything = scorer.score(self._ids)
            self._scale[name] = (everything.mean(), everything.std() + 1e-12)

    def explain(self, ids: np.ndarray) -> dict[str, np.ndarray]:
        """Standardized score of each signal (the hybrid score is their sum)."""
        return {
            name: (scorer.score(ids) - self._scale[name][0]) / self._scale[name][1]
            for name, scorer in self.parts.items()
        }

    def score(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(sum(self.explain(ids).values()))
