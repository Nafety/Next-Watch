"""Cross-validation on the user's ratings: what matters is liking a film, not just having watched it.

Rated films are split in 5 folds, 10 times with different shuffles. Each scorer learns from 4 folds and must:
- rank the hidden films by taste: AUC = chance that a liked film (>= 4) scores above a disliked one (<= 3);
- find the hidden liked films among all unrated catalog films: recall@100.
"""

from collections.abc import Callable

import numpy as np
import polars as pl

from film_reco.config import Settings
from film_reco.models import Scorer

LIKED, DISLIKED = 4.0, 3.0
TOP_K = 100


def taste_auc(scores: np.ndarray, ratings: np.ndarray) -> float:
    """Share of (liked, disliked) pairs ordered correctly; ties count for half."""
    liked, disliked = scores[ratings >= LIKED], scores[ratings <= DISLIKED]
    if len(liked) == 0 or len(disliked) == 0:
        return float("nan")
    diff = liked[:, None] - disliked[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def recall_at_k(target_scores: np.ndarray, other_scores: np.ndarray, k: int = TOP_K) -> float:
    """Share of target films ranked in the top k among targets + others (ties do not push a film down)."""
    if len(target_scores) == 0:
        return float("nan")
    everything = np.sort(np.concatenate([target_scores, other_scores]))
    beaten_by = len(everything) - np.searchsorted(everything, target_scores, side="right")
    return float((beaten_by < k).mean())


def load_rated(settings: Settings) -> pl.DataFrame:
    ratings = pl.read_parquet(settings.data_dir / "ratings.parquet")
    return ratings.filter(pl.col("tmdb_id").is_not_null()).unique("tmdb_id").select("tmdb_id", "rating")


def evaluate(
    settings: Settings,
    make_scorers: Callable[[], list[Scorer]],
    folds: int = 5,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> pl.DataFrame:
    rated = load_rated(settings)
    catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
    unrated = np.setdiff1d(catalog["tmdb_id"].to_numpy(), rated["tmdb_id"].to_numpy())

    rows = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(rated))
        for fold in np.array_split(order, folds):
            test = rated[fold]
            train = rated[np.setdiff1d(order, fold)]
            test_ids, test_ratings = test["tmdb_id"].to_numpy(), test["rating"].to_numpy()
            for scorer in make_scorers():
                scorer.fit(train)
                test_scores, unrated_scores = scorer.score(test_ids), scorer.score(unrated)
                liked = test_ratings >= LIKED
                rows.append(
                    {
                        "scorer": scorer.name,
                        "auc": taste_auc(test_scores, test_ratings),
                        "recall": recall_at_k(
                            test_scores[liked], np.concatenate([unrated_scores, test_scores[~liked]])
                        ),
                    }
                )

    return (
        pl.DataFrame(rows)
        .group_by("scorer", maintain_order=True)
        .agg(
            # ± is the standard error of the mean over all folds
            pl.col("auc").mean().alias("taste AUC"),
            (pl.col("auc").std() / pl.len().sqrt()).alias("± AUC"),
            pl.col("recall").mean().alias(f"recall@{TOP_K}"),
            (pl.col("recall").std() / pl.len().sqrt()).alias("± recall"),
        )
    )
