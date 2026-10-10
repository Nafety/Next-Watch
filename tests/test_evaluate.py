import numpy as np

from film_reco.evaluate import recall_at_k, taste_auc


def test_auc_perfect_and_reversed() -> None:
    ratings = np.array([5.0, 4.0, 3.0, 1.0])
    assert taste_auc(np.array([4.0, 3.0, 2.0, 1.0]), ratings) == 1.0
    assert taste_auc(np.array([1.0, 2.0, 3.0, 4.0]), ratings) == 0.0


def test_auc_ignores_middle_ratings_and_counts_ties_half() -> None:
    ratings = np.array([4.0, 3.5, 3.0])
    assert taste_auc(np.array([1.0, 9.0, 1.0]), ratings) == 0.5


def test_recall_at_k() -> None:
    others = np.arange(10.0)  # scores 0..9
    assert recall_at_k(np.array([9.5, 5.5]), others, k=3) == 0.5  # 1st place in, 5th place out
