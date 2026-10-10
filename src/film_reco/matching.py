"""Picking the TMDB search result that corresponds to a Letterboxd film (title + year)."""

from rapidfuzz import fuzz, utils

from film_reco.tmdb import SearchResult

MIN_SCORE = 80.0


def match_score(title: str, year: int | None, result: SearchResult) -> float:
    """0-100: title similarity (English or original title), minus a penalty when the years differ."""
    similarity = max(
        fuzz.token_sort_ratio(title, candidate, processor=utils.default_process)
        for candidate in (result.title, result.original_title)
    )
    if year is None or result.year is None:
        return similarity - 10
    gap = abs(year - result.year)
    return similarity - (0 if gap == 0 else 5 if gap == 1 else 40)


def best_match(
    title: str, year: int | None, results: list[SearchResult]
) -> tuple[SearchResult, float] | None:
    """Best result if it is convincing enough, else None. Ties go to the most popular film."""
    scored = [(r, match_score(title, year, r)) for r in results]
    if not scored:
        return None
    best = max(scored, key=lambda pair: (pair[1], pair[0].popularity))
    return best if best[1] >= MIN_SCORE else None
