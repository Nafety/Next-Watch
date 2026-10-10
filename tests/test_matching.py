from film_reco.matching import best_match
from film_reco.tmdb import SearchResult


def result(id: int, title: str, date: str, original: str = "", popularity: float = 1.0) -> SearchResult:
    return SearchResult(
        id=id, title=title, original_title=original or title, release_date=date, popularity=popularity
    )


def test_exact_title_and_year() -> None:
    results = [result(1, "Parasite", "1982-03-12"), result(496243, "Parasite", "2019-05-30")]
    match = best_match("Parasite", 2019, results)
    assert match is not None and match[0].id == 496243


def test_original_title_counts() -> None:
    results = [result(194, "Amélie", "2001-04-25", original="Le Fabuleux Destin d'Amélie Poulain")]
    match = best_match("Le Fabuleux Destin d'Amélie Poulain", 2001, results)
    assert match is not None and match[0].id == 194


def test_one_year_off_is_accepted() -> None:
    match = best_match("Fight Club", 1999, [result(550, "Fight Club", "2000-01-01")])
    assert match is not None and match[0].id == 550


def test_wrong_film_is_rejected() -> None:
    assert best_match("The Matrix", 1999, [result(1, "The Matrix Resurrections", "2021-12-16")]) is None


def test_tie_goes_to_most_popular() -> None:
    results = [
        result(1, "Halloween", "2018-10-18", popularity=5),
        result(2, "Halloween", "2018-10-19", popularity=80),
    ]
    match = best_match("Halloween", 2018, results)
    assert match is not None and match[0].id == 2
