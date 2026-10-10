"""Minimal TMDB API client: retries on rate limits and server errors, and caches every response on disk."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class SearchResult(BaseModel):
    id: int
    title: str
    original_title: str = ""
    release_date: str = ""
    popularity: float = 0.0

    @property
    def year(self) -> int | None:
        return int(self.release_date[:4]) if self.release_date[:4].isdigit() else None


class _Named(BaseModel):
    name: str


class _CrewMember(_Named):
    job: str


class _Credits(BaseModel):
    cast: list[_Named] = []
    crew: list[_CrewMember] = []


class _Keywords(BaseModel):
    keywords: list[_Named] = []


class Movie(SearchResult):
    original_language: str = ""
    overview: str = ""
    runtime: int | None = None
    genres: list[_Named] = []
    vote_average: float = 0.0
    vote_count: int = 0
    poster_path: str | None = None
    credits: _Credits = _Credits()
    keywords: _Keywords = _Keywords()

    def to_row(self) -> dict[str, Any]:
        return {
            "tmdb_id": self.id,
            "title": self.title,
            "original_title": self.original_title,
            "year": self.year,
            "original_language": self.original_language,
            "runtime": self.runtime,
            "genres": [g.name for g in self.genres],
            "keywords": [k.name for k in self.keywords.keywords],
            "directors": [c.name for c in self.credits.crew if c.job == "Director"],
            "cast": [c.name for c in self.credits.cast[:5]],
            "overview": self.overview,
            "popularity": self.popularity,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count,
            "poster_path": self.poster_path,
        }


def _retryable(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code == 429 or error.response.status_code >= 500
    return isinstance(error, httpx.TransportError)


class TMDBClient:
    def __init__(self, api_key: str, cache_dir: Path) -> None:
        self._http = httpx.Client(
            base_url="https://api.themoviedb.org/3", params={"api_key": api_key}, timeout=20
        )
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "TMDBClient":
        return self

    def __exit__(self, *_: object) -> None:
        self._http.close()

    @retry(
        retry=retry_if_exception(_retryable), wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6)
    )
    def _fetch(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        response = self._http.get(path, params=params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def _get(self, path: str, **params: str) -> dict[str, Any]:
        # The API key is a client-level parameter, so it never ends up in the cache key
        key = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()
        cached = self._cache / f"{key}.json"
        if cached.exists():
            data: dict[str, Any] = json.loads(cached.read_text(encoding="utf-8"))
            return data
        data = self._fetch(path, params)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data

    def search_movie(self, title: str, year: int | None = None) -> list[SearchResult]:
        params = {"query": title} | ({"primary_release_year": str(year)} if year else {})
        return [SearchResult.model_validate(r) for r in self._get("/search/movie", **params)["results"]]

    def search_tv(self, title: str, year: int | None = None) -> list[SearchResult]:
        """TV shows, with the same fields as films (Letterboxd also lists miniseries)."""
        params = {"query": title} | ({"first_air_date_year": str(year)} if year else {})
        return [
            SearchResult(
                id=r["id"],
                title=r["name"],
                original_title=r.get("original_name", ""),
                release_date=r.get("first_air_date", ""),
                popularity=r.get("popularity", 0.0),
            )
            for r in self._get("/search/tv", **params)["results"]
        ]

    def movie(self, tmdb_id: int) -> Movie:
        return Movie.model_validate(self._get(f"/movie/{tmdb_id}", append_to_response="credits,keywords"))
