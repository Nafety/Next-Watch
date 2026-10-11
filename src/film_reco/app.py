"""Streamlit app: upload a Letterboxd export, get film recommendations.

    uv run streamlit run src/film_reco/app.py
Needs data/app/ (uv run film-reco export-app) and a TMDB API key (TMDB_API_KEY, in .env or as a secret).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import streamlit as st

from film_reco.config import Settings
from film_reco.content import EMBEDDER, encode_films
from film_reco.tmdb import Movie, TMDBClient
from film_reco.web import AppModel, analyse_export

SHOWN = 20
POSTER = "https://image.tmdb.org/t/p/w342"

st.set_page_config(page_title="Film recommendations from Letterboxd", page_icon="🎬", layout="wide")


@st.cache_resource
def resources() -> tuple[Settings, AppModel, TMDBClient]:
    settings = Settings()  # the API key comes from .env or the environment
    tmdb = TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.cache_dir / "tmdb")
    return settings, AppModel(settings.data_dir / "app"), tmdb


@st.cache_resource
def embedder() -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDER, device="cpu")


def details(tmdb: TMDBClient, ids: list[int]) -> list[Movie]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(tmdb.movie, ids))


@st.cache_data(show_spinner=False, max_entries=20)
def analyse(export: bytes) -> tuple[pl.DataFrame, dict[str, int], set[int]]:
    _, model, tmdb = resources()

    def embed(films: list[dict[str, object]]) -> np.ndarray:
        return encode_films(embedder(), films)

    return analyse_export(export, model, tmdb, embed)


def bar(value: float) -> str:
    """A standardized score (about -3..+3) as five dots."""
    filled = int(np.clip(round((value + 1) * 1.25), 0, 5))
    return "●" * filled + "○" * (5 - filled)


st.title("🎬 Film recommendations from your Letterboxd")
st.markdown(
    "Export your data on Letterboxd (**Settings → Data → Export your data**), then drop the ZIP here. "
    "Your ratings stay in this session: nothing is stored."
)
upload = st.file_uploader("Letterboxd export (.zip)", type="zip")

if upload is None:
    with st.expander("How does it work?"):
        st.markdown(
            "Each film gets three scores, then their sum ranks the films you have not rated:\n"
            "- **Quality**: its TMDB rating, corrected for the number of votes;\n"
            "- **Content**: how close its description (genres, keywords, director, cast, synopsis) is to the "
            "films you rated well, and far from those you rated badly;\n"
            "- **Taste**: whether people who rate like you, among 200,000 MovieLens users, liked it.\n\n"
            "Candidates are the 10,000 most voted films on TMDB."
        )
    st.stop()

with st.spinner("Finding your films on TMDB and scoring 10,000 candidates…"):
    scored, counts, in_watchlist = analyse(upload.getvalue())
st.caption(
    f"{counts['rated']} rated films, {counts['found']} found on TMDB"
    + (f" ({counts['miniseries']} miniseries left out)" if counts["miniseries"] else "")
)
if counts["found"] < 10:
    st.warning("Rate at least 10 films on Letterboxd for meaningful recommendations.")
    st.stop()

first, last = int(scored["year"].drop_nulls().min()), int(scored["year"].drop_nulls().max())  # type: ignore[arg-type]
with st.sidebar:
    st.header("Filters")
    low, high = st.slider("Release year", first, last, (first, last))
    genres = sorted({g for gs in scored["genres"] for g in gs})
    wanted = st.multiselect("Genres (any of)", genres)
    hide_watchlist = st.checkbox("Hide my watchlist", value=False)

shown = scored.filter(pl.col("year").is_between(low, high))
if wanted:
    shown = shown.filter(pl.col("genres").list.eval(pl.element().is_in(wanted)).list.any())
if hide_watchlist:
    shown = shown.filter(~pl.col("tmdb_id").is_in(list(in_watchlist)))
shown = shown.head(SHOWN)

tmdb = resources()[2]
films = details(tmdb, shown["tmdb_id"].to_list())
for start in range(0, len(films), 5):
    for column, film, row in zip(
        st.columns(5), films[start : start + 5], shown[start : start + 5].iter_rows(named=True), strict=False
    ):
        with column:
            if film.poster_path:
                st.image(POSTER + film.poster_path, width="stretch")
            directors = ", ".join(c.name for c in film.credits.crew if c.job == "Director")[:40]
            badge = " · 📌 watchlist" if film.id in in_watchlist else ""
            st.markdown(f"**[{film.title}](https://letterboxd.com/tmdb/{film.id}/)** ({film.year}){badge}")
            st.caption(
                f"{directors}  \nQuality {bar(row['quality'])}  \nContent {bar(row['content'])}  \n"
                f"Taste {bar(row['collaborative'])}"
            )

st.divider()
st.caption(
    "Film data and posters from [TMDB](https://www.themoviedb.org). "
    "This product uses the TMDB API but is not endorsed or certified by TMDB. "
    "Taste signal learned from [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)."
)
