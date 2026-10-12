"""Streamlit app: upload a Letterboxd export, browse recommendations, keep favorites.

    uv run streamlit run src/film_reco/app.py
Needs data/app/ (uv run film-reco export-app) and a TMDB API key (TMDB_API_KEY, in .env or as a secret).
"""

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import polars as pl
import streamlit as st

from film_reco.config import Settings
from film_reco.content import EMBEDDER, encode_films
from film_reco.hybrid import combine
from film_reco.tmdb import Movie, TMDBClient
from film_reco.web import AppModel, analyse_export

SHOWN = 20
POSTER = "https://image.tmdb.org/t/p/w342"
MODES = {  # label -> hybrid mode
    "Mix of both": "mix",
    "Close to my ratings": "taste",
    "Public favourites": "public",
}

st.set_page_config(page_title="Film recommendations from Letterboxd", page_icon="🎬", layout="wide")
# Per-visitor state, kept across reruns: favorites (tmdb_id -> scored row) and the current page
st.session_state.setdefault("favorites", {})
st.session_state.setdefault("page", 0)
st.session_state.setdefault("view", None)


@st.cache_resource
def resources() -> tuple[Settings, AppModel, TMDBClient]:
    settings = Settings()  # the API key comes from .env or the environment
    tmdb = TMDBClient(settings.tmdb_api_key.get_secret_value(), settings.cache_dir / "tmdb")
    return settings, AppModel(settings.data_dir / "app"), tmdb


@st.cache_resource
def embedder() -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDER, device="cpu")


def details(ids: list[int]) -> list[Movie]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(resources()[2].movie, ids))


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


def toggle_favorite(row: dict[str, Any]) -> None:
    favorites = st.session_state.favorites
    if row["tmdb_id"] in favorites:
        del favorites[row["tmdb_id"]]
    else:
        favorites[row["tmdb_id"]] = row


def turn_page(step: int) -> None:
    st.session_state.page = max(0, st.session_state.page + step) if step else 0


def grid(rows: list[dict[str, Any]], watchlist: set[int], key: str) -> None:
    """Film cards, 5 per line: poster, title, TMDB rating, the three signals, favorite button."""
    films = details([row["tmdb_id"] for row in rows])
    for start in range(0, len(rows), 5):
        for column, film, row in zip(
            st.columns(5), films[start : start + 5], rows[start : start + 5], strict=False
        ):
            with column:
                if film.poster_path:
                    st.image(POSTER + film.poster_path, width="stretch")
                badge = " · 📌 watchlist" if film.id in watchlist else ""
                st.markdown(
                    f"**[{film.title}](https://letterboxd.com/tmdb/{film.id}/)** ({film.year}){badge}"
                )
                directors = ", ".join(c.name for c in film.credits.crew if c.job == "Director")[:40]
                st.caption(
                    f"{directors}  \n★ **{film.vote_average:.1f}**/10 · {film.vote_count:,} votes  \n"
                    f"Quality {bar(row['quality'])}  \nContent {bar(row['content'])}  \n"
                    f"Taste {bar(row['collaborative'])}"
                )
                liked = film.id in st.session_state.favorites
                st.button(
                    "♥ Remove" if liked else "♡ Favorite",
                    key=f"{key}-{film.id}",
                    on_click=toggle_favorite,
                    args=(row,),
                    type="primary" if liked else "secondary",
                )


def letterboxd_csv(rows: list[dict[str, Any]]) -> str:
    """Favorites in Letterboxd's import format (Settings > Import, or a list's import)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["tmdbID", "Title", "Year"])
    for film in details([row["tmdb_id"] for row in rows]):
        writer.writerow([film.id, film.title, film.year or ""])
    return out.getvalue()


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
    st.header("Recommend")
    label = st.radio(
        "Based on",
        list(MODES),
        help="Close to my ratings: films like the ones you rated well, whatever the public thinks. "
        "Public favourites: the best rated films you have not seen. Mix: both.",
    )
    st.header("Filters")
    low, high = st.slider("Release year", first, last, (first, last))
    wanted = st.multiselect("Genres (any of)", sorted({g for gs in scored["genres"] for g in gs}))
    hide_watchlist = st.checkbox("Hide my watchlist", value=False)

parts = {name: scored[name].to_numpy() for name in ("quality", "content", "collaborative")}
ranked = scored.with_columns(score=combine(parts, MODES[label])).sort("score", descending=True)
ranked = ranked.filter(pl.col("year").is_between(low, high))
if wanted:
    ranked = ranked.filter(pl.col("genres").list.eval(pl.element().is_in(wanted)).list.any())
if hide_watchlist:
    ranked = ranked.filter(~pl.col("tmdb_id").is_in(list(in_watchlist)))

# A new mode, filter or export starts again from the best films
view = (upload.file_id, label, low, high, tuple(wanted), hide_watchlist)
if view != st.session_state.view:
    st.session_state.view, st.session_state.page = view, 0

favorites = st.session_state.favorites
recommendations, saved = st.tabs(["Recommendations", f"♥ Favorites ({len(favorites)})"])

with recommendations:
    start = st.session_state.page * SHOWN
    if start >= len(ranked):
        st.info("No more films with these filters.")
    else:
        st.caption(f"Films {start + 1}–{min(start + SHOWN, len(ranked))} of {len(ranked):,}")
        grid(ranked.slice(start, SHOWN).to_dicts(), in_watchlist, key="reco")
    left, right, _ = st.columns([1, 1, 4])
    left.button("↻ Other films", on_click=turn_page, args=(1,), disabled=start + SHOWN >= len(ranked))
    right.button("⟲ Back to the best", on_click=turn_page, args=(0,), disabled=start == 0)

with saved:
    if not favorites:
        st.info("Click ♡ Favorite on a film to keep it here.")
    else:
        st.download_button(
            "Download for Letterboxd (CSV)",
            letterboxd_csv(list(favorites.values())),
            file_name="favorites.csv",
            mime="text/csv",
            help="Import it on Letterboxd: Settings → Import, or into a list or your watchlist.",
        )
        grid(list(favorites.values()), in_watchlist, key="fav")

st.divider()
st.caption(
    "Film data and posters from [TMDB](https://www.themoviedb.org). "
    "This product uses the TMDB API but is not endorsed or certified by TMDB. "
    "Taste signal learned from [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)."
)
