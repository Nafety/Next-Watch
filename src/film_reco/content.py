"""Step 3: content-based scoring. Films are compared through an embedding of their TMDB description."""

from typing import Any

import numpy as np
import polars as pl

from film_reco.config import Settings

# Embedding model -> prefix it expects before a document (e5 was trained with "passage: ")
EMBEDDERS = {
    "intfloat/multilingual-e5-small": "passage: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "BAAI/bge-small-en-v1.5": "",
}
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


def content_embeddings(settings: Settings, catalog: pl.DataFrame, embedder: str = EMBEDDER) -> np.ndarray:
    """One normalized vector per catalog film (same order), cached in data/ until the catalog changes."""
    cache = settings.data_dir / f"content_embeddings_{embedder.split('/')[-1]}.npz"
    ids = catalog["tmdb_id"].to_numpy()
    if cache.exists():
        stored = np.load(cache)
        if np.array_equal(stored["ids"], ids):
            vectors: np.ndarray = np.asarray(stored["vectors"])
            return vectors

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedder, device="cpu")
    vectors = encode_films(model, list(catalog.iter_rows(named=True)), EMBEDDERS[embedder], progress=True)
    np.savez(cache, ids=ids, vectors=vectors)
    return vectors


def encode_films(
    model: Any, films: list[dict[str, object]], prefix: str = EMBEDDERS[EMBEDDER], progress: bool = False
) -> np.ndarray:
    """Normalized embedding of each film's description (model: a loaded SentenceTransformer)."""
    texts = [prefix + film_text(film) for film in films]
    return np.asarray(
        model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=progress)
    )
