"""Data of the web app (web/public/data/), computed once from the catalog and MovieLens.

catalog.json: catalog columns (ids, years, genres, TMDB ratings), no TMDB texts.
content / collaborative / movielens: int8 vectors with one scale per row (.i8 / .scales.f32), and meta.json.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from film_reco.collaborative import collaborative_embeddings, movielens_vectors
from film_reco.config import Settings
from film_reco.content import EMBEDDER, EMBEDDERS, content_embeddings

# Transformers.js (ONNX) version of each embedding model, and its pooling, for the web app
WEB_EMBEDDERS = {
    "intfloat/multilingual-e5-small": ("Xenova/multilingual-e5-small", "mean"),
    "sentence-transformers/all-MiniLM-L6-v2": ("Xenova/all-MiniLM-L6-v2", "mean"),
    "BAAI/bge-small-en-v1.5": ("Xenova/bge-small-en-v1.5", "cls"),
}


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 values and one float32 scale per row (value = q * scale / 127): 4 times smaller than float32."""
    scales = np.abs(vectors).max(axis=1).astype(np.float32)
    safe = np.where(scales > 0, scales, 1)[:, None]
    return np.round(vectors / safe * 127).astype(np.int8), scales


def dequantize(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(values.astype(np.float32) * scales[:, None] / 127, dtype=np.float32)


def export_web_data(settings: Settings, out: Path = Path("web/public/data")) -> None:
    """The static site's data: catalog columns (JSON), int8 vectors and their scales (binary), meta.json."""
    catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
    out.mkdir(parents=True, exist_ok=True)
    genres = sorted({g for gs in catalog["genres"] for g in gs})
    (out / "catalog.json").write_text(
        json.dumps(
            {
                "ids": catalog["tmdb_id"].to_list(),
                "years": catalog["year"].to_list(),
                "genreNames": genres,
                "genres": [[genres.index(g) for g in gs] for gs in catalog["genres"]],
                "voteAverage": catalog["vote_average"].to_list(),
                "voteCount": catalog["vote_count"].to_list(),
            }
        ),
        encoding="utf-8",
    )
    ml_ids, ml_vectors = movielens_vectors(settings)
    matrices = {
        "content": content_embeddings(settings, catalog),
        "collaborative": collaborative_embeddings(settings, catalog),
        "movielens": ml_vectors,
    }
    for name, vectors in matrices.items():
        values, scales = quantize(vectors)
        (out / f"{name}.i8").write_bytes(values.tobytes())
        (out / f"{name}.scales.f32").write_bytes(scales.tobytes())
    (out / "movielens.ids.i32").write_bytes(ml_ids.astype(np.int32).tobytes())
    model, pooling = WEB_EMBEDDERS[EMBEDDER]
    (out / "meta.json").write_text(
        json.dumps(
            {
                "embedder": {"model": model, "prefix": EMBEDDERS[EMBEDDER], "pooling": pooling},
                "dims": {name: int(v.shape[1]) for name, v in matrices.items()},
            }
        ),
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Web data: {len(catalog)} films, {len(ml_ids)} MovieLens films, {size:.1f} MB in {out}")
