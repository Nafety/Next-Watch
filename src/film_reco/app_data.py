"""Files shipped with the web app (data/app/), computed once from the catalog and MovieLens.

catalog.parquet: per film, only what scoring and filters need (no TMDB texts).
vectors.npz: content and collaborative vectors, in the catalog's order.
movielens.npz: collaborative vectors of every film MovieLens knows, for users' films outside the catalog.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from film_reco.collaborative import collaborative_embeddings, movielens_vectors
from film_reco.config import Settings
from film_reco.content import EMBEDDER, EMBEDDERS, content_embeddings

CATALOG_COLUMNS = ["tmdb_id", "year", "genres", "vote_average", "vote_count", "popularity"]


def export_app_data(settings: Settings) -> None:
    catalog = pl.read_parquet(settings.data_dir / "catalog.parquet")
    out = settings.data_dir / "app"
    out.mkdir(exist_ok=True)
    catalog.select(CATALOG_COLUMNS).write_parquet(out / "catalog.parquet")
    np.savez_compressed(
        out / "vectors.npz",
        ids=catalog["tmdb_id"].to_numpy(),
        content=content_embeddings(settings, catalog).astype(np.float16),
        collaborative=collaborative_embeddings(settings, catalog).astype(np.float16),
    )
    ids, vectors = movielens_vectors(settings)
    np.savez_compressed(out / "movielens.npz", ids=ids, vectors=vectors.astype(np.float16))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"App data: {len(catalog)} catalog films, {len(ids)} MovieLens films, {size:.0f} MB in {out}")


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
