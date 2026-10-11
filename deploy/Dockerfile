# Image for the Hugging Face Space (built from dist/space/, see `film-reco export-app`)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/app/.hf

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
# Download the embedding model at build time, so the app starts fast and offline
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('intfloat/multilingual-e5-small')"
COPY data/app ./data/app
# Spaces run the container as a non-root user, with no writable home
RUN chmod -R a+rX /app
ENV HF_HUB_OFFLINE=1 HOME=/tmp CACHE_DIR=/tmp/tmdb-cache

EXPOSE 7860
CMD [".venv/bin/streamlit", "run", "src/film_reco/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
