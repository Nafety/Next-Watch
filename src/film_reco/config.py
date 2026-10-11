"""Settings read from the environment or the .env file."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tmdb_api_key: SecretStr
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")  # TMDB responses; must be writable
