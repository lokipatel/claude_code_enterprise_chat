"""Application configuration, read from environment variables / .env.

Reads the .env file already present at the project root without ever
writing to it. Field names track the environment variables that are
actually defined there (e.g. NEO4J_USERNAME, not NEO4J_USER).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = Field(validation_alias="NEO4J_URI")
    neo4j_username: str = Field(validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", validation_alias="NEO4J_DATABASE")

    # OpenAI (used by Graphiti for extraction + embeddings)
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    openai_llm_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_LLM_MODEL")
    openai_small_llm_model: str = Field(
        default="gpt-4.1-mini", validation_alias="OPENAI_SMALL_LLM_MODEL"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )

    # Gmail OAuth
    gmail_credentials_path: Path = Field(
        default=PROJECT_ROOT / "credentials.json", validation_alias="GMAIL_CREDENTIALS_PATH"
    )
    gmail_token_path: Path = Field(
        default=PROJECT_ROOT / "token.json", validation_alias="GMAIL_TOKEN_PATH"
    )
    gmail_scopes: tuple[str, ...] = ("https://www.googleapis.com/auth/gmail.readonly",)

    # Incremental-sync checkpoint (tracks which Gmail messages are already ingested)
    sync_state_path: Path = Field(
        default=PROJECT_ROOT / "sync_state.json", validation_alias="SYNC_STATE_PATH"
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton so .env is only parsed once per process."""
    return Settings()  # type: ignore[call-arg]
