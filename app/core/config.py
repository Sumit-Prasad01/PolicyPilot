from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "PolicyPilot"
    app_env: str = "development"

    # Google Gemini (chat + embeddings)
    gemini_api_key: str = ""
    # Optional: falls back to gemini_api_key when empty.
    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    # Optional override for models whose output dimension is not in the built-in map.
    embedding_dimension: int | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_DIMENSION"),
    )

    # Web fallback
    tavily_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "policypilot"
    pinecone_namespace: str = "company-hr-kb"
    pinecone_cloud: str = Field(
        default="aws",
        validation_alias=AliasChoices("PINECONE_CLOUD"),
    )
    pinecone_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("PINECONE_REGION"),
    )

    # Retrieval / retries
    top_k: int = 4
    max_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("MAX_WEB_REWRITES", "MAX_RETRIES"),
    )

    # Admin / uploads
    admin_api_key: str = "change-me-in-production"
    max_upload_mb: int = 25

    # Local paths
    audit_db_path: str = Field(
        default=str(BASE_DIR / "data" / "audit.db"),
        validation_alias=AliasChoices("AUDIT_DB_PATH", "DATABASE_PATH"),
    )
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    @property
    def embedding_api_key(self) -> str:
        return (self.google_api_key or self.gemini_api_key).strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
