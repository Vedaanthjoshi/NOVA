from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for NOVA."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core environment
    nova_env: Literal["development", "testing", "production"] = "development"
    log_level: str = "INFO"

    # Model providers
    openai_api_key: str = ""
    gemini_api_key: str = ""
    default_model: str = "gemini-3.5-flash-lite"

    # Database
    sqlite_db_path: Path = Path("nova_audit.sqlite")

    # Mock environment workspace
    mock_env_dir: Path = Path("./mock_state")


# Singleton instance for system-wide configuration
settings = Settings()

