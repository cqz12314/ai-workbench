from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Workbench API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./data/ai_workbench.db"
    cors_origins: list[str] = ["http://localhost:5173"]
    rag_enabled: bool = True
    agent_enabled: bool = False
    github_enabled: bool = True
    github_write_enabled: bool = False
    github_token: SecretStr | None = None
    github_api_base: str = "https://api.github.com"
    workspace_enabled: bool = False
    workspace_root: str | None = None
    code_index_enabled: bool = False
    git_enabled: bool = False
    litellm_model: str | None = None
    deepseek_api_key: SecretStr | None = None
    deepseek_api_base: str = "https://api.deepseek.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
