from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "anthropic" or "groq" — swap the reasoning-node LLM provider without
    # touching agent/graph.py's call sites.
    llm_provider: str = "groq"

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    groq_api_key: str = ""
    groq_model: str = "allam-2-7b"

    database_url: str = "postgresql+psycopg://darukaa:darukaa@localhost:5432/darukaa"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Minimum number of the 5 core environmental variables that must be
    # known before the agent will commit to a recommendation instead of
    # asking a clarifying question. The challenge brief requires
    # reasoning over >= 3 variables together.
    min_known_variables: int = 3


settings = Settings()