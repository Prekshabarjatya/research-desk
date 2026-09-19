from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Primary LLM provider (Groq).
    groq_api_key: str = ""
    groq_model_fast: str = "openai/gpt-oss-20b"    # parsing, queries, outline
    groq_model_strong: str = "openai/gpt-oss-120b"  # drafting, critique

    # Optional fallback provider, any OpenAI-compatible endpoint.
    fallback_base_url: str = ""
    fallback_api_key: str = ""
    fallback_model_fast: str = ""
    fallback_model_strong: str = ""

    # Literature sources.
    openalex_mailto: str = ""           # polite-pool contact email, no key needed
    semantic_scholar_api_key: str = ""  # optional, raises rate limits
    openalex_api_key: str = ""          # optional; helps when a shared IP is rate limited
    crossref_mailto: str = ""

    # Guardrails. A run fails closed once it crosses either limit.
    max_revisions: int = 4
    max_tokens_per_run: int = 400_000
    min_verified_sources: int = 3       # fewer than this and the run stops before spending tokens
    max_sources: int = 15               # keep only the most relevant verified sources
    abstract_chars: int = 400           # per-source abstract length sent to the model

    # Service. The API refuses every route except /health until a token is set (fail closed).
    api_token: str = ""
    max_active_runs: int = 20
    run_worker: bool = False         # run the worker inside the API process (one-service deploys)
    cors_origins: str = ""           # comma-separated origins allowed to call the API (e.g. a Vercel UI)
    stale_run_seconds: int = 120     # a "running" run with no heartbeat this long is re-queued
    worker_poll_seconds: float = 2.0

    database_url: str = "postgresql://research:research@localhost:5432/research"


settings = Settings()
