from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Primary LLM provider (OpenRouter, OpenAI-compatible).
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_fast: str = "nvidia/nemotron-3.5-lightning:free"    # parsing, queries, outline
    openrouter_model_strong: str = "nvidia/nemotron-3-ultra-550b-a55b:free"  # thesis, drafting, critique
    openrouter_reasoning: bool = True
    # Only this provider's endpoints, no fallbacks. A bare slug like "nvidia" matches all of its
    # endpoints (Lightning's is tagged "nvidia/nvfp4", Ultra's "nvidia"). Empty lets OpenRouter route.
    openrouter_provider: str = "nvidia"
    llm_timeout_seconds: float = 300  # per request; a stalled free endpoint otherwise hangs the run

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
