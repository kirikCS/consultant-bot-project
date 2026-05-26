from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str = "changeme"

    llm_url: str = "http://llm:8080"
    embedder_url: str = "http://embedder:8081"

    med_services_path: str = "/app/data/med_services.json"
    indices_dir: str = "/app/indices"
    sqlite_path: str = "/app/data/bot.db"

    top_k_dense: int = 20
    top_k_sparse: int = 20
    rrf_k: int = 60
    rerank_input: int = 10
    top_n_final: int = 3

    answer_temperature: float = 0.0
    answer_max_tokens: int = 1500
    rerank_temperature: float = 0.0
    rerank_max_tokens: int = 32

    short_term_k: int = 4
    short_term_window: int = 50
    memory_pinned_recent: int = 6
    memory_max_chars: int = 3500

    agent_max_iters: int = 3

    http_timeout: float = 240.0


settings = Settings()
