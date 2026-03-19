from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    db_url: str = Field(..., alias="DB_URL")
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field("models/gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_model_preferences: list[str] = Field(
        default_factory=lambda: [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
        ],
        alias="GEMINI_MODEL_PREFERRED",
    )
    gemini_log_models_on_start: bool = Field(True, alias="GEMINI_LOG_MODELS_ON_START")
    gemini_timeout_seconds: int = Field(20, alias="GEMINI_TIMEOUT_SECONDS")
    gemini_max_output_tokens: int = Field(1024, alias="GEMINI_MAX_OUTPUT_TOKENS")
    gemini_enable_google_search: bool = Field(
        False, alias="GEMINI_ENABLE_GOOGLE_SEARCH"
    )

    # Recommendation batch scheduler
    enable_reco_scheduler: bool = Field(False, alias="ENABLE_RECO_SCHEDULER")
    # Default: every Monday 09:00 Seoul time (crontab format)
    reco_scheduler_cron: str = Field("0 9 * * 1", alias="RECO_SCHEDULER_CRON")
    reco_scheduler_timezone: str = Field("Asia/Seoul", alias="RECO_SCHEDULER_TZ")
    reco_scheduler_top_k: int = Field(4, alias="RECO_SCHEDULER_TOP_K")
    reco_scheduler_search_k: int = Field(20, alias="RECO_SCHEDULER_SEARCH_K")
    reco_scheduler_use_v2: bool = Field(False, alias="RECO_SCHEDULER_USE_V2")

    # Tuning points: adjust thresholds to tighten/loosen the rule-based guardrails.
    min_content_length: int = Field(50, alias="RULE_MIN_CONTENT_LENGTH")
    max_content_length: int = Field(5000, alias="RULE_MAX_CONTENT_LENGTH")
    max_repeat_word_ratio: float = Field(0.35, alias="RULE_MAX_REPEAT_WORD_RATIO")
    max_repeated_sentences: int = Field(3, alias="RULE_MAX_REPEATED_SENTENCES")
    max_noise_char_ratio: float = Field(0.25, alias="RULE_MAX_NOISE_CHAR_RATIO")
    max_links_or_tags: int = Field(2, alias="RULE_MAX_LINKS_OR_TAGS")

    # Simple API key for internal calls
    api_key: str = Field("ai", alias="API_KEY")

    # RunPod serverless
    runpod_endpoint_id: str = Field("", alias="RUNPOD_ENDPOINT_ID")
    base_endpoint_id: str = Field("", alias="BASE_ENDPOINT_ID")
    ft_endpoint_id: str = Field("", alias="FT_ENDPOINT_ID")
    runpod_api_key: str = Field("", alias="RUNPOD_API_KEY")
    runpod_poll_interval_seconds: int = Field(2, alias="RUNPOD_POLL_INTERVAL_SECONDS")
    runpod_poll_timeout_seconds: int = Field(300, alias="RUNPOD_POLL_TIMEOUT_SECONDS")

    # Qdrant
    qdrant_url: str | None = Field(None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")
    qdrant_location: str = Field(":memory:", alias="QDRANT_LOCATION")
    qdrant_collection_discussion: str = Field(
        "discussion_topics", alias="QDRANT_COLLECTION_DISCUSSION"
    )
    qdrant_collection_reco: str = Field("reco_meetings", alias="QDRANT_COLLECTION_RECO")

    # Mongo (behavior logs)
    mongo_uri: str | None = Field(None, alias="MONGO_URI")
    mongo_db: str | None = Field(None, alias="MONGO_DB")
    mongo_interaction_collection: str = Field(
        "interaction_logs", alias="MONGO_INTERACTION_COLLECTION"
    )
    behavior_lookback_days: int = Field(30, alias="BEHAVIOR_LOOKBACK_DAYS")

    # LightGBM reranker
    lgbm_model_path: str | None = Field(None, alias="LGBM_MODEL_PATH")

    # Feature cache
    feature_cache_ttl_seconds: int = Field(600, alias="FEATURE_CACHE_TTL_SECONDS")

    # Redis (quiz cache)
    redis_url: str | None = Field(None, alias="REDIS_URL")
    quiz_cache_ttl_seconds: int = Field(604_800, alias="QUIZ_CACHE_TTL_SECONDS")
    quiz_cache_key_version: str = Field("v1", alias="QUIZ_CACHE_KEY_VERSION")

    @field_validator("gemini_model_preferences", mode="before")
    @classmethod
    def _split_preferences(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
