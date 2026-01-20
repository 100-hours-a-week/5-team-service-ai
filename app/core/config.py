from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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

    # Tuning points: adjust thresholds to tighten/loosen the rule-based guardrails.
    min_content_length: int = Field(50, alias="RULE_MIN_CONTENT_LENGTH")
    max_content_length: int = Field(5000, alias="RULE_MAX_CONTENT_LENGTH")
    max_repeat_word_ratio: float = Field(0.35, alias="RULE_MAX_REPEAT_WORD_RATIO")
    max_repeated_sentences: int = Field(3, alias="RULE_MAX_REPEATED_SENTENCES")
    max_noise_char_ratio: float = Field(0.25, alias="RULE_MAX_NOISE_CHAR_RATIO")
    max_links_or_tags: int = Field(2, alias="RULE_MAX_LINKS_OR_TAGS")

    @field_validator("gemini_model_preferences", mode="before")
    @classmethod
    def _split_preferences(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
