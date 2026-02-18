"""
Application configuration management.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""

    default_provider: str = Field(default="google", description="Default LLM provider")
    default_model: str = Field(default="gemini-2.5-flash", description="Default model name")
    temperature: float = Field(default=0.7, description="Default temperature")


class ExamHelperConfig(BaseModel):
    """Configuration for exam helper specific settings."""

    max_response_words: int = Field(default=200, description="Maximum words in response")


class AppConfig(BaseModel):
    """Main application configuration."""

    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode flag")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    exam_helper: ExamHelperConfig = Field(default_factory=ExamHelperConfig)


class AppConfigLoader:
    """Singleton loader for application configuration."""

    _instance: Optional[AppConfig] = None

    @classmethod
    def load_config(cls) -> AppConfig:
        """Load configuration from environment variables."""
        if cls._instance is None:
            cls._instance = AppConfig(
                environment=os.getenv("ENVIRONMENT", "development"),
                debug=os.getenv("DEBUG", "false").lower() == "true",
                llm=LLMConfig(
                    default_provider=os.getenv("LLM_PROVIDER", "google"),
                    default_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                ),
                exam_helper=ExamHelperConfig(
                    max_response_words=int(os.getenv("MAX_RESPONSE_WORDS", "200")),
                ),
            )
        return cls._instance

    @classmethod
    def app_config(cls) -> AppConfig:
        """Get the current application configuration."""
        return cls.load_config()

    @classmethod
    def reset(cls) -> None:
        """Reset the configuration (useful for testing)."""
        cls._instance = None
