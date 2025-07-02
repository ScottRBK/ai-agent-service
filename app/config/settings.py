"""
Configuration management for the Agent Service.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Application settings with environment variable support."""


    # Application Info
    SERVICE_NAME: str = "AI Agent Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = "AI Agent Service that provides intelligent automation and AI-powered capabilities"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "DEBUG"

    # AZURE OpenAI Configuration
    AZURE_OPENAI_BASE_URL: str = "https://{your-custom-endpoint}.openai.azure.com/"
    AZURE_OPENAI_DEFAULT_MODEL: str = "gpt-4.1-nano"
    AZURE_OPENAI_API_VERSION: str = "2025-03-01-preview"
    AZURE_OPENAI_API_KEY: str = ""

    """Pydantic configuration."""   
    model_config = ConfigDict(
        env_file = ".env",
        case_sensitive = True
    )

settings = Settings()