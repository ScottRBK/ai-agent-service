"""
Configuration management for the Agent Service.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""


    # Application Info
    SERVICE_NAME: str = "Agent Service"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_DESCRIPTION: str = "A lightweight microservice"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "DEBUG"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True

settings = Settings()