"""
Configuration management for the Agent Service.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Application settings with environment variable support."""


    # Application Info
    SERVICE_NAME: str = "AI Agent Service"
    SERVICE_VERSION: str = "0.1.10"
    SERVICE_DESCRIPTION: str = "AI Agent Service that provides intelligent automation and AI-powered capabilities"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "DEBUG"

    # Configuration File Paths
    AGENT_CONFIG_PATH: str = "agent_config.json"
    MCP_CONFIG_PATH: str = "mcp.json"
    PROMPTS_DIR_PATH: str = "prompts"

    # AZURE OpenAI Configuration
    AZURE_OPENAI_BASE_URL: str = "https://{your-custom-endpoint}.openai.azure.com/"
    AZURE_OPENAI_DEFAULT_MODEL: str = "gpt-4.1-nano"
    AZURE_OPENAI_API_VERSION: str = "2025-03-01-preview"
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_MODEL_LIST: str = "gpt-4o-mini|gpt-4.1-nano"

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "qwen3:4b"

    # PostgreSQL configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres_user_here"
    POSTGRES_PASSWORD: str = "your_password_here"
    POSTGRES_DB: str = "postgres"

    # Note: MCP server tokens are handled dynamically via environment variable substitution
    # in ToolRegistry.load_mcp_servers(). No need to declare them here.
    # Example: ${GITHUB_TOKEN}, ${NEW_SERVICE_TOKEN}, etc. will be resolved automatically.

    """Pydantic configuration."""   
    model_config = ConfigDict(
        env_file = ".env",
        case_sensitive = True,
        extra = "ignore"  # Allow extra environment variables for MCP server tokens
    )

settings = Settings()