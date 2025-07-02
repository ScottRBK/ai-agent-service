"""
Provider configuration models and validation.
All Pydantic models for provider configurations.
"""
import os
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class ProviderType(str, Enum):
    """Supported provider types"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


class ProviderConfig(BaseModel):
    """Base configuration for all LLM providers"""
    name: str
    provider_type: ProviderType
    timeout: int = 30
    max_retries: int = 3

    track_usage: bool = False
    log_requests: bool = False
    log_responses: bool = False

class OllamaConfig(ProviderConfig):
    "Ollama-specific configuration"
    provider_type: ProviderType = ProviderType.OLLAMA
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    model_list: list[str] = ["llama3.1:8b", "llama3.1:70b", "qwen3:8b"]

class OpenAIConfig(ProviderConfig):
    "OpenAI-specific configuration"
    provider_type: ProviderType = ProviderType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"
    model_list: list[str] = ["gpt-4o-mini", "gpt-4o"]

class AzureOpenAIConfig(ProviderConfig):
    "Azure-OpenAI-specific configuration"
    provider_type: ProviderType = ProviderType.AZURE_OPENAI
    base_url: str = os.getenv("AZURE_OPENAI_BASE_URL", "https://{your-custom-endpoint}.openai.azure.com/")
    default_model: str = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4.1-nano")
    model_list: list[str] = ["gpt-4.1-nano", "gpt-4o"]
    api_version: str = "2025-03-01-preview"
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
