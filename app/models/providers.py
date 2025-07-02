"""
Provider configuration models and validation.
All Pydantic models for provider configurations.
"""

from pydantic import BaseModel
from enum import Enum

class ProviderType(str, Enum):
    """Supported provider types"""
    OLLAMA = "ollama"


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