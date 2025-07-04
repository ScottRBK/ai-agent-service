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
    AZURE_OPENAI_CC = "azure_openai_cc" # Azure OpenAI Chat Completions

class ProviderConfig(BaseModel):
    """Base configuration for all LLM providers"""
    name: str
    provider_type: ProviderType
    timeout: int = 30
    max_retries: int = 3
    version: str = ""

    track_usage: bool = False
    log_requests: bool = False
    log_responses: bool = False

class OllamaConfig(ProviderConfig):
    "Ollama-specific configuration"
    name: str = "Ollama"
    provider_type: ProviderType = ProviderType.OLLAMA
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:4b")
    model_list: list[str] = ["qwen3:4b", "qwen3:8b", "qwen3:14b"]

class OpenAIConfig(ProviderConfig):
    "OpenAI-specific configuration"
    name: str = "OpenAI"
    provider_type: ProviderType = ProviderType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"
    model_list: list[str] = ["gpt-4o-mini", "gpt-4o"]

class AzureOpenAIConfig(ProviderConfig):
    "Azure-OpenAI-specific configuration"
    name: str = "AzureOpenAI"
    provider_type: ProviderType = ProviderType.AZURE_OPENAI
    base_url: str = os.getenv("AZURE_OPENAI_BASE_URL", "https://{your-custom-endpoint}.openai.azure.com/")
    default_model: str = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4.1-nano")
    model_list: list[str] = os.getenv("AZURE_OPENAI_MODEL_LIST", "gpt-4.1-nano,gpt-4o-mini").split(",")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
