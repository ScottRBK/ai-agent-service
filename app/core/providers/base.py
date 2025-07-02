"""
Base provider interface for LLM providers.
Abstract base class defining the standard interface for all providers.
"""
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from app.models.providers import ProviderConfig

class ProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, provider_name: str, error_code: Optional[str] = None):
        self.message = message
        self.provider_name = provider_name
        self.error_code = error_code
        super().__init__(f"[{provider_name}] {message}")


class ProviderConnectionError(ProviderError):
    """Raised when provider connection fails."""
    pass


class ProviderAPIError(ProviderError):
    """Raised when provider API returns an error."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when provider request times out."""
    pass


class ProviderModelNotFoundError(ProviderError):
    """Raised when requested model is not available."""
    pass


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    def __init__(self, config: ProviderConfig, name: str):
        self.config = config
        self.name = name
        self.provider_type = config.provider_type.value

        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.last_successful_call: Optional[datetime] = None
        self.last_error: Optional[ProviderError] = None
        self.error_count = 0

        self.client = None
        self.initialized = False

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        pass

    @abstractmethod
    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        pass


