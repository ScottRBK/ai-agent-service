"""
Base provider interface for LLM providers.
Abstract base class defining the standard interface for all providers.
"""
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from app.models.providers import ProviderConfig
from app.models.tools import Tool
from app.models.health import HealthStatus

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
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
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
    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        pass

    @abstractmethod
    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        pass

    @abstractmethod 
    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Send input to the provider and return the response."""
        pass

    @abstractmethod
    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass

    async def record_successful_call(self) -> None:
        """Record a successful call."""
        self.total_requests += 1
        self.success_requests += 1
        self.last_successful_call = datetime.now()
