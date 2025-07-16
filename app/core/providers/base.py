"""
Base provider interface for LLM providers.
Abstract base class defining the standard interface for all providers.
"""
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

from app.models.providers import ProviderConfig
from app.models.tools.tools import Tool
from app.models.health import HealthStatus
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.tools.tool_registry import ToolRegistry
import json

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

class ProviderMaxToolIterationsError(ProviderError):
    """Raised when max tool iterations is reached."""
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
        self.max_tool_iterations = 10

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

    async def get_available_tools(self, agent_id: str = None, requested_tools: list[Tool] = None) -> list[dict]:
        """
        Get available tools for the agent, combining regular tools and MCP tools.
        """
        if agent_id:
            agent_manager = AgentToolManager(agent_id)
            tools = await agent_manager.get_available_tools()
            return tools if tools else None
        elif requested_tools:
            tools_list = ToolRegistry.convert_tool_registry_to_chat_completions_format()
            tools = [tool for tool in tools_list if tool["function"]["name"] in requested_tools]
            return tools if tools else None
        return None

    async def execute_tool_call(self, tool_name: str, arguments: dict, agent_id: str = None) -> str:
        """
        Execute a tool call with extracted name and arguments.
        Each provider extracts these from their own API structure.
        """
        if agent_id:
            agent_manager = AgentToolManager(agent_id)
            return await agent_manager.execute_tool(tool_name, arguments)
        else:
            return ToolRegistry.execute_tool_call(tool_name, arguments)
