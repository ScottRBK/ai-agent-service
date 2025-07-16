import importlib.metadata
import json
from app.core.providers.base import BaseProvider, ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.models.providers import OllamaConfig
from app.models.tools.tools import Tool
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from pydantic import ValidationError
from app.utils.logging import logger
from ollama import AsyncClient, ChatResponse, Message
from datetime import datetime
from typing import Sequence

class OllamaProvider(BaseProvider):
    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.client = AsyncClient(host=config.base_url)
        self.config: OllamaConfig = config
        self.version = importlib.metadata.version("ollama")
        # self.config.model_list = self.client.list()


    async def initialize(self) -> None:
        """Initialize resources."""
        try:
            
            logger.info(f"Initializing Ollama Provider {self.config.name}")
            self.client = AsyncClient(host=self.config.base_url)
            models = []
            models_list = await self.client.list()
            for model in models_list:
                models.append(model[1][1].model)
            self.config.model_list = models
            logger.info(f"Ollama Provider {self.config.name} initialized successfully")

        except Exception as e:
            logger.warning(f"Error during initialization Ollama Provider {self.config.name}: {e}")

    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""     
        try:
            await self.client.list()
            return HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, 
                                version=self.version)
        except Exception as e:
            return HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, 
                                version=self.version, error_details=str(e))

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.client:
                self.client = None
                logger.debug(f"Ollama Provider {self.config.name} cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup Ollama Provider {self.config.name} cleanup: {e}")

    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    async def send_chat(self, context: list, model: str, 
                        instructions: str, 
                        tools: list[Tool] = None, 
                        agent_id: str = None) -> str:
        """Send input to the provider and return the response."""
        messages = []

        logger.debug(f"OllamaProvider - send_chat - model: {model}")
        logger.debug(f"OllamaProvider - send_chat - instructions: {instructions}")
        logger.debug(f"OllamaProvider - send_chat - context: {context}")
        logger.debug(f"OllamaProvider - send_chat - tools: {tools}")
        logger.debug(f"OllamaProvider - send_chat - agent_id: {agent_id}")
        logger.debug(f"OllamaProvider - send_chat - self.client: {self.config.base_url}")

        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Get available tools using AgentToolManager if agent_id is provided
        available_tools = await self.get_available_tools(agent_id, tools)
        logger.debug(f"OllamaProvider - send_chat - available tools: {len(available_tools) if available_tools else 0}")

        messages.extend(context)

        response: ChatResponse = await self.client.chat(
            model=model,
            messages=messages,
            tools=available_tools
        )
        await self.record_successful_call()
        logger.debug(f"""OllamaProvider - send_chat - Success: {self.success_requests}, 
                        Total: {self.total_requests}
                        /n messages: {messages}
                        /n response: {response.message.content}""")
        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            if response.message.tool_calls:
                messages.append(response.message)

                logger.debug(f"OllamaProvider - send_chat - tool_calls: {response.message.tool_calls}")

                for tool_call in response.message.tool_calls:
                    tool_result = await self.execute_tool_call(tool_call.function.name, tool_call.function.arguments, agent_id)
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result)
                    })

                response: ChatResponse = await self.client.chat(
                    model=model,
                    messages=messages,
                    tools=available_tools
                )
                await self.record_successful_call()
                logger.debug(f"""OllamaProvider - send_chat - Success: {self.success_requests}, 
                        Total: {self.total_requests}
                        /n messages: {messages}
                        /n response: {response.message.content}""")
                total_tool_iterations += 1
            else:
                break

        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"""OllamaProvider - send_chat - max tool iterations reached: {total_tool_iterations} - check tools and system prompt""")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)

        logger.debug(f"""OllamaProvider - send_chat - completed Total Requests: {self.total_requests}""")
        return response.message.content
    
    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None) -> str:
        """Stream input to the provider and yield the response."""
        pass
