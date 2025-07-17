"""
Azure OpenAI Provider implementation.
Integrates with the Azure OpenAI API.
"""

import importlib.metadata
from app.core.providers.base import BaseProvider, ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.models.providers import AzureOpenAIConfig
from app.models.tools.tools import Tool
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from app.core.agents.agent_tool_manager import AgentToolManager
from app.utils.logging import logger
from openai import AsyncAzureOpenAI
from datetime import datetime
from pydantic import ValidationError
import json
from typing import Optional

class AzureOpenAIProvider(BaseProvider):
    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.config: AzureOpenAIConfig = config
        self.client: AsyncAzureOpenAI = None
        self.version = importlib.metadata.version("openai")

    async def initialize(self) -> None:
        """Initialise resources"""
        try:
            logger.info(f"Intializing AzureAI provider {self.config.name}")
            self.client = AsyncAzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.base_url,
                azure_ad_token=self.config.api_key,
            )
            logger.info(f"AzureAI provider {self.config.name} initialized successfully")

        except Exception as e:
            logger.warning(f"""Error during initialization 
                            AzureOpenAI Provider {self.config.name}: {e}""")
            
    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""
        try:
            # await self.client.responses.create(
            #     model=self.config.default_model,
            #     input=[{"role": "user", "content": "Hello, how are you?"}]
            # )
            logger.info(f"AzureAI provider {self.config.name} - health check")
            models = await self.client.models.list()
            health = HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, version=self.version)
            logger.debug(f"AzureAI provider {self.config.name} - health check - models: {models}")
            logger.info(f"AzureAI provider {self.config.name} - health check - completed {health}")
            return health   
        except Exception as e:
            health = HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, version=self.version, error_details=str(e))
            logger.debug(f"AzureAI provider {self.config.name} - health check - failed: {health}")
            return health

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.client:
                self.client = None
                logger.debug(f"AzureOpenAI Provider {self.config.name} cleaned up")
        except Exception as e:
            logger.warning(f"""Error during cleanup 
                            AzureOpenAI Provider {self.config.name} cleanup: {e}""")


    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None, agent_id: str = None, model_settings: Optional[dict] = None) -> str:
        """Send input to the provider and return the response."""

        logger.debug(f"AzureOpenAIProvider - send_chat - model: {model}")
        logger.debug(f"AzureOpenAIProvider - send_chat - instructions: {instructions}")
        logger.debug(f"AzureOpenAIProvider - send_chat - context: {context}")
        logger.debug(f"AzureOpenAIProvider - send_chat - agent_id: {agent_id}")
        logger.debug(f"AzureOpenAIProvider - send_chat - self.client: {self.client.base_url}")

        available_tools = await self.get_available_tools(agent_id, tools)
        logger.debug(f"AzureOpenAIProvider - send_chat - available tools: {available_tools}")

        request_params = {
            "model": model,
            "instructions": instructions,
            "input": context,
            "tools": available_tools
        }

        if model_settings:
            request_params.update(model_settings)

        response = await self.client.responses.create(**request_params)
        
        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            total_tool_iterations += 1
            tool_messages = []  # Reset tool_messages for each iteration
            
            for output in response.output:
                if output.type == "function_call":
                    logger.debug(f"AzureOpenAIProvider - send_chat - function_call: {output.model_dump_json()}")
                    tool_name = output.name
                    arguments = output.arguments
                    call_id = output.call_id
                    logger.debug(f"AzureOpenAIProvider - send_chat - executing tool call - tool_name: {tool_name}, arguments: {arguments}, call_id: {call_id}")
                    result = await self.execute_tool_call(tool_name, json.loads(arguments), agent_id)
                    tool_messages.append(output)
                    tool_messages.append({"type": "function_call_output", "call_id": call_id,"output": str(result)})

            if tool_messages:
                response = await self.client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=tool_messages,
                    tools=available_tools
                )
                await self.record_successful_call()
                logger.debug(f"""AzureOpenAIProvider - send_chat - Success: {self.success_requests}, 
                        Total: {self.total_requests}
                        /n messages: {tool_messages}
                        /n response: {response.output_text}""")
            else:
                break
                
        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"""AzureOpenAIProvider - send_chat - max tool iterations reached: {total_tool_iterations} - check tools and system prompt""")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)

        logger.debug(f"""AzureOpenAIProvider - send_chat - completed Total Requests: {self.total_requests}""")
        return response.output_text

    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass

    async def get_available_tools(self, agent_id: str = None, requested_tools: list[Tool] = None) -> list[dict]:
        """
        Get available tools for the agent, combining regular tools and MCP tools.
        Caches tools per agent_id to avoid repeated MCP server calls.
        """
        if agent_id:
            # Check cache first
            if agent_id in self.cached_tools:
                return self.cached_tools[agent_id]
            
            # Load and cache tools
            agent_manager = AgentToolManager(agent_id)
            tools = await agent_manager.get_available_tools()
            
            # Convert from chat completions format to response format
            response_tools = self.convert_to_response_format(tools)
            self.cached_tools[agent_id] = response_tools if response_tools else None
            return self.cached_tools[agent_id]
        elif requested_tools:
            tools_list = ToolRegistry.convert_tool_registry_to_response_format()
            tools = [tool for tool in tools_list if tool["name"] in requested_tools]
            return tools if tools else None
        return None

    def convert_to_response_format(self, chat_completions_tools: list[dict]) -> list[dict]:
        """
        Convert tools from chat completions format to response format.
        
        Chat completions format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "description",
                "parameters": {...}
            }
        }
        
        Response format:
        {
            "type": "function", 
            "name": "tool_name",
            "description": "description",
            "parameters": {...}
        }
        """
        if not chat_completions_tools:
            return []
        
        response_tools = []
        for tool in chat_completions_tools:
            if tool.get("type") == "function" and "function" in tool:
                response_tool = {
                    "type": "function",
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"]
                }
                response_tools.append(response_tool)
        
        return response_tools

    async def execute_tool_call(self, tool_name: str, arguments: dict, agent_id: str = None) -> str:
        """
        Execute a tool call using the AgentToolManager if agent_id is available,
        otherwise fall back to the ToolRegistry.
        """
        if agent_id:
            agent_manager = AgentToolManager(agent_id)
            return await agent_manager.execute_tool(tool_name, arguments)
        else:
            return ToolRegistry.execute_tool_call(tool_name, arguments)