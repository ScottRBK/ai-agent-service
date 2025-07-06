"""
Azure OpenAI Provider implementation.
Integrates with the Azure OpenAI API using the Chat Completions API.
"""

import importlib.metadata
import json
from app.core.providers.base import BaseProvider, ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.models.providers import AzureOpenAIConfig
from app.models.tools.tools import Tool
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from pydantic import ValidationError
from app.utils.logging import logger
from openai import AsyncAzureOpenAI
from datetime import datetime


class AzureOpenAIProviderCC(BaseProvider):
    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.config: AzureOpenAIConfig = config
        self.client: AsyncAzureOpenAI = None
        self.version = importlib.metadata.version("openai")

    async def initialize(self) -> None:
        """Initialise resources"""
        try:
            logger.info(f"Intializing AzureAI provider - initialize - {self.config.name}")
            self.client = AsyncAzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.base_url,
                azure_ad_token=self.config.api_key
            )
            
            logger.info(f"AzureAI provider - initialize - {self.config.name} initialized successfully")

        except Exception as e:
            logger.warning(f"""Error during initialization 
                            AzureOpenAI Provider - initialize - {self.config.name}: {e}""")
            
    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""
        try:
            await self.client.chat.completions.create(
                model=self.config.default_model,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=1
            )
            health = HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, version=self.version)
            logger.debug(f"AzureOpenAI Provider {self.config.name} - health_check - health: {health}")
            return health
        except Exception as e:
            health = HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, version=self.version, error_details=str(e))
            logger.debug(f"AzureOpenAI Provider {self.config.name} - health_check - health: {health}")
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

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None) -> str:
        """Send input to the provider and return the response."""
        messages = []

        logger.debug(f"AzureOpenAIProviderCC - send_chat - model: {model}")
        logger.debug(f"AzureOpenAIProviderCC - send_chat - instructions: {instructions}")
        logger.debug(f"AzureOpenAIProviderCC - send_chat - context: {context}")
        logger.debug(f"AzureOpenAIProviderCC - send_chat - tools: {tools}")
        logger.debug(f"AzureOpenAIProviderCC - send_chat - self.client: {self.client.base_url}")

        if instructions:
            messages.append({"role": "system", "content": instructions})

        registered_tools = None
        
        if tools:
            tools_list = ToolRegistry.convert_tool_registry_to_chat_completions_format()
            registered_tools = [tool for tool in tools_list if tool["function"]["name"] in tools]
            logger.debug(f"AzureOpenAIProviderCC - send_chat - registered tools: {registered_tools}")

        messages.extend(context)

        response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=registered_tools
            )
        await self.record_successful_call()

        logger.debug(f"""AzureOpenAIProviderCC - send_chat - Success: {self.success_requests}, 
                        Total: {self.total_requests}
                        /n messages: {messages}
                        /n response: {response.choices[0].message.content}""")
        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            total_tool_iterations += 1
            if response.choices[0].message.tool_calls:

                messages.append(response.choices[0].message)

                for tool_call in response.choices[0].message.tool_calls:

                    logger.debug(f"AzureOpenAIProviderCC - send_chat - tool_call: {tool_call}")

                    tool_result = ToolRegistry.execute_tool_call(tool_call.function.name, json.loads(tool_call.function.arguments))
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.id
                    })
                    logger.debug(f"AzureOpenAIProviderCC - send_chat - tool result: {tool_result}")
                    
                response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=registered_tools)

                await self.record_successful_call()
                
                logger.debug(f"""AzureOpenAIProviderCC - send_chat - Success: {self.success_requests}, 
                        Total: {self.total_requests}
                        /n messages: {messages}
                        /n response: {response.choices[0].message.content}""")
            else:
                break
        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"""AzureOpenAIProviderCC - send_chat - max tool iterations reached: {total_tool_iterations} - check tools and system prompt""")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)
        
        logger.debug(f"""AzureOpenAIProviderCC - send_chat - completed Total Requests: {self.total_requests}""")
        return response.choices[0].message.content

    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass
