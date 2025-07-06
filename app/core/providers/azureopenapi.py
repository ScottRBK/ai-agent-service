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
from app.utils.logging import logger
from openai import AsyncAzureOpenAI
from datetime import datetime
from pydantic import ValidationError
import json

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

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Send input to the provider and return the response."""

        logger.debug(f"AzureOpenAIProvider - send_chat - model: {model}")
        logger.debug(f"AzureOpenAIProvider - send_chat - instructions: {instructions}")
        logger.debug(f"AzureOpenAIProvider - send_chat - context: {context}")
        logger.debug(f"AzureOpenAIProvider - send_chat - tools: {tools}")
        logger.debug(f"AzureOpenAIProvider - send_chat - self.client: {self.client.base_url}")
        
        registered_tools = None
        if tools:
            tools_list = ToolRegistry.convert_tool_registry_to_response_format()
            registered_tools = [tool for tool in tools_list if tool["name"] in tools_list]
            logger.debug(f"AzureOpenAIProvider - send_chat - registered tools: {registered_tools}")

        response = await self.client.responses.create(
            model=model,
            instructions=instructions,
            input=context,
            tools=registered_tools

        )
        tool_messages = []
        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            total_tool_iterations += 1
            for output in response.output:
                if output.type == "function_call":
                    logger.debug(f"AzureOpenAIProvider - send_chat - function_call: {output.model_dump_json(indent=2)}")
                    tool_name = output.name
                    arguments = output.arguments
                    call_id = output.call_id
                    logger.debug(f"AzureOpenAIProvider - send_chat - executing tool call - tool_name: {tool_name}, arguments: {arguments}, call_id: {call_id}")
                    result = ToolRegistry.execute_tool_call(tool_name, json.loads(arguments))
                    tool_messages.append(output)
                    tool_messages.append({"type": "function_call_output", "call_id": call_id,"output": str(result)})

            if tool_messages:
                response = await self.client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=tool_messages,
                    tools=registered_tools
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


    
    async def execute_tool_call(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool_entry = TOOL_REGISTRY[tool_name]
        params_model = tool_entry["params_model"]
        implementation = tool_entry["implementation"]

        try:
            validated_args = params_model(**arguments)
        except ValidationError as e:
            raise ValueError(f"Argument validation error: {e}")

        return implementation(**validated_args.model_dump())