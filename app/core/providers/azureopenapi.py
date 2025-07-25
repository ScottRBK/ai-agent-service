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
from typing import Optional, AsyncGenerator

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

    def _prepare_request_params(self, model: str, instructions: str, context: list, available_tools: list, model_settings: Optional[dict] = None) -> dict:
        """Prepare request parameters for Azure OpenAI Response API."""
        request_params = {
            "model": model,
            "instructions": instructions,
            "input": context,
            "tools": available_tools,
            **(model_settings if model_settings else {})
        }
        return request_params

    async def _execute_tool_calls(self, context: list, tool_calls: list, agent_id: str) -> int:
        """Execute tool calls and append results to context. Returns number of tools executed."""
        tool_count = 0
        for tool_call in tool_calls:
            logger.debug(f"AzureOpenAIProvider - executing tool call - tool_name: {tool_call.name}, arguments: {tool_call.arguments}, call_id: {tool_call.call_id}")
            
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"AzureOpenAIProvider - invalid JSON in tool arguments: {e}")
                error_result = f"Error: Invalid tool arguments format - {str(e)}"
                context.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": error_result})
                continue

            try:
                result = await self.execute_tool_call(tool_call.name, arguments, agent_id)
                context.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(result)})
                logger.debug(f"AzureOpenAIProvider - tool call result: {result}")
                tool_count += 1
            except Exception as e:
                logger.error(f"AzureOpenAIProvider - tool execution failed for {tool_call.name}: {e}")
                error_result = f"Error: Tool execution failed - {str(e)}"
                context.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": error_result})
        
        return tool_count

    async def _process_streaming_response(self, response) -> AsyncGenerator[tuple[str, str, list], None]:
        """Process streaming response chunks and yield content plus final results."""
        final_tool_calls = []
        completed_event = None
        assistant_response = []

        try:
            async for event in response:
                if event.type == "response.output_text.delta":
                    yield ("content", event.delta, [])

                if event.type == "response.completed":
                    completed_event = event

        except Exception as e:
            logger.error(f"AzureOpenAIProvider - _process_streaming_response - streaming error: {e}")
            yield ("error", f"Error: Streaming interrupted - {str(e)}", [])
            return

        if not completed_event or not completed_event.response:
            logger.warning("AzureOpenAIProvider - _process_streaming_response - no completed event received")
            yield ("final", "", [])
            return

        try:
            for output in completed_event.response.output:
                if output.type == "function_call":
                    final_tool_calls.append(output)
                    logger.debug(f"AzureOpenAIProvider - _process_streaming_response - final_tool_calls: {final_tool_calls}")

                if output.type == "message":
                    for content in output.content:
                        assistant_response.append(content.text)
                    logger.debug(f"AzureOpenAIProvider - _process_streaming_response - assistant_response: {assistant_response}")

        except Exception as e:
            logger.error(f"AzureOpenAIProvider - _process_streaming_response - error processing response output: {e}")
            yield ("error", f"Error: Failed to process response - {str(e)}", [])
            return

        yield ("final", "\\n".join(assistant_response), final_tool_calls)

    async def _handle_tool_calls_streaming(self, context: list, model: str, available_tools: list, 
                                         agent_id: str, model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Handle tool calling iterations for streaming responses."""
        total_tool_iterations = 0
        local_context = context.copy()
        
        # Extract instructions if present in context for Response API
        instructions = ""
        if local_context and local_context[0].get("role") == "system":
            instructions = local_context[0]["content"]
            local_context = local_context[1:]  # Remove system message from context

        while total_tool_iterations < self.max_tool_iterations:
            try:
                request_params = self._prepare_request_params(model, instructions, local_context, available_tools, model_settings)
                request_params["stream"] = True
                response = await self.client.responses.create(**request_params)
                await self.record_successful_call()

            except Exception as e:
                logger.error(f"AzureOpenAIProvider - _handle_tool_calls_streaming - API call failed: {e}")
                yield f"Error: Failed to communicate with Azure OpenAI - {str(e)}"
                return

            final_tool_calls = []
            assistant_response = ""

            async for chunk_type, content, tool_calls in self._process_streaming_response(response):
                if chunk_type == "content":
                    yield content
                elif chunk_type == "error":
                    yield content
                    return
                elif chunk_type == "final":
                    assistant_response = content
                    final_tool_calls = tool_calls

            if assistant_response:
                local_context.append({"role": "assistant", "content": assistant_response})

            if not final_tool_calls:
                break

            tool_count = await self._execute_tool_calls(local_context, final_tool_calls, agent_id)
            total_tool_iterations += tool_count

        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"AzureOpenAIProvider - _handle_tool_calls_streaming - max tool iterations reached: {total_tool_iterations}")
            yield f"Warning: Maximum tool iterations ({self.max_tool_iterations}) reached. Some tools may not have been executed."

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None, agent_id: str = None, model_settings: Optional[dict] = None) -> str:
        """Send input to the provider and return the response."""

        logger.debug(f"AzureOpenAIProvider - send_chat - model: {model}")
        logger.debug(f"AzureOpenAIProvider - send_chat - instructions: {instructions}")
        logger.debug(f"AzureOpenAIProvider - send_chat - context: {context}")
        logger.debug(f"AzureOpenAIProvider - send_chat - agent_id: {agent_id}")
        logger.debug(f"AzureOpenAIProvider - send_chat - self.client: {self.client.base_url}")

        available_tools = await self.get_available_tools(agent_id, tools)
        logger.debug(f"AzureOpenAIProvider - send_chat - available tools: {available_tools}")

        request_params = self._prepare_request_params(model, instructions, context, available_tools, model_settings)
        response = await self.client.responses.create(**request_params)
        
        total_tool_iterations = 0
        while total_tool_iterations < self.max_tool_iterations:
            total_tool_iterations += 1
            tool_messages = []  
            
            function_call_outputs = []
            for output in response.output:
                if output.type == "function_call":
                    logger.debug(f"AzureOpenAIProvider - send_chat - function_call: {output.model_dump_json()}")
                    function_call_outputs.append(output)
                    tool_messages.append(output)
            
            if function_call_outputs:
                await self._execute_tool_calls(tool_messages, function_call_outputs, agent_id)

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

    async def send_chat_with_streaming(self, context: list, model: str, instructions: str, tools: list[Tool] = None, agent_id: str = None, model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Send input to the provider and return the response."""

        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - model: {model}")
        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - instructions: {instructions}")
        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - context: {context}")
        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - agent_id: {agent_id}")
        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - self.client: {self.client.base_url}")

        available_tools = await self.get_available_tools(agent_id, tools)
        logger.debug(f"AzureOpenAIProvider - send_chat_with_streaming - available tools: {len(available_tools) if available_tools else 0}")
        
        logger.info(f"AzureOpenAIProvider - send_chat_with_streaming - starting streaming request to azure openai")
        
        # Prepare context with instructions - for Response API, instructions are passed separately
        context_with_instructions = context.copy()
        if instructions:
            # For Response API, we need to pass instructions separately, but include them in context for tool calls
            context_with_instructions.insert(0, {"role": "system", "content": instructions})
        
        async for content in self._handle_tool_calls_streaming(context_with_instructions, model, available_tools, agent_id, model_settings):
            yield content

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
