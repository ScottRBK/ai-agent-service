"""
OpenRouter Provider implementation.
Integrates with the OpenRouter API using the OpenAI-compatible Chat Completions API.
Aligns naming and flow with AzureOpenAIProviderCC for consistency.
"""

import importlib.metadata
import json
from app.core.providers.base import BaseProvider, ProviderMaxToolIterationsError, ProviderAPIError, ProviderConnectionError
from app.models.health import HealthStatus
from app.models.providers import OpenRouterConfig
from app.models.tools.tools import Tool
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from app.core.agents.agent_tool_manager import AgentToolManager
from app.utils.logging import logger
from openai import AsyncOpenAI
from datetime import datetime
from typing import Optional, AsyncGenerator, List

class OpenRouterProvider(BaseProvider):
    def __init__(self, config: OpenRouterConfig):
        super().__init__(config)
        self.config: OpenRouterConfig = config
        self.client: AsyncOpenAI = None
        try:
            self.version = importlib.metadata.version("openai")
        except Exception:
            self.version = "unknown"

    async def initialize(self) -> None:
        """Initialise resources"""
        try:
            logger.info(f"Intializing OpenRouter provider - initialize - {self.config.name}")
            if not self.config.api_key:
                raise ProviderConnectionError("Missing OPENROUTER_API_KEY", self.config.name)

            self.client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key
            )
            # populate model_list best-effort
            try:
                models = await self.client.models.list()
                names = [m.id for m in getattr(models, "data", []) or [] if getattr(m, "id", None)]
                if names:
                    self.config.model_list = names
            except Exception as e:
                logger.debug(f"OpenRouter provider - initialize - models.list failed: {e}")

            logger.info(f"OpenRouter provider - initialize - {self.config.name} initialized successfully")

        except Exception as e:
            logger.warning(f"""Error during initialization 
                            OpenRouter Provider - initialize - {self.config.name}: {e}""")

    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""
        try:
            await self.client.chat.completions.create(
                model=self.config.default_model,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=1
            )
            health = HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, version=self.version)

            return health
        except Exception as e:
            health = HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, version=self.version, error_details=str(e))

            return health

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.client:
                self.client = None

        except Exception as e:
            logger.warning(f"""Error during cleanup 
                            OpenRouter Provider {self.config.name} cleanup: {e}""")

    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    def _prepare_messages(self, instructions: str, context: list) -> list:
        """Prepare messages list with system instructions and context."""
        messages = []
        
        if instructions:
            messages.append({"role": "system", "content": instructions})
        
        messages.extend(context)
        return messages

    async def _parse_streaming_tool_calls(self, response) -> AsyncGenerator[tuple[str, str, list], None]:
        """Parse streaming response chunks and yield content plus final tool calls."""
        assistant_content = ""
        tool_calls = []
        current_tool_call = {"id": None, "name": None, "args": ""}
        arg_buffer = ""
        
        try:
            async for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield ("content", delta.content, [])
                        assistant_content += delta.content

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.id and tc.id != current_tool_call["id"]:
                                # New tool call, save the previous one if it exists
                                if current_tool_call["name"]:
                                    tool_calls.append(current_tool_call.copy())
                                current_tool_call = {"id": tc.id, "name": None, "args": ""}
                                arg_buffer = ""
                            
                            if tc.function.name:
                                current_tool_call["name"] = tc.function.name
                            if tc.function.arguments:
                                arg_buffer += tc.function.arguments
                                current_tool_call["args"] = arg_buffer
                        
                    if chunk.choices[0].finish_reason == "tool_calls":
                        break
        except Exception as e:
            logger.error(f"OpenRouterProvider - _parse_streaming_tool_calls - streaming error: {e}")
            yield ("error", f"Error: Streaming interrupted - {str(e)}", [])
            return

        if current_tool_call["name"]:
            tool_calls.append(current_tool_call)

        yield ("final", assistant_content, tool_calls)

    async def _execute_tool_calls(self, messages: list, tool_calls: list, agent_id: str) -> int:
        """Execute tool calls and append results to messages. Returns number of tools executed."""
        tool_count = 0
        for tool_call in tool_calls:
            if tool_call["name"]:
                # Validate JSON before parsing
                try:
                    args = json.loads(tool_call["args"])
                    logger.info(f"Tool call: {tool_call['name']} - {tool_call['args']}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in tool call args: {tool_call['args']}")
                    logger.error(f"Error: {e}")
                    continue

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["args"]
                        }
                    }]
                })             
                
                try:
                    tool_result = await self.execute_tool_call(tool_call["name"], args, agent_id)
                    logger.info(f"Tool result: {str(tool_result)}")

                    messages.append({       
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call["id"]
                    })
                    tool_count += 1
                except Exception as e:
                    logger.error(f"OpenRouterProvider - _execute_tool_calls - error executing tool call: {e}")
                    continue

        return tool_count

    async def _handle_tool_calls_streaming(self, messages: list, model: str, available_tools: list, 
                                         agent_id: str, model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Handle tool calling iterations for streaming responses."""
        total_tool_iterations = 0

        while total_tool_iterations < self.max_tool_iterations:
            try:
                request_params = {
                    "model": model,
                    "messages": messages,
                    "tools": available_tools,
                    "stream": True,
                    **(model_settings if model_settings else {})
                }

                response = await self.client.chat.completions.create(**request_params)
            except Exception as e:
                logger.error(f"OpenRouterProvider - _handle_tool_calls_streaming - API call failed: {e}")
                yield f"Error: Failed to communicate with OpenRouter - {str(e)}"
                return

            assistant_content = ""
            tool_calls = []

            async for chunk_type, content, parsed_tool_calls in self._parse_streaming_tool_calls(response):
                if chunk_type == "content":
                    yield content
                elif chunk_type == "error":
                    yield content
                    return
                elif chunk_type == "final":
                    assistant_content = content
                    tool_calls = parsed_tool_calls

            if assistant_content.strip():         
                messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls were made, break
            if not tool_calls:
                break

            tool_count = await self._execute_tool_calls(messages, tool_calls, agent_id)
            total_tool_iterations += tool_count

        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"OpenRouterProvider - _handle_tool_calls_streaming - max tool iterations reached: {total_tool_iterations}")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None, agent_id: str = None, model_settings: Optional[dict] = None) -> str:
        """Send input to the provider and return the response."""

        messages = self._prepare_messages(instructions, context)
        available_tools = await self.get_available_tools(agent_id, tools)
    

        request_params = {
            "model": model,
            "messages": messages,
            "tools": available_tools,
            **(model_settings if model_settings else {})
        }

        response = await self.client.chat.completions.create(**request_params)
        await self.record_successful_call()

        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            total_tool_iterations += 1
            if response.choices[0].message.tool_calls:

                messages.append(response.choices[0].message)

                for tool_call in response.choices[0].message.tool_calls:

                    logger.debug(f"OpenRouterProvider - send_chat - tool_call: {tool_call}")

                    tool_result = await self.execute_tool_call(tool_call.function.name, json.loads(tool_call.function.arguments), agent_id)
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.id
                    })
                    logger.debug(f"OpenRouterProvider - send_chat - tool result: {tool_result}")
                    
                request_params["messages"] = messages
                response = await self.client.chat.completions.create(**request_params)

                await self.record_successful_call()

            else:
                break
        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"""OpenRouterProvider - send_chat - max tool iterations reached: {total_tool_iterations} - check tools and system prompt""")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)
        
        logger.debug(f"""OpenRouterProvider - send_chat - completed Total Requests: {self.total_requests}""")
        return response.choices[0].message.content

    async def send_chat_with_streaming(self, context: list, model: str, instructions: str, tools: list[Tool] = None, agent_id: str = None, model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Send input to the provider and return the response."""
        

        messages = self._prepare_messages(instructions, context)
        available_tools = await self.get_available_tools(agent_id, tools)
        
        logger.info(f"OpenRouterProvider - send_chat_with_streaming - starting streaming request to openrouter")
        
        async for content in self._handle_tool_calls_streaming(messages, model, available_tools, agent_id, model_settings):
            yield content

    async def embed(self, text: str, model: str) -> list[float]:
        """Embed text using the provider."""
        response = await self.client.embeddings.create(
            model=model,
            input=text
        )
        
        # Validate response structure
        if not hasattr(response, 'data') or response.data is None:
            raise AttributeError("Response missing 'data' attribute")
        
        if len(response.data) == 0:
            raise IndexError("Response data is empty")
        
        if not hasattr(response.data[0], 'embedding') or response.data[0].embedding is None:
            raise AttributeError("Response data missing 'embedding' attribute")
        
        return response.data[0].embedding
    
    async def rerank(self, model: str, query: str, candidates: List[str]) -> List[float]:
        return await super().rerank(model, query, candidates)