import importlib.metadata
import json
from app.core.providers.base import BaseProvider, ProviderMaxToolIterationsError, ProviderConnectionError
from app.models.health import HealthStatus
from app.models.providers import OllamaConfig
from app.models.tools.tools import Tool
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from pydantic import ValidationError
from app.utils.logging import logger
from ollama import AsyncClient, ChatResponse, Message
from datetime import datetime
from typing import List, Optional, AsyncGenerator

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
    
    def _prepare_messages(self, instructions: str, context: list) -> list:
        """Prepare messages list with system instructions and context."""
        messages = []
        
        if instructions:
            messages.append({"role": "system", "content": instructions})
        
        messages.extend(context)
        return messages
    
    async def _process_streaming_response(self, response: ChatResponse) -> AsyncGenerator[tuple[str, str, list], None]:
        """Process streaming response chunks and yield content plus final results."""
        assistant_content = ""
        tool_calls_to_process = []
        
        async for chunk in response:
            if chunk.message.content:
                content_chunk = chunk.message.content
                assistant_content += content_chunk
                yield ("content", content_chunk, [])

            if chunk.message.tool_calls:
                logger.info(f"OllamaProvider - tool_calls: {chunk.message.tool_calls}")
                tool_calls_to_process.extend(chunk.message.tool_calls)
        
        yield ("final", assistant_content, tool_calls_to_process)

    async def _execute_tool_calls(self, messages: list, tool_calls_to_process: list, agent_id: str) -> int:
        """Execute tool calls and append results to messages. Returns number of tools executed."""
        tool_count = 0
        for tool_call in tool_calls_to_process:
            logger.info(f"OllamaProvider - tool_call: {tool_call.function.name}")
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }],
            })
            tool_result = await self.execute_tool_call(tool_call.function.name, tool_call.function.arguments, agent_id)               
            messages.append({
                "role": "tool",
                "content": str(tool_result)
            })
            tool_count += 1
        return tool_count

    async def _handle_tool_calls_streaming(self, messages: list, model: str, available_tools: list, 
                                         agent_id: str, model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Handle tool calling iterations for streaming responses."""
        total_tool_iterations = 0
        
        while total_tool_iterations < self.max_tool_iterations:
            request_params = {
                "model": model,
                "messages": messages,
                "tools": available_tools,
                "stream": True,
                **({"options": model_settings} if model_settings else {})
            }
            
            response: ChatResponse = await self.client.chat(**request_params)
            await self.record_successful_call()
            
            assistant_content = ""
            tool_calls_to_process = []
            
            async for chunk_type, content, tool_calls in self._process_streaming_response(response):
                if chunk_type == "content":
                    yield content
                elif chunk_type == "final":
                    assistant_content = content
                    tool_calls_to_process = tool_calls

            if assistant_content.strip():
                messages.append({"role": "assistant", "content": assistant_content})

            if not tool_calls_to_process:
                break

            tool_count = await self._execute_tool_calls(messages, tool_calls_to_process, agent_id)
            total_tool_iterations += tool_count

        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"OllamaProvider - max tool iterations reached: {total_tool_iterations}")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)
    
    async def send_chat_with_streaming(self, context: list, model: str, 
                        instructions: str, 
                        tools: list[Tool] = None, 
                        agent_id: str = None,
                        model_settings: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """Send input to the provider and return the response."""

        logger.debug(f"OllamaProvider - send_chat_with_streaming - model: {model}")
        logger.debug(f"OllamaProvider - send_chat_with_streaming - instructions: {instructions}")
        logger.debug(f"OllamaProvider - send_chat_with_streaming - context: {context}")
        logger.debug(f"OllamaProvider - send_chat_with_streaming - tools: {tools}")
        logger.debug(f"OllamaProvider - send_chat_with_streaming - agent_id: {agent_id}")
        logger.debug(f"OllamaProvider - send_chat_with_streaming - self.client: {self.config.base_url}")

        messages = self._prepare_messages(instructions, context)
        available_tools = await self.get_available_tools(agent_id, tools)
        logger.debug(f"OllamaProvider - send_chat_with_streaming - available tools: {len(available_tools) if available_tools else 0}")
        
        logger.info(f"OllamaProvider - send_chat_with_streaming - starting streaming request to ollama")
        
        async for content in self._handle_tool_calls_streaming(messages, model, available_tools, agent_id, model_settings):
            yield content

    async def send_chat(self, context: list, model: str, 
                        instructions: str, 
                        tools: list[Tool] = None, 
                        agent_id: str = None,
                        model_settings: Optional[dict] = None) -> str:
        """Send input to the provider and return the response."""

        messages = self._prepare_messages(instructions, context)
        available_tools = await self.get_available_tools(agent_id, tools)
        
        request_params = {
            "model": model,
            "messages": messages,
            "tools": available_tools
        }
        if model_settings:
            request_params["options"] = model_settings

        logger.info(f"OllamaProvider - send_chat - sending request to ollama")
        response: ChatResponse = await self.client.chat(**request_params)

        await self.record_successful_call()

        total_tool_iterations = 0
        for _ in range(self.max_tool_iterations):
            if response.message.tool_calls:
                
                messages.append(response.message)

                tool_count = await self._execute_tool_calls(messages, response.message.tool_calls, agent_id)
                total_tool_iterations += tool_count

                request_params["messages"] = messages
                response: ChatResponse = await self.client.chat(**request_params)

                await self.record_successful_call()
            else:
                break

        if total_tool_iterations >= self.max_tool_iterations:
            logger.error(f"""OllamaProvider - send_chat - max tool iterations reached: {total_tool_iterations} - check tools and system prompt""")
            raise ProviderMaxToolIterationsError(f"Max tool iterations reached: {total_tool_iterations}", self.config.name)

        logger.debug(f"""OllamaProvider - send_chat - completed Total Requests: {self.total_requests}""")
        return response.message.content
 
    async def embed(self, text: str, model: str) -> list[float]:
        """"Generate embeddings using Ollama"""
        response = await self.client.embed(
            model=model,
            input=text
        )
        
        # Validate response structure
        if not hasattr(response, 'embeddings'):
            raise AttributeError("Response missing 'embeddings' attribute")
        
        if response.embeddings is None:
            raise TypeError("Embeddings is None")
        
        if len(response.embeddings) == 0:
            raise IndexError("Embeddings list is empty")
        
        return response.embeddings[0]
    
    async def rerank(self, model: str, query: str, candidates: List[str]) -> List[float]:
        """
        Rerank candidates using specialized Ollama reranking models.
        
        Supports models like dengcao/Qwen3-Reranker-4B:Q8_0 which are specifically
        designed for re-ranking tasks.
        """
        if not self.client:
            raise ProviderConnectionError("Ollama client not initialized", self.name)
        
        try:
            scores = []
            
            # Process each candidate with the re-ranking model
            for candidate in candidates:
                # Format input for re-ranking model using system/user/assistant format
                # This format works better with Qwen re-ranking models
                rerank_input = (
                    f"<|system|>You are a relevance scoring model. Output only a decimal number between 0.0 and 1.0 "
                    f"where 0.0 means completely irrelevant and 1.0 means perfectly relevant."
                    f"<|user|>Query: {query}\nDocument: {candidate}<|assistant|>"
                )
                
                response = await self.client.generate(
                    model=model,
                    prompt=rerank_input,
                    options={
                        "temperature": 0.0,  # Deterministic scoring
                        "num_predict": 5,    # Only need a score
                        "stop": ["\n", " ", "<|"]  # Stop after score
                    }
                )
                
                # Extract score from response
                # Re-ranking models typically output a relevance score
                try:
                    score_text = response['response'].strip()
                    logger.debug(f"Raw rerank response: '{score_text}'")
                    
                    # Handle different score formats
                    # Try to extract numeric value from various formats
                    import re
                    # Look for patterns like "0.95", "Score: 0.95", "Relevance: 0.8", etc.
                    numeric_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', score_text)
                    if numeric_match:
                        score = float(numeric_match.group())
                        # Normalize to 0-1 range if needed
                        score = max(0.0, min(1.0, score))
                    else:
                        # If no numeric value found, use default
                        logger.warning(f"No numeric score found in rerank response: '{score_text}'")
                        score = 0.5
                    
                    scores.append(score)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not parse score from reranking model: {e}")
                    scores.append(0.5)  # Default middle score
            
            await self.record_successful_call()
            logger.info(f"Ollama reranked {len(candidates)} candidates with model {model}")
            return scores
            
        except Exception as e:
            logger.error(f"Ollama reranking failed: {e}")
            # Return equal scores on failure
            return [0.5] * len(candidates)