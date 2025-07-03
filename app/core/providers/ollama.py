from app.core.providers.base import BaseProvider
from app.models.providers import OllamaConfig
from app.models.tools import Tool
from app.utils.logging import logging
from ollama import AsyncClient
from datetime import datetime

class OllamaProvider(BaseProvider):
    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.client = AsyncClient(host=config.base_url)
        self.config: OllamaConfig = config
        # self.config.model_list = self.client.list()

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.client:
                self.client = None
                logging.debug(f"Ollama Provider {self.config.name} cleaned up")
        except Exception as e:
            logging.warning(f"Error during cleanup Ollama Provider {self.config.name} cleanup: {e}")

    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Send input to the provider and return the response."""
        messages = []

        if instructions:
            messages.append({"role": "system", "content": instructions})

        messages.extend(context)

        response = await self.client.chat(
            model=model,
            messages=messages,
        )

        await self.record_successful_call()

        logging.debug(f"OllamaProvider - send_chat - Success: {self.success_requests}, Total: {self.total_requests} /n response: {response.model_dump_json(indent=2)}")

        return response['message']['content']
    
    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass