"""
Azure OpenAI Provider implementation.
Integrates with the Azure OpenAI API.
"""

from app.core.providers.base import BaseProvider
from app.models.providers import AzureOpenAIConfig
from app.models.tools import Tool
from app.utils.logging import logging
from openai import AzureOpenAI
from datetime import datetime


class AzureOpenAIProvider(BaseProvider):
    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=config.base_url,
            azure_ad_token=config.api_key
        )
        self.config.model_list = self.client.models.list()
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.client:
                self.client = None
                logging.debug(f"AzureOpenAI Provider {self.config.name} cleaned up")
        except Exception as e:
            logging.warning(f"Error during cleanup AzureOpenAI Provider {self.config.name} cleanup: {e}")


    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Send input to the provider and return the response."""
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=context,
            tools=tools

        )

        await self.record_successful_call()

        logging.debug(f"AzureOpenAIProvider - send_input - Success: {self.success_requests}, Total: {self.total_requests} /n response: {response.model_dump_json(indent=2)}")

        return response.output_text

    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass
