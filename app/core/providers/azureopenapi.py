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
        """Cleanup resources and close connections."""
        pass

    async def get_model_list(self) -> list[str]:
        """Get a list of available models."""
        return self.config.model_list

    async def send_input(self, context: str, model: str, instructions: str, tools: list[Tool]) -> str:
        """Send input to the provider and return the response."""
        response = self.client.responses.create(
            model=model,
            instructions=instructions,
            input=context,
            tools=tools

        )
        self.total_requests += 1
        self.success_requests += 1
        self.last_successful_call = datetime.now()

        logging.debug(f"AzureOpenAIProvider - send_input - Success: {self.success_requests}, Total: {self.total_requests} /n response: {response.model_dump_json(indent=2)}")

        return response.output_text

    async def stream_input(self, context: str, model: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass
