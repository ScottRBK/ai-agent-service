"""
Azure OpenAI Provider implementation.
Integrates with the Azure OpenAI API using the Chat Completions API.
"""

import importlib.metadata
from app.core.providers.base import BaseProvider    
from app.models.health import HealthStatus
from app.models.providers import AzureOpenAIConfig
from app.models.tools import Tool
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
            logger.info(f"Intializing AzureAI provider {self.config.name}")
            self.client = AsyncAzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.base_url,
                azure_ad_token=self.config.api_key
            )
            
            logger.info(f"AzureAI provider {self.config.name} initialized successfully")

        except Exception as e:
            logger.warning(f"""Error during initialization 
                            AzureOpenAI Provider {self.config.name}: {e}""")
            
    async def health_check(self) -> HealthStatus:
        """Check the health of the provider."""
        try:
            await self.client.chat.completions.create(
                model=self.config.default_model,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=1
            )
            return HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, version=self.version)
        except Exception as e:
            return HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, version=self.version, error_details=str(e))

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

        messages.extend(context)

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        await self.record_successful_call()

        logger.debug(f"""AzureOpenAIProviderCC - send_chat - Success: {self.success_requests}, 
                      Total: {self.total_requests} 
                      /n response: {response.model_dump_json(indent=2)}""")

        return response.choices[0].message.content

    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool]) -> str:
        """Stream input to the provider and yield the response."""
        pass
