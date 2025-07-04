import importlib.metadata
from app.core.providers.base import BaseProvider
from app.models.health import HealthStatus
from app.models.providers import OllamaConfig
from app.models.tools import Tool
from app.utils.logging import logger
from ollama import AsyncClient
from datetime import datetime

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
            return HealthStatus(status="healthy", timestamp=datetime.now(), service=self.config.name, version=self.version)
        except Exception as e:
            return HealthStatus(status="unhealthy", timestamp=datetime.now(), service=self.config.name, version=self.version, error_details=str(e))

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

    async def send_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None) -> str:
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

        logger.debug(f"""OllamaProvider - send_chat - Success: {self.success_requests}, 
                      Total: {self.total_requests} 
                      /n response: {response.model_dump_json(indent=2)}""")

        return response['message']['content']
    
    async def stream_chat(self, context: list, model: str, instructions: str, tools: list[Tool] = None) -> str:
        """Stream input to the provider and yield the response."""
        pass