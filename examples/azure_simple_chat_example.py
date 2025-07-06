"""
This example shows how to use the AzureOpenAIProvider to send a message to the Azure OpenAI API.
"""
import asyncio
import app.main
from app.models.providers import ProviderConfig, AzureOpenAIConfig
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.utils.logging import logger
from app.config.settings import settings

async def main():

    provider = AzureOpenAIProvider(AzureOpenAIConfig())
    await provider.initialize()
    logger.debug(f"Settings: {settings.AZURE_OPENAI_BASE_URL}")
    logger.debug(f"Provider Config: {provider.config}")
    model = provider.config.default_model
    health = await provider.health_check()
    logger.debug(f"Health: {health}")
    logger.debug(f"Provider Config: {provider.config}")    

    instructions = "You are a helpful assistant. Please respond to the user's input."
    context = []
    message = await asyncio.to_thread(input, "You:")
    
    context.append({"role": "user", "content": message})

    while True:
        response = await provider.send_chat(
            model=model,
            instructions=instructions,
            context=context,
            tools=[]
        )
        await asyncio.to_thread(print, f"\n\033[31m{provider.name}: \033[32m{response}\n")
        context.append({"role": "assistant", "content": response})
        message = await asyncio.to_thread(input, "\033[37mYou:")
        context.append({"role": "user", "content": message})

if __name__ == "__main__":
    asyncio.run(main())
