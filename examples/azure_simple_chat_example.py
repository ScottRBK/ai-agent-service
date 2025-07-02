import asyncio
import app.main
from app.models.providers import ProviderConfig, AzureOpenAIConfig
from app.core.providers.azureopenapi import AzureOpenAIProvider
provider = AzureOpenAIProvider(AzureOpenAIConfig())



instructions = "You are a helpful assistant. Please respond to the user's input."
context = []

async def main():
    message = await asyncio.to_thread(input, "You:")
    context.append({"role": "user", "content": message})

    while True:
        response = await provider.send_input(
            context,
            provider.config.default_model,
        instructions,
        []
    )
        await asyncio.to_thread(print, response)
        context.append({"role": "assistant", "content": response})
        message = await asyncio.to_thread(input, "You:")
        context.append({"role": "user", "content": message})

if __name__ == "__main__":
    asyncio.run(main())
