"""
This example shows how to use the OllamaProvider to send a message to the Ollama API.
"""
import asyncio
import app.main
from app.models.providers import ProviderConfig, OllamaConfig
from app.core.providers.ollama import OllamaProvider


async def main():
    provider = OllamaProvider(OllamaConfig())

    instructions = "You are a helpful assistant. Please respond to the user's input."
    context = []

    message = await asyncio.to_thread(input, "You:")
    context.append({"role": "user", "content": message})

    while True:
        response = await provider.send_chat(
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
