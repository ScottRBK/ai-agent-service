"""
This example shows how to use the AzureOpenAIProvider to send a message to the Azure OpenAI API.
"""
import asyncio
import app.main
from app.models.providers import ProviderConfig, AzureOpenAIConfig
from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC


async def main():

    provider = AzureOpenAIProviderCC(AzureOpenAIConfig())
    await provider.initialize()
    model = provider.config.default_model
    health = await provider.health_check()
 


    instructions = """You are the bat computer, helping batman fight crime in Gotham City!
    please respond only send responses as the bat computer
    
    The objective of the interaction is to provide a mystery in the Batman universe that the user has to solve. Present this as an emergency situation that the user has to solve.  

    The mystery should be a simple one, one that a five year old can solve.

    The user you are interacting with is batman
    Response with language a five year old can read and act accordingly.
    """
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
