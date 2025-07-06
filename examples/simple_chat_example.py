"""
This example shows how to use the AzureOpenAIProvider to send a message to the Azure OpenAI API.
"""
import asyncio
import app.main
import questionary
from app.core.tools.function_calls.date_tool import DateTool
from app.core.tools.function_calls.arithmetic_tool import ArithmeticTool
from app.core.providers.manager import ProviderManager
from app.utils.logging import logger

async def select_provider(provider_manager: ProviderManager):
    """Displays the available providers and prompts the user to select them in the CLI."""
    providers = provider_manager.list_providers()
    provider_choices = [
        questionary.Choice(title=info['name'], value=key)
        for key, info in providers.items()
    ]
    selected_provider = await asyncio.to_thread(
        lambda: questionary.select(
            "Please select a provider:",
            choices=provider_choices
        ).ask()
    )
    print(f"You selected: {selected_provider}")
    return selected_provider

async def select_model(provider):
    """Prompts the user to select a model from the provider."""
    models = await provider.get_model_list()
    for model in models:
        print(model)
    model_choices = [
        questionary.Choice(title=model, value=model) for model in models
    ]
    selected_model = await asyncio.to_thread(
        lambda: questionary.select(
            "Please select a model:",
            choices=model_choices
        ).ask()
    )
    print(f"You selected model: {selected_model}")
    return selected_model

async def main():

    provider_manager = ProviderManager()
    
    selected_provider = await select_provider(provider_manager)
    
    if not selected_provider:
        print("No provider selected. Exiting.")
        return
        
    provider_info = provider_manager.get_provider(selected_provider)
    provider_class = provider_info["class"]
    config_class = provider_info["config_class"]
    provider = provider_class(config_class())
    await provider.initialize()

    model = await select_model(provider)
    if not model:
        print(f"No model selected. Using default model {provider.config.default_model}.")
  
    instructions = "You are a helpful assistant. Please respond to the user's input."
    # tools = [DateTool.get_current_datetime_json_schema()]
    # available_functions = {"get_current_datetime": DateTool.get_current_datetime}

    context = []
    message = await asyncio.to_thread(input, "You:")

    tools = ["get_current_datetime"]
    
    context.append({"role": "user", "content": message})

    while True:
        response = await provider.send_chat(
            model=model,
            instructions=instructions,
            context=context,
            tools=tools,
            # available_functions=available_functions
        )
        await asyncio.to_thread(print, f"\n\033[31m{provider.name}: \033[32m{response}\n")
        context.append({"role": "assistant", "content": response})
        message = await asyncio.to_thread(input, "\033[37mYou:")
        context.append({"role": "user", "content": message})

if __name__ == "__main__":
    asyncio.run(main())
