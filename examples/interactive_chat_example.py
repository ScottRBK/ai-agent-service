"""
Interactive chat example with provider and model selection.
This example shows how to dynamically select providers and models for chat.

Generated with Claude-4-sonnet in cursor as a quick example (worked first time)
"""
import asyncio
import app.main
from typing import Dict, Type, Any

# Import all available providers and configs
from app.models.providers import OllamaConfig, AzureOpenAIConfig
from app.core.providers.ollama import OllamaProvider
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
from app.core.providers.base import BaseProvider


class ProviderManager:
    """Manages available providers and provides selection interface."""
    
    def __init__(self):
        self.providers = {
            "1": {
                "name": "Ollama",
                "description": "Local Ollama instance",
                "class": OllamaProvider,
                "config_class": OllamaConfig
            },
            "2": {
                "name": "Azure OpenAI",
                "description": "Azure OpenAI API (Legacy)",
                "class": AzureOpenAIProvider,
                "config_class": AzureOpenAIConfig
            },
            "3": {
                "name": "Azure OpenAI Chat Completions",
                "description": "Azure OpenAI Chat Completions API",
                "class": AzureOpenAIProviderCC,
                "config_class": AzureOpenAIConfig
            }
        }
    
    def display_providers(self) -> None:
        """Display available providers to the user."""
        print("\n" + "="*50)
        print("Available Providers:")
        print("="*50)
        for key, provider in self.providers.items():
            print(f"{key}. {provider['name']}")
            print(f"   {provider['description']}")
        print("="*50)
    
    async def get_user_provider_choice(self) -> str:
        """Get provider choice from user with validation."""
        while True:
            try:
                choice = await asyncio.to_thread(
                    input, 
                    f"\nSelect provider (1-{len(self.providers)}) or 'q' to quit: "
                )
                
                if choice.lower() == 'q':
                    print("Goodbye!")
                    exit(0)
                
                if choice in self.providers:
                    return choice
                else:
                    print(f"Invalid choice. Please select 1-{len(self.providers)} or 'q'")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                exit(0)
    
    async def create_provider(self, choice: str) -> BaseProvider:
        """Create and initialize the selected provider."""
        provider_info = self.providers[choice]
        config_class = provider_info["config_class"]
        provider_class = provider_info["class"]
        
        print(f"\nInitializing {provider_info['name']}...")
        
        try:
            config = config_class()
            provider = provider_class(config)
            await provider.initialize()
            print(f"✓ {provider_info['name']} initialized successfully")
            return provider
        except Exception as e:
            print(f"✗ Failed to initialize {provider_info['name']}: {e}")
            raise


class ModelSelector:
    """Handles model selection for providers."""
    
    @staticmethod
    async def display_models(provider: BaseProvider) -> None:
        """Display available models for the provider."""
        try:
            models = await provider.get_model_list()
            print(f"\n{'='*50}")
            print(f"Available models for {provider.name}:")
            print(f"{'='*50}")
            
            if isinstance(models, list) and len(models) > 0:
                for i, model in enumerate(models, 1):
                    # Handle different model formats
                    if isinstance(model, dict):
                        model_name = model.get('name', model.get('id', str(model)))
                    else:
                        model_name = str(model)
                    print(f"{i}. {model_name}")
            else:
                print("No models available or could not retrieve model list")
                print(f"Using default model: {provider.config.default_model}")
                return provider.config.default_model
            
            print(f"{'='*50}")
            return models
        except Exception as e:
            print(f"Error retrieving models: {e}")
            print(f"Using default model: {provider.config.default_model}")
            return [provider.config.default_model]
    
    @staticmethod
    async def get_user_model_choice(models: list, provider: BaseProvider) -> str:
        """Get model choice from user with validation."""
        if not models or len(models) == 0:
            return provider.config.default_model
        
        while True:
            try:
                choice = await asyncio.to_thread(
                    input, 
                    f"\nSelect model (1-{len(models)}) or press Enter for default ({provider.config.default_model}): "
                )
                
                if choice.strip() == "":
                    return provider.config.default_model
                
                try:
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(models):
                        selected_model = models[model_index]
                        # Handle different model formats
                        if isinstance(selected_model, dict):
                            return selected_model.get('name', selected_model.get('id', str(selected_model)))
                        else:
                            return str(selected_model)
                    else:
                        print(f"Invalid choice. Please select 1-{len(models)}")
                except ValueError:
                    print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\nUsing default model...")
                return provider.config.default_model


async def chat_loop(provider: BaseProvider, model: str) -> None:
    """Main chat interaction loop."""
    print(f"\n{'='*50}")
    print(f"Chat started with {provider.name} using model: {model}")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation")
    print(f"{'='*50}")
    
    instructions = "You are a helpful assistant. Please respond to the user's input."
    context = []
    
    try:
        # Get initial message
        message = await asyncio.to_thread(input, "\n\033[36mYou: \033[0m")
        
        if message.lower() in ['quit', 'exit']:
            return
        
        context.append({"role": "user", "content": message})
        
        while True:
            try:
                response = await provider.send_chat(
                    context=context,
                    model=model,
                    instructions=instructions,
                    tools=[]
                )
                
                print(f"\n\033[32m{provider.name}: \033[0m{response}\n")
                context.append({"role": "assistant", "content": response})
                
                message = await asyncio.to_thread(input, "\033[36mYou: \033[0m")
                
                if message.lower() in ['quit', 'exit']:
                    break
                
                context.append({"role": "user", "content": message})
                
            except Exception as e:
                print(f"\n\033[31mError during chat: {e}\033[0m")
                print("Please try again or type 'quit' to exit.")
                continue
                
    except (KeyboardInterrupt, EOFError):
        print("\n\nConversation ended.")
    finally:
        await provider.cleanup()


async def main():
    """Main application entry point."""
    print("\nInteractive Chat Example")
    print("This example allows you to select from available providers and models.")
    
    try:
        # Initialize provider manager
        provider_manager = ProviderManager()
        
        # Display and select provider
        provider_manager.display_providers()
        provider_choice = await provider_manager.get_user_provider_choice()
        
        # Create and initialize provider
        provider = await provider_manager.create_provider(provider_choice)
        
        # Display and select model
        model_selector = ModelSelector()
        models = await model_selector.display_models(provider)
        selected_model = await model_selector.get_user_model_choice(models, provider)
        
        print(f"\n✓ Selected: {provider.name} with model '{selected_model}'")
        
        # Start chat loop
        await chat_loop(provider, selected_model)
        
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")
    finally:
        print("\nThank you for using the interactive chat example!")


if __name__ == "__main__":
    asyncio.run(main()) 