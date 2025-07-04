"""
Manager class that handles the management of providers and their interactions
The class is responsible for generating and managing provider instances, as well as handling their lifecycle and interactions.
methods
- list_providers: Returns a list of all registered provider instances.
- get_provider: Retrieves a specific provider instance.
- create_provider: Adds a new provider instance.
- remove_provider: Removes a provider instance.
"""

from app.models.providers import OllamaConfig, AzureOpenAIConfig
from app.core.providers.ollama import OllamaProvider
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
from app.core.providers.base import BaseProvider

class ProviderManager:
    def __init__(self):
        self.providers = {
            "ollama": {
                "name": "Ollama",
                "description": "Local Ollama instance",
                "class": OllamaProvider,
                "config_class": OllamaConfig
            },
            "azure_openai": {
                "name": "Azure OpenAI",
                "description": "Azure OpenAI API (Legacy)",
                "class": AzureOpenAIProvider,
                "config_class": AzureOpenAIConfig
            },
            "azure_openai_cc": {
                "name": "Azure OpenAI Chat Completions",
                "description": "Azure OpenAI Chat Completions API",
                "class": AzureOpenAIProviderCC,
                "config_class": AzureOpenAIConfig
            }
        }

    def list_providers(self):
        """Returns a list of all registered provider instances."""
        return self.providers

    def get_provider(self, provider_id):
        """Retrieves a specific provider instance."""
        return self.providers.get(provider_id)

    def create_provider(self, provider_id, name, description, provider_class, config_class):
        """Adds a new provider instance."""
        self.providers[provider_id] = {
            "name": name,
            "description": description,
            "class": provider_class,
            "config_class": config_class
        }

    def remove_provider(self, provider_id):
        """Removes a provider instance."""
        self.providers.pop(provider_id, None)
