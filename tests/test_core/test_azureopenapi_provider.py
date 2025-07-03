"""
Tests for Azure OpenAI provider.

This module tests the Azure OpenAI provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.models.providers import AzureOpenAIConfig

@pytest.fixture
def mock_config():
    return AzureOpenAIConfig(
        name="test-provider",
        api_version="2023-05-15",
        base_url="https://example.openai.azure.com/",
        api_key="test-key",
        model_list=["gpt-35-turbo", "gpt-4"]
    )

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_initialization(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    assert provider.client is mock_client
    assert provider.config.model_list == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_get_model_list(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    models = await provider.get_model_list()
    assert models == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_cleanup_noop(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    # Should not raise
    await provider.cleanup()

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_send_chat_returns_response(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Hello, world!"
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_azure_openai.return_value = mock_client

    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    result = await provider.send_chat(model="gpt-35-turbo", 
                                      instructions="instructions", 
                                      context="hi", tools=[])
    assert result == "Hello, world!"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-35-turbo",
        instructions="instructions",
        input="hi",
        tools=[]
    )

