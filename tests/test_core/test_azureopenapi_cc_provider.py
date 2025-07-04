"""
Tests for Azure OpenAI CC provider.

This module tests the Azure OpenAI CC provider, ensuring it
interacts with the Chat Completions API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
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

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_initialization(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProviderCC(mock_config)
    await provider.initialize()
    assert provider.client is mock_client
    assert provider.config.model_list == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_get_model_list(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProviderCC(mock_config)
    models = await provider.get_model_list()
    assert models == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_cleanup_noop(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProviderCC(mock_config)
    # Should not raise
    await provider.cleanup()

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_send_chat_returns_response(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    
    # Create a proper mock response that matches ChatCompletion structure
    mock_message = MagicMock()
    mock_message.content = "Hello, world!"
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_azure_openai.return_value = mock_client

    provider = AzureOpenAIProviderCC(mock_config)
    await provider.initialize()
    
    context = [{"role": "user", "content": "hi"}]
    result = await provider.send_chat(
        context=context,
        model="gpt-35-turbo", 
        instructions="You are a helpful assistant", 
        tools=[]
    )
    
    assert result == "Hello, world!"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "hi"}
        ]
    )

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_send_chat_without_instructions(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    
    # Create a proper mock response that matches ChatCompletion structure
    mock_message = MagicMock()
    mock_message.content = "Response without instructions"
    
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_azure_openai.return_value = mock_client

    provider = AzureOpenAIProviderCC(mock_config)
    await provider.initialize()
    
    context = [{"role": "user", "content": "hello"}]
    result = await provider.send_chat(
        context=context,
        model="gpt-4", 
        instructions=None, 
        tools=None
    )
    
    assert result == "Response without instructions"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "hello"}
        ]
    ) 