"""
Tests for Ollama provider.

This module tests the Ollama provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.core.providers.ollama import OllamaProvider
from app.models.providers import OllamaConfig

@pytest.fixture
def mock_config():
    return OllamaConfig(
        name="test-provider",
        base_url="http://localhost:11434",
        model_list=["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    )

@patch("app.core.providers.ollama.OllamaProvider")
def test_initialization(mock_ollama, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    mock_ollama.return_value = mock_client
    provider = OllamaProvider(mock_config)
    # assert provider.client is mock_client
    assert provider.config.model_list == ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]

@patch("app.core.providers.azureopenapi.AzureOpenAI")
@pytest.mark.asyncio
async def test_get_model_list(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    models = await provider.get_model_list()
    assert models == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi.AzureOpenAI")
@pytest.mark.asyncio
async def test_cleanup_noop(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    # Should not raise
    await provider.cleanup()

@patch("app.core.providers.azureopenapi.AzureOpenAI")
@pytest.mark.asyncio
async def test_send_chat_returns_response(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Hello, world!"
    mock_response.model_dump_json.return_value = '{"output_text": "Hello, world!"}'
    mock_client.responses.create.return_value = mock_response
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client

    provider = AzureOpenAIProvider(mock_config)
    result = await provider.send_chat("hi", "gpt-35-turbo", "instructions", [])
    assert result == "Hello, world!"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-35-turbo",
        instructions="instructions",
        input="hi",
        tools=[]
    )

