"""
Tests for Ollama provider.

This module tests the Ollama provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.core.providers.ollama import OllamaProvider
from app.models.providers import OllamaConfig



@pytest.fixture
def mock_config():
    return OllamaConfig(
        name="test-provider",
        base_url="http://localhost:11434",
        model_list=["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    )



@patch("app.core.providers.ollama.AsyncClient")
@pytest.mark.asyncio
async def test_initialization(mock_ollama, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    mock_ollama.return_value = mock_client
    provider = OllamaProvider(mock_config)
    await provider.initialize()
    assert provider.client is mock_client
    assert provider.config.model_list == ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]

@patch("app.core.providers.ollama.AsyncClient")
@pytest.mark.asyncio
async def test_get_model_list(mock_ollama_client, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    mock_ollama_client.return_value = mock_client
    provider = OllamaProvider(mock_config)
    models = await provider.get_model_list()
    assert models == ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]

@patch("app.core.providers.ollama.OllamaProvider")
@pytest.mark.asyncio
async def test_cleanup_noop(mock_ollama_provider, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    mock_ollama_provider.return_value = mock_client
    provider = OllamaProvider(mock_config)
    # Should not raise
    await provider.cleanup()

@patch("app.core.providers.ollama.AsyncClient")
@pytest.mark.asyncio
async def test_send_chat_returns_response(mock_ollama_client, mock_config):
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simulate the response structure returned by the actual chat method
    mock_response.__getitem__.side_effect = lambda key: {"message": {"content": "Hello, world!"}}[key]
    mock_client.chat = AsyncMock(return_value=mock_response)
    mock_ollama_client.return_value = mock_client

    provider = OllamaProvider(mock_config)
    await provider.initialize()
    result = await provider.send_chat(
        context=[{"role": "user", "content": "hi"}],
        model="llama3.1:8b",
        instructions=None
    )
    # The Ollama chat API expects a list of messages, not 'instructions'
    mock_client.chat.assert_called_once_with(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": "hi"}]
    )
    assert result == "Hello, world!"

