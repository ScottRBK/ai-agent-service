"""
Tests for Ollama provider.

This module tests the Ollama provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from app.core.providers.ollama import OllamaProvider
from app.models.providers import OllamaConfig
from app.core.tools.function_calls.date_tool import DateTool
from datetime import datetime
from ollama import ChatResponse, Message

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
async def test_health_check_healthy(mock_ollama, mock_config):
    mock_client = MagicMock()
    mock_client.list = AsyncMock(return_value=["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"])
    mock_ollama.return_value = mock_client
    provider = OllamaProvider(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "healthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is None

@patch("app.core.providers.ollama.AsyncClient")
@pytest.mark.asyncio
async def test_health_check_unhealthy(mock_ollama, mock_config):
    mock_client = MagicMock()
    mock_client.list = AsyncMock(side_effect=Exception("Test error"))
    mock_ollama.return_value = mock_client
    provider = OllamaProvider(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "unhealthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is "Test error"



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
    mock_ollama_client.return_value = mock_client
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Hello, world!"
    mock_message.tool_calls = []
    mock_response.message = mock_message
    mock_client.chat = AsyncMock(return_value=mock_response)

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
        messages=[{"role": "user", "content": "hi"}],
        tools=None
    )
    assert result == "Hello, world!"
