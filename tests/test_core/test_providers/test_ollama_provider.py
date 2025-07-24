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
from app.core.providers.base import ProviderMaxToolIterationsError

@pytest.fixture
def mock_config():
    return OllamaConfig(
        name="test-provider",
        base_url="http://localhost:11434",
        model_list=["llama3.1:8b", "qwen3:4b", "qwen3:8b", "qwen3:14b"]
    )

@pytest.fixture
def mock_client():
    return AsyncMock()

@pytest.fixture
def provider(mock_config):
    with patch('app.core.providers.ollama.AsyncClient'):
        return OllamaProvider(mock_config)



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

### send_chat_with_streaming
@pytest.mark.asyncio
async def test_send_chat_with_streaming_basic_response(provider, mock_client):
    # Arrange
    provider.client = mock_client
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_first_yield'):
                self._first_yield = True
                return MagicMock(message=MagicMock(content="Hello", tool_calls=None))
            elif not hasattr(self, '_second_yield'):
                self._second_yield = True
                return MagicMock(message=MagicMock(content=" world", tool_calls=None))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    # Act
    chunks = []
    async for chunk in provider.send_chat_with_streaming(
        context=[{"role": "user", "content": "Hi"}],
        model="test-model",
        instructions="Be helpful"
    ):
        chunks.append(chunk)
    
    # Assert
    assert chunks == ["Hello", " world"]
    mock_client.chat.assert_called_once()

@pytest.mark.asyncio
async def test_send_chat_with_streaming_with_tool_calls(provider, mock_client):
    # Arrange
    provider.client = mock_client
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="I'll help", tool_calls=[mock_tool_call]))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    with patch.object(provider, 'execute_tool_call', return_value="tool_result"):
        # Act
        chunks = []
        async for chunk in provider.send_chat_with_streaming(
            context=[{"role": "user", "content": "Use tool"}],
            model="test-model",
            instructions="Be helpful",
            agent_id="test_agent"
        ):
            chunks.append(chunk)
        
        # Assert
        assert chunks == ["I'll help"]
        assert provider.execute_tool_call.called

@pytest.mark.asyncio
async def test_send_chat_with_streaming_max_iterations_exceeded(provider, mock_client):
    # Arrange
    provider.client = mock_client
    provider.max_tool_iterations = 1
    
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="", tool_calls=[mock_tool_call]))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    with patch.object(provider, 'execute_tool_call', return_value="tool_result"):
        # Act & Assert
        with pytest.raises(ProviderMaxToolIterationsError):
            async for _ in provider.send_chat_with_streaming(
                context=[{"role": "user", "content": "Loop"}],
                model="test-model",
                instructions="Be helpful"
            ):
                pass

@pytest.mark.asyncio
async def test_send_chat_with_streaming_model_settings(provider, mock_client):
    # Arrange
    provider.client = mock_client
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="Response", tool_calls=None))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    model_settings = {"temperature": 0.7, "max_tokens": 100}
    
    # Act
    async for _ in provider.send_chat_with_streaming(
        context=[{"role": "user", "content": "Hi"}],
        model="test-model",
        instructions="Be helpful",
        model_settings=model_settings
    ):
        break
    
    # Assert
    call_args = mock_client.chat.call_args[1]
    assert call_args["options"] == model_settings

@pytest.mark.asyncio
async def test_send_chat_with_streaming_multiple_tool_calls(provider, mock_client):
    # Arrange
    provider.client = mock_client
    mock_tool_call1 = MagicMock()
    mock_tool_call1.function.name = "tool1"
    mock_tool_call1.function.arguments = '{"arg": "value1"}'
    
    mock_tool_call2 = MagicMock()
    mock_tool_call2.function.name = "tool2"
    mock_tool_call2.function.arguments = '{"arg": "value2"}'
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="Using tools", tool_calls=[mock_tool_call1, mock_tool_call2]))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    with patch.object(provider, 'execute_tool_call', return_value="tool_result"):
        # Act
        chunks = []
        async for chunk in provider.send_chat_with_streaming(
            context=[{"role": "user", "content": "Use multiple tools"}],
            model="test-model",
            instructions="Be helpful",
            agent_id="test_agent"
        ):
            chunks.append(chunk)
        
        # Assert
        assert chunks == ["Using tools"]
        assert provider.execute_tool_call.call_count == 2

@pytest.mark.asyncio
async def test_send_chat_with_streaming_no_content_only_tools(provider, mock_client):
    # Arrange
    provider.client = mock_client
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"arg": "value"}'
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="", tool_calls=[mock_tool_call]))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    with patch.object(provider, 'execute_tool_call', return_value="tool_result"):
        # Act
        chunks = []
        async for chunk in provider.send_chat_with_streaming(
            context=[{"role": "user", "content": "Tool only"}],
            model="test-model",
            instructions="Be helpful",
            agent_id="test_agent"
        ):
            chunks.append(chunk)
        
        # Assert
        assert chunks == []  # No content chunks
        assert provider.execute_tool_call.called

@pytest.mark.asyncio
async def test_send_chat_with_streaming_with_instructions(provider, mock_client):
    # Arrange
    provider.client = mock_client
    
    # Create a proper async generator
    class MockAsyncGenerator:
        def __aiter__(self):
            return self
        
        async def __anext__(self):
            if not hasattr(self, '_yielded'):
                self._yielded = True
                return MagicMock(message=MagicMock(content="Response", tool_calls=None))
            else:
                raise StopAsyncIteration
    
    mock_response = MockAsyncGenerator()
    mock_client.chat.return_value = mock_response
    
    # Act
    async for _ in provider.send_chat_with_streaming(
        context=[{"role": "user", "content": "Hi"}],
        model="test-model",
        instructions="You are a helpful assistant"
    ):
        break
    
    # Assert
    call_args = mock_client.chat.call_args[1]
    messages = call_args["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
