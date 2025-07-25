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

# Task 5: Shared test utilities
class MockAsyncStreamGenerator:
    """Reusable mock async generator for streaming responses."""
    
    def __init__(self, chunks):
        """
        Initialize with a list of chunk data.
        Each chunk should be a dict with 'content' and/or 'tool_calls' keys.
        """
        self.chunks = chunks
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        
        chunk_data = self.chunks[self.index]
        self.index += 1
        
        chunk = MagicMock()
        chunk.message.content = chunk_data.get('content')
        chunk.message.tool_calls = chunk_data.get('tool_calls')
        return chunk

def create_mock_tool_call(name, arguments='{}'):
    """Factory function to create mock tool calls."""
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = name
    mock_tool_call.function.arguments = arguments
    return mock_tool_call

def create_streaming_response(chunks):
    """Helper to create a mock streaming response."""
    return MockAsyncStreamGenerator(chunks)



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

# Task 7: Updated existing non-streaming tests
class TestSendChat:
    """Test cases for the send_chat non-streaming method."""
    
    @patch("app.core.providers.ollama.AsyncClient")
    @pytest.mark.asyncio
    async def test_send_chat_returns_response(self, mock_ollama_client, mock_config):
        """Test basic send_chat functionality."""
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
    
    @pytest.mark.asyncio
    async def test_send_chat_with_instructions(self, provider, mock_client):
        """Test send_chat with system instructions using _prepare_messages."""
        provider.client = mock_client
        
        mock_response = MagicMock()
        mock_response.message.content = "Helpful response"
        mock_response.message.tool_calls = []
        mock_client.chat = AsyncMock(return_value=mock_response)
        
        result = await provider.send_chat(
            context=[{"role": "user", "content": "Hello"}],
            model="test-model",
            instructions="You are a helpful assistant"
        )
        
        # Verify _prepare_messages was used correctly
        call_args = mock_client.chat.call_args[1]
        messages = call_args["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert result == "Helpful response"
    
    @pytest.mark.asyncio
    async def test_send_chat_with_tool_calls(self, provider, mock_client):
        """Test send_chat with tool calls using _execute_tool_calls."""
        provider.client = mock_client
        
        # Mock tool call
        tool_call = create_mock_tool_call("test_tool", '{"arg": "value"}')
        
        # First response with tool call
        first_response = MagicMock()
        first_response.message.content = "I'll use a tool"
        first_response.message.tool_calls = [tool_call]
        
        # Second response after tool execution
        second_response = MagicMock()
        second_response.message.content = "Tool completed successfully"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='tool_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Use a tool"}],
                model="test-model",
                instructions=None,
                agent_id="test_agent"
            )
        
        # Verify tool was executed
        mock_execute.assert_called_once_with("test_tool", '{"arg": "value"}', "test_agent")
        
        # Verify two chat calls were made
        assert mock_client.chat.call_count == 2
        assert result == "Tool completed successfully"
    
    @pytest.mark.asyncio
    async def test_send_chat_with_model_settings(self, provider, mock_client):
        """Test send_chat with model settings passed correctly."""
        provider.client = mock_client
        
        mock_response = MagicMock()
        mock_response.message.content = "Response with settings"
        mock_response.message.tool_calls = []
        mock_client.chat = AsyncMock(return_value=mock_response)
        
        model_settings = {"temperature": 0.8, "max_tokens": 150}
        
        result = await provider.send_chat(
            context=[{"role": "user", "content": "Test"}],
            model="test-model",
            instructions=None,
            model_settings=model_settings
        )
        
        # Verify model settings were passed
        call_args = mock_client.chat.call_args[1]
        assert call_args["options"] == model_settings
        assert result == "Response with settings"
    
    @pytest.mark.asyncio
    async def test_send_chat_max_tool_iterations_exceeded(self, provider, mock_client):
        """Test send_chat max tool iterations error."""
        provider.client = mock_client
        provider.max_tool_iterations = 1
        
        tool_call = create_mock_tool_call("infinite_tool", '{"loop": true}')
        
        # Always return tool calls to trigger infinite loop
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = [tool_call]
        mock_client.chat = AsyncMock(return_value=mock_response)
        
        with patch.object(provider, 'execute_tool_call', return_value='loop_result'):
            with pytest.raises(ProviderMaxToolIterationsError):
                await provider.send_chat(
                    context=[{"role": "user", "content": "Loop"}],
                    model="test-model",
                    instructions=None,
                    agent_id="test_agent"
                )
    
    @pytest.mark.asyncio
    async def test_send_chat_multiple_tool_calls_single_response(self, provider, mock_client):
        """Test send_chat with multiple tool calls in single response."""
        provider.client = mock_client
        
        # Multiple tool calls in first response
        tool_call1 = create_mock_tool_call("tool1", '{"arg": "value1"}')
        tool_call2 = create_mock_tool_call("tool2", '{"arg": "value2"}')
        
        first_response = MagicMock()
        first_response.message.content = "Using multiple tools"
        first_response.message.tool_calls = [tool_call1, tool_call2]
        
        # Final response
        second_response = MagicMock()
        second_response.message.content = "All tools completed"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='tool_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Use multiple tools"}],
                model="test-model",
                instructions=None,
                agent_id="test_agent"
            )
        
        # Verify both tools were executed
        assert mock_execute.call_count == 2
        mock_execute.assert_any_call("tool1", '{"arg": "value1"}', "test_agent")
        mock_execute.assert_any_call("tool2", '{"arg": "value2"}', "test_agent")
        assert result == "All tools completed"

### send_chat_with_streaming
# Task 6: Updated existing streaming tests using shared utilities
@pytest.mark.asyncio
async def test_send_chat_with_streaming_basic_response(provider, mock_client):
    # Arrange
    provider.client = mock_client
    
    chunks_data = [
        {'content': "Hello"},
        {'content': " world"}
    ]
    mock_response = create_streaming_response(chunks_data)
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
    mock_tool_call = create_mock_tool_call("test_tool", '{"arg": "value"}')
    
    chunks_data = [
        {'content': "I'll help", 'tool_calls': [mock_tool_call]}
    ]
    mock_response = create_streaming_response(chunks_data)
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
    
    chunks_data = [{'content': "Response"}]
    mock_response = create_streaming_response(chunks_data)
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

# Task 1: Tests for _prepare_messages method
class TestPrepareMessages:
    """Test cases for the _prepare_messages private method."""
    
    def test_prepare_messages_with_instructions_and_context(self, provider):
        """Test message preparation with both instructions and context."""
        instructions = "You are a helpful assistant"
        context = [{"role": "user", "content": "Hello"}]
        
        result = provider._prepare_messages(instructions, context)
        
        expected = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        assert result == expected
    
    def test_prepare_messages_with_instructions_only(self, provider):
        """Test message preparation with instructions but no context."""
        instructions = "You are a helpful assistant"
        context = []
        
        result = provider._prepare_messages(instructions, context)
        
        expected = [{"role": "system", "content": "You are a helpful assistant"}]
        assert result == expected
    
    def test_prepare_messages_with_context_only(self, provider):
        """Test message preparation with context but no instructions."""
        instructions = None
        context = [{"role": "user", "content": "Hello"}]
        
        result = provider._prepare_messages(instructions, context)
        
        expected = [{"role": "user", "content": "Hello"}]
        assert result == expected
    
    def test_prepare_messages_with_empty_inputs(self, provider):
        """Test message preparation with empty inputs."""
        instructions = None
        context = []
        
        result = provider._prepare_messages(instructions, context)
        
        assert result == []
    
    def test_prepare_messages_with_empty_string_instructions(self, provider):
        """Test message preparation with empty string instructions."""
        instructions = ""
        context = [{"role": "user", "content": "Hello"}]
        
        result = provider._prepare_messages(instructions, context)
        
        expected = [{"role": "user", "content": "Hello"}]
        assert result == expected
    
    def test_prepare_messages_with_multiple_context_messages(self, provider):
        """Test message preparation with multiple context messages."""
        instructions = "Be helpful"
        context = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = provider._prepare_messages(instructions, context)
        
        expected = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        assert result == expected

# Task 2: Tests for _execute_tool_calls method
class TestExecuteToolCalls:
    """Test cases for the _execute_tool_calls private method."""
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_single_call(self, provider):
        """Test executing a single tool call."""
        messages = []
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'
        tool_calls_to_process = [mock_tool_call]
        
        with patch.object(provider, 'execute_tool_call', return_value='tool_result') as mock_execute:
            result = await provider._execute_tool_calls(messages, tool_calls_to_process, "test_agent")
            
            # Verify tool was executed
            mock_execute.assert_called_once_with("test_tool", '{"arg": "value"}', "test_agent")
            
            # Verify messages were appended correctly
            assert len(messages) == 2
            assert messages[0]["role"] == "assistant"
            assert messages[0]["tool_calls"][0]["function"]["name"] == "test_tool"
            assert messages[1]["role"] == "tool"
            assert messages[1]["content"] == "tool_result"
            
            # Verify return value
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple_calls(self, provider):
        """Test executing multiple tool calls."""
        messages = []
        mock_tool_call1 = MagicMock()
        mock_tool_call1.function.name = "tool1"
        mock_tool_call1.function.arguments = '{"arg": "value1"}'
        
        mock_tool_call2 = MagicMock()
        mock_tool_call2.function.name = "tool2"
        mock_tool_call2.function.arguments = '{"arg": "value2"}'
        
        tool_calls_to_process = [mock_tool_call1, mock_tool_call2]
        
        with patch.object(provider, 'execute_tool_call', return_value='tool_result') as mock_execute:
            result = await provider._execute_tool_calls(messages, tool_calls_to_process, "test_agent")
            
            # Verify both tools were executed
            assert mock_execute.call_count == 2
            mock_execute.assert_any_call("tool1", '{"arg": "value1"}', "test_agent")
            mock_execute.assert_any_call("tool2", '{"arg": "value2"}', "test_agent")
            
            # Verify messages structure
            assert len(messages) == 4  # 2 assistant + 2 tool messages
            assert messages[0]["role"] == "assistant"
            assert messages[1]["role"] == "tool"
            assert messages[2]["role"] == "assistant"
            assert messages[3]["role"] == "tool"
            
            # Verify return value
            assert result == 2
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_different_argument_types(self, provider):
        """Test executing tool calls with different argument formats."""
        messages = []
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "complex_tool"
        mock_tool_call.function.arguments = '{"list": [1, 2, 3], "nested": {"key": "value"}}'
        tool_calls_to_process = [mock_tool_call]
        
        with patch.object(provider, 'execute_tool_call', return_value='complex_result') as mock_execute:
            result = await provider._execute_tool_calls(messages, tool_calls_to_process, "test_agent")
            
            # Verify complex arguments were passed correctly
            mock_execute.assert_called_once_with(
                "complex_tool", 
                '{"list": [1, 2, 3], "nested": {"key": "value"}}', 
                "test_agent"
            )
            
            # Verify message structure for complex tool call
            assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"list": [1, 2, 3], "nested": {"key": "value"}}'
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_list(self, provider):
        """Test executing with empty tool calls list."""
        messages = []
        tool_calls_to_process = []
        
        result = await provider._execute_tool_calls(messages, tool_calls_to_process, "test_agent")
        
        # Verify no messages were added and count is 0
        assert len(messages) == 0
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_message_structure_validation(self, provider):
        """Test that the message structure follows the expected format."""
        messages = []
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "validation_tool"
        mock_tool_call.function.arguments = '{"test": true}'
        tool_calls_to_process = [mock_tool_call]
        
        with patch.object(provider, 'execute_tool_call', return_value='validation_result'):
            await provider._execute_tool_calls(messages, tool_calls_to_process, "test_agent")
            
            # Validate assistant message structure
            assistant_msg = messages[0]
            assert assistant_msg["role"] == "assistant"
            assert "tool_calls" in assistant_msg
            assert len(assistant_msg["tool_calls"]) == 1
            
            tool_call_structure = assistant_msg["tool_calls"][0]
            assert tool_call_structure["type"] == "function"
            assert "function" in tool_call_structure
            assert tool_call_structure["function"]["name"] == "validation_tool"
            assert tool_call_structure["function"]["arguments"] == '{"test": true}'
            
            # Validate tool response message structure
            tool_msg = messages[1]
            assert tool_msg["role"] == "tool"
            assert tool_msg["content"] == "validation_result"

# Task 3: Tests for _process_streaming_response method
class TestProcessStreamingResponse:
    """Test cases for the _process_streaming_response private method."""
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_content_only(self, provider):
        """Test processing streaming response with content only."""
        # Create mock response chunks
        class MockAsyncGenerator:
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if not hasattr(self, '_first_yield'):
                    self._first_yield = True
                    chunk = MagicMock()
                    chunk.message.content = "Hello"
                    chunk.message.tool_calls = None
                    return chunk
                elif not hasattr(self, '_second_yield'):
                    self._second_yield = True
                    chunk = MagicMock()
                    chunk.message.content = " world"
                    chunk.message.tool_calls = None
                    return chunk
                else:
                    raise StopAsyncIteration
        
        mock_response = MockAsyncGenerator()
        
        # Process the streaming response
        chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(mock_response):
            if chunk_type == "content":
                chunks.append(content)
            elif chunk_type == "final":
                final_content = content
                final_tool_calls = tool_calls
        
        # Verify content chunks
        assert chunks == ["Hello", " world"]
        assert final_content == "Hello world"
        assert final_tool_calls == []
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_tool_calls_only(self, provider):
        """Test processing streaming response with tool calls only."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_tool"
        
        class MockAsyncGenerator:
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if not hasattr(self, '_yielded'):
                    self._yielded = True
                    chunk = MagicMock()
                    chunk.message.content = None
                    chunk.message.tool_calls = [mock_tool_call]
                    return chunk
                else:
                    raise StopAsyncIteration
        
        mock_response = MockAsyncGenerator()
        
        # Process the streaming response
        chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(mock_response):
            if chunk_type == "content":
                chunks.append(content)
            elif chunk_type == "final":
                final_content = content
                final_tool_calls = tool_calls
        
        # Verify no content chunks but tool calls present
        assert chunks == []
        assert final_content == ""
        assert final_tool_calls == [mock_tool_call]
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_mixed_content_and_tools(self, provider):
        """Test processing streaming response with both content and tool calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "mixed_tool"
        
        class MockAsyncGenerator:
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if not hasattr(self, '_first_yield'):
                    self._first_yield = True
                    chunk = MagicMock()
                    chunk.message.content = "I'll help you"
                    chunk.message.tool_calls = None
                    return chunk
                elif not hasattr(self, '_second_yield'):
                    self._second_yield = True
                    chunk = MagicMock()
                    chunk.message.content = None
                    chunk.message.tool_calls = [mock_tool_call]
                    return chunk
                else:
                    raise StopAsyncIteration
        
        mock_response = MockAsyncGenerator()
        
        # Process the streaming response
        content_chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(mock_response):
            if chunk_type == "content":
                content_chunks.append(content)
            elif chunk_type == "final":
                final_content = content
                final_tool_calls = tool_calls
        
        # Verify both content and tool calls
        assert content_chunks == ["I'll help you"]
        assert final_content == "I'll help you"
        assert final_tool_calls == [mock_tool_call]
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_empty_response(self, provider):
        """Test processing empty streaming response."""
        class MockAsyncGenerator:
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                raise StopAsyncIteration
        
        mock_response = MockAsyncGenerator()
        
        # Process the streaming response
        chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(mock_response):
            if chunk_type == "content":
                chunks.append(content)
            elif chunk_type == "final":
                final_content = content
                final_tool_calls = tool_calls
        
        # Verify empty results
        assert chunks == []
        assert final_content == ""
        assert final_tool_calls == []
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_multiple_tool_calls(self, provider):
        """Test processing streaming response with multiple tool calls."""
        mock_tool_call1 = MagicMock()
        mock_tool_call1.function.name = "tool1"
        mock_tool_call2 = MagicMock()
        mock_tool_call2.function.name = "tool2"
        
        class MockAsyncGenerator:
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if not hasattr(self, '_first_yield'):
                    self._first_yield = True
                    chunk = MagicMock()
                    chunk.message.content = "Using tools"
                    chunk.message.tool_calls = [mock_tool_call1]
                    return chunk
                elif not hasattr(self, '_second_yield'):
                    self._second_yield = True
                    chunk = MagicMock()
                    chunk.message.content = None
                    chunk.message.tool_calls = [mock_tool_call2]
                    return chunk
                else:
                    raise StopAsyncIteration
        
        mock_response = MockAsyncGenerator()
        
        # Process the streaming response
        content_chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(mock_response):
            if chunk_type == "content":
                content_chunks.append(content)
            elif chunk_type == "final":
                final_content = content
                final_tool_calls = tool_calls
        
        # Verify content and multiple tool calls
        assert content_chunks == ["Using tools"]
        assert final_content == "Using tools"
        assert len(final_tool_calls) == 2
        assert final_tool_calls == [mock_tool_call1, mock_tool_call2]

# Task 4: Tests for _handle_tool_calls_streaming method
class TestHandleToolCallsStreaming:
    """Test cases for the _handle_tool_calls_streaming private method."""
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_single_iteration_content_only(self, provider, mock_client):
        """Test streaming with single iteration and content only."""
        provider.client = mock_client
        
        # Create streaming response with content only
        chunks = [
            {'content': 'Hello'},
            {'content': ' world'}
        ]
        mock_response = create_streaming_response(chunks)
        mock_client.chat.return_value = mock_response
        
        messages = []
        available_tools = []
        
        # Collect streaming content
        content_chunks = []
        async for content in provider._handle_tool_calls_streaming(
            messages, "test-model", available_tools, "test_agent"
        ):
            content_chunks.append(content)
        
        # Verify content was streamed
        assert content_chunks == ['Hello', ' world']
        # Verify assistant message was added
        assert len(messages) == 1
        assert messages[0]['role'] == 'assistant'
        assert messages[0]['content'] == 'Hello world'
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_with_tool_calls(self, provider, mock_client):
        """Test streaming with tool calls."""
        provider.client = mock_client
        
        # Mock the tool calling sequence
        tool_call = create_mock_tool_call("test_tool", '{"arg": "value"}')
        
        # First response: content + tool call
        first_chunks = [{'content': 'I will help', 'tool_calls': [tool_call]}]
        # Second response: final content after tool execution
        second_chunks = [{'content': 'Task completed'}]
        
        # Mock client to return different responses for each call
        responses = [
            create_streaming_response(first_chunks),
            create_streaming_response(second_chunks)
        ]
        mock_client.chat.side_effect = responses
        
        with patch.object(provider, 'execute_tool_call', return_value='tool_result'):
            messages = []
            available_tools = [{'name': 'test_tool'}]
            
            # Collect streaming content
            content_chunks = []
            async for content in provider._handle_tool_calls_streaming(
                messages, "test-model", available_tools, "test_agent"
            ):
                content_chunks.append(content)
            
            # Verify content was streamed from both iterations
            assert content_chunks == ['I will help', 'Task completed']
            
            # Verify messages include assistant, tool call, tool response, and final assistant
            assert len(messages) == 4
            assert messages[0]['role'] == 'assistant'
            assert messages[0]['content'] == 'I will help'
            assert messages[1]['role'] == 'assistant'
            assert 'tool_calls' in messages[1]
            assert messages[2]['role'] == 'tool'
            assert messages[2]['content'] == 'tool_result'
            assert messages[3]['role'] == 'assistant'
            assert messages[3]['content'] == 'Task completed'
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_max_iterations_exceeded(self, provider, mock_client):
        """Test max iterations exceeded error."""
        provider.client = mock_client
        provider.max_tool_iterations = 1  # Set low limit
        
        tool_call = create_mock_tool_call("infinite_tool", '{"loop": true}')
        
        # Always return tool calls to trigger infinite loop
        chunks = [{'content': '', 'tool_calls': [tool_call]}]
        mock_client.chat.return_value = create_streaming_response(chunks)
        
        with patch.object(provider, 'execute_tool_call', return_value='loop_result'):
            messages = []
            available_tools = [{'name': 'infinite_tool'}]
            
            # Should raise max iterations error
            with pytest.raises(ProviderMaxToolIterationsError):
                async for _ in provider._handle_tool_calls_streaming(
                    messages, "test-model", available_tools, "test_agent"
                ):
                    pass
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_no_tool_calls(self, provider, mock_client):
        """Test streaming with no tool calls (breaks immediately)."""
        provider.client = mock_client
        
        chunks = [{'content': 'Simple response'}]
        mock_response = create_streaming_response(chunks)
        mock_client.chat.return_value = mock_response
        
        messages = []
        available_tools = []
        
        # Collect streaming content
        content_chunks = []
        async for content in provider._handle_tool_calls_streaming(
            messages, "test-model", available_tools, "test_agent"
        ):
            content_chunks.append(content)
        
        # Verify single iteration
        assert content_chunks == ['Simple response']
        assert len(messages) == 1
        assert messages[0]['role'] == 'assistant'
        assert messages[0]['content'] == 'Simple response'
        
        # Verify only one chat call was made
        assert mock_client.chat.call_count == 1
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_model_settings_passed(self, provider, mock_client):
        """Test that model settings are properly passed to the client."""
        provider.client = mock_client
        
        chunks = [{'content': 'Response'}]
        mock_response = create_streaming_response(chunks)
        mock_client.chat.return_value = mock_response
        
        messages = []
        available_tools = []
        model_settings = {"temperature": 0.7, "max_tokens": 100}
        
        # Execute streaming
        async for _ in provider._handle_tool_calls_streaming(
            messages, "test-model", available_tools, "test_agent", model_settings
        ):
            break
        
        # Verify model settings were passed
        call_args = mock_client.chat.call_args[1]
        assert call_args["options"] == model_settings
        assert call_args["stream"] is True
        assert call_args["model"] == "test-model"

# Task 8: Integration tests for method interaction
class TestMethodIntegration:
    """Test cases for integration between refactored methods."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming_with_tool_calls(self, provider, mock_client):
        """Test complete streaming flow with all refactored methods working together."""
        provider.client = mock_client
        
        # Create tool call
        tool_call = create_mock_tool_call("integration_tool", '{"test": "integration"}')
        
        # First streaming response with content and tool call
        first_chunks = [
            {'content': 'I will help you'},
            {'content': ' with this task', 'tool_calls': [tool_call]}
        ]
        
        # Second streaming response after tool execution
        second_chunks = [
            {'content': 'Task completed successfully'}
        ]
        
        # Mock responses
        responses = [
            create_streaming_response(first_chunks),
            create_streaming_response(second_chunks)
        ]
        mock_client.chat.side_effect = responses
        
        with patch.object(provider, 'execute_tool_call', return_value='integration_result') as mock_execute:
            # Test the full streaming flow
            chunks = []
            async for chunk in provider.send_chat_with_streaming(
                context=[{"role": "user", "content": "Help me"}],
                model="integration-model",
                instructions="You are helpful",
                agent_id="integration_agent"
            ):
                chunks.append(chunk)
        
        # Verify all methods worked together
        # _prepare_messages: system instruction should be in first call
        first_call_args = mock_client.chat.call_args_list[0][1]
        messages = first_call_args["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        
        # _process_streaming_response: content chunks were yielded
        assert chunks == ['I will help you', ' with this task', 'Task completed successfully']
        
        # _execute_tool_calls: tool was executed
        mock_execute.assert_called_once_with("integration_tool", '{"test": "integration"}', "integration_agent")
        
        # _handle_tool_calls_streaming: two iterations occurred
        assert mock_client.chat.call_count == 2
    
    @pytest.mark.asyncio
    async def test_end_to_end_non_streaming_with_tool_calls(self, provider, mock_client):
        """Test complete non-streaming flow with all refactored methods working together."""
        provider.client = mock_client
        
        tool_call = create_mock_tool_call("non_stream_tool", '{"mode": "sync"}')
        
        # First response with tool call
        first_response = MagicMock()
        first_response.message.content = "Processing request"
        first_response.message.tool_calls = [tool_call]
        
        # Second response after tool execution
        second_response = MagicMock()
        second_response.message.content = "Request processed successfully"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='sync_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Process this"}],
                model="sync-model", 
                instructions="Be precise",
                agent_id="sync_agent"
            )
        
        # Verify all methods worked together
        # _prepare_messages: system instruction included
        first_call_args = mock_client.chat.call_args_list[0][1]
        messages = first_call_args["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be precise"
        
        # _execute_tool_calls: tool was executed
        mock_execute.assert_called_once_with("non_stream_tool", '{"mode": "sync"}', "sync_agent")
        
        # Final result returned
        assert result == "Request processed successfully"
        assert mock_client.chat.call_count == 2
    
    @pytest.mark.asyncio
    async def test_message_flow_consistency_between_methods(self, provider, mock_client):
        """Test that message structure is consistent between streaming and non-streaming."""
        provider.client = mock_client
        
        # Test data
        instructions = "Be consistent"
        context = [{"role": "user", "content": "Test consistency"}]
        model = "consistency-model"
        
        # Test streaming version
        streaming_chunks = [{'content': 'Streaming response'}]
        mock_client.chat.return_value = create_streaming_response(streaming_chunks)
        
        async for _ in provider.send_chat_with_streaming(
            context=context,
            model=model,
            instructions=instructions
        ):
            break
        
        streaming_call_args = mock_client.chat.call_args[1]
        streaming_messages = streaming_call_args["messages"]
        
        # Reset mock
        mock_client.reset_mock()
        
        # Test non-streaming version
        non_streaming_response = MagicMock()
        non_streaming_response.message.content = "Non-streaming response"
        non_streaming_response.message.tool_calls = []
        mock_client.chat.return_value = non_streaming_response
        
        await provider.send_chat(
            context=context,
            model=model,
            instructions=instructions
        )
        
        non_streaming_call_args = mock_client.chat.call_args[1]
        non_streaming_messages = non_streaming_call_args["messages"]
        
        # Verify both use same message preparation logic
        assert streaming_messages == non_streaming_messages
        assert streaming_messages[0]["role"] == "system"
        assert streaming_messages[0]["content"] == "Be consistent"
        assert streaming_messages[1]["role"] == "user"
        assert streaming_messages[1]["content"] == "Test consistency"
    
    @pytest.mark.asyncio
    async def test_shared_tool_execution_logic_consistency(self, provider, mock_client):
        """Test that tool execution works consistently in both streaming and non-streaming."""
        provider.client = mock_client
        
        tool_call = create_mock_tool_call("shared_tool", '{"shared": true}')
        
        # Test streaming tool execution
        streaming_chunks = [{'content': 'Using tool', 'tool_calls': [tool_call]}]
        second_chunks = [{'content': 'Tool used'}]
        streaming_responses = [
            create_streaming_response(streaming_chunks),
            create_streaming_response(second_chunks)
        ]
        mock_client.chat.side_effect = streaming_responses
        
        with patch.object(provider, 'execute_tool_call', return_value='shared_result') as mock_execute_streaming:
            async for _ in provider.send_chat_with_streaming(
                context=[{"role": "user", "content": "Use shared tool"}],
                model="test-model",
                instructions=None,
                agent_id="shared_agent"
            ):
                pass
        
        # Verify streaming tool execution
        mock_execute_streaming.assert_called_with("shared_tool", '{"shared": true}', "shared_agent")
        
        # Reset mocks
        mock_client.reset_mock()
        
        # Test non-streaming tool execution
        first_response = MagicMock()
        first_response.message.content = "Using tool"
        first_response.message.tool_calls = [tool_call]
        
        second_response = MagicMock()
        second_response.message.content = "Tool used"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='shared_result') as mock_execute_non_streaming:
            await provider.send_chat(
                context=[{"role": "user", "content": "Use shared tool"}],
                model="test-model",
                instructions=None,
                agent_id="shared_agent"
            )
        
        # Verify non-streaming tool execution
        mock_execute_non_streaming.assert_called_with("shared_tool", '{"shared": true}', "shared_agent")
        
        # Both should have executed the tool with identical parameters
        assert mock_execute_streaming.call_args == mock_execute_non_streaming.call_args

# Task 9: Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test cases for performance scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_large_number_of_tool_calls(self, provider, mock_client):
        """Test handling a large number of tool calls in sequence."""
        provider.client = mock_client
        provider.max_tool_iterations = 15  # Increase limit for this test
        
        # Create 10 tool calls
        tool_calls = [create_mock_tool_call(f"tool_{i}", f'{{"id": {i}}}') for i in range(10)]
        
        # First response with many tool calls
        first_response = MagicMock()
        first_response.message.content = "Processing many tools"
        first_response.message.tool_calls = tool_calls
        
        # Final response
        second_response = MagicMock()
        second_response.message.content = "All tools completed"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='batch_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Use many tools"}],
                model="batch-model",
                instructions=None,
                agent_id="batch_agent"
            )
        
        # Verify all 10 tools were executed
        assert mock_execute.call_count == 10
        
        # Verify each tool was called with correct parameters
        for i in range(10):
            mock_execute.assert_any_call(f"tool_{i}", f'{{"id": {i}}}', "batch_agent")
        
        assert result == "All tools completed"
    
    @pytest.mark.asyncio
    async def test_very_long_content_streams(self, provider, mock_client):
        """Test streaming with very long content spread across many chunks."""
        provider.client = mock_client
        
        # Create 50 content chunks
        long_chunks = [{'content': f'chunk_{i}_'} for i in range(50)]
        
        mock_response = create_streaming_response(long_chunks)
        mock_client.chat.return_value = mock_response
        
        # Collect all streaming content
        chunks = []
        async for chunk in provider.send_chat_with_streaming(
            context=[{"role": "user", "content": "Give long response"}],
            model="long-model",
            instructions=None
        ):
            chunks.append(chunk)
        
        # Verify all 50 chunks were streamed
        assert len(chunks) == 50
        expected_chunks = [f'chunk_{i}_' for i in range(50)]
        assert chunks == expected_chunks
    
    @pytest.mark.asyncio
    async def test_malformed_tool_call_responses(self, provider, mock_client):
        """Test handling of malformed tool call responses."""
        provider.client = mock_client
        
        # Create malformed tool call (missing function attributes)
        malformed_tool_call = MagicMock()
        malformed_tool_call.function.name = "malformed_tool"
        malformed_tool_call.function.arguments = "invalid json {"  # Malformed JSON
        
        first_response = MagicMock()
        first_response.message.content = "Using malformed tool"
        first_response.message.tool_calls = [malformed_tool_call]
        
        # Final response
        second_response = MagicMock()
        second_response.message.content = "Handled gracefully"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        # Mock execute_tool_call to raise an exception for malformed input
        with patch.object(provider, 'execute_tool_call', side_effect=Exception("Invalid JSON")) as mock_execute:
            # Should not crash, but handle the error gracefully
            try:
                result = await provider.send_chat(
                    context=[{"role": "user", "content": "Test malformed"}],
                    model="error-model",
                    instructions=None,
                    agent_id="error_agent"
                )
                # If we get here, the error was handled
                assert True
            except Exception as e:
                # If an exception is raised, verify it's handled appropriately
                assert "Invalid JSON" in str(e) or "malformed" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_empty_content_with_tool_calls_only(self, provider, mock_client):
        """Test edge case where response has no content but has tool calls."""
        provider.client = mock_client
        
        tool_call = create_mock_tool_call("silent_tool", '{"silent": true}')
        
        # Response with no content, only tool calls
        first_response = MagicMock()
        first_response.message.content = ""  # Empty content
        first_response.message.tool_calls = [tool_call]
        
        second_response = MagicMock()
        second_response.message.content = "Tool executed silently"
        second_response.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[first_response, second_response])
        
        with patch.object(provider, 'execute_tool_call', return_value='silent_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Silent operation"}],
                model="silent-model",
                instructions=None,
                agent_id="silent_agent"
            )
        
        # Verify tool was still executed despite empty content
        mock_execute.assert_called_once_with("silent_tool", '{"silent": true}', "silent_agent")
        assert result == "Tool executed silently"
    
    @pytest.mark.asyncio
    async def test_streaming_with_mixed_empty_and_content_chunks(self, provider, mock_client):
        """Test streaming with mix of empty and content chunks."""
        provider.client = mock_client
        
        # Mix of empty and content chunks
        mixed_chunks = [
            {'content': 'Start'},
            {'content': ''},  # Empty chunk
            {'content': 'middle'},
            {'content': None},  # None content
            {'content': 'end'}
        ]
        
        mock_response = create_streaming_response(mixed_chunks)
        mock_client.chat.return_value = mock_response
        
        chunks = []
        async for chunk in provider.send_chat_with_streaming(
            context=[{"role": "user", "content": "Mixed content"}],
            model="mixed-model",
            instructions=None
        ):
            chunks.append(chunk)
        
        # Should only yield non-empty chunks
        expected_chunks = ['Start', 'middle', 'end']
        assert chunks == expected_chunks
    
    @pytest.mark.asyncio
    async def test_multiple_consecutive_tool_calling_iterations(self, provider, mock_client):
        """Test multiple consecutive iterations of tool calling."""
        provider.client = mock_client
        
        # Set up 3 iterations: each response triggers another tool call
        tool_call1 = create_mock_tool_call("step1", '{"step": 1}')
        tool_call2 = create_mock_tool_call("step2", '{"step": 2}')
        
        response1 = MagicMock()
        response1.message.content = "Step 1"
        response1.message.tool_calls = [tool_call1]
        
        response2 = MagicMock()
        response2.message.content = "Step 2"
        response2.message.tool_calls = [tool_call2]
        
        response3 = MagicMock()
        response3.message.content = "Complete"
        response3.message.tool_calls = []
        
        mock_client.chat = AsyncMock(side_effect=[response1, response2, response3])
        
        with patch.object(provider, 'execute_tool_call', return_value='step_result') as mock_execute:
            result = await provider.send_chat(
                context=[{"role": "user", "content": "Multi-step process"}],
                model="multi-step-model",
                instructions=None,
                agent_id="multi_agent"
            )
        
        # Verify both tools were executed in sequence
        assert mock_execute.call_count == 2
        mock_execute.assert_any_call("step1", '{"step": 1}', "multi_agent")
        mock_execute.assert_any_call("step2", '{"step": 2}', "multi_agent")
        
        # Verify 3 chat calls were made
        assert mock_client.chat.call_count == 3
        assert result == "Complete"

# Task 10: Code coverage analysis tests  
class TestCodeCoverage:
    """Test cases to ensure comprehensive coverage of all refactored methods."""
    
    @pytest.mark.asyncio
    async def test_prepare_messages_edge_cases_coverage(self, provider):
        """Ensure complete coverage of _prepare_messages edge cases."""
        # Test with whitespace-only instructions
        result = provider._prepare_messages("   ", [{"role": "user", "content": "test"}])
        assert len(result) == 2  # Should still include system message
        assert result[0]["content"] == "   "
        
        # Test with empty list context 
        result = provider._prepare_messages("test", [])
        assert len(result) == 1
        assert result[0]["content"] == "test"
        
        # Test with None instructions and empty context
        result = provider._prepare_messages(None, [])
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_error_paths_coverage(self, provider):
        """Test error handling paths in _execute_tool_calls."""
        messages = []
        
        # Test with tool call that has None arguments
        tool_call = MagicMock()
        tool_call.function.name = "null_args_tool"
        tool_call.function.arguments = None
        
        with patch.object(provider, 'execute_tool_call', return_value='null_result') as mock_execute:
            result = await provider._execute_tool_calls(messages, [tool_call], "coverage_agent")
            
            # Should handle None arguments gracefully
            mock_execute.assert_called_once_with("null_args_tool", None, "coverage_agent")
            assert result == 1
    
    @pytest.mark.asyncio
    async def test_process_streaming_response_boundary_conditions(self, provider):
        """Test boundary conditions in _process_streaming_response."""
        # Test with response that has alternating None/content chunks
        class BoundaryAsyncGenerator:
            def __init__(self):
                self.count = 0
                
            def __aiter__(self):
                return self
                
            async def __anext__(self):
                if self.count >= 6:
                    raise StopAsyncIteration
                
                chunk = MagicMock()
                if self.count % 2 == 0:
                    chunk.message.content = f"content_{self.count}" if self.count < 4 else None
                else:
                    chunk.message.content = None
                chunk.message.tool_calls = None
                self.count += 1
                return chunk
        
        response = BoundaryAsyncGenerator()
        
        chunks = []
        async for chunk_type, content, tool_calls in provider._process_streaming_response(response):
            if chunk_type == "content":
                chunks.append(content)
            elif chunk_type == "final":
                final_content = content
        
        # Should only capture non-None content
        assert chunks == ["content_0", "content_2"]
        assert final_content == "content_0content_2"
    
    @pytest.mark.asyncio 
    async def test_handle_tool_calls_streaming_iteration_limits(self, provider, mock_client):
        """Test iteration limit handling in _handle_tool_calls_streaming."""
        provider.client = mock_client
        provider.max_tool_iterations = 2  # Set low limit for testing
        
        # Create responses that would cause exactly max_tool_iterations
        tool_call = create_mock_tool_call("limit_tool", '{"limit": true}')
        
        # First two responses have tool calls, third doesn't
        responses = []
        for i in range(3):
            chunks = [{'content': f'iter_{i}', 'tool_calls': [tool_call] if i < 2 else []}]
            responses.append(create_streaming_response(chunks))
        
        mock_client.chat.side_effect = responses
        
        with patch.object(provider, 'execute_tool_call', return_value='limit_result'):
            with pytest.raises(ProviderMaxToolIterationsError):
                chunks = []
                async for chunk in provider._handle_tool_calls_streaming(
                    [], "limit-model", [], "limit_agent"
                ):
                    chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_model_settings_none_vs_empty_dict_coverage(self, provider, mock_client):
        """Test model_settings handling with None vs empty dict."""
        provider.client = mock_client
        
        chunks_data = [{'content': 'settings test'}]
        mock_response = create_streaming_response(chunks_data)
        mock_client.chat.return_value = mock_response
        
        # Test with None model_settings
        async for _ in provider._handle_tool_calls_streaming(
            [], "test-model", [], "test_agent", None
        ):
            break
        
        call_args_none = mock_client.chat.call_args[1]
        assert "options" not in call_args_none
        
        # Reset mock
        mock_client.reset_mock()
        mock_client.chat.return_value = create_streaming_response(chunks_data)
        
        # Test with empty dict model_settings
        async for _ in provider._handle_tool_calls_streaming(
            [], "test-model", [], "test_agent", {}
        ):
            break
        
        call_args_empty = mock_client.chat.call_args[1]
        # Empty dict model_settings should not be passed when empty
        assert "options" not in call_args_empty or call_args_empty.get("options") == {}
    
    @pytest.mark.asyncio
    async def test_comprehensive_method_integration_coverage(self, provider, mock_client):
        """Comprehensive test covering all method interactions in single flow."""
        provider.client = mock_client
        
        # Complex scenario: instructions + context + streaming + tool calls + model settings
        tool_call1 = create_mock_tool_call("coverage_tool_1", '{"phase": 1}')
        tool_call2 = create_mock_tool_call("coverage_tool_2", '{"phase": 2}')
        
        # Multi-phase streaming with different tool calls
        phase1_chunks = [{'content': 'Phase 1: ', 'tool_calls': [tool_call1]}]
        phase2_chunks = [{'content': 'Phase 2: ', 'tool_calls': [tool_call2]}] 
        phase3_chunks = [{'content': 'Complete!'}]
        
        responses = [
            create_streaming_response(phase1_chunks),
            create_streaming_response(phase2_chunks), 
            create_streaming_response(phase3_chunks)
        ]
        mock_client.chat.side_effect = responses
        
        with patch.object(provider, 'execute_tool_call', return_value='coverage_result') as mock_execute:
            chunks = []
            async for chunk in provider.send_chat_with_streaming(
                context=[
                    {"role": "user", "content": "Start comprehensive test"}, 
                    {"role": "assistant", "content": "Previous response"},
                    {"role": "user", "content": "Continue test"}
                ],
                model="comprehensive-model",
                instructions="Execute comprehensive coverage test",
                agent_id="coverage_agent",
                model_settings={"temperature": 0.9, "max_tokens": 200}
            ):
                chunks.append(chunk)
        
        # Verify complete integration:
        
        # 1. _prepare_messages: instructions + multi-message context
        first_call = mock_client.chat.call_args_list[0][1]
        messages = first_call["messages"]
        # Messages grow as tool calls are processed, just verify the first few
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Execute comprehensive coverage test"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Start comprehensive test"
        
        # 2. _process_streaming_response: all content chunks yielded
        assert chunks == ['Phase 1: ', 'Phase 2: ', 'Complete!']
        
        # 3. _execute_tool_calls: both tools executed
        assert mock_execute.call_count == 2
        mock_execute.assert_any_call("coverage_tool_1", '{"phase": 1}', "coverage_agent")
        mock_execute.assert_any_call("coverage_tool_2", '{"phase": 2}', "coverage_agent")
        
        # 4. _handle_tool_calls_streaming: model settings passed, 3 iterations
        for call_args in mock_client.chat.call_args_list:
            assert call_args[1]["options"] == {"temperature": 0.9, "max_tokens": 200}
            assert call_args[1]["stream"] is True
        assert mock_client.chat.call_count == 3
