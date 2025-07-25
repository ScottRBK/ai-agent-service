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
async def test_health_check_healthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProviderCC(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "healthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is None

@patch("app.core.providers.azureopenapi_cc.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_health_check_unhealthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Test error"))
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProviderCC(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "unhealthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is "Test error"

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
    mock_message.tool_calls = []
    
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
    mock_client.chat.completions.create.assert_awaited_with(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "hi"},
        ],
        tools=None
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
    mock_message.tool_calls = []
    
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
        ],
        tools=None
    )

# Add missing import
from unittest.mock import MagicMock

# ============================================================================
# New Private Method Tests (following Ollama refactoring pattern)
# ============================================================================

class TestPrepareMessages:
    """Tests for _prepare_messages method"""
    
    def test_prepare_messages_with_instructions_and_context(self):
        """Test message preparation with both instructions and context"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15", 
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        instructions = "You are a helpful assistant"
        context = [{"role": "user", "content": "Hello"}]
        
        # Act
        result = provider._prepare_messages(instructions, context)
        
        # Assert
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are a helpful assistant"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_prepare_messages_with_instructions_only(self):
        """Test message preparation with instructions only"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        instructions = "You are a helpful assistant"
        context = []
        
        # Act
        result = provider._prepare_messages(instructions, context)
        
        # Assert
        assert len(result) == 1
        assert result[0] == {"role": "system", "content": "You are a helpful assistant"}

    def test_prepare_messages_with_context_only(self):
        """Test message preparation with context only"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        instructions = ""
        context = [{"role": "user", "content": "Hello"}]
        
        # Act
        result = provider._prepare_messages(instructions, context)
        
        # Assert
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_prepare_messages_with_none_instructions(self):
        """Test message preparation with None instructions"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        instructions = None
        context = [{"role": "user", "content": "Hello"}]
        
        # Act
        result = provider._prepare_messages(instructions, context)
        
        # Assert
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_prepare_messages_empty_both(self):
        """Test message preparation with empty instructions and context"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        instructions = ""
        context = []
        
        # Act
        result = provider._prepare_messages(instructions, context)
        
        # Assert
        assert len(result) == 0

class TestParseStreamingToolCalls:
    """Tests for _parse_streaming_tool_calls method"""
    
    @pytest.mark.asyncio
    async def test_parse_streaming_tool_calls_single_tool(self):
        """Test parsing single tool call from streaming response"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        # Create mock streaming response
        async def mock_response():
            # Content chunk
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "I'll help you"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            yield chunk1
            
            # Tool call chunk
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].delta.tool_calls = [MagicMock()]
            chunk2.choices[0].delta.tool_calls[0].id = "call_123"
            chunk2.choices[0].delta.tool_calls[0].function.name = "test_tool"
            chunk2.choices[0].delta.tool_calls[0].function.arguments = '{"param": "value"}'
            chunk2.choices[0].finish_reason = "tool_calls"
            yield chunk2
        
        # Act
        results = []
        async for chunk_type, content, tool_calls in provider._parse_streaming_tool_calls(mock_response()):
            results.append((chunk_type, content, tool_calls))
        
        # Assert
        assert len(results) == 2
        assert results[0] == ("content", "I'll help you", [])
        assert results[1][0] == "final"
        assert results[1][1] == "I'll help you"
        assert len(results[1][2]) == 1
        assert results[1][2][0]["id"] == "call_123"
        assert results[1][2][0]["name"] == "test_tool"
        assert results[1][2][0]["args"] == '{"param": "value"}'

    @pytest.mark.asyncio
    async def test_parse_streaming_tool_calls_incremental_arguments(self):
        """Test parsing tool call with incremental argument building"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15", 
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        async def mock_response():
            # First tool call chunk with ID and name
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = None
            chunk1.choices[0].delta.tool_calls = [MagicMock()]
            chunk1.choices[0].delta.tool_calls[0].id = "call_123"
            chunk1.choices[0].delta.tool_calls[0].function.name = "test_tool"
            chunk1.choices[0].delta.tool_calls[0].function.arguments = '{"param":'
            chunk1.choices[0].finish_reason = None
            yield chunk1
            
            # Second chunk with more arguments
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].delta.tool_calls = [MagicMock()]
            chunk2.choices[0].delta.tool_calls[0].id = "call_123"
            chunk2.choices[0].delta.tool_calls[0].function.name = None
            chunk2.choices[0].delta.tool_calls[0].function.arguments = ' "value"}'
            chunk2.choices[0].finish_reason = "tool_calls"
            yield chunk2
        
        # Act
        results = []
        async for chunk_type, content, tool_calls in provider._parse_streaming_tool_calls(mock_response()):
            results.append((chunk_type, content, tool_calls))
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == "final"
        assert len(results[0][2]) == 1
        assert results[0][2][0]["args"] == '{"param": "value"}'

    @pytest.mark.asyncio
    async def test_parse_streaming_tool_calls_multiple_tools(self):
        """Test parsing multiple tool calls"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key", 
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        async def mock_response():
            # First tool call
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = None
            chunk1.choices[0].delta.tool_calls = [MagicMock()]
            chunk1.choices[0].delta.tool_calls[0].id = "call_1"
            chunk1.choices[0].delta.tool_calls[0].function.name = "tool1"
            chunk1.choices[0].delta.tool_calls[0].function.arguments = '{"param1": "value1"}'
            chunk1.choices[0].finish_reason = None
            yield chunk1
            
            # Second tool call
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].delta.tool_calls = [MagicMock()]
            chunk2.choices[0].delta.tool_calls[0].id = "call_2"
            chunk2.choices[0].delta.tool_calls[0].function.name = "tool2"
            chunk2.choices[0].delta.tool_calls[0].function.arguments = '{"param2": "value2"}'
            chunk2.choices[0].finish_reason = "tool_calls"
            yield chunk2
        
        # Act
        results = []
        async for chunk_type, content, tool_calls in provider._parse_streaming_tool_calls(mock_response()):
            results.append((chunk_type, content, tool_calls))
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == "final"
        assert len(results[0][2]) == 2
        assert results[0][2][0]["id"] == "call_1"
        assert results[0][2][1]["id"] == "call_2"

    @pytest.mark.asyncio
    async def test_parse_streaming_tool_calls_with_error(self):
        """Test error handling during streaming"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        async def mock_response():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Hello"
            yield chunk
            raise Exception("Streaming error")
        
        # Act
        results = []
        async for chunk_type, content, tool_calls in provider._parse_streaming_tool_calls(mock_response()):
            results.append((chunk_type, content, tool_calls))
        
        # Assert
        assert len(results) == 2
        assert results[0] == ("content", "Hello", [])
        assert results[1][0] == "error"
        assert "Error: Streaming interrupted" in results[1][1]

    @pytest.mark.asyncio
    async def test_parse_streaming_tool_calls_content_only(self):
        """Test parsing content-only response without tool calls"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        async def mock_response():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"
            chunk1.choices[0].delta.tool_calls = None
            chunk1.choices[0].finish_reason = None
            yield chunk1
            
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = " world"
            chunk2.choices[0].delta.tool_calls = None
            chunk2.choices[0].finish_reason = "stop"
            yield chunk2
        
        # Act
        results = []
        async for chunk_type, content, tool_calls in provider._parse_streaming_tool_calls(mock_response()):
            results.append((chunk_type, content, tool_calls))
        
        # Assert
        assert len(results) == 3
        assert results[0] == ("content", "Hello", [])
        assert results[1] == ("content", " world", [])
        assert results[2] == ("final", "Hello world", [])

class TestExecuteToolCallsCC:
    """Tests for _execute_tool_calls method"""
    
    @pytest.mark.asyncio
    async def test_execute_tool_calls_single_tool(self):
        """Test executing a single tool call with message creation"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        provider.execute_tool_call = AsyncMock(return_value="tool_result")
        
        messages = []
        tool_calls = [
            {"id": "call_123", "name": "test_tool", "args": '{"param": "value"}'}
        ]
        agent_id = "test_agent"
        
        # Act
        tool_count = await provider._execute_tool_calls(messages, tool_calls, agent_id)
        
        # Assert
        assert tool_count == 1
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "call_123"
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "tool_result"
        assert messages[1]["tool_call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_multiple_tools(self):
        """Test executing multiple tool calls"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        provider.execute_tool_call = AsyncMock(side_effect=["result1", "result2"])
        
        messages = []
        tool_calls = [
            {"id": "call_1", "name": "tool1", "args": '{"param1": "value1"}'},
            {"id": "call_2", "name": "tool2", "args": '{"param2": "value2"}'}
        ]
        agent_id = "test_agent"
        
        # Act
        tool_count = await provider._execute_tool_calls(messages, tool_calls, agent_id)
        
        # Assert
        assert tool_count == 2
        assert len(messages) == 4  # 2 assistant messages + 2 tool result messages

    @pytest.mark.asyncio
    async def test_execute_tool_calls_invalid_json(self):
        """Test handling invalid JSON in tool arguments"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        
        messages = []
        tool_calls = [
            {"id": "call_123", "name": "test_tool", "args": "invalid json"}
        ]
        agent_id = "test_agent"
        
        # Act
        tool_count = await provider._execute_tool_calls(messages, tool_calls, agent_id)
        
        # Assert
        assert tool_count == 0
        assert len(messages) == 0  # No messages added due to invalid JSON

class TestHandleToolCallsStreamingCC:
    """Tests for _handle_tool_calls_streaming method"""
    
    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_single_iteration(self):
        """Test single iteration without tool calls"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        provider.client = AsyncMock()
        
        # Mock streaming response
        async def mock_response():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Hello"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            yield chunk
        
        provider.client.chat.completions.create.return_value = mock_response()
        
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-35-turbo"
        available_tools = []
        agent_id = "test_agent"
        
        # Act
        results = []
        async for content in provider._handle_tool_calls_streaming(messages, model, available_tools, agent_id):
            results.append(content)
        
        # Assert
        assert "Hello" in results
        provider.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tool_calls_streaming_max_iterations(self):
        """Test max iterations handling"""
        # Arrange
        from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC
        from app.models.providers import AzureOpenAIConfig
        from app.core.providers.base import ProviderMaxToolIterationsError
        
        config = AzureOpenAIConfig(
            name="test-provider",
            api_version="2023-05-15",
            base_url="https://example.openai.azure.com/",
            api_key="test-key",
            model_list=["gpt-35-turbo"]
        )
        provider = AzureOpenAIProviderCC(config)
        provider.client = AsyncMock()
        provider.max_tool_iterations = 1
        provider.execute_tool_call = AsyncMock(return_value="result")
        
        # Mock response that always returns tool calls
        async def mock_response():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.tool_calls = [MagicMock()]
            chunk.choices[0].delta.tool_calls[0].id = "call_123"
            chunk.choices[0].delta.tool_calls[0].function.name = "test_tool"
            chunk.choices[0].delta.tool_calls[0].function.arguments = '{"param": "value"}'
            chunk.choices[0].finish_reason = "tool_calls"
            yield chunk
        
        provider.client.chat.completions.create.return_value = mock_response()
        
        messages = [{"role": "user", "content": "Use tools"}]
        model = "gpt-35-turbo"
        available_tools = [{"type": "function", "name": "test_tool"}]
        agent_id = "test_agent"
        
        # Act & Assert
        with pytest.raises(ProviderMaxToolIterationsError):
            async for content in provider._handle_tool_calls_streaming(messages, model, available_tools, agent_id):
                pass