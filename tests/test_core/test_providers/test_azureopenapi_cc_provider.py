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
        """Test max iterations graceful handling"""
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
        provider.max_tool_iterations = 1
        provider.execute_tool_call = AsyncMock(return_value="result")
        
        # First response with tool calls (triggers limit)
        async def mock_first_response():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            chunk.choices[0].delta.tool_calls = [MagicMock()]
            chunk.choices[0].delta.tool_calls[0].id = "call_123"
            chunk.choices[0].delta.tool_calls[0].function.name = "test_tool"
            chunk.choices[0].delta.tool_calls[0].function.arguments = '{"param": "value"}'
            chunk.choices[0].finish_reason = "tool_calls"
            yield chunk
        
        # Final response after hitting limit (without tools)
        async def mock_final_response():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Max tool calls reached."
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            yield chunk
        
        # Mock responses in sequence
        provider.client.chat.completions.create.side_effect = [
            mock_first_response(),
            mock_final_response()
        ]
        
        messages = [{"role": "user", "content": "Use tools"}]
        model = "gpt-35-turbo"
        available_tools = [{"type": "function", "name": "test_tool"}]
        agent_id = "test_agent"
        
        # Act - collect all streaming chunks
        with patch('app.core.providers.azureopenapi_cc.logger') as mock_logger:
            chunks = []
            async for content in provider._handle_tool_calls_streaming(messages, model, available_tools, agent_id):
                chunks.append(content)
        
        # Assert
        # Verify tool was executed once (hitting the limit)
        provider.execute_tool_call.assert_called_once_with("test_tool", {'param': 'value'}, agent_id)
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        assert "max tool iterations reached" in mock_logger.warning.call_args[0][0].lower()
        
        # Verify graceful response was streamed
        assert "Max tool calls reached." in chunks
        
        # Verify two chat calls were made (initial + final without tools)
        assert provider.client.chat.completions.create.call_count == 2


# ============================================================================
# Embedding Function Tests
# ============================================================================

class TestEmbedFunction:
    """Test cases for the embed method"""
    
    @pytest.fixture
    def provider(self, mock_config):
        """Create provider instance for testing"""
        return AzureOpenAIProviderCC(mock_config)
    
    @pytest.mark.asyncio
    async def test_embed_successful_generation(self, provider):
        """Test successful embedding generation"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        text = "This is a test text for embedding"
        model = "text-embedding-3-small"
        
        result = await provider.embed(text, model)
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=text
        )
    
    @pytest.mark.asyncio
    async def test_embed_with_different_models(self, provider):
        """Test embedding with different Azure OpenAI models"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        models_to_test = [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002",
            "custom-embedding-model"
        ]
        
        for model in models_to_test:
            mock_response = MagicMock()
            # Different dimensions for different models
            dimensions = 1536 if "ada-002" in model else 3072 if "large" in model else 1536
            mock_response.data = [MagicMock(embedding=[0.1] * dimensions)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            text = f"Test text for {model}"
            result = await provider.embed(text, model)
            
            assert len(result) == dimensions
            assert result == [0.1] * dimensions
            mock_client.embeddings.create.assert_called_with(
                model=model,
                input=text
            )
    
    @pytest.mark.asyncio
    async def test_embed_with_long_text(self, provider):
        """Test embedding with long text content"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        # Create long text (8KB)
        long_text = "This is a very long text for embedding. " * 200  # ~8KB
        model = "text-embedding-3-small"
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(long_text, model)
        
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=long_text
        )
    
    @pytest.mark.asyncio
    async def test_embed_with_unicode_text(self, provider):
        """Test embedding with Unicode text"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        unicode_text = "Hello 世界! 🌍 This contains émojis and ñoñ-ASCII çharacters"
        model = "text-embedding-3-small"
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(unicode_text, model)
        
        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=unicode_text
        )
    
    @pytest.mark.asyncio
    async def test_embed_with_empty_text(self, provider):
        """Test embedding with empty text"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        empty_text = ""
        model = "text-embedding-3-small"
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.0] * 1536)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(empty_text, model)
        
        assert len(result) == 1536
        assert all(x == 0.0 for x in result)
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=empty_text
        )
    
    @pytest.mark.asyncio
    async def test_embed_dimension_consistency(self, provider):
        """Test embedding dimension consistency for Chat Completions API"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        model = "text-embedding-3-small"
        texts = [
            "Short text",
            "This is a medium length text with some more content to test consistency",
            "This is a very long text that contains multiple sentences and should still produce the same dimensional embedding as shorter texts for consistency in vector storage and search operations. The embedding model should maintain consistent dimensions regardless of input length."
        ]
        
        for text in texts:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            result = await provider.embed(text, model)
            
            assert len(result) == 1536, f"Expected 1536 dimensions for '{text[:50]}...'"
            assert all(isinstance(x, (int, float)) for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_api_error_handling(self, provider):
        """Test embedding API error handling"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        # Test different error scenarios
        error_scenarios = [
            Exception("Network connection failed"),
            Exception("API key invalid"),
            Exception("Rate limit exceeded"),
            Exception("Model not found"),
            Exception("Input too long")
        ]
        
        for error in error_scenarios:
            mock_client.embeddings.create = AsyncMock(side_effect=error)
            
            with pytest.raises(Exception) as exc_info:
                await provider.embed(text, model)
            
            assert str(exc_info.value) == str(error)
            mock_client.embeddings.create.assert_called_with(
                model=model,
                input=text
            )
    
    @pytest.mark.asyncio
    async def test_embed_response_format_validation(self, provider):
        """Test validation of embedding response format"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        # Test various valid response formats
        valid_embeddings = [
            [0.1, 0.2, 0.3],  # Standard format
            [1.0, -0.5, 0.0, 0.8],  # With negative values
            [0.123456789] * 100,  # High precision floats
        ]
        
        for embedding in valid_embeddings:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=embedding)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            result = await provider.embed(text, model)
            
            assert result == embedding
            assert all(isinstance(x, (int, float)) for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_malformed_response_handling(self, provider):
        """Test handling of malformed embedding responses"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        # Test malformed responses
        malformed_responses = [
            MagicMock(data=None),  # None data
            MagicMock(data=[]),    # Empty data list
            MagicMock(data=[MagicMock(embedding=None)]),  # None embedding
            MagicMock(),  # Missing data attribute
        ]
        
        for mock_response in malformed_responses:
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            with pytest.raises((AttributeError, IndexError, TypeError)):
                await provider.embed(text, model)
    
    @pytest.mark.asyncio
    async def test_embed_multiple_data_entries(self, provider):
        """Test handling response with multiple data entries (taking first one)"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        # Mock response with multiple data entries
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),  # First entry (should be returned)
            MagicMock(embedding=[0.4, 0.5, 0.6]),  # Second entry (should be ignored)
            MagicMock(embedding=[0.7, 0.8, 0.9])   # Third entry (should be ignored)
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(text, model)
        
        # Should return only the first embedding
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=text
        )
    
    @pytest.mark.asyncio
    async def test_embed_numerical_precision(self, provider):
        """Test handling of high-precision floating point embeddings"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        # High precision float embeddings
        high_precision_embeddings = [
            0.123456789012345,
            -0.987654321098765,
            1.23456789e-10,
            -9.87654321e+5
        ]
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=high_precision_embeddings)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(text, model)
        
        assert result == high_precision_embeddings
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_large_dimension_models(self, provider):
        """Test embedding with large dimension models"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        text = "Test text for large dimension model"
        model = "text-embedding-3-large"
        
        # Simulate large dimension embedding (3072 for text-embedding-3-large)
        large_embedding = [0.001 * i for i in range(3072)]
        
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=large_embedding)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await provider.embed(text, model)
        
        assert len(result) == 3072
        assert result == large_embedding
        mock_client.embeddings.create.assert_called_once_with(
            model=model,
            input=text
        )
    
    @pytest.mark.asyncio
    async def test_embed_concurrent_requests(self, provider):
        """Test concurrent embedding requests"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        import asyncio
        
        model = "text-embedding-3-small"
        texts = [f"Text {i}" for i in range(5)]
        
        # Mock responses for each request
        async def mock_embed_side_effect(model, input):
            # Different embeddings for each text
            text_index = int(input.split()[-1])
            embedding = [0.1 * text_index] * 3
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=embedding)]
            return mock_response
        
        mock_client.embeddings.create = AsyncMock(side_effect=mock_embed_side_effect)
        
        # Execute concurrent requests
        tasks = [provider.embed(text, model) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            expected_value = 0.1 * i
            assert all(x == expected_value for x in result)
        
        assert mock_client.embeddings.create.call_count == 5
    
    @pytest.mark.asyncio
    async def test_embed_with_special_characters(self, provider):
        """Test embedding with special characters and edge cases"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        special_texts = [
            "Text with\nnewlines\tand\ttabs",
            "Text with \"quotes\" and 'apostrophes'",
            "Text with mathematical symbols: ∑∫∂∇∆∞",
            "Text with code: def func(x): return x ** 2",
            "Mixed: English 中文 العربية русский ελληνικά"
        ]
        
        model = "text-embedding-3-small"
        
        for text in special_texts:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            result = await provider.embed(text, model)
            
            assert len(result) == 1536
            assert all(isinstance(x, (int, float)) for x in result)
            mock_client.embeddings.create.assert_called_with(
                model=model,
                input=text
            )
    
    @pytest.mark.asyncio
    async def test_embed_with_none_client(self, provider):
        """Test embedding when client is None"""
        provider.client = None
        
        text = "Test text"
        model = "text-embedding-3-small"
        
        with pytest.raises(AttributeError):
            await provider.embed(text, model)
    
    @pytest.mark.asyncio
    async def test_embed_method_signature_consistency(self, provider):
        """Test that embed method signature matches expected pattern"""
        mock_client = AsyncMock()
        provider.client = mock_client
        
        # Verify the method signature matches the base provider pattern
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        
        # Test with positional arguments as expected in base provider signature
        result = await provider.embed("test text", "test-model")
        
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="test-model",
            input="test text"
        )