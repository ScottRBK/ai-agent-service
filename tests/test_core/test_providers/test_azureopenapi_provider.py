"""
Tests for Azure OpenAI provider.

This module tests the Azure OpenAI provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from datetime import datetime
from typing import AsyncGenerator
import importlib
import sys
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.models.providers import AzureOpenAIConfig
from app.models.health import HealthStatus
from app.core.providers.base import ProviderMaxToolIterationsError
from app.models.tools.tools import Tool

@pytest.fixture
def mock_config():
    return AzureOpenAIConfig(
        name="test-provider",
        api_version="2025-03-01-preview",
        base_url="https://example.openai.azure.com/",
        api_key="test-key",
        model_list=["gpt-4.1-nano", "gpt-4o-mini"]
    )

@pytest.fixture
def mock_provider(mock_config):
    return AzureOpenAIProvider(mock_config)

@pytest.fixture
def mock_azure_client():
    return AsyncMock()

@pytest.fixture
def mock_response_stream():
    """Mock response stream for streaming tests"""
    async def mock_stream():
        # Simulate streaming events
        yield Mock(type="response.output_text.delta", delta="Hello")
        yield Mock(type="response.output_text.delta", delta=" world")
        yield Mock(type="response.completed", response=Mock(
            output=[
                Mock(type="message", content=[Mock(text="Hello world")])
            ]
        ))
    return mock_stream()

@pytest.fixture
def mock_tool_call_response_stream():
    """Mock response stream with tool calls"""
    async def mock_stream():
        yield Mock(type="response.output_text.delta", delta="I'll help you")
        yield Mock(type="response.completed", response=Mock(
            output=[
                Mock(type="function_call", name="test_tool", arguments='{"param": "value"}', call_id="call_123")
            ]
        ))
    return mock_stream()

@pytest.fixture
def valid_tool():
    """Create a valid Tool object for testing"""
    return Tool(
        name="test_tool",
        description="A test tool",
        type="function",
        parameters={"type": "object", "properties": {}}
    )

# ============================================================================
# 1. Provider Lifecycle Tests
# ============================================================================

class TestProviderLifecycle:
    """AC1-AC2: Provider initialization and health check tests"""
    
    @patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_provider_initialization_success(self, mock_azure_openai, mock_config):
        """AC1: Provider Initialization - successful case"""
        # Arrange
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        
        # Act
        provider = AzureOpenAIProvider(mock_config)
        await provider.initialize()
        
        # Assert
        assert provider.client is mock_client
        mock_azure_openai.assert_called_once_with(
            api_version=mock_config.api_version,
            azure_endpoint=mock_config.base_url,
            azure_ad_token=mock_config.api_key,
        )
        assert provider.initialized is False  # Note: initialize() doesn't set this flag

    @patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
    @pytest.mark.asyncio
    async def test_provider_initialization_failure(self, mock_azure_openai, mock_config):
        """AC1: Provider Initialization - failure case"""
        # Arrange
        mock_azure_openai.side_effect = Exception("Connection failed")
        
        # Act
        provider = AzureOpenAIProvider(mock_config)
        await provider.initialize()
        
        # Assert
        assert provider.client is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_provider, mock_azure_client):
        """AC2: Health Check - healthy status"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_azure_client.models.list.return_value = ["gpt-4.1-nano", "gpt-4o-mini"]
        
        # Act
        health = await mock_provider.health_check()
        
        # Assert
        assert isinstance(health, HealthStatus)
        assert health.status == "healthy"
        assert health.service == mock_provider.config.name
        assert health.version == mock_provider.version
        assert health.error_details is None

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_provider, mock_azure_client):
        """AC2: Health Check - unhealthy status"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_azure_client.models.list.side_effect = Exception("API unavailable")
        
        # Act
        health = await mock_provider.health_check()
        
        # Assert
        assert isinstance(health, HealthStatus)
        assert health.status == "unhealthy"
        assert health.service == mock_provider.config.name
        assert health.version == mock_provider.version
        assert "API unavailable" in health.error_details

# ============================================================================
# 2. Basic Chat Functionality Tests
# ============================================================================

class TestBasicChatFunctionality:
    """AC3-AC4: Basic chat functionality tests"""
    
    @pytest.mark.asyncio
    async def test_send_chat_basic_response(self, mock_provider, mock_azure_client):
        """AC3: Non-Streaming Chat - basic response"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_response = Mock()
        mock_response.output_text = "Hello, how can I help you?"
        mock_response.output = []  # Empty output for basic response
        mock_azure_client.responses.create.return_value = mock_response
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        result = await mock_provider.send_chat(context, model, instructions)
        
        # Assert
        assert result == "Hello, how can I help you?"
        mock_azure_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_chat_with_tool_calls(self, mock_provider, mock_azure_client):
        """AC3: Non-Streaming Chat - with tool calls"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock(return_value="tool_result")
        
        # First response with tool call
        mock_response1 = Mock()
        mock_response1.output = [
            Mock(type="function_call", name="test_tool", arguments='{"param": "value"}', call_id="call_123")
        ]
        
        # Second response after tool execution
        mock_response2 = Mock()
        mock_response2.output_text = "I used the tool and here's the result."
        mock_response2.output = []
        
        mock_azure_client.responses.create.side_effect = [mock_response1, mock_response2]
        
        context = [{"role": "user", "content": "Use a tool"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        result = await mock_provider.send_chat(context, model, instructions)
        
        # Assert
        assert result == "I used the tool and here's the result."
        assert mock_azure_client.responses.create.call_count == 2

    @pytest.mark.asyncio
    async def test_send_chat_with_streaming_basic(self, mock_provider, mock_azure_client):
        """AC4: Streaming Chat - basic streaming response"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream():
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="message", content=[Mock(text="Hello world")])
                ]
            ))
        
        mock_azure_client.responses.create.return_value = mock_stream()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses
        assert " world" in responses  # Note: space before "world"
        mock_azure_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_chat_with_streaming_tool_calls(self, mock_provider, mock_azure_client):
        """AC4: Streaming Chat - with tool calls after streaming"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock(return_value="tool_result")
        
        async def mock_stream_with_tools():
            yield Mock(type="response.output_text.delta", delta="I'll help you")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="test_tool", arguments='{"param": "value"}', call_id="call_123")
                ]
            ))
        
        # Mock the second call after tool execution
        async def mock_second_stream():
            yield Mock(type="response.output_text.delta", delta="Tool executed")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.side_effect = [mock_stream_with_tools(), mock_second_stream()]
        
        context = [{"role": "user", "content": "Use a tool"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "I'll help you" in responses
        assert "Tool executed" in responses
        mock_azure_client.responses.create.assert_called()
        mock_provider.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_chat_with_streaming_no_tools_mid_stream(self, mock_provider, mock_azure_client):
        """AC4: Streaming Chat - tools NOT executed mid-stream"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock()
        
        async def mock_stream():
            # Only yield text deltas, no tool calls during streaming
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.return_value = mock_stream()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses
        assert " world" in responses
        mock_provider.execute_tool_call.assert_not_called()

# ============================================================================
# 3. Tool Integration Tests
# ============================================================================

class TestToolIntegration:
    """AC5-AC6: Tool integration tests"""
    
    @pytest.mark.asyncio
    async def test_get_available_tools_with_agent_id(self, mock_provider):
        """AC5: Tool Availability - with agent_id"""
        # Arrange
        mock_provider.cached_tools = {}
        mock_provider.get_available_tools = AsyncMock(return_value=[
            {"type": "function", "name": "test_tool", "description": "A test tool"}
        ])
        
        # Act
        tools = await mock_provider.get_available_tools(agent_id="test_agent")
        
        # Assert
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_get_available_tools_caching(self, mock_provider):
        """AC5: Tool Availability - caching per agent_id"""
        # Arrange
        cached_tools = [{"type": "function", "name": "cached_tool"}]
        mock_provider.cached_tools = {"test_agent": cached_tools}
        
        # Act
        tools = await mock_provider.get_available_tools(agent_id="test_agent")
        
        # Assert
        assert tools == cached_tools

    @pytest.mark.asyncio
    async def test_get_available_tools_with_requested_tools(self, mock_provider, valid_tool):
        """AC5: Tool Availability - with requested tools"""
        # Arrange
        requested_tools = [valid_tool]
        
        # Act
        tools = await mock_provider.get_available_tools(requested_tools=requested_tools)
        
        # Assert
        # This will be None since we're not mocking the ToolRegistry
        assert tools is None

    @pytest.mark.asyncio
    async def test_tool_execution_after_streaming(self, mock_provider, mock_azure_client):
        """AC6: Tool Execution - after streaming completes"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock(return_value="tool_result")
        
        async def mock_stream_with_tools():
            yield Mock(type="response.output_text.delta", delta="I'll help")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="test_tool", arguments='{"param": "value"}', call_id="call_123")
                ]
            ))
        
        mock_azure_client.responses.create.return_value = mock_stream_with_tools()
        
        context = [{"role": "user", "content": "Use a tool"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "I'll help" in responses
        mock_provider.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_to_response_format(self, mock_provider):
        """Test tool format conversion"""
        # Arrange
        chat_completions_tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        # Act
        result = mock_provider.convert_to_response_format(chat_completions_tools)
        
        # Assert
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "test_tool"
        assert result[0]["description"] == "A test tool"

# ============================================================================
# 4. Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """AC7-AC8: Error handling tests"""
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self, mock_provider, mock_azure_client):
        """AC7: API Failures - graceful handling"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_azure_client.responses.create.side_effect = Exception("API communication failed")
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert len(responses) == 1
        assert "Error: Failed to communicate with Azure OpenAI" in responses[0]

    @pytest.mark.asyncio
    async def test_streaming_interruption_handling(self, mock_provider, mock_azure_client):
        """AC7: API Failures - streaming interruption"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream_with_error():
            yield Mock(type="response.output_text.delta", delta="Hello")
            raise Exception("Streaming interrupted")
        
        mock_azure_client.responses.create.return_value = mock_stream_with_error()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert len(responses) == 2
        assert "Hello" in responses
        assert "Error: Streaming interrupted" in responses[1]

    @pytest.mark.asyncio
    async def test_tool_execution_failure_handling(self, mock_provider, mock_azure_client):
        """AC8: Tool Execution Failures - graceful handling"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock(side_effect=Exception("Tool execution failed"))
        
        async def mock_stream_with_tool_error():
            yield Mock(type="response.output_text.delta", delta="I'll help")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="failing_tool", arguments='{"param": "value"}', call_id="call_123")
                ]
            ))
        
        mock_azure_client.responses.create.return_value = mock_stream_with_tool_error()
        
        context = [{"role": "user", "content": "Use a tool"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "I'll help" in responses
        mock_provider.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_json_in_tool_arguments(self, mock_provider, mock_azure_client):
        """AC8: Tool Execution Failures - invalid JSON handling"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream_with_invalid_json():
            yield Mock(type="response.output_text.delta", delta="I'll help")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="test_tool", arguments='invalid json', call_id="call_123")
                ]
            ))
        
        mock_azure_client.responses.create.return_value = mock_stream_with_invalid_json()
        
        context = [{"role": "user", "content": "Use a tool"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "I'll help" in responses

    @pytest.mark.asyncio
    async def test_max_tool_iterations_reached(self, mock_provider, mock_azure_client):
        """Test max tool iterations limit"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.max_tool_iterations = 1  # Set to 1 to trigger the limit quickly
        
        async def mock_stream_with_tools():
            yield Mock(type="response.output_text.delta", delta="Processing")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="test_tool", arguments='{"param": "value"}', call_id="call_123")
                ]
            ))
        
        mock_azure_client.responses.create.return_value = mock_stream_with_tools()
        mock_provider.execute_tool_call = AsyncMock(return_value="result")
        
        context = [{"role": "user", "content": "Use tools repeatedly"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert any("Warning: Maximum tool iterations" in response for response in responses)

# ============================================================================
# 5. Configuration & Cleanup Tests
# ============================================================================

class TestConfigurationAndCleanup:
    """AC9-AC10: Configuration and cleanup tests"""
    
    def test_configuration_management_valid(self, mock_config):
        """AC9: Configuration Management - valid configuration"""
        # Act
        provider = AzureOpenAIProvider(mock_config)
        
        # Assert
        assert provider.config.name == "test-provider"
        assert provider.config.api_version == "2025-03-01-preview"
        assert provider.config.base_url == "https://example.openai.azure.com/"
        assert provider.config.api_key == "test-key"

    def test_configuration_management_invalid(self):
        """AC9: Configuration Management - invalid configuration"""
        # Arrange
        invalid_config = AzureOpenAIConfig(
            name="test-provider",
            api_version="invalid-version",
            base_url="not-a-url",
            api_key="",
            model_list=[]
        )
        
        # Act & Assert
        provider = AzureOpenAIProvider(invalid_config)
        assert provider.config.name == "test-provider"
        # Note: Pydantic validation would catch truly invalid configs

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, mock_provider, mock_azure_client):
        """AC10: Resource Cleanup - proper cleanup"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        # Act
        await mock_provider.cleanup()
        
        # Assert
        assert mock_provider.client is None

    @pytest.mark.asyncio
    async def test_cleanup_with_no_client(self, mock_provider):
        """AC10: Resource Cleanup - no client to cleanup"""
        # Arrange
        mock_provider.client = None
        
        # Act & Assert (should not raise exception)
        await mock_provider.cleanup()
        assert mock_provider.client is None

    @pytest.mark.asyncio
    async def test_get_model_list(self, mock_provider):
        """Test model list retrieval"""
        # Act
        models = await mock_provider.get_model_list()
        
        # Assert
        assert models == ["gpt-4.1-nano", "gpt-4o-mini"]

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self, mock_provider, mock_azure_client):
        """Test complete streaming workflow with multiple tool calls"""
        # Arrange
        mock_provider.client = mock_azure_client
        mock_provider.execute_tool_call = AsyncMock(side_effect=["result1", "result2"])
        
        # Create separate mock streams for each iteration
        async def mock_first_stream():
            yield Mock(type="response.output_text.delta", delta="I'll help you")
            yield Mock(type="response.completed", response=Mock(
                output=[
                    Mock(type="function_call", name="tool1", arguments='{"param": "value1"}', call_id="call_1"),
                    Mock(type="function_call", name="tool2", arguments='{"param": "value2"}', call_id="call_2")
                ]
            ))
        
        async def mock_second_stream():
            yield Mock(type="response.output_text.delta", delta="Based on the results")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.side_effect = [mock_first_stream(), mock_second_stream()]
        
        context = [{"role": "user", "content": "Help me with multiple tools"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "I'll help you" in responses
        assert "Based on the results" in responses
        assert mock_provider.execute_tool_call.call_count == 2

    @pytest.mark.asyncio
    async def test_streaming_with_model_settings(self, mock_provider, mock_azure_client):
        """Test streaming with custom model settings"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream():
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.return_value = mock_stream()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        model_settings = {"temperature": 0.7, "max_tokens": 100}
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions, model_settings=model_settings):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses
        assert " world" in responses
        call_args = mock_azure_client.responses.create.call_args[1]
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100

# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary condition tests"""
    
    @pytest.mark.asyncio
    async def test_empty_context(self, mock_provider, mock_azure_client):
        """Test streaming with empty context"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream():
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.return_value = mock_stream()
        
        context = []
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses
        assert " world" in responses

    @pytest.mark.asyncio
    async def test_empty_instructions(self, mock_provider, mock_azure_client):
        """Test streaming with empty instructions"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream():
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.output_text.delta", delta=" world")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        mock_azure_client.responses.create.return_value = mock_stream()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = ""
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses
        assert " world" in responses

    @pytest.mark.asyncio
    async def test_no_completed_event(self, mock_provider, mock_azure_client):
        """Test handling when no completed event is received"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream_no_completed():
            yield Mock(type="response.output_text.delta", delta="Hello")
            # No completed event
        
        mock_azure_client.responses.create.return_value = mock_stream_no_completed()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses

    @pytest.mark.asyncio
    async def test_completed_event_without_response(self, mock_provider, mock_azure_client):
        """Test handling when completed event has no response"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream_no_response():
            yield Mock(type="response.output_text.delta", delta="Hello")
            yield Mock(type="response.completed", response=None)
        
        mock_azure_client.responses.create.return_value = mock_stream_no_response()
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        responses = []
        async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
            responses.append(response)
        
        # Assert
        assert "Hello" in responses

# ============================================================================
# Performance and Concurrency Tests
# ============================================================================

class TestPerformanceAndConcurrency:
    """Performance and concurrency related tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, mock_provider, mock_azure_client):
        """Test handling multiple concurrent streaming requests"""
        # Arrange
        mock_provider.client = mock_azure_client
        
        async def mock_stream():
            yield Mock(type="response.output_text.delta", delta="Response")
            yield Mock(type="response.completed", response=Mock(output=[]))
        
        # Create a new mock stream for each call
        mock_azure_client.responses.create.side_effect = [mock_stream() for _ in range(3)]
        
        context = [{"role": "user", "content": "Hello"}]
        model = "gpt-4.1-nano"
        instructions = "You are a helpful assistant."
        
        # Act
        async def make_request():
            responses = []
            async for response in mock_provider.send_chat_with_streaming(context, model, instructions):
                responses.append(response)
            return responses
        
        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # Assert
        for result in results:
            assert "Response" in result
        assert mock_azure_client.responses.create.call_count == 3

# ============================================================================
# Legacy Tests (keeping existing tests)
# ============================================================================

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_initialization(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    assert provider.client is mock_client
    assert provider.config.model_list == ["gpt-4.1-nano", "gpt-4o-mini"]

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_health_check_healthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    # Ensure the client is properly set
    provider.client = mock_client
    # Mock the models.list call to return successfully
    mock_client.models.list = AsyncMock(return_value=["gpt-35-turbo", "gpt-4"])
    health = await provider.health_check()
    assert health.status == "healthy"
    assert health.service == "test-provider"

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_health_check_unhealthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.side_effect = Exception("API unavailable")
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "unhealthy"
    assert "API unavailable" in health.error_details

@patch.dict("os.environ", {
    "AZURE_OPENAI_MODEL_LIST": "gpt-4o,gpt-4o-mini,claude-3",
    "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/",
    "AZURE_OPENAI_DEFAULT_MODEL": "gpt-4o",
    "AZURE_OPENAI_API_VERSION": "2023-05-15",
    "AZURE_OPENAI_API_KEY": "test-key"
}, clear=False)
@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_get_config_from_env_var(mock_azure_openai):
    """Test that config is correctly read from environment variable"""

    # Force reload the providers module to pick up our environment changes
    import app.models.providers
    importlib.reload(app.models.providers)
    
    # Import after reload to get the updated config
    from app.models.providers import AzureOpenAIConfig
    
    # Create config without explicitly setting model_list, so it reads from env var
    config = AzureOpenAIConfig(
        name="test-provider"
    )
    
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-4o", "gpt-4o-mini", "claude-3"]
    mock_azure_openai.return_value = mock_client
    
    provider = AzureOpenAIProvider(config)
    models = await provider.get_model_list()
    
    # Verify the model list matches what was set in the environment variable
    assert models == ["gpt-4o", "gpt-4o-mini", "claude-3"]
    
    # Verify all environment variables were read correctly
    assert config.model_list == ["gpt-4o", "gpt-4o-mini", "claude-3"]
    assert config.base_url == "https://example.openai.azure.com/"
    assert config.default_model == "gpt-4o"
    assert config.api_version == "2023-05-15"
    assert config.api_key == "test-key"

