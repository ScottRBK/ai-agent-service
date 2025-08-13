"""
Unit tests for BaseProvider agent_instance handling and agent context passing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.providers.base import BaseProvider
from app.models.providers import ProviderConfig, ProviderType


class MockProvider(BaseProvider):
    """Mock implementation of BaseProvider for testing"""
    
    async def health_check(self):
        return Mock(is_healthy=True)
    
    async def cleanup(self):
        pass
    
    async def get_model_list(self):
        return ["test-model"]
    
    async def send_chat(self, context, model, instructions, tools):
        return "test response"
    
    async def send_chat_with_streaming(self, context, model, instructions, tools):
        yield "test"
        yield " response"
    
    async def embed(self, text):
        return [0.1, 0.2, 0.3]
    
    async def rerank(self, model, query, candidates):
        return [0.8, 0.6, 0.4]


class TestBaseProviderAgentContext:
    """Test cases for BaseProvider agent context handling"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock provider configuration"""
        return ProviderConfig(
            name="test_provider",
            provider_type=ProviderType.AZURE_OPENAI_CC,
            api_key="test_key",
            endpoint="https://test.openai.azure.com/",
            default_model="gpt-4"
        )
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent instance"""
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.user_id = "test_user"
        agent.session_id = "test_session"
        agent.knowledge_base = Mock()
        return agent
    
    def test_init_agent_instance_none(self, mock_config):
        """Test BaseProvider initialization sets agent_instance to None"""
        provider = MockProvider(mock_config)
        assert provider.agent_instance is None
    
    def test_agent_instance_assignment(self, mock_config, mock_agent):
        """Test agent_instance can be assigned"""
        provider = MockProvider(mock_config)
        provider.agent_instance = mock_agent
        assert provider.agent_instance == mock_agent
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_agent_instance_parameter(self, mock_config, mock_agent):
        """Test execute_tool_call uses provided agent_instance parameter"""
        provider = MockProvider(mock_config)
        provider.agent_instance = Mock()  # Different agent stored
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            result = await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent",
                agent_instance=mock_agent  # Explicit agent_instance
            )
            
            # Should use provided agent_instance, not stored one
            mock_manager_class.assert_called_once_with("test_agent", agent_instance=mock_agent)
            mock_agent_manager.execute_tool.assert_called_once_with("test_tool", {"param": "value"})
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_stored_agent_instance(self, mock_config, mock_agent):
        """Test execute_tool_call falls back to stored agent_instance"""
        provider = MockProvider(mock_config)
        provider.agent_instance = mock_agent
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            result = await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent"
                # No agent_instance parameter provided
            )
            
            # Should use stored agent_instance
            mock_manager_class.assert_called_once_with("test_agent", agent_instance=mock_agent)
            mock_agent_manager.execute_tool.assert_called_once_with("test_tool", {"param": "value"})
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_no_agent_instance_available(self, mock_config):
        """Test execute_tool_call when no agent_instance is available"""
        provider = MockProvider(mock_config)
        provider.agent_instance = None
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            result = await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent"
            )
            
            # Should pass None as agent_instance
            mock_manager_class.assert_called_once_with("test_agent", agent_instance=None)
            mock_agent_manager.execute_tool.assert_called_once_with("test_tool", {"param": "value"})
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_without_agent_id(self, mock_config):
        """Test execute_tool_call without agent_id falls back to ToolRegistry"""
        provider = MockProvider(mock_config)
        
        with patch('app.core.providers.base.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="registry result")
            
            result = await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}
                # No agent_id provided
            )
            
            mock_registry.execute_tool_call.assert_called_once_with("test_tool", {"param": "value"})
            assert result == "registry result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_tool_tracking(self, mock_config, mock_agent):
        """Test execute_tool_call tracks tool calls when enabled"""
        provider = MockProvider(mock_config)
        provider.config.track_tool_calls = True
        provider.agent_instance = mock_agent
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent"
            )
            
            # Should track the tool call
            assert len(provider.tool_calls_made) == 1
            assert provider.tool_calls_made[0]["tool_name"] == "test_tool"
            assert provider.tool_calls_made[0]["arguments"] == {"param": "value"}
            assert provider.tool_calls_made[0]["results"] == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_without_tool_tracking(self, mock_config, mock_agent):
        """Test execute_tool_call doesn't track when disabled"""
        provider = MockProvider(mock_config)
        provider.config.track_tool_calls = False
        provider.agent_instance = mock_agent
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent"
            )
            
            # Should not track the tool call
            assert len(provider.tool_calls_made) == 0
    
    def test_get_tool_calls_made(self, mock_config):
        """Test get_tool_calls_made returns tracked calls"""
        provider = MockProvider(mock_config)
        provider.tool_calls_made = [
            {"tool_name": "tool1", "arguments": {"a": 1}},
            {"tool_name": "tool2", "arguments": {"b": 2}}
        ]
        
        calls = provider.get_tool_calls_made()
        assert len(calls) == 2
        assert calls[0]["tool_name"] == "tool1"
        assert calls[1]["tool_name"] == "tool2"
    
    def test_clear_tool_calls_made(self, mock_config):
        """Test clear_tool_calls_made clears the list"""
        provider = MockProvider(mock_config)
        provider.tool_calls_made = [{"tool_name": "tool1", "arguments": {"a": 1}}]
        
        provider.clear_tool_calls_made()
        assert len(provider.tool_calls_made) == 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_agent_instance_precedence(self, mock_config):
        """Test that explicit agent_instance parameter takes precedence over stored one"""
        provider = MockProvider(mock_config)
        
        stored_agent = Mock()
        stored_agent.agent_id = "stored_agent"
        
        explicit_agent = Mock()
        explicit_agent.agent_id = "explicit_agent"
        
        provider.agent_instance = stored_agent
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(return_value="tool result")
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent",
                agent_instance=explicit_agent
            )
            
            # Should use explicit agent_instance, not stored one
            mock_manager_class.assert_called_once_with("test_agent", agent_instance=explicit_agent)
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_error_handling(self, mock_config, mock_agent):
        """Test execute_tool_call handles errors gracefully"""
        provider = MockProvider(mock_config)
        provider.agent_instance = mock_agent
        
        mock_agent_manager = Mock()
        mock_agent_manager.execute_tool = AsyncMock(side_effect=Exception("Tool execution failed"))
        
        with patch('app.core.providers.base.AgentToolManager') as mock_manager_class:
            mock_manager_class.return_value = mock_agent_manager
            
            # Should return error string instead of raising exception (centralized error handling)
            result = await provider.execute_tool_call(
                "test_tool", 
                {"param": "value"}, 
                agent_id="test_agent"
            )
            
            assert "Error executing tool test_tool: Tool execution failed" in result


class TestBaseProviderReranking:
    """Test cases for BaseProvider re-ranking functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock provider configuration"""
        return ProviderConfig(
            name="test_provider",
            provider_type=ProviderType.AZURE_OPENAI_CC,
            api_key="test_key",
            endpoint="https://test.openai.azure.com/",
            default_model="gpt-4"
        )
    
    @pytest.mark.asyncio
    async def test_custom_rerank_implementation(self, mock_config):
        """Test that MockProvider has custom re-ranking implementation"""
        provider = MockProvider(mock_config)
        
        scores = await provider.rerank(
            "test-model",
            "test query",
            ["high relevance", "medium relevance", "low relevance"]
        )
        
        # Should use MockProvider's custom implementation, not BaseProvider default
        assert scores == [0.8, 0.6, 0.4]
    
    @pytest.mark.asyncio
    async def test_rerank_with_empty_candidates(self, mock_config):
        """Test MockProvider re-ranking with empty candidates list"""
        provider = MockProvider(mock_config)
        
        scores = await provider.rerank(
            "test-model",
            "test query", 
            []
        )
        
        # MockProvider returns fixed scores regardless of input
        assert scores == [0.8, 0.6, 0.4]
    
    @pytest.mark.asyncio
    async def test_rerank_single_candidate(self, mock_config):
        """Test MockProvider re-ranking with single candidate"""
        provider = MockProvider(mock_config)
        
        scores = await provider.rerank(
            "test-model",
            "test query",
            ["single document"]
        )
        
        # MockProvider has custom implementation, not BaseProvider default
        assert scores == [0.8, 0.6, 0.4]
    
    @pytest.mark.asyncio
    async def test_rerank_with_different_model_names(self, mock_config):
        """Test re-ranking with various model names"""
        provider = MockProvider(mock_config)
        
        models_to_test = [
            "gpt-4",
            "custom-rerank-model:latest",
            "reranker-v2.0",
            "specialized/rerank:Q8_0"
        ]
        
        for model in models_to_test:
            scores = await provider.rerank(
                model,
                "test query",
                ["doc1", "doc2"]
            )
            
            # MockProvider should return same scores regardless of model or input
            assert scores == [0.8, 0.6, 0.4]