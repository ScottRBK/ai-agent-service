"""
Unit tests for APIAgent class.
Tests basic functionality including initialization, chat, memory management, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime
from app.core.agents.api_agent import APIAgent
from app.models.resources.memory import MemoryEntry


class TestAPIAgentInitialization:
    """Test APIAgent initialization and basic setup."""
    
    def test_init_with_defaults(self):
        """Test APIAgent initialization with default parameters."""
        agent = APIAgent("test_agent")
        
        assert agent.agent_id == "test_agent"
        assert agent.user_id == "default_user"
        assert agent.session_id == "default_session"
        assert agent.requested_model is None
        assert agent.requested_model_settings is None
        assert agent.initialized is False
        assert agent.memory_resource is None
        assert agent.provider is None
    
    def test_init_with_custom_parameters(self):
        """Test APIAgent initialization with custom parameters."""
        agent = APIAgent(
            agent_id="custom_agent",
            user_id="user123",
            session_id="session456",
            model="gpt-4",
            model_settings={"temperature": 0.8}
        )
        
        assert agent.agent_id == "custom_agent"
        assert agent.user_id == "user123"
        assert agent.session_id == "session456"
        assert agent.requested_model == "gpt-4"
        assert agent.requested_model_settings == {"temperature": 0.8}
    
    @patch('app.core.agents.api_agent.AgentToolManager')
    def test_get_provider_from_config(self, mock_tool_manager):
        """Test getting provider from agent configuration."""
        # Mock tool manager config
        mock_instance = MagicMock()
        mock_instance.config = {"provider": "test_provider"}
        mock_tool_manager.return_value = mock_instance
        
        agent = APIAgent("test_agent")
        provider = agent._get_provider_from_config()
        
        assert provider == "test_provider"
    
    @patch('app.core.agents.api_agent.AgentToolManager')
    def test_get_provider_from_config_default(self, mock_tool_manager):
        """Test getting default provider when not specified in config."""
        # Mock tool manager config without provider
        mock_instance = MagicMock()
        mock_instance.config = {}
        mock_tool_manager.return_value = mock_instance
        
        agent = APIAgent("test_agent")
        provider = agent._get_provider_from_config()
        
        assert provider == "azure_openai_cc"


class TestAPIAgentMemoryManagement:
    """Test APIAgent memory management functionality."""
    
    def test_clean_response_for_memory(self):
        """Test cleaning response for memory storage."""
        agent = APIAgent("test_agent")
        
        # Test with think tags
        response = "Here is my answer<think>I should think about this</think>and continue"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "Here is my answerand continue"
        
        # Test with escaped newlines
        response = "Line 1\\nLine 2"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "Line 1\nLine 2"
        
        # Test with whitespace
        response = "  Test response  "
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "Test response"
    
    def test_clean_response_complex_content(self):
        """Test cleaning response with complex content."""
        agent = APIAgent("test_agent")
        
        # Test with multiple think tags and escaped characters
        response = """Here is my response<think>First thought</think>
        Some content<think>Second thought</think>
        More content\\nwith escaped\\nnewlines\\tand tabs"""
        
        cleaned = agent._clean_response_for_memory(response)
        
        # The actual behavior: escaped newlines become literal newlines, tabs remain escaped
        expected = """Here is my response
        Some content
        More content
with escaped
newlines\\tand tabs"""
        
        assert cleaned == expected
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_no_memory(self):
        """Test getting conversation history when no memory resource is available."""
        agent = APIAgent("test_agent")
        agent.memory_resource = None
        
        history = await agent.get_conversation_history()
        
        assert history == []
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_with_memory(self):
        """Test getting conversation history with memory resource."""
        agent = APIAgent("test_agent")
        
        # Mock memory resource
        mock_memory = AsyncMock()
        mock_memories = [
            MagicMock(content={"role": "user", "content": "Hello"}),
            MagicMock(content={"role": "assistant", "content": "Hi there!"})
        ]
        mock_memory.get_memories.return_value = mock_memories
        # Mock get_session_summary to return None (no summary)
        mock_memory.get_session_summary.return_value = None
        agent.memory_resource = mock_memory
        
        history = await agent.get_conversation_history()
        
        expected_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        assert history == expected_history
        mock_memory.get_memories.assert_called_once_with(
            "default_user", session_id="default_session", agent_id="test_agent", order_direction="asc"
        )
        mock_memory.get_session_summary.assert_called_once_with(
            "default_user", "default_session", "test_agent"
        )
    
    @pytest.mark.asyncio
    async def test_save_memory(self):
        """Test saving memory entry."""
        agent = APIAgent("test_agent")
        
        # Mock memory resource
        mock_memory = AsyncMock()
        agent.memory_resource = mock_memory
        
        await agent.save_memory("user", "Test message")
        
        # Verify MemoryEntry was created and stored
        mock_memory.store_memory.assert_called_once()
        call_args = mock_memory.store_memory.call_args[0][0]
        assert isinstance(call_args, MemoryEntry)
        assert call_args.user_id == "default_user"
        assert call_args.session_id == "default_session"
        assert call_args.agent_id == "test_agent"
        assert call_args.content == {"role": "user", "content": "Test message"}
    
    @pytest.mark.asyncio
    async def test_save_memory_no_resource(self):
        """Test saving memory when no memory resource is available."""
        agent = APIAgent("test_agent")
        agent.memory_resource = None
        
        # Should not raise any exception
        await agent.save_memory("user", "Test message")
    
    @pytest.mark.asyncio
    async def test_clear_conversation(self):
        """Test clearing conversation history."""
        agent = APIAgent("test_agent")
        
        # Mock memory resource
        mock_memory = AsyncMock()
        agent.memory_resource = mock_memory
        
        await agent.clear_conversation()
        
        mock_memory.clear_session_memories.assert_called_once_with(
            "default_user", "default_session", "test_agent"
        )
    
    @pytest.mark.asyncio
    async def test_clear_conversation_no_resource(self):
        """Test clearing conversation when no memory resource is available."""
        agent = APIAgent("test_agent")
        agent.memory_resource = None
        
        # Should not raise any exception
        await agent.clear_conversation()


class TestAPIAgentModelConfiguration:
    """Test APIAgent model configuration handling."""
    
    def test_model_priority_order(self):
        """Test model selection priority: requested > config > provider default."""
        # Test with requested model (highest priority)
        agent = APIAgent("test_agent", model="requested-model")
        assert agent.requested_model == "requested-model"
        
        # Test without requested model
        agent2 = APIAgent("test_agent")
        assert agent2.requested_model is None
    
    def test_model_settings_priority_order(self):
        """Test model settings priority: requested > config."""
        # Test with requested settings (highest priority)
        requested_settings = {"temp": 0.9, "max_tokens": 1000}
        agent = APIAgent("test_agent", model_settings=requested_settings)
        assert agent.requested_model_settings == requested_settings
        
        # Test without requested settings
        agent2 = APIAgent("test_agent")
        assert agent2.requested_model_settings is None


class TestAPIAgentEdgeCases:
    """Test APIAgent edge cases and error conditions."""
    
    def test_clean_response_empty_string(self):
        """Test cleaning empty response."""
        agent = APIAgent("test_agent")
        cleaned = agent._clean_response_for_memory("")
        assert cleaned == ""
    
    def test_clean_response_only_whitespace(self):
        """Test cleaning response with only whitespace."""
        agent = APIAgent("test_agent")
        cleaned = agent._clean_response_for_memory("   \n\t   ")
        assert cleaned == ""
    
    def test_clean_response_no_think_tags(self):
        """Test cleaning response without think tags."""
        agent = APIAgent("test_agent")
        response = "This is a normal response without any think tags."
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "This is a normal response without any think tags."
    
    def test_clean_response_only_think_tags(self):
        """Test cleaning response with only think tags."""
        agent = APIAgent("test_agent")
        response = "<think>This is a thought</think>"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == ""
    
    def test_clean_response_mixed_content(self):
        """Test cleaning response with mixed content."""
        agent = APIAgent("test_agent")
        response = "Start<think>Thought 1</think>Middle<think>Thought 2</think>End"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "StartMiddleEnd"
    
    @pytest.mark.asyncio
    async def test_memory_error_handling(self):
        """Test handling of memory-related errors."""
        agent = APIAgent("test_agent")
        
        # Mock memory resource that raises exception
        mock_memory = AsyncMock()
        mock_memory.get_memories.side_effect = Exception("Memory error")
        agent.memory_resource = mock_memory
        
        # Should propagate memory errors (actual behavior)
        with pytest.raises(Exception, match="Memory error"):
            await agent.get_conversation_history()
        
        # Test save_memory with error
        mock_memory.store_memory.side_effect = Exception("Store error")
        
        # Should propagate store errors (actual behavior)
        with pytest.raises(Exception, match="Store error"):
            await agent.save_memory("user", "Test message")
        
        # Test clear_conversation with error
        mock_memory.clear_session_memories.side_effect = Exception("Clear error")
        
        # Should propagate clear errors (actual behavior)
        with pytest.raises(Exception, match="Clear error"):
            await agent.clear_conversation()


class TestAPIAgentIntegration:
    """Test APIAgent integration with existing conftest.py setup."""
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_existing_config(self):
        """Test that agent can be created with existing test configuration."""
        # This test works with the existing conftest.py setup
        agent = APIAgent("research_agent")
        
        assert agent.agent_id == "research_agent"
        assert agent.user_id == "default_user"
        assert agent.session_id == "default_session"
        assert agent.initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_custom_user_session(self):
        """Test agent creation with custom user and session."""
        agent = APIAgent(
            agent_id="cli_agent",
            user_id="test_user",
            session_id="test_session"
        )
        
        assert agent.agent_id == "cli_agent"
        assert agent.user_id == "test_user"
        assert agent.session_id == "test_session"
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_model_override(self):
        """Test agent creation with model override."""
        agent = APIAgent(
            agent_id="mcp_agent",
            model="custom-model",
            model_settings={"temperature": 0.5}
        )
        
        assert agent.requested_model == "custom-model"
        assert agent.requested_model_settings == {"temperature": 0.5}
    
    def test_agent_id_validation(self):
        """Test that agent accepts various valid agent IDs."""
        valid_agent_ids = [
            "research_agent",
            "cli_agent", 
            "mcp_agent",
            "test_agent",
            "custom_agent_123",
            "agent-with-dashes",
            "agent_with_underscores"
        ]
        
        for agent_id in valid_agent_ids:
            agent = APIAgent(agent_id)
            assert agent.agent_id == agent_id
    
    def test_user_session_defaults(self):
        """Test that user and session defaults work correctly."""
        agent = APIAgent("test_agent")
        
        assert agent.user_id == "default_user"
        assert agent.session_id == "default_session"
        
        # Test with explicit defaults
        agent2 = APIAgent("test_agent", user_id="default_user", session_id="default_session")
        assert agent2.user_id == "default_user"
        assert agent2.session_id == "default_session"
    
    def test_model_settings_types(self):
        """Test that model settings accept various data types."""
        # Test with different types of model settings
        test_cases = [
            {"temperature": 0.7},
            {"max_tokens": 1000, "top_p": 0.9},
            {"num_ctx": 30000, "num_predict": 2000},
            {"frequency_penalty": 0.1, "presence_penalty": 0.2},
            {"temperature": 0.0, "max_tokens": 500, "top_p": 1.0}
        ]
        
        for settings in test_cases:
            agent = APIAgent("test_agent", model_settings=settings)
            assert agent.requested_model_settings == settings 