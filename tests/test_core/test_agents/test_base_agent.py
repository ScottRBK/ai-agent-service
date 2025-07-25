"""
Unit tests for BaseAgent class.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.base_agent import BaseAgent
from app.models.resources.memory import MemoryEntry, MemorySessionSummary


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration"""
    return {
        "agent_id": "test_agent",
        "provider": "azure_openai_cc",
        "model": "gpt-4",
        "model_settings": {"temperature": 0.7}
    }


@pytest.fixture
def mock_provider():
    """Mock provider"""
    provider = Mock()
    provider.initialize = AsyncMock()
    provider.send_chat = AsyncMock(return_value="Test response")
    provider.send_chat_with_streaming = AsyncMock()
    provider.config = Mock(default_model="gpt-3.5-turbo")
    return provider


@pytest.fixture
def mock_memory_resource():
    """Mock memory resource"""
    resource = Mock()
    resource.store_memory = AsyncMock()
    resource.get_memories = AsyncMock(return_value=[])
    resource.get_session_summary = AsyncMock(return_value=None)
    return resource


class TestBaseAgent:
    """Test cases for BaseAgent"""
    
    @pytest.mark.asyncio
    async def test_init(self):
        """Test BaseAgent initialization"""
        agent = BaseAgent(
            agent_id="test_agent",
            user_id="test_user",
            session_id="test_session"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.user_id == "test_user"
        assert agent.session_id == "test_session"
        assert agent.memory_resource is None
        assert not agent.initialized
        assert agent.conversation_history == []
        assert agent.provider is None
    
    @pytest.mark.asyncio
    async def test_init_with_model_settings(self):
        """Test BaseAgent initialization with custom model settings"""
        agent = BaseAgent(
            agent_id="test_agent",
            model="gpt-4",
            model_settings={"temperature": 0.8}
        )
        
        assert agent.requested_model == "gpt-4"
        assert agent.requested_model_settings == {"temperature": 0.8}
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_provider, mock_agent_config):
        """Test successful agent initialization"""
        agent = BaseAgent("test_agent")
        
        with patch.object(agent.tool_manager, 'config', mock_agent_config), \
             patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])), \
             patch.object(agent.resource_manager, 'get_model_config', return_value=("gpt-4", {"temperature": 0.7})), \
             patch.object(agent.resource_manager, 'get_memory_resource', AsyncMock(return_value=None)), \
             patch.object(agent.provider_manager, 'get_provider', return_value={
                 "class": Mock(return_value=mock_provider),
                 "config_class": Mock
             }):
            
            await agent.initialize()
            
            assert agent.initialized
            assert agent.provider == mock_provider
            assert agent.model == "gpt-4"
            assert agent.model_settings == {"temperature": 0.7}
            mock_provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_with_memory(self, mock_provider, mock_memory_resource):
        """Test agent initialization with memory resource"""
        agent = BaseAgent("test_agent")
        
        with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])), \
             patch.object(agent.resource_manager, 'get_model_config', return_value=(None, None)), \
             patch.object(agent.resource_manager, 'get_memory_resource', AsyncMock(return_value=mock_memory_resource)), \
             patch.object(agent.provider_manager, 'get_provider', return_value={
                 "class": Mock(return_value=mock_provider),
                 "config_class": Mock
             }):
            
            await agent.initialize()
            
            assert agent.memory_resource == mock_memory_resource
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_provider):
        """Test that initialize is idempotent"""
        agent = BaseAgent("test_agent")
        agent.initialized = True
        
        await agent.initialize()
        # Should return early without doing anything
        assert agent.initialized
    
    @pytest.mark.asyncio
    async def test_save_memory_with_resource(self, mock_memory_resource):
        """Test saving memory when resource is available"""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        agent.memory_resource = mock_memory_resource
        
        await agent.save_memory("user", "Hello")
        
        mock_memory_resource.store_memory.assert_called_once()
        call_args = mock_memory_resource.store_memory.call_args[0][0]
        assert isinstance(call_args, MemoryEntry)
        assert call_args.user_id == "user1"
        assert call_args.session_id == "session1"
        assert call_args.content == {"role": "user", "content": "Hello"}
    
    @pytest.mark.asyncio
    async def test_save_memory_without_resource(self):
        """Test saving memory when no resource is available"""
        agent = BaseAgent("test_agent")
        agent.memory_resource = None
        
        # Should not raise exception
        await agent.save_memory("user", "Hello")
    
    @pytest.mark.asyncio
    async def test_load_memory_without_resource(self):
        """Test loading memory when no resource is available"""
        agent = BaseAgent("test_agent")
        agent.memory_resource = None
        
        history = await agent.load_memory()
        assert history == []
    
    @pytest.mark.asyncio
    async def test_load_memory_with_messages(self, mock_memory_resource):
        """Test loading memory with messages"""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        agent.memory_resource = mock_memory_resource
        
        # Mock memory entries
        mock_memories = [
            Mock(content={"role": "user", "content": "Hello"}),
            Mock(content={"role": "assistant", "content": "Hi there"})
        ]
        mock_memory_resource.get_memories.return_value = mock_memories
        
        history = await agent.load_memory()
        
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there"}
    
    @pytest.mark.asyncio
    async def test_load_memory_with_summary(self, mock_memory_resource):
        """Test loading memory with summary"""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        agent.memory_resource = mock_memory_resource
        
        # Mock summary
        mock_summary = Mock(summary="Previous conversation summary")
        mock_memory_resource.get_session_summary.return_value = mock_summary
        
        # Mock memory entries
        mock_memories = [
            Mock(content={"role": "user", "content": "New message"})
        ]
        mock_memory_resource.get_memories.return_value = mock_memories
        
        history = await agent.load_memory()
        
        assert len(history) == 2
        assert history[0] == {"role": "system", "content": "Previous conversation summary"}
        assert history[1] == {"role": "user", "content": "New message"}
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, mock_memory_resource):
        """Test get_conversation_history alias"""
        agent = BaseAgent("test_agent")
        agent.memory_resource = mock_memory_resource
        
        mock_memory_resource.get_memories.return_value = []
        
        history = await agent.get_conversation_history()
        assert history == []
        # Verify it calls load_memory internally
        mock_memory_resource.get_memories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clean_response_for_memory(self):
        """Test response cleaning"""
        agent = BaseAgent("test_agent")
        
        # Test removing think tags
        response = "Hello <think>internal thought</think> world"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "Hello  world"
        
        # Test replacing newline characters
        response = "Line 1\\nLine 2"
        cleaned = agent._clean_response_for_memory(response)
        assert cleaned == "Line 1\nLine 2"
    
    @pytest.mark.asyncio
    async def test_trigger_memory_compression_without_resource(self):
        """Test memory compression when no resource available"""
        agent = BaseAgent("test_agent")
        agent.memory_resource = None
        
        # Should not raise exception
        await agent._trigger_memory_compression()
    
    @pytest.mark.asyncio
    async def test_trigger_memory_compression_with_resource(self, mock_memory_resource):
        """Test memory compression with resource"""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        agent.memory_resource = mock_memory_resource
        
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent') as mock_compression_agent_class:
            mock_compression_agent = Mock()
            mock_compression_agent.compress_conversation = AsyncMock()
            mock_compression_agent_class.return_value = mock_compression_agent
            
            await agent._trigger_memory_compression()
            
            mock_compression_agent.compress_conversation.assert_called_once_with(
                "test_agent",
                {
                    "threshold_tokens": 10000,
                    "recent_messages_to_keep": 10,
                    "enabled": True
                },
                "user1",
                "session1"
            )
    
    @pytest.mark.asyncio
    async def test_trigger_memory_compression_with_custom_config(self, mock_memory_resource):
        """Test memory compression with custom config"""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        agent.memory_resource = mock_memory_resource
        
        custom_config = {
            "threshold_tokens": 5000,
            "recent_messages_to_keep": 5,
            "enabled": True
        }
        
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent') as mock_compression_agent_class:
            mock_compression_agent = Mock()
            mock_compression_agent.compress_conversation = AsyncMock()
            mock_compression_agent_class.return_value = mock_compression_agent
            
            await agent._trigger_memory_compression(custom_config)
            
            mock_compression_agent.compress_conversation.assert_called_once_with(
                "test_agent",
                custom_config,
                "user1",
                "session1"
            )
    
    @pytest.mark.asyncio
    async def test_get_provider_from_config(self, mock_agent_config):
        """Test getting provider from config"""
        agent = BaseAgent("test_agent")
        
        with patch.object(agent.tool_manager, 'config', mock_agent_config):
            provider_id = agent._get_provider_from_config()
            assert provider_id == "azure_openai_cc"
        
        # Test default fallback
        with patch.object(agent.tool_manager, 'config', {}):
            provider_id = agent._get_provider_from_config()
            assert provider_id == "azure_openai_cc"