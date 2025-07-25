"""
Tests to validate memory-enabled vs non-memory agent behavior after BaseAgent refactoring.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.base_agent import BaseAgent
from app.core.agents.api_agent import APIAgent
from app.core.agents.cli_agent import CLIAgent


class TestMemoryBehavior:
    """Test memory behavior differences between agents with and without memory configured."""
    
    @pytest.mark.asyncio
    async def test_base_agent_without_memory(self):
        """Test BaseAgent behavior when no memory resource is configured."""
        agent = BaseAgent("test_agent")
        agent.memory_resource = None
        
        # Memory operations should handle gracefully
        await agent.save_memory("user", "test message")  # Should not raise
        history = await agent.load_memory()
        assert history == []
        
        # Compression should be skipped
        await agent._trigger_memory_compression()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_base_agent_with_memory(self):
        """Test BaseAgent behavior when memory resource is configured."""
        agent = BaseAgent("test_agent", user_id="user1", session_id="session1")
        
        # Mock memory resource
        mock_memory = AsyncMock()
        mock_memory.get_memories.return_value = []
        mock_memory.get_session_summary.return_value = None
        mock_memory.store_memory = AsyncMock()
        agent.memory_resource = mock_memory
        
        # Memory operations should work
        await agent.save_memory("user", "test message")
        mock_memory.store_memory.assert_called_once()
        
        history = await agent.load_memory()
        assert history == []
        mock_memory.get_memories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_agent_memory_inheritance(self):
        """Test that APIAgent properly inherits memory behavior from BaseAgent."""
        agent = APIAgent("test_agent")
        
        # Without memory resource
        agent.memory_resource = None
        history = await agent.get_conversation_history()
        assert history == []
        
        # With memory resource
        mock_memory = AsyncMock()
        mock_memory.get_memories.return_value = [
            Mock(content={"role": "user", "content": "Hello"})
        ]
        mock_memory.get_session_summary.return_value = None
        agent.memory_resource = mock_memory
        
        history = await agent.get_conversation_history()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}
    
    @pytest.mark.asyncio
    async def test_cli_agent_memory_inheritance(self):
        """Test that CLIAgent properly inherits memory behavior from BaseAgent."""
        agent = CLIAgent("test_agent")
        
        # Without memory resource
        agent.memory_resource = None
        history = await agent.load_memory()
        assert history == []
        
        # With memory resource
        mock_memory = AsyncMock()
        mock_memory.get_memories.return_value = [
            Mock(content={"role": "assistant", "content": "Hi there"})
        ]
        mock_memory.get_session_summary.return_value = None
        agent.memory_resource = mock_memory
        
        history = await agent.load_memory()
        assert len(history) == 1
        assert history[0] == {"role": "assistant", "content": "Hi there"}
    
    @pytest.mark.asyncio
    async def test_memory_error_handling_consistency(self):
        """Test that all agents handle memory errors consistently."""
        # Test BaseAgent error handling
        base_agent = BaseAgent("test_agent")
        mock_memory = AsyncMock()
        mock_memory.get_memories.side_effect = Exception("Memory error")
        base_agent.memory_resource = mock_memory
        
        # Should return empty list, not raise exception
        history = await base_agent.load_memory()
        assert history == []
        
        # Test APIAgent error handling (should inherit same behavior)
        api_agent = APIAgent("test_agent")  
        api_agent.memory_resource = mock_memory
        
        history = await api_agent.get_conversation_history()
        assert history == []
        
        # Test CLIAgent error handling (should inherit same behavior)
        cli_agent = CLIAgent("test_agent")
        cli_agent.memory_resource = mock_memory
        
        history = await cli_agent.load_memory()
        assert history == []
    
    @pytest.mark.asyncio
    async def test_compression_behavior_consistency(self):
        """Test that compression behavior is consistent across agents."""
        # Mock memory resource
        mock_memory = AsyncMock()
        
        # Test BaseAgent compression
        base_agent = BaseAgent("test_agent")
        base_agent.memory_resource = mock_memory
        
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent') as mock_compression_class:
            mock_compression_agent = Mock()
            mock_compression_agent.compress_conversation = AsyncMock()
            mock_compression_class.return_value = mock_compression_agent
            
            await base_agent._trigger_memory_compression()
            mock_compression_agent.compress_conversation.assert_called_once()
        
        # Test without memory resource - should not trigger compression
        base_agent.memory_resource = None
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent') as mock_compression_class:
            mock_compression_agent = Mock()
            mock_compression_class.return_value = mock_compression_agent
            
            await base_agent._trigger_memory_compression()
            mock_compression_class.assert_not_called()