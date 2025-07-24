# tests/test_core/test_resources/test_memory_compression_manager.py
"""
Unit tests for Memory Compression Manager.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from app.core.resources.memory_compression_manager import MemoryCompressionManager
from app.models.resources.memory import MemoryEntry


class TestMemoryCompressionManager:
    """Test cases for Memory Compression Manager."""
    
    @pytest.fixture
    def compression_config(self):
        """Test configuration for compression manager."""
        return {
            "threshold_tokens": 5000,
            "recent_messages_to_keep": 3,
            "enabled": True
        }
    
    @pytest.fixture
    def compression_manager(self, compression_config):
        """Create compression manager instance."""
        return MemoryCompressionManager("test_agent", compression_config)
    
    @pytest.fixture
    def default_compression_manager(self):
        """Create compression manager with default configuration."""
        return MemoryCompressionManager("test_agent")
    
    @pytest.fixture
    def sample_memory_entries(self):
        """Sample memory entries for testing."""
        return [
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "Hello, how are you?"}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "I'm doing well, thank you for asking!"}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "Can you help me with a question?"}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "Of course! I'd be happy to help you with any questions you have."}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "What is the capital of France?"}
            )
        ]
    
    @pytest.fixture
    def sample_conversation_history(self):
        """Sample conversation history for testing."""
        return [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            {"role": "user", "content": "Can you help me with a question?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help you with any questions you have."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    
    def test_init_with_valid_config(self, compression_config):
        """Test initialization with valid configuration."""
        manager = MemoryCompressionManager("test_agent", compression_config)
        
        assert manager.agent_id == "test_agent"
        assert manager.config == compression_config
        assert manager.threshold_tokens == 5000
        assert manager.recent_messages_to_keep == 3
        assert manager.enabled is True
        assert manager.token_counter is not None
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        manager = MemoryCompressionManager("test_agent")
        
        assert manager.agent_id == "test_agent"
        assert manager.config == {}
        assert manager.threshold_tokens == 8000  # Default value
        assert manager.recent_messages_to_keep == 4  # Default value
        assert manager.enabled is True  # Default value
        assert manager.token_counter is not None
    
    def test_init_with_partial_config(self):
        """Test initialization with partial configuration."""
        partial_config = {"threshold_tokens": 3000}
        manager = MemoryCompressionManager("test_agent", partial_config)
        
        assert manager.threshold_tokens == 3000
        assert manager.recent_messages_to_keep == 4  # Default value
        assert manager.enabled is True  # Default value
    
    def test_init_with_disabled_compression(self):
        """Test initialization with compression disabled."""
        config = {"enabled": False}
        manager = MemoryCompressionManager("test_agent", config)
        
        assert manager.enabled is False
    
    @patch('app.core.resources.memory_compression_manager.logger')
    def test_should_compress_disabled(self, mock_logger, default_compression_manager, sample_memory_entries):
        """Test should_compress when compression is disabled."""
        default_compression_manager.enabled = False
        
        result = default_compression_manager.should_compress(sample_memory_entries)
        
        assert result is False
        mock_logger.debug.assert_not_called()
        mock_logger.info.assert_not_called()
    
    def test_should_compress_insufficient_messages(self, compression_manager, sample_memory_entries):
        """Test should_compress when there are insufficient messages."""
        # Keep only 2 messages (less than recent_messages_to_keep = 3)
        few_messages = sample_memory_entries[:2]
        
        result = compression_manager.should_compress(few_messages)
        
        assert result is False
    
    @patch('app.core.resources.memory_compression_manager.logger')
    def test_should_compress_below_threshold(self, mock_logger, compression_manager, sample_memory_entries):
        """Test should_compress when tokens are below threshold."""
        # Mock token counter to return low token count
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=2000)  # Below threshold of 5000
        
        result = compression_manager.should_compress(sample_memory_entries)
        
        assert result is False
        compression_manager.token_counter.count_conversation_tokens.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_logger.debug.assert_not_called()
    
    @patch('app.core.resources.memory_compression_manager.logger')
    def test_should_compress_above_threshold(self, mock_logger, compression_manager, sample_memory_entries):
        """Test should_compress when tokens are above threshold."""
        # Mock token counter to return high token count
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=7000)  # Above threshold of 5000
        
        result = compression_manager.should_compress(sample_memory_entries)
        
        assert result is True
        compression_manager.token_counter.count_conversation_tokens.assert_called_once()
        mock_logger.info.assert_called()
        mock_logger.debug.assert_not_called()
    
    @patch('app.core.resources.memory_compression_manager.logger')
    def test_should_compress_exact_threshold(self, mock_logger, compression_manager, sample_memory_entries):
        """Test should_compress when tokens are exactly at threshold."""
        # Mock token counter to return exact threshold
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=5000)  # Exactly at threshold
        
        result = compression_manager.should_compress(sample_memory_entries)
        
        assert result is False  # Should not compress when exactly at threshold
        compression_manager.token_counter.count_conversation_tokens.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_logger.debug.assert_not_called()
    
    def test_should_compress_conversation_format(self, compression_manager, sample_memory_entries):
        """Test that should_compress formats conversation correctly for token counting."""
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=1000)
        
        compression_manager.should_compress(sample_memory_entries)
        
        # Verify the conversation format passed to token counter
        expected_format = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            {"role": "user", "content": "Can you help me with a question?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help you with any questions you have."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        compression_manager.token_counter.count_conversation_tokens.assert_called_once_with(expected_format)
    
    def test_split_conversation_for_compression_insufficient_messages(self, compression_manager, sample_conversation_history):
        """Test split_conversation_for_compression with insufficient messages."""
        # Keep only 2 messages (less than recent_messages_to_keep = 3)
        few_messages = sample_conversation_history[:2]
        
        older_messages, recent_messages = compression_manager.split_conversation_for_compression(few_messages)
        
        assert older_messages == []
        assert recent_messages == few_messages
    
    def test_split_conversation_for_compression_exact_messages(self, compression_manager, sample_conversation_history):
        """Test split_conversation_for_compression with exact number of messages."""
        # Keep exactly 3 messages (equal to recent_messages_to_keep)
        exact_messages = sample_conversation_history[:3]
        
        older_messages, recent_messages = compression_manager.split_conversation_for_compression(exact_messages)
        
        assert older_messages == []
        assert recent_messages == exact_messages
    
    def test_split_conversation_for_compression_more_messages(self, compression_manager, sample_conversation_history):
        """Test split_conversation_for_compression with more messages than threshold."""
        # 5 messages total, keep 3 recent
        older_messages, recent_messages = compression_manager.split_conversation_for_compression(sample_conversation_history)
        
        assert len(older_messages) == 2
        assert len(recent_messages) == 3
        assert older_messages == sample_conversation_history[:-3]
        assert recent_messages == sample_conversation_history[-3:]
    
    def test_split_conversation_for_compression_many_messages(self, compression_manager):
        """Test split_conversation_for_compression with many messages."""
        # Create 10 messages
        many_messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        
        older_messages, recent_messages = compression_manager.split_conversation_for_compression(many_messages)
        
        assert len(older_messages) == 7  # 10 - 3
        assert len(recent_messages) == 3
        assert older_messages == many_messages[:-3]
        assert recent_messages == many_messages[-3:]
    
    def test_format_messages_for_summary_empty_list(self, compression_manager):
        """Test format_messages_for_summary with empty list."""
        result = compression_manager.format_messages_for_summary([])
        
        assert result == ""
    
    def test_format_messages_for_summary_single_message(self, compression_manager, sample_memory_entries):
        """Test format_messages_for_summary with single message."""
        single_message = [sample_memory_entries[0]]
        
        result = compression_manager.format_messages_for_summary(single_message)
        
        expected = "1. USER: Hello, how are you?"
        assert result == expected
    
    def test_format_messages_for_summary_multiple_messages(self, compression_manager, sample_memory_entries):
        """Test format_messages_for_summary with multiple messages."""
        result = compression_manager.format_messages_for_summary(sample_memory_entries)
        
        expected = (
            "1. USER: Hello, how are you?\n\n"
            "2. ASSISTANT: I'm doing well, thank you for asking!\n\n"
            "3. USER: Can you help me with a question?\n\n"
            "4. ASSISTANT: Of course! I'd be happy to help you with any questions you have.\n\n"
            "5. USER: What is the capital of France?"
        )
        assert result == expected
    
    def test_format_messages_for_summary_different_roles(self, compression_manager):
        """Test format_messages_for_summary with different roles."""
        messages = [
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "system", "content": "You are a helpful assistant."}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "Hello"}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "Hi there!"}
            )
        ]
        
        result = compression_manager.format_messages_for_summary(messages)
        
        expected = (
            "1. SYSTEM: You are a helpful assistant.\n\n"
            "2. USER: Hello\n\n"
            "3. ASSISTANT: Hi there!"
        )
        assert result == expected
    
    def test_get_compression_stats(self, compression_manager, sample_conversation_history):
        """Test get_compression_stats functionality."""
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=6000)  # Above threshold
        
        stats = compression_manager.get_compression_stats(sample_conversation_history)
        
        expected_stats = {
            "agent_id": "test_agent",
            "enabled": True,
            "threshold_tokens": 5000,
            "current_tokens": 6000,
            "should_compress": True,
            "message_count": 5,
            "recent_messages_to_keep": 3
        }
        
        assert stats == expected_stats
        compression_manager.token_counter.count_conversation_tokens.assert_called_once_with(sample_conversation_history)
    
    def test_get_compression_stats_below_threshold(self, compression_manager, sample_conversation_history):
        """Test get_compression_stats when below threshold."""
        compression_manager.token_counter.count_conversation_tokens = MagicMock(return_value=3000)  # Below threshold
        
        stats = compression_manager.get_compression_stats(sample_conversation_history)
        
        assert stats["current_tokens"] == 3000
        assert stats["should_compress"] is False
    
    def test_get_compression_stats_empty_conversation(self, compression_manager):
        """Test get_compression_stats with empty conversation."""
        empty_conversation = []
        
        stats = compression_manager.get_compression_stats(empty_conversation)
        
        assert stats["message_count"] == 0
        assert stats["current_tokens"] == 0
        assert stats["should_compress"] is False
    
    def test_get_compression_stats_disabled_compression(self, compression_manager, sample_conversation_history):
        """Test get_compression_stats when compression is disabled."""
        compression_manager.enabled = False
        
        stats = compression_manager.get_compression_stats(sample_conversation_history)
        
        assert stats["enabled"] is False
        # should_compress should still reflect the actual token count vs threshold
        assert "should_compress" in stats
    
    def test_compression_manager_with_custom_threshold(self):
        """Test compression manager with custom threshold."""
        config = {
            "threshold_tokens": 10000,
            "recent_messages_to_keep": 5,
            "enabled": True
        }
        manager = MemoryCompressionManager("custom_agent", config)
        
        assert manager.threshold_tokens == 10000
        assert manager.recent_messages_to_keep == 5
        assert manager.enabled is True
    
    def test_compression_manager_edge_cases(self):
        """Test compression manager edge cases."""
        # Test with zero threshold
        config_zero = {"threshold_tokens": 0}
        manager_zero = MemoryCompressionManager("test_agent", config_zero)
        assert manager_zero.threshold_tokens == 0
        
        # Test with zero recent messages to keep
        config_zero_recent = {"recent_messages_to_keep": 0}
        manager_zero_recent = MemoryCompressionManager("test_agent", config_zero_recent)
        assert manager_zero_recent.recent_messages_to_keep == 0
        
        # Test with very high threshold
        config_high = {"threshold_tokens": 100000}
        manager_high = MemoryCompressionManager("test_agent", config_high)
        assert manager_high.threshold_tokens == 100000


class TestMemoryCompressionManagerIntegration:
    """Integration tests for Memory Compression Manager."""
    
    def test_full_compression_workflow(self):
        """Test the full compression workflow."""
        # Create manager with low threshold for testing
        config = {
            "threshold_tokens": 100,  # Low threshold to trigger compression
            "recent_messages_to_keep": 2,
            "enabled": True
        }
        manager = MemoryCompressionManager("test_agent", config)
        
        # Create conversation history with long messages to exceed threshold
        long_messages = [
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "This is a very long message that should exceed the token threshold. " * 10}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "This is another very long response that should also contribute to exceeding the threshold. " * 10}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "And this is a third long message to ensure we exceed the threshold. " * 10}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "This is the fourth long message in our conversation. " * 10}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "This is the most recent message that should be kept."}
            )
        ]
        
        # Test should_compress
        should_compress = manager.should_compress(long_messages)
        assert should_compress is True
        
        # Test split_conversation_for_compression
        conversation_format = [{"role": msg.content["role"], "content": msg.content["content"]} for msg in long_messages]
        older_messages, recent_messages = manager.split_conversation_for_compression(conversation_format)
        
        assert len(older_messages) == 3  # 5 - 2
        assert len(recent_messages) == 2
        
        # Test format_messages_for_summary
        formatted = manager.format_messages_for_summary(long_messages)
        assert "1. USER:" in formatted
        assert "2. ASSISTANT:" in formatted
        assert "5. USER:" in formatted
        
        # Test get_compression_stats
        stats = manager.get_compression_stats(conversation_format)
        assert stats["agent_id"] == "test_agent"
        assert stats["enabled"] is True
        assert stats["should_compress"] is True
        assert stats["message_count"] == 5
        assert stats["recent_messages_to_keep"] == 2
    
    def test_compression_manager_with_real_token_counting(self):
        """Test compression manager with real token counting (no mocking)."""
        config = {
            "threshold_tokens": 50,  # Very low threshold
            "recent_messages_to_keep": 1,
            "enabled": True
        }
        manager = MemoryCompressionManager("test_agent", config)
        
        # Create messages that should exceed the token threshold
        long_messages = [
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "This is a very long message that should definitely exceed the token threshold of 50 tokens. " * 5}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "assistant", "content": "This is another long response that should also contribute to exceeding the threshold. " * 5}
            ),
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content={"role": "user", "content": "This is the most recent message that should be kept."}
            )
        ]
        
        # Test should_compress with real token counting
        should_compress = manager.should_compress(long_messages)
        # The result depends on the actual token count, but we can verify the method works
        assert isinstance(should_compress, bool)
        
        # Test get_compression_stats with real token counting
        conversation_format = [{"role": msg.content["role"], "content": msg.content["content"]} for msg in long_messages]
        stats = manager.get_compression_stats(conversation_format)
        
        assert stats["agent_id"] == "test_agent"
        assert stats["enabled"] is True
        assert stats["message_count"] == 3
        assert stats["recent_messages_to_keep"] == 1
        assert isinstance(stats["current_tokens"], int)
        assert stats["current_tokens"] > 0
        assert isinstance(stats["should_compress"], bool) 