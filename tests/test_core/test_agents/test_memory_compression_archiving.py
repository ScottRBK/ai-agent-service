"""
Unit tests for knowledge base archiving functionality in MemoryCompressionAgent.
Tests the archiving behavior when compressing conversation history.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.core.agents.memory_compression_agent import MemoryCompressionAgent
from app.models.resources.memory import MemoryEntry, MemorySessionSummary
from app.models.resources.knowledge_base import DocumentType


@pytest.fixture
def mock_provider():
    """Mock AI provider for summary generation"""
    provider = Mock()
    provider.initialize = AsyncMock()
    provider.send_chat = AsyncMock(return_value="Generated summary of the conversation")
    provider.cleanup = AsyncMock()
    return provider


@pytest.fixture
def mock_memory_resource():
    """Mock memory resource with conversation history"""
    resource = Mock()
    
    # Sample conversation history
    sample_messages = [
        MemoryEntry(
            id="msg1",
            user_id="user123",
            session_id="session456",
            agent_id="parent_agent",
            role="user",
            content="Hello, how are you?",
            created_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        ),
        MemoryEntry(
            id="msg2",
            user_id="user123",
            session_id="session456",
            agent_id="parent_agent",
            role="assistant",
            content="I'm doing well, thank you for asking!",
            created_at=datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        )
    ]
    
    resource.get_memories = AsyncMock(return_value=sample_messages)
    resource.get_session_summary = AsyncMock(return_value=None)
    resource.store_session_summary = AsyncMock()
    resource.delete_memory = AsyncMock()
    
    return resource


@pytest.fixture
def mock_knowledge_base_resource():
    """Mock knowledge base resource for archiving"""
    resource = Mock()
    resource.ingest_document = AsyncMock()
    return resource


@pytest.fixture
def compression_agent():
    """Create MemoryCompressionAgent instance"""
    with patch('app.core.agents.memory_compression_agent.BaseAgent.__init__'):
        agent = MemoryCompressionAgent()
        agent.initialized = True
        agent.provider_manager = Mock()
        agent.provider_id = "azure_openai_cc"  # Set provider_id attribute
        agent.model = "gpt-4"
        agent.model_settings = {"temperature": 0.7}
        agent.system_prompt = "You are a helpful assistant."
        agent.agent_id = "memory_compression_agent"
        agent._clean_response_for_memory = lambda x: x  # Simple pass-through
        return agent


@pytest.fixture
def compression_config_with_archiving():
    """Compression configuration with archiving enabled"""
    return {
        "archive_conversations": True,
        "compression_threshold": 8000,
        "recent_message_count": 10
    }


@pytest.fixture
def compression_config_without_archiving():
    """Compression configuration with archiving disabled"""
    return {
        "archive_conversations": False,
        "compression_threshold": 8000,
        "recent_message_count": 10
    }


class TestMemoryCompressionArchiving:
    """Test cases for knowledge base archiving functionality"""
    
    @pytest.mark.asyncio
    async def test_archive_when_enabled_and_kb_exists(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        mock_knowledge_base_resource,
        compression_config_with_archiving
    ):
        """Test that conversation is archived when archive_conversations=True and KB resource exists"""
        
        # Setup provider manager to return our mock provider
        provider_info = {
            "class": lambda config: mock_provider,
            "config_class": Mock
        }
        compression_agent.provider_manager.get_provider.return_value = provider_info
        
        # Mock the compression manager to require compression
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = True
            mock_manager.split_conversation_for_compression.return_value = (
                [mock_memory_resource.get_memories.return_value[0]],  # older_messages
                [mock_memory_resource.get_memories.return_value[1]]   # recent_messages
            )
            mock_manager.format_messages_for_summary.return_value = "Formatted messages"
            
            # Mock the structured summary response
            structured_response = """## SUMMARY
Test conversation summary

## TOPICS
testing, conversation

## ENTITIES
user, assistant

## DECISIONS
None

## QUESTIONS
None"""
            mock_provider.send_chat.return_value = structured_response
            
            # Execute compression with archiving
            await compression_agent.compress_conversation(
                parent_agent_id="parent_agent",
                compression_config=compression_config_with_archiving,
                user_id="user123",
                session_id="session456",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=mock_knowledge_base_resource
            )
            
            # Verify knowledge base ingest_document was called with correct parameters
            mock_knowledge_base_resource.ingest_document.assert_called_once()
            call_args = mock_knowledge_base_resource.ingest_document.call_args
            
            assert call_args[1]["user_id"] == "user123"
            assert call_args[1]["namespace_type"] == "conversations"
            assert call_args[1]["doc_type"] == DocumentType.CONVERSATION
            assert call_args[1]["source"] == "session:session456"
            assert "session456" in call_args[1]["title"]
            
            # Verify content includes summary and metadata
            content = call_args[1]["content"]
            assert "Test conversation summary" in content
            assert "Topics Discussed" in content
            
            # Verify metadata structure
            metadata = call_args[1]["metadata"]
            assert metadata["session_id"] == "session456"
            assert metadata["agent_id"] == "parent_agent"
            assert "conversation_topics" in metadata
            assert "compression_timestamp" in metadata


    @pytest.mark.asyncio
    async def test_no_archive_when_disabled(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        mock_knowledge_base_resource,
        compression_config_without_archiving
    ):
        """Test that conversation is NOT archived when archive_conversations=False"""
        
        # Setup provider manager
        provider_info = {
            "class": lambda config: mock_provider,
            "config_class": Mock
        }
        compression_agent.provider_manager.get_provider.return_value = provider_info
        
        # Mock compression manager
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = True
            mock_manager.split_conversation_for_compression.return_value = (
                [mock_memory_resource.get_memories.return_value[0]],
                [mock_memory_resource.get_memories.return_value[1]]
            )
            mock_manager.format_messages_for_summary.return_value = "Formatted messages"
            
            # Execute compression without archiving
            await compression_agent.compress_conversation(
                parent_agent_id="parent_agent",
                compression_config=compression_config_without_archiving,
                user_id="user123",
                session_id="session456",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=mock_knowledge_base_resource
            )
            
            # Verify knowledge base ingest_document was NOT called
            mock_knowledge_base_resource.ingest_document.assert_not_called()


    @pytest.mark.asyncio
    async def test_no_archive_when_kb_resource_is_none(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        compression_config_with_archiving
    ):
        """Test that conversation is NOT archived when knowledge_base_resource is None"""
        
        # Setup provider manager
        provider_info = {
            "class": lambda config: mock_provider,
            "config_class": Mock
        }
        compression_agent.provider_manager.get_provider.return_value = provider_info
        
        # Mock compression manager
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = True
            mock_manager.split_conversation_for_compression.return_value = (
                [mock_memory_resource.get_memories.return_value[0]],
                [mock_memory_resource.get_memories.return_value[1]]
            )
            mock_manager.format_messages_for_summary.return_value = "Formatted messages"
            
            # Execute compression with None knowledge_base_resource
            await compression_agent.compress_conversation(
                parent_agent_id="parent_agent",
                compression_config=compression_config_with_archiving,
                user_id="user123",
                session_id="session456",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=None  # This should prevent archiving
            )
            
            # Since we can't verify a method wasn't called on None, 
            # we can verify the flow completed successfully without errors
            # The main test is that no exception was raised


    @pytest.mark.asyncio
    async def test_archive_with_correct_document_parameters(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        mock_knowledge_base_resource,
        compression_config_with_archiving
    ):
        """Test that ingest_document is called with all correct parameters"""
        
        # Setup provider manager
        provider_info = {
            "class": lambda config: mock_provider,
            "config_class": Mock
        }
        compression_agent.provider_manager.get_provider.return_value = provider_info
        
        # Mock compression manager
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = True
            mock_manager.split_conversation_for_compression.return_value = (
                [mock_memory_resource.get_memories.return_value[0]],
                [mock_memory_resource.get_memories.return_value[1]]
            )
            mock_manager.format_messages_for_summary.return_value = "Formatted messages"
            
            # Mock structured response with rich metadata
            structured_response = """## SUMMARY
Detailed conversation about project planning

## TOPICS
project planning, deadlines, resources

## ENTITIES
John, Project Alpha, Python

## DECISIONS
Use Python for backend development
Set deadline for next Friday

## QUESTIONS
What about testing strategy?"""
            mock_provider.send_chat.return_value = structured_response
            
            # Execute compression
            await compression_agent.compress_conversation(
                parent_agent_id="test_agent",
                compression_config=compression_config_with_archiving,
                user_id="user456",
                session_id="session789",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=mock_knowledge_base_resource
            )
            
            # Verify all parameters passed to ingest_document
            call_args = mock_knowledge_base_resource.ingest_document.call_args
            kwargs = call_args[1]
            
            # Required parameters
            assert kwargs["user_id"] == "user456"
            assert kwargs["namespace_type"] == "conversations"
            assert kwargs["doc_type"] == DocumentType.CONVERSATION
            assert kwargs["source"] == "session:session789"
            
            # Title should contain session info
            assert "session789" in kwargs["title"]
            
            # Content should be structured
            content = kwargs["content"]
            assert "# Conversation Summary" in content
            assert "Detailed conversation about project planning" in content
            assert "Topics Discussed:" in content
            assert "Key Decisions:" in content
            assert "Open Questions:" in content
            
            # Metadata should contain all extracted information
            metadata = kwargs["metadata"]
            assert metadata["session_id"] == "session789"
            assert metadata["agent_id"] == "test_agent"
            assert "project planning" in metadata["conversation_topics"]
            assert "John" in metadata["entities_mentioned"]
            assert "Use Python for backend development" in metadata["decisions_made"]
            assert "What about testing strategy?" in metadata["open_questions"]
            assert "compression_timestamp" in metadata
            assert "date_range" in metadata


    @pytest.mark.asyncio
    async def test_archive_failure_handled_gracefully(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        mock_knowledge_base_resource,
        compression_config_with_archiving
    ):
        """Test that archive failures are handled gracefully without crashing compression"""
        
        # Setup provider manager
        provider_info = {
            "class": lambda config: mock_provider,
            "config_class": Mock
        }
        compression_agent.provider_manager.get_provider.return_value = provider_info
        
        # Make knowledge base ingest_document raise an exception
        mock_knowledge_base_resource.ingest_document.side_effect = Exception("KB archival failed")
        
        # Mock compression manager
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = True
            mock_manager.split_conversation_for_compression.return_value = (
                [mock_memory_resource.get_memories.return_value[0]],
                [mock_memory_resource.get_memories.return_value[1]]
            )
            mock_manager.format_messages_for_summary.return_value = "Formatted messages"
            
            structured_response = """## SUMMARY
Test summary

## TOPICS
test

## ENTITIES
user

## DECISIONS
None

## QUESTIONS
None"""
            mock_provider.send_chat.return_value = structured_response
            
            # Execute compression - should not raise exception despite KB failure
            await compression_agent.compress_conversation(
                parent_agent_id="parent_agent",
                compression_config=compression_config_with_archiving,
                user_id="user123",
                session_id="session456",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=mock_knowledge_base_resource
            )
            
            # Verify that compression continued successfully
            # (storing summary and deleting old messages should still happen)
            mock_memory_resource.store_session_summary.assert_called_once()
            mock_memory_resource.delete_memory.assert_called()


    @pytest.mark.asyncio
    async def test_no_archiving_when_no_compression_needed(
        self, 
        compression_agent, 
        mock_provider,
        mock_memory_resource,
        mock_knowledge_base_resource,
        compression_config_with_archiving
    ):
        """Test that no archiving occurs when compression is not needed"""
        
        # Mock compression manager to indicate no compression needed
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionManager') as mock_manager_class:
            mock_manager = mock_manager_class.return_value
            mock_manager.should_compress.return_value = False  # No compression needed
            
            # Execute compression
            result = await compression_agent.compress_conversation(
                parent_agent_id="parent_agent",
                compression_config=compression_config_with_archiving,
                user_id="user123",
                session_id="session456",
                parent_memory_resource=mock_memory_resource,
                knowledge_base_resource=mock_knowledge_base_resource
            )
            
            # Verify no archiving occurred
            mock_knowledge_base_resource.ingest_document.assert_not_called()
            
            # Verify early return with empty string
            assert result == ""