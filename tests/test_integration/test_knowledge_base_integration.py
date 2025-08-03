"""
Integration tests for Knowledge Base integration with memory compression.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.core.agents.api_agent import APIAgent
from app.models.resources.knowledge_base import DocumentType, Document, DocumentChunk, SearchResult
import json
import tempfile
import os


class TestKnowledgeBaseIntegration:
    """Integration tests for knowledge base functionality."""
    
    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent configuration with KB settings."""
        return {
            "agent_id": "test_knowledge_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4o-mini",
            "resources": ["memory", "knowledge_base"],
            "allowed_regular_tools": ["search_knowledge_base", "list_documents"],
            "resource_config": {
                "memory": {
                    "compression": {
                        "enabled": True,
                        "threshold_tokens": 100,  # Low threshold for testing
                        "recent_messages_to_keep": 2,
                        "archive_conversations": True
                    }
                },
                "knowledge_base": {
                    "vector_provider": "pgvector",
                    "enable_cross_session_context": True,
                    "search_triggers": ["what did we discuss", "previous conversation"]
                }
            }
        }
    
    @pytest.fixture(autouse=True)
    def mock_config_file(self, mock_agent_config):
        """Mock the agent config file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([mock_agent_config], f)
            config_path = f.name
        
        with patch('app.core.agents.agent_tool_manager.AgentToolManager._load_config') as mock_load:
            mock_load.return_value = mock_agent_config
            yield
        
        os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_memory_to_knowledge_base_flow(self, mock_agent_config):
        """Test automatic archival of compressed conversations"""
        # Mock the knowledge base resource
        mock_kb = AsyncMock()
        mock_kb.ingest_document = AsyncMock(return_value="doc_123")
        mock_kb.search = AsyncMock(return_value=[
            SearchResult(
                document=Document(
                    id="doc_123",
                    namespace="conversations:test_user",
                    doc_type=DocumentType.CONVERSATION,
                    title="Test Conversation",
                    content="Archived conversation content",
                    metadata={"session_id": "test_session"}
                ),
                chunk=DocumentChunk(
                    id="chunk_1",
                    document_id="doc_123",
                    namespace="conversations:test_user",
                    content="Important discussion about features",
                    chunk_index=0
                ),
                score=0.95
            )
        ])
        
        # Mock memory resource
        mock_memory = AsyncMock()
        mock_memory.get_memories = AsyncMock(return_value=[
            MagicMock(id="1", content={"role": "user", "content": f"Important discussion about feature {i}"})
            for i in range(10)
        ])
        mock_memory.get_session_summary = AsyncMock(return_value=None)
        mock_memory.store_session_summary = AsyncMock()
        mock_memory.delete_memory = AsyncMock()
        
        # Mock compression agent
        with patch('app.core.agents.memory_compression_agent.MemoryCompressionAgent', create=True) as mock_compression_class:
            mock_compression_agent = AsyncMock()
            mock_compression_agent.initialize = AsyncMock()
            mock_compression_agent.compress_conversation = AsyncMock(return_value="Compressed summary")
            mock_compression_class.return_value = mock_compression_agent
            
            # Create agent with mocked resources
            agent = APIAgent("test_knowledge_agent", "test_user", "test_session")
            agent.knowledge_base = mock_kb
            agent.memory = mock_memory
            agent.tool_manager = MagicMock()
            agent.tool_manager.config = mock_agent_config
            
            await agent.initialize()
            
            # Trigger compression
            await agent._trigger_memory_compression()
            
            # Verify compression was called with KB resource
            mock_compression_agent.compress_conversation.assert_called_once()
            call_args = mock_compression_agent.compress_conversation.call_args
            assert call_args.kwargs.get("knowledge_base_resource") == mock_kb
            
            # Verify search works
            search_results = await mock_kb.search(
                "Important discussion",
                namespaces=["conversations:test_user"]
            )
            
            assert len(search_results) > 0
            assert search_results[0].document.metadata["session_id"] == "test_session"
            assert search_results[0].document.doc_type == DocumentType.CONVERSATION
    
    @pytest.mark.asyncio
    async def test_cross_session_context(self, mock_agent_config):
        """Test cross-session context retrieval"""
        # Mock knowledge base with past conversation data
        mock_kb = AsyncMock()
        mock_kb.search = AsyncMock(return_value=[
            SearchResult(
                document=Document(
                    id="past_doc",
                    namespace="conversations:test_user",
                    doc_type=DocumentType.CONVERSATION,
                    title="Past Conversation",
                    content="Past conversation about JWT authentication",
                    metadata={
                        "session_id": "session1",
                        "date_range": {
                            "start": "2024-01-01",
                            "end": "2024-01-01"
                        }
                    }
                ),
                chunk=DocumentChunk(
                    id="chunk_1",
                    document_id="past_doc",
                    namespace="conversations:test_user",
                    content="Discussion about JWT tokens with 1-hour expiration using RS256",
                    chunk_index=0
                ),
                score=0.85
            )
        ])
        
        # Mock memory
        mock_memory = AsyncMock()
        mock_memory.get_memories = AsyncMock(return_value=[])
        mock_memory.get_session_summary = AsyncMock(return_value=None)
        
        # Create agent
        agent = APIAgent("test_knowledge_agent", "test_user", "session2")
        agent.knowledge_base = mock_kb
        agent.memory = mock_memory
        agent.tool_manager = MagicMock()
        agent.tool_manager.config = mock_agent_config
        
        await agent.initialize()
        
        # Load memory to prepare agent state
        await agent.load_memory()
        
        # Simulate user asking about past conversations
        test_message = "What did we discuss about authentication?"
        should_search = agent._should_search_cross_session(test_message)
        assert should_search is True
        
        # Get cross-session context
        context = await agent._get_cross_session_context(test_message, "session2")
        assert context is not None
        assert "JWT tokens" in context
        assert "RS256" in context
        
        # Verify the search was called with correct params (no metadata_filters in search)
        mock_kb.search.assert_called_with(
            query=test_message,
            namespaces=["conversations:test_user"],
            limit=10,
            use_reranking=True
        )