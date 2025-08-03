# tests/test_core/test_resources/test_knowledge_base.py
"""
Unit tests for KnowledgeBaseResource.
"""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Optional, Dict, Any

from app.core.resources.knowledge_base import KnowledgeBaseResource
from app.core.resources.base import ResourceError, ResourceType
from app.models.resources.knowledge_base import (
    Document, DocumentChunk, SearchResult, SearchFilters, DocumentType
)


class TestKnowledgeBaseResource:
    """Test cases for KnowledgeBaseResource."""
    
    @pytest.fixture
    def knowledge_base_config(self):
        """Test configuration for knowledge base resource."""
        return {
            "vector_provider": "pgvector",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking": {
                "text": "simple",
                "markdown": "simple"
            },
            "rerank_limit": 50
        }
    
    @pytest.fixture
    def knowledge_base_resource(self, knowledge_base_config):
        """Create knowledge base resource instance."""
        with patch('app.core.resources.knowledge_base.PGVectorProvider'):
            return KnowledgeBaseResource("test_kb", knowledge_base_config)
    
    @pytest.fixture
    def mock_vector_provider(self):
        """Mock vector provider."""
        provider = AsyncMock()
        provider.initialize = AsyncMock()
        provider.cleanup = AsyncMock()
        provider.health_check = AsyncMock(return_value=True)
        provider.store_document = AsyncMock()
        provider.store_chunks = AsyncMock()
        provider.search_similar = AsyncMock()
        provider.get_document = AsyncMock()
        provider.list_documents = AsyncMock()
        provider.delete_document = AsyncMock(return_value=True)
        return provider
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        return provider
    
    @pytest.fixture
    def mock_chat_provider(self):
        """Mock chat provider."""
        provider = AsyncMock()
        return provider
    
    @pytest.fixture
    def mock_rerank_provider(self):
        """Mock rerank provider."""
        provider = AsyncMock()
        provider.rerank = AsyncMock(return_value=[0.9, 0.8, 0.7])
        return provider
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return Document(
            id="doc-123",
            namespace="test_namespace:text-embedding-ada-002",
            doc_type=DocumentType.TEXT,
            source="test.txt",
            title="Test Document",
            content="This is a test document. It has multiple sentences. This is for testing chunking.",
            metadata={"author": "test"}
        )
    
    @pytest.fixture
    def sample_document_chunk(self):
        """Sample document chunk for testing."""
        return DocumentChunk(
            id="chunk-123",
            document_id="doc-123",
            namespace="test_namespace:text-embedding-ada-002",
            chunk_index=0,
            content="This is a test document.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"chunk_count": 3}
        )
    
    @pytest.fixture
    def sample_search_result(self, sample_document_chunk, sample_document):
        """Sample search result for testing."""
        return SearchResult(
            chunk=sample_document_chunk,
            score=0.95,
            document=sample_document
        )

    # Initialization and Configuration Tests
    
    def test_init_with_valid_config(self, knowledge_base_config):
        """Test initialization with valid configuration."""
        with patch('app.core.resources.knowledge_base.PGVectorProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider
            
            resource = KnowledgeBaseResource("test_kb", knowledge_base_config)
            
            assert resource.resource_id == "test_kb"
            assert resource.chunk_size == 1000
            assert resource.chunk_overlap == 200
            assert resource.chunking_config == {"text": "simple", "markdown": "simple"}
            assert resource.rerank_limit == 50
            assert resource.vector_provider == mock_provider
            assert resource.chat_provider is None
            assert resource.embedding_provider is None
            assert resource.embedding_model is None
            assert resource.rerank_provider is None
            assert resource.rerank_model is None
            mock_provider_class.assert_called_once_with(knowledge_base_config)
    
    def test_init_with_default_config(self):
        """Test initialization with minimal configuration."""
        config = {"vector_provider": "pgvector"}
        
        with patch('app.core.resources.knowledge_base.PGVectorProvider') as mock_provider_class:
            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider
            
            resource = KnowledgeBaseResource("test_kb", config)
            
            assert resource.chunk_size == 1000  # default
            assert resource.chunk_overlap == 200  # default
            assert resource.chunking_config == {}  # default
            assert resource.rerank_limit == 50  # default
    
    def test_init_with_unknown_vector_provider(self):
        """Test initialization fails with unknown vector provider."""
        config = {"vector_provider": "unknown_provider"}
        
        with pytest.raises(ResourceError, match="Unknown vector provider: unknown_provider"):
            KnowledgeBaseResource("test_kb", config)
    
    def test_get_resource_type(self, knowledge_base_resource):
        """Test resource type identification."""
        assert knowledge_base_resource._get_resource_type() == ResourceType.KNOWLEDGE_BASE

    # Provider Injection Tests
    
    def test_set_chat_provider(self, knowledge_base_resource, mock_chat_provider):
        """Test chat provider injection."""
        knowledge_base_resource.set_chat_provider(mock_chat_provider)
        assert knowledge_base_resource.chat_provider == mock_chat_provider
    
    def test_set_embedding_provider(self, knowledge_base_resource, mock_embedding_provider):
        """Test embedding provider injection."""
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        assert knowledge_base_resource.embedding_provider == mock_embedding_provider
        assert knowledge_base_resource.embedding_model == "text-embedding-ada-002"
    
    def test_set_rerank_provider(self, knowledge_base_resource, mock_rerank_provider):
        """Test rerank provider injection."""
        knowledge_base_resource.set_rerank_provider(mock_rerank_provider, "gpt-4o-mini")
        assert knowledge_base_resource.rerank_provider == mock_rerank_provider
        assert knowledge_base_resource.rerank_model == "gpt-4o-mini"

    # Lifecycle Management Tests
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, knowledge_base_resource, mock_vector_provider):
        """Test successful initialization."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        await knowledge_base_resource.initialize()
        
        assert knowledge_base_resource.initialized is True
        mock_vector_provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, knowledge_base_resource, mock_vector_provider):
        """Test initialization failure."""
        mock_vector_provider.initialize.side_effect = Exception("Vector store connection failed")
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        with pytest.raises(ResourceError, match="Failed to initialize knowledge base"):
            await knowledge_base_resource.initialize()
        
        assert knowledge_base_resource.initialized is False
    
    @pytest.mark.asyncio
    async def test_cleanup_success(self, knowledge_base_resource, mock_vector_provider):
        """Test successful cleanup."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        await knowledge_base_resource.cleanup()
        
        mock_vector_provider.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, knowledge_base_resource, mock_vector_provider):
        """Test cleanup with exception (should not raise)."""
        mock_vector_provider.cleanup.side_effect = Exception("Cleanup failed")
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        # Should not raise an exception
        await knowledge_base_resource.cleanup()
        
        mock_vector_provider.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, knowledge_base_resource, mock_vector_provider):
        """Test successful health check."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        result = await knowledge_base_resource.health_check()
        
        assert result is True
        mock_vector_provider.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, knowledge_base_resource, mock_vector_provider):
        """Test health check failure."""
        mock_vector_provider.health_check.side_effect = Exception("Health check failed")
        knowledge_base_resource.vector_provider = mock_vector_provider
        
        result = await knowledge_base_resource.health_check()
        
        assert result is False

    # Embedding Generation Tests
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, knowledge_base_resource, mock_embedding_provider):
        """Test successful embedding generation."""
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        
        result = await knowledge_base_resource._generate_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_embedding_provider.embed.assert_called_once_with(
            model="text-embedding-ada-002", 
            text="test text"
        )
    
    @pytest.mark.asyncio
    async def test_generate_embedding_no_provider(self, knowledge_base_resource):
        """Test embedding generation without provider."""
        with pytest.raises(ResourceError, match="No embedding provider configured"):
            await knowledge_base_resource._generate_embedding("test text")
    
    @pytest.mark.asyncio
    async def test_generate_embedding_no_model(self, knowledge_base_resource, mock_embedding_provider):
        """Test embedding generation without model."""
        knowledge_base_resource.embedding_provider = mock_embedding_provider
        # Don't set embedding_model
        
        with pytest.raises(ResourceError, match="No embedding model configured"):
            await knowledge_base_resource._generate_embedding("test text")

    # Chunking Tests
    
    def test_chunk_text_simple_strategy(self, knowledge_base_resource):
        """Test text chunking with simple strategy."""
        text = "This is a test. It has multiple sentences. This is for testing."
        
        with patch.object(knowledge_base_resource, '_get_chunking_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.chunk.return_value = ["Chunk 1", "Chunk 2"]
            mock_get_strategy.return_value = mock_strategy
            
            result = knowledge_base_resource._chunk_text(text, DocumentType.TEXT)
            
            assert result == ["Chunk 1", "Chunk 2"]
            mock_strategy.chunk.assert_called_once_with(text, 1000, 200)
    
    def test_get_chunking_strategy_simple(self, knowledge_base_resource):
        """Test getting simple chunking strategy."""
        # Test that the strategy is correctly instantiated 
        from app.core.resources.chunking.simple import SimpleChunkingStrategy
        result = knowledge_base_resource._get_chunking_strategy(DocumentType.TEXT)
        assert isinstance(result, SimpleChunkingStrategy)
    
    def test_get_chunking_strategy_configured(self, knowledge_base_resource):
        """Test getting configured chunking strategy."""
        knowledge_base_resource.chunking_config = {"text": "simple"}
        
        from app.core.resources.chunking.simple import SimpleChunkingStrategy
        result = knowledge_base_resource._get_chunking_strategy(DocumentType.TEXT)
        assert isinstance(result, SimpleChunkingStrategy)
    
    def test_get_chunking_strategy_unknown_fallback(self, knowledge_base_resource):
        """Test unknown chunking strategy falls back to simple."""
        knowledge_base_resource.chunking_config = {"text": "unknown_strategy"}
        
        from app.core.resources.chunking.simple import SimpleChunkingStrategy
        result = knowledge_base_resource._get_chunking_strategy(DocumentType.TEXT)
        assert isinstance(result, SimpleChunkingStrategy)

    # Namespace Creation Tests
    
    def test_create_model_namespace(self, knowledge_base_resource):
        """Test model namespace creation."""
        knowledge_base_resource.embedding_model = "text-embedding-ada-002"
        
        result = knowledge_base_resource._create_model_namespace("test_namespace")
        
        assert result == "test_namespace:text-embedding-ada-002"
    
    def test_create_model_namespace_no_model(self, knowledge_base_resource):
        """Test model namespace creation without embedding model."""
        with pytest.raises(ResourceError, match="No embedding model configured"):
            knowledge_base_resource._create_model_namespace("test_namespace")

    # Document Ingestion Tests
    
    @pytest.mark.asyncio
    async def test_ingest_document_success(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider):
        """Test successful document ingestion."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        
        # Mock UUID generation - patch where uuid is imported in the function
        with patch('uuid.uuid4') as mock_uuid:
            # Create mock UUID objects that return the expected string when str() is called
            mock_doc_uuid = Mock()
            mock_doc_uuid.__str__ = Mock(return_value='doc-123')
            mock_chunk1_uuid = Mock()
            mock_chunk1_uuid.__str__ = Mock(return_value='chunk-1')
            mock_chunk2_uuid = Mock()
            mock_chunk2_uuid.__str__ = Mock(return_value='chunk-2')
            
            mock_uuid.side_effect = [mock_doc_uuid, mock_chunk1_uuid, mock_chunk2_uuid]
            
            # Mock chunking
            with patch.object(knowledge_base_resource, '_chunk_text') as mock_chunk:
                mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
                
                # Mock record_successful_call
                with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
                    result = await knowledge_base_resource.ingest_document(
                        content="Test content",
                        namespace="test_namespace",
                        doc_type=DocumentType.TEXT,
                        source="test.txt",
                        title="Test Document",
                        metadata={"author": "test"}
                    )
                    
                    assert result == "doc-123"
                    
                    # Verify document storage
                    mock_vector_provider.store_document.assert_called_once()
                    stored_doc = mock_vector_provider.store_document.call_args[0][0]
                    assert stored_doc.id == "doc-123"
                    assert stored_doc.namespace == "test_namespace:text-embedding-ada-002"
                    assert stored_doc.content == "Test content"
                    
                    # Verify chunk storage
                    mock_vector_provider.store_chunks.assert_called_once()
                    stored_chunks = mock_vector_provider.store_chunks.call_args[0][0]
                    assert len(stored_chunks) == 2
                    assert stored_chunks[0].content == "Chunk 1"
                    assert stored_chunks[1].content == "Chunk 2"
                    
                    # Verify embedding calls
                    assert mock_embedding_provider.embed.call_count == 2
                    
                    mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest_document_no_embedding_model(self, knowledge_base_resource):
        """Test document ingestion without embedding model."""
        with pytest.raises(ResourceError, match="No embedding model configured"):
            await knowledge_base_resource.ingest_document(
                content="Test content",
                namespace="test_namespace",
                doc_type=DocumentType.TEXT
            )
    
    @pytest.mark.asyncio
    async def test_ingest_document_embedding_failure(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider):
        """Test document ingestion with embedding failure."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        mock_embedding_provider.embed.side_effect = Exception("Embedding failed")
        
        with patch.object(knowledge_base_resource, '_chunk_text') as mock_chunk:
            mock_chunk.return_value = ["Chunk 1"]
            
            with patch.object(knowledge_base_resource, 'record_failed_call') as mock_record_failed:
                with pytest.raises(ResourceError, match="Failed to ingest document"):
                    await knowledge_base_resource.ingest_document(
                        content="Test content",
                        namespace="test_namespace",
                        doc_type=DocumentType.TEXT
                    )
                
                mock_record_failed.assert_called_once()

    # Search Tests
    
    @pytest.mark.asyncio
    async def test_search_success_without_reranking(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider, sample_search_result):
        """Test successful search without reranking."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        
        mock_vector_provider.search_similar.return_value = [sample_search_result]
        
        with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
            result = await knowledge_base_resource.search(
                query="test query",
                namespaces=["test_namespace"],
                doc_types=[DocumentType.TEXT],
                limit=10,
                use_reranking=False
            )
            
            assert len(result) == 1
            assert result[0] == sample_search_result
            
            # Verify embedding generation
            mock_embedding_provider.embed.assert_called_once_with(
                model="text-embedding-ada-002",
                text="test query"
            )
            
            # Verify vector search
            mock_vector_provider.search_similar.assert_called_once()
            call_args = mock_vector_provider.search_similar.call_args
            query_embedding = call_args[0][0]
            filters = call_args[0][1]
            
            assert query_embedding == [0.1, 0.2, 0.3, 0.4]
            assert filters.namespaces == ["test_namespace:text-embedding-ada-002"]
            assert filters.doc_types == [DocumentType.TEXT]
            assert filters.limit == 10
            
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_success_with_reranking(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider, mock_rerank_provider, sample_search_result):
        """Test successful search with reranking."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        knowledge_base_resource.set_rerank_provider(mock_rerank_provider, "gpt-4o-mini")
        
        # Create multiple search results for reranking
        results = [sample_search_result for _ in range(5)]
        mock_vector_provider.search_similar.return_value = results
        
        with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
            result = await knowledge_base_resource.search(
                query="test query",
                limit=3,
                use_reranking=True
            )
            
            assert len(result) <= 3
            
            # Verify reranking was called
            mock_rerank_provider.rerank.assert_called_once()
            
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_no_namespaces(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider):
        """Test search without namespace filtering."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        
        mock_vector_provider.search_similar.return_value = []
        
        await knowledge_base_resource.search(query="test query")
        
        # Verify filters have no namespaces
        call_args = mock_vector_provider.search_similar.call_args
        filters = call_args[0][1]
        assert filters.namespaces is None
    
    @pytest.mark.asyncio
    async def test_search_embedding_failure(self, knowledge_base_resource, mock_embedding_provider):
        """Test search with embedding failure."""
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        mock_embedding_provider.embed.side_effect = Exception("Embedding failed")
        
        with patch.object(knowledge_base_resource, 'record_failed_call') as mock_record_failed:
            with pytest.raises(ResourceError, match="Search failed"):
                await knowledge_base_resource.search(query="test query")
            
            mock_record_failed.assert_called_once()

    # Reranking Tests
    
    @pytest.mark.asyncio
    async def test_rerank_results_success(self, knowledge_base_resource, mock_rerank_provider, sample_search_result):
        """Test successful result reranking."""
        knowledge_base_resource.set_rerank_provider(mock_rerank_provider, "gpt-4o-mini")
        
        # Create separate result objects to avoid mutation issues
        from copy import deepcopy
        results = [deepcopy(sample_search_result) for _ in range(3)]
        # Set different initial scores
        results[0].score = 0.5
        results[1].score = 0.6
        results[2].score = 0.4
        
        mock_rerank_provider.rerank.return_value = [0.9, 0.8, 0.7]
        
        reranked = await knowledge_base_resource._rerank_results("test query", results, 2)
        
        assert len(reranked) == 2
        # Results should be sorted by rerank score (highest first)
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.8
        
        mock_rerank_provider.rerank.assert_called_once_with(
            "gpt-4o-mini",
            "test query",
            [result.chunk.content for result in results]
        )
    
    @pytest.mark.asyncio
    async def test_rerank_results_failure_fallback(self, knowledge_base_resource, mock_rerank_provider, sample_search_result):
        """Test reranking failure falls back to original results."""
        knowledge_base_resource.set_rerank_provider(mock_rerank_provider, "gpt-4o-mini")
        mock_rerank_provider.rerank.side_effect = Exception("Reranking failed")
        
        results = [sample_search_result for _ in range(3)]
        
        reranked = await knowledge_base_resource._rerank_results("test query", results, 2)
        
        # Should return first 2 original results
        assert len(reranked) == 2
        assert reranked == results[:2]

    # Document Management Tests
    
    @pytest.mark.asyncio
    async def test_get_document_success(self, knowledge_base_resource, mock_vector_provider, sample_document):
        """Test successful document retrieval."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.get_document.return_value = sample_document
        
        with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
            result = await knowledge_base_resource.get_document("doc-123")
            
            assert result == sample_document
            mock_vector_provider.get_document.assert_called_once_with("doc-123")
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, knowledge_base_resource, mock_vector_provider):
        """Test document retrieval when not found."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.get_document.return_value = None
        
        result = await knowledge_base_resource.get_document("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_document_failure(self, knowledge_base_resource, mock_vector_provider):
        """Test document retrieval failure."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.get_document.side_effect = Exception("Database error")
        
        with patch.object(knowledge_base_resource, 'record_failed_call') as mock_record_failed:
            with pytest.raises(ResourceError, match="Failed to get document"):
                await knowledge_base_resource.get_document("doc-123")
            
            mock_record_failed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_documents_success(self, knowledge_base_resource, mock_vector_provider, sample_document):
        """Test successful document listing."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.embedding_model = "text-embedding-ada-002"
        mock_vector_provider.list_documents.return_value = [sample_document]
        
        with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
            result = await knowledge_base_resource.list_documents("test_namespace")
            
            assert len(result) == 1
            assert result[0] == sample_document
            
            # Verify model namespace was used
            mock_vector_provider.list_documents.assert_called_once_with(
                "test_namespace:text-embedding-ada-002"
            )
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_documents_failure(self, knowledge_base_resource, mock_vector_provider):
        """Test document listing failure."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.embedding_model = "text-embedding-ada-002"
        mock_vector_provider.list_documents.side_effect = Exception("Database error")
        
        with patch.object(knowledge_base_resource, 'record_failed_call') as mock_record_failed:
            with pytest.raises(ResourceError, match="Failed to list documents"):
                await knowledge_base_resource.list_documents("test_namespace")
            
            mock_record_failed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self, knowledge_base_resource, mock_vector_provider):
        """Test successful document deletion."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.delete_document.return_value = True
        
        with patch.object(knowledge_base_resource, 'record_successful_call') as mock_record:
            result = await knowledge_base_resource.delete_document("doc-123")
            
            assert result is True
            mock_vector_provider.delete_document.assert_called_once_with("doc-123")
            mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, knowledge_base_resource, mock_vector_provider):
        """Test document deletion when not found."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.delete_document.return_value = False
        
        result = await knowledge_base_resource.delete_document("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_document_failure(self, knowledge_base_resource, mock_vector_provider):
        """Test document deletion failure."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        mock_vector_provider.delete_document.side_effect = Exception("Database error")
        
        with patch.object(knowledge_base_resource, 'record_failed_call') as mock_record_failed:
            with pytest.raises(ResourceError, match="Failed to delete document"):
                await knowledge_base_resource.delete_document("doc-123")
            
            mock_record_failed.assert_called_once()

    # Configuration Edge Cases
    
    def test_chunking_config_variations(self, knowledge_base_config):
        """Test various chunking configuration scenarios."""
        # Test with custom chunking config
        knowledge_base_config["chunking"] = {
            "text": "simple",
            "markdown": "simple",
            "json": "simple"
        }
        
        with patch('app.core.resources.knowledge_base.PGVectorProvider'):
            resource = KnowledgeBaseResource("test_kb", knowledge_base_config)
            assert resource.chunking_config["text"] == "simple"
            assert resource.chunking_config["markdown"] == "simple"
            assert resource.chunking_config["json"] == "simple"
    
    def test_rerank_limit_configuration(self, knowledge_base_config):
        """Test rerank limit configuration."""
        knowledge_base_config["rerank_limit"] = 100
        
        with patch('app.core.resources.knowledge_base.PGVectorProvider'):
            resource = KnowledgeBaseResource("test_kb", knowledge_base_config)
            assert resource.rerank_limit == 100

    # Error Handling Edge Cases
    
    @pytest.mark.asyncio
    async def test_search_rerank_limit_logic(self, knowledge_base_resource, mock_vector_provider, mock_embedding_provider, mock_rerank_provider):
        """Test search limit logic with reranking."""
        knowledge_base_resource.vector_provider = mock_vector_provider
        knowledge_base_resource.set_embedding_provider(mock_embedding_provider, "text-embedding-ada-002")
        knowledge_base_resource.set_rerank_provider(mock_rerank_provider, "gpt-4o-mini")
        knowledge_base_resource.rerank_limit = 50
        
        # Mock empty results to focus on limit logic
        mock_vector_provider.search_similar.return_value = []
        
        # Test with limit less than rerank_limit
        await knowledge_base_resource.search(query="test", limit=10, use_reranking=True)
        
        # Should use rerank_limit (50) for initial search
        call_args = mock_vector_provider.search_similar.call_args
        filters = call_args[0][1]
        assert filters.limit == 50
        
        # Test with limit greater than rerank_limit
        await knowledge_base_resource.search(query="test", limit=100, use_reranking=True)
        
        # Should use the requested limit (100)
        call_args = mock_vector_provider.search_similar.call_args
        filters = call_args[0][1]
        assert filters.limit == 100