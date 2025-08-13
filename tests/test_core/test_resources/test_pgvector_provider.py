"""
Tests for PGVectorProvider.

This module tests the PGVectorProvider class, ensuring it
properly handles vector storage operations and database interactions.
"""

import pytest
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from sqlalchemy.exc import SQLAlchemyError
from typing import List

from app.core.resources.vector_providers.pgvector_provider import PGVectorProvider, DocumentTable, ChunkTable
from app.models.resources.knowledge_base import Document, DocumentChunk, SearchResult, SearchFilters, DocumentType


@pytest.fixture
def mock_config():
    """Mock configuration for PGVectorProvider"""
    return {
        "connection_string": "postgresql://user:pass@localhost:5432/test_db"
    }


@pytest.fixture
def mock_invalid_config():
    """Mock invalid configuration for testing"""
    return {}


@pytest.fixture
def provider(mock_config):
    """Create PGVectorProvider instance with mock config"""
    return PGVectorProvider(mock_config)


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine"""
    return MagicMock()


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session"""
    session = MagicMock()
    session.query.return_value = session
    session.filter.return_value = session
    session.order_by.return_value = session
    session.limit.return_value = session
    session.all.return_value = []
    session.first.return_value = None
    session.scalar.return_value = 0.5
    return session


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return Document(
        id="test-doc-123",
        user_id="test_user",
        namespace_type="test_namespace",
        embedding_model="text-embedding-ada-002",
        namespace_qualifier=None,
        doc_type=DocumentType.TEXT,
        source="test_source.txt",
        title="Test Document",
        content="This is a test document content",
        metadata={"test": "metadata"}
    )


@pytest.fixture
def sample_document_chunk():
    """Sample document chunk for testing"""
    return DocumentChunk(
        id="test-chunk-123",
        document_id="test-doc-123",
        user_id="test_user",
        namespace_type="test_namespace",
        embedding_model="text-embedding-ada-002",
        namespace_qualifier=None,
        chunk_index=0,
        content="This is a test chunk",
        embedding=[0.1] * 2560,  # 2560 dimensions
        metadata={"chunk_count": 1}
    )


@pytest.fixture
def sample_search_filters():
    """Sample search filters for testing"""
    return SearchFilters(
        user_id="test_user",
        namespace_types=["test_namespace"],
        doc_types=[DocumentType.TEXT],
        embedding_model="text-embedding-ada-002",
        limit=10
    )


# ============================================================================
# 1. Provider Initialization Tests
# ============================================================================

class TestProviderInitialization:
    """Test provider initialization and configuration"""
    
    def test_init_with_valid_config(self, mock_config):
        """Test initialization with valid configuration"""
        provider = PGVectorProvider(mock_config)
        assert provider.config == mock_config
        assert provider.connection_string == mock_config["connection_string"]
        assert provider.engine is None
        assert provider.SessionLocal is None
        assert not provider.initialized
    
    def test_init_with_invalid_config(self, mock_invalid_config):
        """Test initialization with invalid configuration"""
        with pytest.raises(ValueError, match="PostgreSQL connection string is required"):
            PGVectorProvider(mock_invalid_config)
    
    def test_init_with_none_connection_string(self):
        """Test initialization with None connection string"""
        config = {"connection_string": None}
        with pytest.raises(ValueError, match="PostgreSQL connection string is required"):
            PGVectorProvider(config)
    
    def test_init_with_empty_connection_string(self):
        """Test initialization with empty connection string"""
        config = {"connection_string": ""}
        with pytest.raises(ValueError, match="PostgreSQL connection string is required"):
            PGVectorProvider(config)

    @patch('app.core.resources.vector_providers.pgvector_provider.create_engine')
    @patch('app.core.resources.vector_providers.pgvector_provider.sessionmaker')
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_sessionmaker, mock_create_engine, provider, mock_engine):
        """Test successful initialization"""
        mock_create_engine.return_value = mock_engine
        mock_session_class = MagicMock()
        mock_sessionmaker.return_value = mock_session_class
        
        with patch('app.core.resources.vector_providers.pgvector_provider.Base.metadata.create_all'), \
             patch('sqlalchemy.event.listens_for'):
            await provider.initialize()
        
        assert provider.engine is mock_engine
        assert provider.SessionLocal is mock_session_class
        assert provider.initialized
        mock_create_engine.assert_called_once_with(provider.connection_string)
        mock_sessionmaker.assert_called_once_with(autocommit=False, autoflush=False, bind=mock_engine)

    @patch('app.core.resources.vector_providers.pgvector_provider.create_engine')
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_create_engine, provider):
        """Test initialization failure"""
        mock_create_engine.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            await provider.initialize()
        
        assert provider.engine is None
        assert provider.SessionLocal is None
        assert not provider.initialized


# ============================================================================
# 2. Cleanup and Health Check Tests
# ============================================================================

class TestCleanupAndHealthCheck:
    """Test cleanup and health check functionality"""
    
    @pytest.mark.asyncio
    async def test_cleanup_with_engine(self, provider, mock_engine):
        """Test cleanup when engine exists"""
        provider.engine = mock_engine
        
        await provider.cleanup()
        
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_without_engine(self, provider):
        """Test cleanup when no engine exists"""
        provider.engine = None
        
        # Should not raise exception
        await provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_cleanup_with_engine_error(self, provider, mock_engine):
        """Test cleanup when engine disposal fails"""
        provider.engine = mock_engine
        mock_engine.dispose.side_effect = Exception("Disposal failed")
        
        # Should not raise exception, just log error
        await provider.cleanup()
        
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_no_engine(self, provider):
        """Test health check when no engine exists"""
        provider.engine = None
        
        result = await provider.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, provider, mock_engine):
        """Test successful health check"""
        provider.engine = mock_engine
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        
        result = await provider.health_check()
        
        assert result is True
        mock_connection.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider, mock_engine):
        """Test health check failure"""
        provider.engine = mock_engine
        mock_engine.connect.side_effect = Exception("Connection failed")
        
        result = await provider.health_check()
        
        assert result is False


# ============================================================================
# 3. Session Management Tests
# ============================================================================

class TestSessionManagement:
    """Test database session management"""
    
    def test_get_session_success(self, provider):
        """Test successful session creation"""
        mock_session_class = MagicMock()
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        provider.SessionLocal = mock_session_class
        
        result = provider._get_session()
        
        assert result is mock_session
        mock_session_class.assert_called_once()
    
    def test_get_session_not_initialized(self, provider):
        """Test session creation when not initialized"""
        provider.SessionLocal = None
        
        with pytest.raises(RuntimeError, match="Provider not initialized"):
            provider._get_session()


# ============================================================================
# 4. Document Storage Tests
# ============================================================================

class TestDocumentStorage:
    """Test document storage functionality"""
    
    @pytest.mark.asyncio
    async def test_store_document_success(self, provider, mock_session, sample_document):
        """Test successful document storage"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        result = await provider.store_document(sample_document)
        
        assert result == sample_document.id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_document_with_none_metadata(self, provider, mock_session):
        """Test document storage with None metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        doc = Document(
            id="test-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            doc_type=DocumentType.TEXT,
            content="Test content",
            metadata=None
        )
        
        result = await provider.store_document(doc)
        
        assert result == doc.id
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_document_sqlalchemy_error(self, provider, mock_session, sample_document):
        """Test document storage with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.store_document(sample_document)
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_document_with_complex_metadata(self, provider, mock_session):
        """Test document storage with complex metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        complex_metadata = {
            "nested": {"key": "value"},
            "array": [1, 2, 3],
            "boolean": True,
            "number": 42.5
        }
        doc = Document(
            id="test-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            doc_type=DocumentType.JSON,
            content="Test content",
            metadata=complex_metadata
        )
        
        result = await provider.store_document(doc)
        
        assert result == doc.id
        mock_session.add.assert_called_once()
        # Verify metadata was JSON serialized
        call_args = mock_session.add.call_args[0][0]
        assert call_args.doc_metadata == complex_metadata


# ============================================================================
# 5. Chunk Storage Tests
# ============================================================================

class TestChunkStorage:
    """Test chunk storage functionality"""
    
    @pytest.mark.asyncio
    async def test_store_chunks_single_chunk(self, provider, mock_session, sample_document_chunk):
        """Test storing a single chunk"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        chunks = [sample_document_chunk]
        
        result = await provider.store_chunks(chunks)
        
        assert result == [sample_document_chunk.id]
        assert mock_session.add.call_count == 1
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_chunks_multiple_chunks(self, provider, mock_session):
        """Test storing multiple chunks"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        chunks = []
        expected_ids = []
        
        for i in range(3):
            chunk_id = f"chunk-{i}"
            chunk = DocumentChunk(
                id=chunk_id,
                document_id="test-doc-123",
                user_id="test_user",
                namespace_type="test_namespace",
                embedding_model="text-embedding-ada-002",
                namespace_qualifier=None,
                chunk_index=i,
                content=f"Chunk {i} content",
                embedding=[0.1 * (i + 1)] * 2560,
                metadata={"chunk_number": i}
            )
            chunks.append(chunk)
            expected_ids.append(chunk_id)
        
        result = await provider.store_chunks(chunks)
        
        assert result == expected_ids
        assert mock_session.add.call_count == 3
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_chunks_empty_list(self, provider, mock_session):
        """Test storing empty chunk list"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        chunks = []
        
        result = await provider.store_chunks(chunks)
        
        assert result == []
        mock_session.add.assert_not_called()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_chunks_with_none_metadata(self, provider, mock_session):
        """Test storing chunk with None metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        chunk = DocumentChunk(
            id="chunk-123",
            document_id="test-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            chunk_index=0,
            content="Test content",
            embedding=[0.1] * 2560,
            metadata=None
        )
        
        result = await provider.store_chunks([chunk])
        
        assert result == [chunk.id]
        mock_session.add.assert_called_once()
        # Verify metadata handling
        call_args = mock_session.add.call_args[0][0]
        assert call_args.chunk_metadata is None
    
    @pytest.mark.asyncio
    async def test_store_chunks_sqlalchemy_error(self, provider, mock_session, sample_document_chunk):
        """Test chunk storage with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.store_chunks([sample_document_chunk])
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_chunks_embedding_dimensions(self, provider, mock_session):
        """Test chunk storage with correct embedding dimensions"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Test with exactly 2560 dimensions
        chunk = DocumentChunk(
            id="chunk-123",
            document_id="test-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            chunk_index=0,
            content="Test content",
            embedding=[0.1] * 2560,  # Exactly 2560 dimensions
            metadata={"test": "data"}
        )
        
        result = await provider.store_chunks([chunk])
        
        assert result == [chunk.id]
        call_args = mock_session.add.call_args[0][0]
        assert len(call_args.embedding) == 2560


# ============================================================================
# 6. Document Retrieval Tests
# ============================================================================

class TestDocumentRetrieval:
    """Test document retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_get_document_found(self, provider, mock_session):
        """Test retrieving existing document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock database row
        mock_row = MagicMock()
        mock_row.id = "test-doc-123"
        mock_row.user_id = "test_user"
        mock_row.namespace_type = "test_namespace"
        mock_row.embedding_model = "text-embedding-ada-002"
        mock_row.namespace_qualifier = None
        mock_row.doc_type = "text"
        mock_row.source = "test.txt"
        mock_row.title = "Test Document"
        mock_row.content = "Test content"
        mock_row.doc_metadata = {"test": "metadata"}
        mock_row.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_row
        
        result = await provider.get_document("test-doc-123")
        
        assert result is not None
        assert result.id == "test-doc-123"
        assert result.user_id == "test_user"
        assert result.namespace_type == "test_namespace"
        assert result.embedding_model == "text-embedding-ada-002"
        assert result.doc_type == DocumentType.TEXT
        assert result.source == "test.txt"
        assert result.title == "Test Document"
        assert result.content == "Test content"
        assert result.metadata == {"test": "metadata"}
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, provider, mock_session):
        """Test retrieving non-existent document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        result = await provider.get_document("non-existent-id")
        
        assert result is None
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_document_with_none_metadata(self, provider, mock_session):
        """Test retrieving document with None metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        mock_row = MagicMock()
        mock_row.id = "test-doc-123"
        mock_row.user_id = "test_user"
        mock_row.namespace_type = "test_namespace"
        mock_row.embedding_model = "text-embedding-ada-002"
        mock_row.namespace_qualifier = None
        mock_row.doc_type = "text"
        mock_row.source = None
        mock_row.title = None
        mock_row.content = "Test content"
        mock_row.doc_metadata = None
        mock_row.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_row
        
        result = await provider.get_document("test-doc-123")
        
        assert result is not None
        assert result.metadata is None
        assert result.source is None
        assert result.title is None
    
    @pytest.mark.asyncio
    async def test_get_document_sqlalchemy_error(self, provider, mock_session):
        """Test document retrieval with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.get_document("test-doc-123")
        
        mock_session.close.assert_called_once()


# ============================================================================
# 7. Chunk Retrieval Tests
# ============================================================================

class TestChunkRetrieval:
    """Test chunk retrieval functionality"""
    
    @pytest.mark.asyncio
    async def test_get_chunks_found(self, provider, mock_session):
        """Test retrieving chunks for existing document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock chunk rows
        mock_chunks = []
        for i in range(2):
            mock_chunk = MagicMock()
            mock_chunk.id = f"chunk-{i}"
            mock_chunk.document_id = "test-doc-123"
            mock_chunk.user_id = "test_user"
            mock_chunk.namespace_type = "test_namespace"
            mock_chunk.embedding_model = "text-embedding-ada-002"
            mock_chunk.namespace_qualifier = None
            mock_chunk.chunk_index = i
            mock_chunk.content = f"Chunk {i} content"
            mock_chunk.embedding.to_list.return_value = [0.1] * 2560
            mock_chunk.chunk_metadata = {"chunk_number": i}
            mock_chunk.created_at = datetime.now(timezone.utc)
            mock_chunks.append(mock_chunk)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_chunks
        
        result = await provider.get_chunks("test-doc-123")
        
        assert len(result) == 2
        assert result[0].id == "chunk-0"
        assert result[0].chunk_index == 0
        assert result[0].embedding == [0.1] * 2560
        assert result[1].id == "chunk-1"
        assert result[1].chunk_index == 1
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_chunks_empty_result(self, provider, mock_session):
        """Test retrieving chunks for document with no chunks"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        
        result = await provider.get_chunks("test-doc-123")
        
        assert result == []
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_chunks_with_none_metadata(self, provider, mock_session):
        """Test retrieving chunks with None metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.document_id = "test-doc-123"
        mock_chunk.user_id = "test_user"
        mock_chunk.namespace_type = "test_namespace"
        mock_chunk.embedding_model = "text-embedding-ada-002"
        mock_chunk.namespace_qualifier = None
        mock_chunk.chunk_index = 0
        mock_chunk.content = "Test content"
        mock_chunk.embedding.to_list.return_value = [0.1] * 2560
        mock_chunk.chunk_metadata = None
        mock_chunk.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk]
        
        result = await provider.get_chunks("test-doc-123")
        
        assert len(result) == 1
        assert result[0].metadata is None
    
    @pytest.mark.asyncio
    async def test_get_chunks_sqlalchemy_error(self, provider, mock_session):
        """Test chunk retrieval with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.get_chunks("test-doc-123")
        
        mock_session.close.assert_called_once()


# ============================================================================
# 8. Vector Search Tests
# ============================================================================

class TestVectorSearch:
    """Test vector similarity search functionality"""
    
    @pytest.mark.asyncio
    async def test_search_similar_basic(self, provider, mock_session, sample_search_filters):
        """Test basic vector similarity search"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock query results
        mock_chunk_row = MagicMock()
        mock_chunk_row.id = "chunk-123"
        mock_chunk_row.document_id = "doc-123"
        mock_chunk_row.user_id = "test_user"
        mock_chunk_row.namespace_type = "test_namespace"
        mock_chunk_row.embedding_model = "text-embedding-ada-002"
        mock_chunk_row.namespace_qualifier = None
        mock_chunk_row.chunk_index = 0
        mock_chunk_row.content = "Test chunk content"
        mock_chunk_row.embedding.to_list.return_value = [0.1] * 2560
        mock_chunk_row.chunk_metadata = {"test": "metadata"}
        mock_chunk_row.created_at = datetime.now(timezone.utc)
        
        mock_doc_row = MagicMock()
        mock_doc_row.id = "doc-123"
        mock_doc_row.user_id = "test_user"
        mock_doc_row.namespace_type = "test_namespace"
        mock_doc_row.embedding_model = "text-embedding-ada-002"
        mock_doc_row.namespace_qualifier = None
        mock_doc_row.doc_type = "text"
        mock_doc_row.source = "test.txt"
        mock_doc_row.title = "Test Document"
        mock_doc_row.content = "Test document content"
        mock_doc_row.doc_metadata = {"doc": "metadata"}
        mock_doc_row.created_at = datetime.now(timezone.utc)
        
        # Use a MagicMock that returns itself for chaining  
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [(mock_chunk_row, mock_doc_row)]
        mock_session.query.return_value = mock_query
        
        # Mock distance calculation
        mock_session.query.return_value.filter.return_value.scalar.return_value = 0.3
        
        query_embedding = [0.5] * 2560
        result = await provider.search_similar(query_embedding, sample_search_filters)
        
        assert len(result) == 1
        assert result[0].chunk.id == "chunk-123"
        assert result[0].document.id == "doc-123"
        assert result[0].score == 0.7  # 1.0 - 0.3
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_with_namespace_filter(self, provider, mock_session):
        """Test search with namespace filtering"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        filters = SearchFilters(
            user_id="test_user",
            namespace_types=["namespace1", "namespace2"],
            embedding_model="text-embedding-ada-002",
            limit=5
        )
        query_embedding = [0.5] * 2560
        
        await provider.search_similar(query_embedding, filters)
        
        # Verify namespace filter was applied
        mock_session.query.return_value.join.return_value.filter.assert_called()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_with_doc_type_filter(self, provider, mock_session):
        """Test search with document type filtering"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        filters = SearchFilters(
            doc_types=[DocumentType.TEXT, DocumentType.MARKDOWN],
            embedding_model="text-embedding-ada-002",
            limit=5
        )
        query_embedding = [0.5] * 2560
        
        await provider.search_similar(query_embedding, filters)
        
        # Verify document type filter was applied
        filter_calls = mock_session.query.return_value.join.return_value.filter.call_args_list
        assert len(filter_calls) >= 1
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_no_filters(self, provider, mock_session):
        """Test search without any filters"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        filters = SearchFilters(
            embedding_model="text-embedding-ada-002",
            limit=10
        )
        query_embedding = [0.5] * 2560
        
        await provider.search_similar(query_embedding, filters)
        
        # Verify no additional filters were applied
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_empty_results(self, provider, mock_session, sample_search_filters):
        """Test search with no matching results"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        query_embedding = [0.5] * 2560
        result = await provider.search_similar(query_embedding, sample_search_filters)
        
        assert result == []
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_with_limit(self, provider, mock_session):
        """Test search with custom limit"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Use a MagicMock that returns itself for chaining
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query
        
        filters = SearchFilters(
            embedding_model="text-embedding-ada-002",
            limit=50
        )
        query_embedding = [0.5] * 2560
        
        await provider.search_similar(query_embedding, filters)
        
        # Verify limit was applied
        mock_query.limit.assert_called_with(50)
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_sqlalchemy_error(self, provider, mock_session, sample_search_filters):
        """Test search with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        query_embedding = [0.5] * 2560
        
        with pytest.raises(SQLAlchemyError):
            await provider.search_similar(query_embedding, sample_search_filters)
        
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_embedding_dimensions(self, provider, mock_session, sample_search_filters):
        """Test search with various embedding dimensions"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        # Test with exactly 2560 dimensions
        query_embedding = [0.5] * 2560
        await provider.search_similar(query_embedding, sample_search_filters)
        
        mock_session.close.assert_called_once()


# ============================================================================
# 9. Document Deletion Tests
# ============================================================================

class TestDocumentDeletion:
    """Test document deletion functionality"""
    
    @pytest.mark.asyncio
    async def test_delete_document_found(self, provider, mock_session):
        """Test deleting existing document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        mock_doc = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc
        
        result = await provider.delete_document("test-doc-123")
        
        assert result is True
        mock_session.delete.assert_called_once_with(mock_doc)
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, provider, mock_session):
        """Test deleting non-existent document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        result = await provider.delete_document("non-existent-id")
        
        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_sqlalchemy_error(self, provider, mock_session):
        """Test document deletion with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        mock_doc = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc
        mock_session.delete.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.delete_document("test-doc-123")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


# ============================================================================
# 10. Document Listing Tests
# ============================================================================

class TestDocumentListing:
    """Test document listing functionality"""
    
    @pytest.mark.asyncio
    async def test_list_documents_found(self, provider, mock_session):
        """Test listing documents in namespace"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock document rows
        mock_docs = []
        for i in range(3):
            mock_doc = MagicMock()
            mock_doc.id = f"doc-{i}"
            mock_doc.user_id = "test_user"
            mock_doc.namespace_type = "test_namespace"
            mock_doc.embedding_model = "text-embedding-ada-002"
            mock_doc.namespace_qualifier = None
            mock_doc.doc_type = "text"
            mock_doc.source = f"file{i}.txt"
            mock_doc.title = f"Document {i}"
            mock_doc.content = f"Content {i}"
            mock_doc.doc_metadata = {"index": i}
            mock_doc.created_at = datetime.now(timezone.utc)
            mock_docs.append(mock_doc)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_docs
        
        result = await provider.list_documents("test_user", "test_namespace", "text-embedding-ada-002")
        
        assert len(result) == 3
        assert result[0].id == "doc-0"
        assert result[1].id == "doc-1"
        assert result[2].id == "doc-2"
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_documents_empty_namespace(self, provider, mock_session):
        """Test listing documents in empty namespace"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        
        result = await provider.list_documents("test_user", "empty_namespace", "text-embedding-ada-002")
        
        assert result == []
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_documents_with_none_metadata(self, provider, mock_session):
        """Test listing documents with None metadata"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        mock_doc = MagicMock()
        mock_doc.id = "doc-123"
        mock_doc.user_id = "test_user"
        mock_doc.namespace_type = "test_namespace"
        mock_doc.embedding_model = "text-embedding-ada-002"
        mock_doc.namespace_qualifier = None
        mock_doc.doc_type = "text"
        mock_doc.source = None
        mock_doc.title = None
        mock_doc.content = "Test content"
        mock_doc.doc_metadata = None
        mock_doc.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_doc]
        
        result = await provider.list_documents("test_user", "test_namespace", "text-embedding-ada-002")
        
        assert len(result) == 1
        assert result[0].metadata is None
        assert result[0].source is None
        assert result[0].title is None
    
    @pytest.mark.asyncio
    async def test_list_documents_sqlalchemy_error(self, provider, mock_session):
        """Test document listing with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError):
            await provider.list_documents("test_user", "test_namespace", "text-embedding-ada-002")
        
        mock_session.close.assert_called_once()


# ============================================================================
# 11. Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and comprehensive error handling"""
    
    @pytest.mark.asyncio
    async def test_complex_jsonb_metadata_document(self, provider, mock_session):
        """Test handling complex JSONB metadata in document"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        complex_metadata = {
            "nested": {"key": "value", "number": 123},
            "array": [1, 2, 3],
            "boolean": True,
            "null_value": None
        }
        
        mock_row = MagicMock()
        mock_row.id = "test-doc-123"
        mock_row.user_id = "test_user"
        mock_row.namespace_type = "test_namespace"
        mock_row.embedding_model = "text-embedding-ada-002"
        mock_row.namespace_qualifier = None
        mock_row.doc_type = "text"
        mock_row.source = "test.txt"
        mock_row.title = "Test Document"
        mock_row.content = "Test content"
        mock_row.doc_metadata = complex_metadata  # Complex JSONB data
        mock_row.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_row
        
        # Should handle complex JSONB metadata correctly
        result = await provider.get_document("test-doc-123")
        assert result is not None
        assert result.metadata == complex_metadata
    
    @pytest.mark.asyncio
    async def test_complex_jsonb_metadata_chunk(self, provider, mock_session):
        """Test handling complex JSONB metadata in chunk"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        complex_metadata = {
            "chunk_info": {"index": 0, "total": 3},
            "processing": ["nlp", "embedding"],
            "confidence": 0.95
        }
        
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-123"
        mock_chunk.document_id = "test-doc-123"
        mock_chunk.user_id = "test_user"
        mock_chunk.namespace_type = "test_namespace"
        mock_chunk.embedding_model = "text-embedding-ada-002"
        mock_chunk.namespace_qualifier = None
        mock_chunk.chunk_index = 0
        mock_chunk.content = "Test content"
        mock_chunk.embedding.to_list.return_value = [0.1] * 2560
        mock_chunk.chunk_metadata = complex_metadata  # Complex JSONB data
        mock_chunk.created_at = datetime.now(timezone.utc)
        
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk]
        
        # Should handle complex JSONB metadata correctly
        results = await provider.get_chunks("test-doc-123")
        assert len(results) == 1
        assert results[0].metadata == complex_metadata
    
    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, provider, mock_session):
        """Test handling Unicode content"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        unicode_doc = Document(
            id="unicode-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            doc_type=DocumentType.TEXT,
            content="Test with Unicode: 你好世界 🌍 émojis",
            metadata={"unicode": "测试数据"}
        )
        
        result = await provider.store_document(unicode_doc)
        
        assert result == unicode_doc.id
        mock_session.add.assert_called_once()
        call_args = mock_session.add.call_args[0][0]
        assert "你好世界" in call_args.content
        assert "🌍" in call_args.content
    
    @pytest.mark.asyncio
    async def test_large_embedding_vectors(self, provider, mock_session):
        """Test handling large embedding vectors"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Create chunk with exactly 2560 dimensions (max expected)
        large_embedding = [float(i) / 2560 for i in range(2560)]
        chunk = DocumentChunk(
            id="large-chunk-123",
            document_id="test-doc-123",
            user_id="test_user",
            namespace_type="test_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            chunk_index=0,
            content="Test content",
            embedding=large_embedding,
            metadata={"size": "large"}
        )
        
        result = await provider.store_chunks([chunk])
        
        assert result == [chunk.id]
        call_args = mock_session.add.call_args[0][0]
        assert len(call_args.embedding) == 2560
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, provider, mock_session):
        """Test handling concurrent database operations"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Simulate concurrent access by having multiple session instances
        import asyncio
        
        async def store_document(doc_id):
            doc = Document(
                id=doc_id,
                user_id="test_user",
                namespace_type="concurrent_namespace",
                embedding_model="text-embedding-ada-002",
                namespace_qualifier=None,
                doc_type=DocumentType.TEXT,
                content=f"Content for {doc_id}"
            )
            return await provider.store_document(doc)
        
        # Execute concurrent operations
        tasks = [store_document(f"doc-{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert mock_session.add.call_count == 3
        assert mock_session.commit.call_count == 3
    
    @pytest.mark.asyncio
    async def test_search_with_zero_vector(self, provider, mock_session):
        """Test search with zero embedding vector"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        zero_embedding = [0.0] * 2560
        filters = SearchFilters(
            embedding_model="text-embedding-ada-002",
            limit=10
        )
        
        result = await provider.search_similar(zero_embedding, filters)
        
        assert result == []
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_with_normalized_vector(self, provider, mock_session):
        """Test search with normalized embedding vector"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        mock_session.query.return_value.join.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        # Create normalized vector (L2 norm = 1)
        import math
        dim = 2560
        value = 1.0 / math.sqrt(dim)
        normalized_embedding = [value] * dim
        
        filters = SearchFilters(
            embedding_model="text-embedding-ada-002",
            limit=10
        )
        
        result = await provider.search_similar(normalized_embedding, filters)
        
        assert result == []
        mock_session.close.assert_called_once()


# ============================================================================
# 12. Performance and Stress Tests
# ============================================================================

class TestPerformanceAndStress:
    """Test performance scenarios and stress conditions"""
    
    @pytest.mark.asyncio
    async def test_large_batch_chunk_storage(self, provider, mock_session):
        """Test storing large batch of chunks"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Create 100 chunks
        chunks = []
        for i in range(100):
            chunk = DocumentChunk(
                id=f"batch-chunk-{i}",
                document_id="batch-doc-123",
                user_id="test_user",
                namespace_type="batch_namespace",
                embedding_model="text-embedding-ada-002",
                namespace_qualifier=None,
                chunk_index=i,
                content=f"Batch chunk {i} content",
                embedding=[0.01 * i] * 2560,
                metadata={"batch_index": i}
            )
            chunks.append(chunk)
        
        result = await provider.store_chunks(chunks)
        
        assert len(result) == 100
        assert mock_session.add.call_count == 100
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_large_document_content(self, provider, mock_session):
        """Test storing document with large content"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Create document with 10MB of content
        large_content = "A" * (10 * 1024 * 1024)  # 10MB
        doc = Document(
            id="large-doc-123",
            user_id="test_user",
            namespace_type="large_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            doc_type=DocumentType.TEXT,
            content=large_content,
            metadata={"size": "10MB"}
        )
        
        result = await provider.store_document(doc)
        
        assert result == doc.id
        call_args = mock_session.add.call_args[0][0]
        assert len(call_args.content) == 10 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_search_with_large_result_set(self, provider, mock_session, sample_search_filters):
        """Test search that would return large result set"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock large result set
        mock_results = []
        for i in range(1000):
            mock_chunk = MagicMock()
            mock_chunk.id = f"chunk-{i}"
            mock_chunk.document_id = f"doc-{i // 10}"
            mock_chunk.user_id = "test_user"
            mock_chunk.namespace_type = "large_namespace"
            mock_chunk.embedding_model = "text-embedding-ada-002"
            mock_chunk.namespace_qualifier = None
            mock_chunk.chunk_index = i % 10
            mock_chunk.content = f"Chunk {i} content"
            mock_chunk.embedding.to_list.return_value = [0.1] * 2560
            mock_chunk.chunk_metadata = {"index": i}
            mock_chunk.created_at = datetime.now(timezone.utc)
            
            mock_doc = MagicMock()
            mock_doc.id = f"doc-{i // 10}"
            mock_doc.user_id = "test_user"
            mock_doc.namespace_type = "large_namespace"
            mock_doc.embedding_model = "text-embedding-ada-002"
            mock_doc.namespace_qualifier = None
            mock_doc.doc_type = "text"
            mock_doc.source = f"file{i}.txt"
            mock_doc.title = f"Document {i}"
            mock_doc.content = f"Document {i} content"
            mock_doc.doc_metadata = {"doc_index": i // 10}
            mock_doc.created_at = datetime.now(timezone.utc)
            
            mock_results.append((mock_chunk, mock_doc))
        
        # Use a MagicMock that returns itself for chaining
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_results
        mock_session.query.return_value = mock_query
        
        # Also set up the distance query separately
        mock_session.query.return_value.filter.return_value.scalar.return_value = 0.5
        
        # Set high limit to test large result handling
        filters = SearchFilters(
            user_id="test_user",
            namespace_types=["large_namespace"],
            embedding_model="text-embedding-ada-002",
            limit=1000
        )
        query_embedding = [0.5] * 2560
        
        result = await provider.search_similar(query_embedding, filters)
        
        assert len(result) == 1000
        assert all(isinstance(r, SearchResult) for r in result)


# ============================================================================
# 13. Integration and Workflow Tests
# ============================================================================

class TestIntegrationAndWorkflow:
    """Test complete workflows and integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, provider, mock_session):
        """Test complete document storage and retrieval workflow"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Step 1: Store document
        doc = Document(
            id="workflow-doc-123",
            user_id="test_user",
            namespace_type="workflow_namespace",
            embedding_model="text-embedding-ada-002",
            namespace_qualifier=None,
            doc_type=DocumentType.MARKDOWN,
            source="workflow.md",
            title="Workflow Test",
            content="# Workflow Test\nThis is a test document.",
            metadata={"workflow": "test"}
        )
        
        store_result = await provider.store_document(doc)
        assert store_result == doc.id
        
        # Step 2: Store chunks
        chunks = [
            DocumentChunk(
                id="workflow-chunk-1",
                document_id=doc.id,
                user_id=doc.user_id,
                namespace_type=doc.namespace_type,
                embedding_model=doc.embedding_model,
                namespace_qualifier=doc.namespace_qualifier,
                chunk_index=0,
                content="# Workflow Test",
                embedding=[0.1] * 2560,
                metadata={"chunk_type": "header"}
            ),
            DocumentChunk(
                id="workflow-chunk-2",
                document_id=doc.id,
                user_id=doc.user_id,
                namespace_type=doc.namespace_type,
                embedding_model=doc.embedding_model,
                namespace_qualifier=doc.namespace_qualifier,
                chunk_index=1,
                content="This is a test document.",
                embedding=[0.2] * 2560,
                metadata={"chunk_type": "content"}
            )
        ]
        
        chunk_result = await provider.store_chunks(chunks)
        assert len(chunk_result) == 2
        
        # Verify total operations
        assert mock_session.add.call_count == 3  # 1 doc + 2 chunks
        assert mock_session.commit.call_count == 2  # 1 for doc, 1 for chunks
    
    @pytest.mark.asyncio
    async def test_search_and_retrieval_workflow(self, provider, mock_session):
        """Test search and detailed retrieval workflow"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock search results
        mock_chunk_row = MagicMock()
        mock_chunk_row.id = "found-chunk-123"
        mock_chunk_row.document_id = "found-doc-123"
        mock_chunk_row.user_id = "test_user"
        mock_chunk_row.namespace_type = "search_namespace"
        mock_chunk_row.embedding_model = "text-embedding-ada-002"
        mock_chunk_row.namespace_qualifier = None
        mock_chunk_row.chunk_index = 0
        mock_chunk_row.content = "Found content"
        mock_chunk_row.embedding.to_list.return_value = [0.3] * 2560
        mock_chunk_row.chunk_metadata = {"found": True}
        mock_chunk_row.created_at = datetime.now(timezone.utc)
        
        mock_doc_row = MagicMock()
        mock_doc_row.id = "found-doc-123"
        mock_doc_row.user_id = "test_user"
        mock_doc_row.namespace_type = "search_namespace"
        mock_doc_row.embedding_model = "text-embedding-ada-002"
        mock_doc_row.namespace_qualifier = None
        mock_doc_row.doc_type = "text"
        mock_doc_row.source = "found.txt"
        mock_doc_row.title = "Found Document"
        mock_doc_row.content = "Full found document content"
        mock_doc_row.doc_metadata = {"searchable": True}
        mock_doc_row.created_at = datetime.now(timezone.utc)
        
        # Use a MagicMock that returns itself for chaining
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [(mock_chunk_row, mock_doc_row)]
        mock_session.query.return_value = mock_query
        
        # Also set up the distance query separately
        mock_session.query.return_value.filter.return_value.scalar.return_value = 0.2
        
        # Step 1: Search
        filters = SearchFilters(
            user_id="search_user",
            namespace_types=["search_namespace"],
            embedding_model="text-embedding-ada-002",
            limit=10
        )
        query_embedding = [0.5] * 2560
        search_results = await provider.search_similar(query_embedding, filters)
        
        assert len(search_results) == 1
        found_result = search_results[0]
        assert found_result.score == 0.8  # 1.0 - 0.2
        
        # Step 2: Get full document
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc_row
        full_doc = await provider.get_document(found_result.document.id)
        
        assert full_doc is not None
        assert full_doc.id == "found-doc-123"
        
        # Step 3: Get all chunks for document
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_chunk_row]
        doc_chunks = await provider.get_chunks(found_result.document.id)
        
        assert len(doc_chunks) == 1
        assert doc_chunks[0].id == "found-chunk-123"
    
    @pytest.mark.asyncio
    async def test_cleanup_workflow(self, provider, mock_session):
        """Test document cleanup workflow"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock existing document
        mock_doc = MagicMock()
        mock_doc.id = "cleanup-doc-123"
        mock_doc.user_id = "test_user"
        mock_doc.namespace_type = "test_namespace"
        mock_doc.embedding_model = "text-embedding-ada-002"
        mock_doc.namespace_qualifier = None
        mock_doc.doc_type = "text"
        mock_doc.source = "cleanup.txt"
        mock_doc.title = "Cleanup Document"
        mock_doc.content = "Document to be cleaned up"
        mock_doc.doc_metadata = {"cleanup": True}
        mock_doc.created_at = datetime.now(timezone.utc)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc
        
        # Step 1: Verify document exists
        existing_doc = await provider.get_document("cleanup-doc-123")
        # Reset mock for delete operation
        mock_session.reset_mock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_doc
        
        # Step 2: Delete document (cascades to chunks)
        delete_result = await provider.delete_document("cleanup-doc-123")
        
        assert delete_result is True
        mock_session.delete.assert_called_once_with(mock_doc)
        mock_session.commit.assert_called_once()


# ============================================================================
# 14. Bulk User Deletion Tests
# ============================================================================

class TestBulkUserDeletion:
    """Test bulk deletion of all documents for a user"""
    
    @pytest.mark.asyncio
    async def test_delete_all_documents_for_user_success(self, provider, mock_session):
        """Test successful deletion of all documents for a user"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock that chunks and documents exist and are deleted
        mock_chunk_query = MagicMock()
        mock_doc_query = MagicMock()
        
        # Set up session.query to return different query objects for different tables
        query_side_effects = [mock_chunk_query, mock_doc_query]
        mock_session.query.side_effect = query_side_effects
        
        # Configure chunk query chain
        mock_chunk_query.filter.return_value = mock_chunk_query
        mock_chunk_query.delete.return_value = 5  # 5 chunks deleted
        
        # Configure document query chain  
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.delete.return_value = 2  # 2 documents deleted
        
        result = await provider.delete_all_documents_for_user("test_user")
        
        assert result is True
        
        # Verify chunk deletion was called first
        assert mock_session.query.call_args_list[0][0][0] == ChunkTable
        mock_chunk_query.filter.assert_called_once()
        mock_chunk_query.delete.assert_called_once()
        
        # Verify document deletion was called second
        assert mock_session.query.call_args_list[1][0][0] == DocumentTable
        mock_doc_query.filter.assert_called_once()
        mock_doc_query.delete.assert_called_once()
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_all_documents_for_user_isolates_users(self, provider, mock_session):
        """Test that deletion only affects specified user's documents"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock query objects
        mock_chunk_query = MagicMock()
        mock_doc_query = MagicMock()
        
        query_side_effects = [mock_chunk_query, mock_doc_query]
        mock_session.query.side_effect = query_side_effects
        
        # Configure query chains
        mock_chunk_query.filter.return_value = mock_chunk_query
        mock_chunk_query.delete.return_value = 3
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.delete.return_value = 1
        
        await provider.delete_all_documents_for_user("specific_user")
        
        # Verify that the filter was called with the correct user_id for chunks
        chunk_filter_call = mock_chunk_query.filter.call_args[0][0]
        assert hasattr(chunk_filter_call, 'left')  # SQLAlchemy comparison object
        assert chunk_filter_call.left.name == 'user_id'
        
        # Verify that the filter was called with the correct user_id for documents
        doc_filter_call = mock_doc_query.filter.call_args[0][0]
        assert hasattr(doc_filter_call, 'left')  # SQLAlchemy comparison object  
        assert doc_filter_call.left.name == 'user_id'
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_all_documents_for_user_no_documents(self, provider, mock_session):
        """Test deletion when no documents exist for the user"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock query objects
        mock_chunk_query = MagicMock()
        mock_doc_query = MagicMock()
        
        query_side_effects = [mock_chunk_query, mock_doc_query]
        mock_session.query.side_effect = query_side_effects
        
        # Configure query chains to return 0 deletions
        mock_chunk_query.filter.return_value = mock_chunk_query
        mock_chunk_query.delete.return_value = 0  # No chunks deleted
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.delete.return_value = 0  # No documents deleted
        
        result = await provider.delete_all_documents_for_user("nonexistent_user")
        
        assert result is True  # Method should still return True (successful operation)
        mock_chunk_query.delete.assert_called_once()
        mock_doc_query.delete.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_all_documents_for_user_sqlalchemy_error(self, provider, mock_session):
        """Test deletion with SQLAlchemy error"""
        provider.SessionLocal = MagicMock(return_value=mock_session)
        
        # Mock query to raise error on chunk deletion
        mock_chunk_query = MagicMock()
        mock_session.query.return_value = mock_chunk_query
        mock_chunk_query.filter.return_value = mock_chunk_query
        mock_chunk_query.delete.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(SQLAlchemyError, match="Database error"):
            await provider.delete_all_documents_for_user("test_user")
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()