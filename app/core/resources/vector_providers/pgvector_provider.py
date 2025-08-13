
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import Vector, HALFVEC
from sqlalchemy import Index

from .base import VectorStoreProvider
from app.models.resources.knowledge_base import Document, DocumentChunk, SearchResult, SearchFilters, DocumentType
from app.utils.logging import logger

Base = declarative_base()

# Note: With JSONB columns, metadata is automatically parsed by PostgreSQL

class DocumentTable(Base):
    __tablename__ = 'knowledge_documents'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  
    user_id = Column(String(100), nullable=False, index=True)
    namespace_type = Column(String(50), nullable=False, index=True)
    embedding_model = Column(String(100), nullable=False, index=True)
    namespace_qualifier = Column(String(255), nullable=True)
    doc_type = Column(String(50), nullable=False)
    source = Column(String(500))
    title = Column(String(255))
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSONB)  # JSON stored as JSONB for efficient queries
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    chunks = relationship("ChunkTable", back_populates="document", cascade="all, delete-orphan")

class ChunkTable(Base):
    __tablename__ = 'knowledge_chunks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4())) 
    document_id = Column(String(36), ForeignKey('knowledge_documents.id'), nullable=False)
    user_id = Column(String(100), nullable=False, index=True)
    namespace_type = Column(String(50), nullable=False, index=True)
    embedding_model = Column(String(100), nullable=False, index=True)
    namespace_qualifier = Column(String(255), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(HALFVEC(2560), nullable=False)
    chunk_metadata = Column(JSONB)  # JSON stored as JSONB for efficient queries
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    document = relationship("DocumentTable", back_populates="chunks")

hnsw_index = Index(
    'ix_chunks_embedding_hnsw',
    ChunkTable.embedding,
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'halfvec_cosine_ops'}
)

# Composite index for namespace columns
namespace_composite_index = Index(
    'ix_documents_namespace_composite',
    DocumentTable.user_id,
    DocumentTable.namespace_type,
    DocumentTable.embedding_model
)

chunk_namespace_composite_index = Index(
    'ix_chunks_namespace_composite',
    ChunkTable.user_id,
    ChunkTable.namespace_type,
    ChunkTable.embedding_model
)

# Temporal index for conversation queries
temporal_index = Index(
    'ix_documents_created_at',
    DocumentTable.created_at.desc()
)

# Composite index for namespace and doc_type
namespace_doctype_index = Index(
    'ix_documents_namespace_doctype',
    DocumentTable.user_id,
    DocumentTable.namespace_type,
    DocumentTable.doc_type
)

# GIN index for metadata searching
metadata_gin_index = Index(
    'ix_documents_metadata_gin',
    DocumentTable.doc_metadata,
    postgresql_using='gin'
)


class PGVectorProvider(VectorStoreProvider):
    """PostgresSQL with pgvector implementation"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.engine = None
        self.SessionLocal = None
        self.connection_string = config.get('connection_string')

        if not self.connection_string:
            raise ValueError("PostgreSQL connection string is required")
        
    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables."""
        try:
            self.engine = create_engine(self.connection_string)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Register pgvector types
            from sqlalchemy import event
            from pgvector.psycopg import register_vector
            
            @event.listens_for(self.engine, "connect")
            def connect(dbapi_connection, connection_record):
                register_vector(dbapi_connection)
            
            # Create tables and indexes
            Base.metadata.create_all(bind=self.engine)
            self.initialized = True
            logger.info("PGVector provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PGVector provider: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup PostgreSQL connection."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("PGVector provider cleaned up")
        except Exception as e:
            logger.error(f"Error during PGVector cleanup: {e}")

    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        try:
            if not self.engine:
                return False
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
            
        except Exception as e:
            logger.error(f"PGVector health check failed: {e}")
            return False
        
    def _get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise RuntimeError("Provider not initialized")
        return self.SessionLocal()
    
    async def store_document(self, document: Document) -> str:
        """Store a document and return its ID."""
        try:
            session = self._get_session()
            
            # Document ID is always provided by client
            db_doc = DocumentTable(
                id=document.id,
                user_id=document.user_id,
                namespace_type=document.namespace_type,
                embedding_model=document.embedding_model,
                namespace_qualifier=document.namespace_qualifier,
                doc_type=document.doc_type.value,
                source=document.source,
                title=document.title,
                content=document.content,
                doc_metadata=document.metadata if document.metadata else None
            )
            
            session.add(db_doc)
            session.commit()
            return document.id
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to store document: {e}")
            raise
        finally:
            session.close()

    async def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Store document chunks with embeddings."""
        try:
            session = self._get_session()
            chunk_ids = []
            
            for chunk in chunks:
                # Chunk ID is always provided by client
                db_chunk = ChunkTable(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    user_id=chunk.user_id,
                    namespace_type=chunk.namespace_type,
                    embedding_model=chunk.embedding_model,
                    namespace_qualifier=chunk.namespace_qualifier,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding=chunk.embedding,
                    chunk_metadata=chunk.metadata if chunk.metadata else None
                )
                session.add(db_chunk)
                chunk_ids.append(chunk.id)
            
            session.commit()
            return chunk_ids
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to store chunks: {e}")
            raise
        finally:
            session.close()

    async def search_similar(self, 
                           query_embedding: List[float], 
                           filters: SearchFilters) -> List[SearchResult]:
        """Search for similar chunks using vector similarity."""
        try:
            session = self._get_session()
            
            query = session.query(ChunkTable, DocumentTable).join(
                DocumentTable, ChunkTable.document_id == DocumentTable.id
            )
            
            # Apply structured namespace filters
            if filters.user_id:
                query = query.filter(ChunkTable.user_id == filters.user_id)
            if filters.namespace_types:
                query = query.filter(ChunkTable.namespace_type.in_(filters.namespace_types))
            if filters.embedding_model:
                query = query.filter(ChunkTable.embedding_model == filters.embedding_model)
            
            # Apply document type filter
            if filters.doc_types:
                doc_type_values = [dt.value for dt in filters.doc_types]
                query = query.filter(DocumentTable.doc_type.in_(doc_type_values))
            
            # Note: All embeddings are now fixed at 4096 dimensions
            
            # Order by cosine similarity and limit
            query = query.order_by(
                ChunkTable.embedding.cosine_distance(query_embedding)
            ).limit(filters.limit)
            
            results = []
            for chunk_row, doc_row in query.all():
                # Calculate similarity score (1 - cosine_distance)
                distance_query = session.query(ChunkTable.embedding.cosine_distance(query_embedding)
                    ).filter(ChunkTable.id == chunk_row.id)
                distance = distance_query.scalar()
                score = 1.0 - distance
                                    
                # Convert to Pydantic models
                chunk = DocumentChunk(
                    id=chunk_row.id,
                    document_id=chunk_row.document_id,
                    user_id=chunk_row.user_id,
                    namespace_type=chunk_row.namespace_type,
                    embedding_model=chunk_row.embedding_model,
                    namespace_qualifier=chunk_row.namespace_qualifier,
                    chunk_index=chunk_row.chunk_index,
                    content=chunk_row.content,
                    embedding=chunk_row.embedding.to_list(),
                    metadata=chunk_row.chunk_metadata,
                    created_at=chunk_row.created_at
                )
                
                # Handle DocumentType conversion safely
                try:
                    doc_type = DocumentType(doc_row.doc_type)
                except (ValueError, TypeError):
                    doc_type = DocumentType.TEXT
                
                document = Document(
                    id=doc_row.id,
                    user_id=doc_row.user_id,
                    namespace_type=doc_row.namespace_type,
                    embedding_model=doc_row.embedding_model,
                    namespace_qualifier=doc_row.namespace_qualifier,
                    doc_type=doc_type,
                    source=doc_row.source,
                    title=doc_row.title,
                    content=doc_row.content,
                    metadata=doc_row.doc_metadata,
                    created_at=doc_row.created_at
                )
                
                results.append(SearchResult(chunk=chunk, score=score, document=document))
            
            return results
            
        except SQLAlchemyError as e:
            logger.error(f"Search failed: {e}")
            raise
        finally:
            session.close()

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        try:
            session = self._get_session()
            
            doc_row = session.query(DocumentTable).filter(
                DocumentTable.id == document_id
            ).first()
            
            if not doc_row:
                return None
            
            # Handle DocumentType conversion safely
            try:
                doc_type = DocumentType(doc_row.doc_type)
            except (ValueError, TypeError):
                # If doc_type is not a valid enum value, default to a safe value
                doc_type = DocumentType.TEXT
            
            # Parse metadata safely
            metadata = doc_row.doc_metadata
            
            return Document(
                id=doc_row.id,
                user_id=doc_row.user_id,
                namespace_type=doc_row.namespace_type,
                embedding_model=doc_row.embedding_model,
                namespace_qualifier=doc_row.namespace_qualifier,
                doc_type=doc_type,
                source=doc_row.source,
                title=doc_row.title,
                content=doc_row.content,
                metadata=metadata,
                created_at=doc_row.created_at
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get document: {e}")
            raise
        finally:
            session.close()

    async def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        try:
            session = self._get_session()
            
            chunk_rows = session.query(ChunkTable).filter(
                ChunkTable.document_id == document_id
            ).order_by(ChunkTable.chunk_index).all()
            
            chunks = []
            for chunk_row in chunk_rows:
                chunks.append(DocumentChunk(
                    id=chunk_row.id,
                    document_id=chunk_row.document_id,
                    user_id=chunk_row.user_id,
                    namespace_type=chunk_row.namespace_type,
                    embedding_model=chunk_row.embedding_model,
                    namespace_qualifier=chunk_row.namespace_qualifier,
                    chunk_index=chunk_row.chunk_index,
                    content=chunk_row.content,
                    embedding=chunk_row.embedding.to_list(),
                    metadata=chunk_row.chunk_metadata,
                    created_at=chunk_row.created_at
                ))
            
            return chunks
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get chunks: {e}")
            raise
        finally:
            session.close()

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            session = self._get_session()
            
            doc = session.query(DocumentTable).filter(
                DocumentTable.id == document_id
            ).first()
            
            if doc:
                session.delete(doc)  # Cascades to chunks
                session.commit()
                return True
            return False
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to delete document: {e}")
            raise
        finally:
            session.close()

    async def delete_all_documents_for_user(self, user_id: str) -> bool:
        """Delete all documents for a user."""
        try:
            session = self._get_session()
            
            logger.info(f"PGVectorProvider deleting all documents for user {user_id}")
            # First delete all chunks for the user
            session.query(ChunkTable).filter(
                ChunkTable.user_id == user_id
            ).delete()
            
            # Then delete all documents for the user
            session.query(DocumentTable).filter(
                DocumentTable.user_id == user_id
            ).delete()
            
            session.commit()
            logger.info(f"All documents deleted for user {user_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Failed to delete all documents for user: {e}")
            raise
        finally:
            session.close()
    
    async def list_documents(self, user_id: str, namespace_type: str, embedding_model: str) -> List[Document]:
        """List documents for a user in a specific namespace type and embedding model."""
        try:
            session = self._get_session()
            
            doc_rows = session.query(DocumentTable).filter(
                DocumentTable.user_id == user_id,
                DocumentTable.namespace_type == namespace_type,
                DocumentTable.embedding_model == embedding_model
            ).order_by(DocumentTable.created_at.desc()).all()
            
            documents = []
            for doc_row in doc_rows:
                # Handle DocumentType conversion safely
                try:
                    doc_type = DocumentType(doc_row.doc_type)
                except (ValueError, TypeError):
                    doc_type = DocumentType.TEXT
                
                documents.append(Document(
                    id=doc_row.id,
                    user_id=doc_row.user_id,
                    namespace_type=doc_row.namespace_type,
                    embedding_model=doc_row.embedding_model,
                    namespace_qualifier=doc_row.namespace_qualifier,
                    doc_type=doc_type,
                    source=doc_row.source,
                    title=doc_row.title,
                    content=doc_row.content,
                    metadata=doc_row.doc_metadata,
                    created_at=doc_row.created_at
                ))
            
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to list documents: {e}")
            raise
        finally:
            session.close()