from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.resources.knowledge_base import Document, DocumentChunk, SearchResult, SearchFilters

class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup connections and resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass
    
    @abstractmethod
    async def store_document(self, document: Document) -> str:
        """Store a document and return its ID."""
        pass
    
    @abstractmethod
    async def store_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """Store document chunks with embeddings."""
        pass
    
    @abstractmethod
    async def search_similar(self, 
                           query_embedding: List[float], 
                           filters: SearchFilters) -> List[SearchResult]:
        """Search for similar chunks using vector similarity."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    async def get_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        pass
    
    @abstractmethod
    async def list_documents(self, user_id: str, namespace_type: str, embedding_model: str) -> List[Document]:
        """List documents for a user in a specific namespace type and embedding model."""
        pass