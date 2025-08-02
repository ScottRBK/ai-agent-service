from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.models.resources.knowledge_base import DocumentType

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, **config):
        """Initialize chunking strategy with configuration."""
        self.config = config
    
    @abstractmethod
    def chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of words/characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def supports_document_type(self, doc_type: DocumentType) -> bool:
        """Check if this strategy supports the given document type."""
        pass
    
    def get_strategy_name(self) -> str:
        """Get the name of this chunking strategy."""
        return self.__class__.__name__.replace("ChunkingStrategy", "").lower()