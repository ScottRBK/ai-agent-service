from typing import List
from .base import ChunkingStrategy
from app.models.resources.knowledge_base import DocumentType

class TokenAwareChunkingStrategy(ChunkingStrategy):
    """Token-aware chunking strategy that counts actual tokens."""
    
    def supports_document_type(self, doc_type: DocumentType) -> bool:
        """Supports all text-based document types."""
        return doc_type in [DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.HTML, DocumentType.JSON]
    
    def chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text using precise token counting."""
        # TODO: Implement token-aware chunking
        # - Use tiktoken to count actual tokens
        # - Respect sentence boundaries when possible
        # - Ensure chunks don't exceed token limits
        raise NotImplementedError("Token-aware chunking not yet implemented")