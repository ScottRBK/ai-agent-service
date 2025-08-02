import re
from typing import List
from .base import ChunkingStrategy
from app.models.resources.knowledge_base import DocumentType

class SimpleChunkingStrategy(ChunkingStrategy):
    """Simple sentence-based chunking strategy."""
    
    def supports_document_type(self, doc_type: DocumentType) -> bool:
        """This strategy supports all document types."""
        return True
    
    def chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into chunks with overlap using sentence boundaries."""
        
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-chunk_overlap:]
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks