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
            if not sentence: continue
            
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Take last chunk_overlap characters for overlap
                overlap_text = current_chunk[-chunk_overlap:].lstrip() if chunk_overlap > 0 and len(current_chunk) > chunk_overlap else current_chunk.lstrip() if chunk_overlap > 0 else ""
                current_chunk = (overlap_text + " " if overlap_text else "") + sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks