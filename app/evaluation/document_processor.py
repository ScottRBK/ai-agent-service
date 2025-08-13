"""Document processor for evaluation framework - extends production DocumentLoader"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.core.resources.document_loader import DocumentLoader
from app.core.resources.chunking.base import ChunkingStrategy
from app.core.resources.chunking.simple import SimpleChunkingStrategy
from app.models.resources.knowledge_base import DocumentType
from app.config.settings import settings


class DocumentProcessor(DocumentLoader):
    """Process documents for evaluation"""
    
    @staticmethod
    def load_document(filepath: str) -> str:
        """Load document content based on file type
        
        Evaluation-specific wrapper that handles EVALUATION_INPUT_DIR resolution
        before delegating to the parent DocumentLoader.
        
        Args:
            filepath: Path to the document file (absolute or relative to EVALUATION_INPUT_DIR)
            
        Returns:
            Document content as string
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(filepath)
        
        # Evaluation-specific path resolution
        if not path.is_absolute():
            input_dir = Path(settings.EVALUATION_INPUT_DIR)
            path = input_dir / filepath
        
        # Delegate to parent's load_file method
        return DocumentLoader.load_file(path)
    
    @staticmethod
    def parse_frontmatter(content: str) -> Tuple[Dict, str]:
        """Extract YAML frontmatter from document if present
        
        Evaluation-specific wrapper that maintains backward compatibility
        while delegating to the parent's parse_metadata method.
        
        Args:
            content: Document content
            
        Returns:
            Tuple of (metadata_dict, content_without_frontmatter)
        """
        # Use parent's parse_metadata with TEXT type for generic frontmatter parsing
        return DocumentLoader.parse_metadata(content, DocumentType.TEXT)
    
    @staticmethod
    def detect_type(filepath: str) -> DocumentType:
        """Detect document type from file extension
        
        Evaluation-specific wrapper that converts string path to Path object
        before delegating to parent's detect_document_type.
        
        Args:
            filepath: Path to the document (string)
            
        Returns:
            DocumentType enum value
        """
        path = Path(filepath)
        return DocumentLoader.detect_document_type(path)
    
    @staticmethod
    async def chunk_document(
        content: str,
        doc_type: DocumentType,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Chunk document using existing strategies
        
        Args:
            content: Document content
            doc_type: Type of document
            chunk_size: Maximum size of chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        #TODO: Implement different strategies based on doc_type
        # For now, use SimpleChunkingStrategy for all types
        # Future: Select strategy based on doc_type
        strategy = SimpleChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = strategy.chunk(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks
    
    @staticmethod
    def create_contexts_from_chunks(
        chunks: List[str],
        context_size: int = 3,
        max_contexts: int = 5
    ) -> List[List[str]]:
        """Group chunks into contexts for synthesis
        
        Args:
            chunks: List of document chunks
            context_size: Number of chunks per context
            max_contexts: Maximum number of contexts to create
            
        Returns:
            List of contexts (each context is list of chunks)
        """
        if not chunks:
            return []
        
        contexts = []
        stride = max(1, context_size // 2)  
        
        for i in range(0, len(chunks), stride):
            if len(contexts) >= max_contexts:
                break
            
            # Get context_size chunks starting from position i
            context_chunks = chunks[i:i + context_size]
            
            # Only add if we have at least one chunk
            if context_chunks:
                contexts.append(context_chunks)
        
        return contexts