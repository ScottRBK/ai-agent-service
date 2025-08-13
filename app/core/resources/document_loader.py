"""Production document loader for knowledge base and resource operations"""

import json
import yaml
import csv
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from app.models.resources.knowledge_base import DocumentType


class DocumentLoader:
    """Production document loader for knowledge base operations
    
    Provides core document loading capabilities for production use,
    including file loading, type detection, and metadata extraction.
    """
    
    @staticmethod
    def load_file(filepath: Path) -> str:
        """Load file content with format detection
        
        Args:
            filepath: Path object to the document file
            
        Returns:
            Document content as string
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        # Text-based formats including code files
        if suffix in ['.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', 
                      '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go',
                      '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
                      '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
                      '.r', '.lua', '.dart', '.elm', '.clj', '.ex', '.exs']:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        # JSON
        elif suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Flatten JSON to readable text
                return json.dumps(data, indent=2)
        
        # YAML
        elif suffix in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                # Convert to readable text
                return yaml.dump(data, default_flow_style=False)
        
        # CSV
        elif suffix == '.csv':
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                lines = []
                for row in reader:
                    lines.append(', '.join(row))
                return '\n'.join(lines)
        
        # HTML
        elif suffix in ['.html', '.htm']:
            with open(filepath, 'r', encoding='utf-8') as f:
                # For now, return raw HTML
                # Future: Consider HTML-to-text conversion
                return f.read()
        
        # Try as plain text for unknown extensions
        else:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type: {suffix}")
    
    @staticmethod
    def detect_document_type(filepath: Path) -> DocumentType:
        """Detect document type from file extension
        
        Args:
            filepath: Path object to the document
            
        Returns:
            DocumentType enum value
        """
        suffix = filepath.suffix.lower()
        
        # Markdown
        if suffix in ['.md', '.markdown']:
            return DocumentType.MARKDOWN
        
        # Code files
        elif suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', 
                        '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.rb', 
                        '.php', '.swift', '.kt', '.scala', '.sh', '.bash',
                        '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.r', '.lua',
                        '.dart', '.elm', '.clj', '.ex', '.exs']:
            return DocumentType.CODE
        
        # JSON
        elif suffix == '.json':
            return DocumentType.JSON
        
        # CSV
        elif suffix == '.csv':
            return DocumentType.CSV
        
        # HTML
        elif suffix in ['.html', '.htm']:
            return DocumentType.HTML
        
        # PDF
        elif suffix == '.pdf':
            return DocumentType.PDF
        
        # Images
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']:
            return DocumentType.IMAGE
        
        # Audio
        elif suffix in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            return DocumentType.AUDIO
        
        # Video
        elif suffix in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']:
            return DocumentType.VIDEO
        
        # Plain text (including .txt and unknown extensions)
        else:
            return DocumentType.TEXT
    
    @staticmethod
    def parse_metadata(content: str, doc_type: DocumentType = DocumentType.TEXT) -> Tuple[Dict[str, Any], str]:
        """Extract metadata from document content
        
        Supports:
        - YAML frontmatter (Markdown files)
        - JSON metadata fields
        - Future: Other metadata formats
        
        Args:
            content: Document content
            doc_type: Type of document for context-aware parsing
            
        Returns:
            Tuple of (metadata_dict, content_without_metadata)
        """
        metadata = {}
        content_without_metadata = content
        
        # Check for YAML frontmatter (common in Markdown)
        if doc_type in [DocumentType.MARKDOWN, DocumentType.TEXT]:
            # Pattern for YAML frontmatter: ---\n...\n---
            pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
            match = re.match(pattern, content, re.DOTALL)
            
            if match:
                frontmatter_str = match.group(1)
                content_without_metadata = match.group(2)
                
                try:
                    metadata = yaml.safe_load(frontmatter_str) or {}
                except yaml.YAMLError:
                    # If YAML parsing fails, keep original content
                    content_without_metadata = content
        
        # For JSON documents, extract top-level metadata fields if present
        elif doc_type == DocumentType.JSON:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Extract common metadata fields
                    metadata_fields = ['metadata', 'meta', '_metadata', 'info']
                    for field in metadata_fields:
                        if field in data:
                            metadata = data[field] if isinstance(data[field], dict) else {}
                            break
            except json.JSONDecodeError:
                pass
        
        return metadata, content_without_metadata
    
    @staticmethod
    def prepare_for_ingestion(
        filepath: Path,
        additional_metadata: Dict[str, Any] = None
    ) -> Tuple[str, DocumentType, Dict[str, Any]]:
        """Prepare a document for ingestion into knowledge base
        
        Combines file loading, type detection, and metadata extraction
        into a single operation for convenience.
        
        Args:
            filepath: Path to the document
            additional_metadata: Extra metadata to merge with extracted metadata
            
        Returns:
            Tuple of (content, doc_type, combined_metadata)
        """
        # Load the file
        content = DocumentLoader.load_file(filepath)
        
        # Detect document type
        doc_type = DocumentLoader.detect_document_type(filepath)
        
        # Extract metadata
        extracted_metadata, clean_content = DocumentLoader.parse_metadata(content, doc_type)
        
        # Combine metadata
        combined_metadata = {
            'original_filename': filepath.name,
            'file_extension': filepath.suffix,
            **extracted_metadata
        }
        
        if additional_metadata:
            combined_metadata.update(additional_metadata)
        
        return clean_content, doc_type, combined_metadata