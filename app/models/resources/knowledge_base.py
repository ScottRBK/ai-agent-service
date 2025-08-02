from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class DocumentType(str, Enum):
    """Type of document."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    WEB = "web"
    CONVERSATION = "conversation"
    MARKDOWN = "markdown"


class Document(BaseModel):
    """Document model with client-generated UUID."""
    id: str                           
    namespace: str
    doc_type: DocumentType
    source: Optional[str] = None
    title: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None  

class DocumentChunk(BaseModel):
    """Document chunk model with client-generated UUID."""
    id: str                           
    document_id: str
    namespace: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None  

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    document: Optional[Document] = None


class SearchFilters(BaseModel):
    namespaces: Optional[List[str]] = None
    doc_types: Optional[List[DocumentType]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    limit: int = 10

    
