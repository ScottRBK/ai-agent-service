# Document Management API Implementation Specification

## Overview
Implement REST API endpoints for document management within the knowledge base system. This provides HTTP endpoints for document ingestion, search, and management operations that can be used by external clients or the agent service UI.

## Phase 1: API Models

### 1.1 API Models
**File:** `app/models/documents.py`

Create minimal API-specific models:
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.resources.knowledge_base import DocumentType

class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion"""
    document_id: str
    title: Optional[str]
    doc_type: DocumentType
    namespace: str
    chunks_created: int

class SearchRequest(BaseModel):
    """Request model for knowledge base search"""
    query: str
    namespaces: Optional[List[str]] = None
    doc_types: Optional[List[DocumentType]] = None
    limit: int = Field(10, ge=1, le=50)
    use_reranking: bool = True
    user_id: str = "default_user"
    session_id: str = "default_session"

class SearchResultItem(BaseModel):
    """Individual search result item"""
    document_id: Optional[str]
    title: Optional[str]
    content: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    results: List[SearchResultItem]
    total_results: int
```

## Phase 2: Document API Endpoints

### 2.1 Document API Endpoints
**File:** `app/api/routes/documents.py`

Create document management endpoints:
```python
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
from datetime import datetime
from app.models.resources.knowledge_base import DocumentType, Document
from app.models.documents import (
    DocumentIngestionResponse, SearchRequest, SearchResponse, SearchResultItem
)
from app.core.agents.api_agent import APIAgent

router = APIRouter(prefix="/agents/{agent_id}/documents", tags=["documents"])

@router.post("/", response_model=DocumentIngestionResponse)
async def ingest_document(
    agent_id: str,
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    doc_type: DocumentType = Form(DocumentType.TEXT),
    title: Optional[str] = Form(None),
    user_id: str = Form("default_user"),
    session_id: str = Form("default_session")
):
    """Ingest a document into the agent's knowledge base"""
    
    # Validate input
    if not file and not content:
        raise HTTPException(400, "Either file or content must be provided")
    
    # Get content
    if file:
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        title = title or file.filename
    
    # Create agent to access knowledge base
    agent = APIAgent(agent_id=agent_id, user_id=user_id, session_id=session_id)
    
    try:
        await agent.initialize()
        
        if not agent.knowledge_base:
            raise HTTPException(400, f"Agent {agent_id} does not have knowledge base configured")
        
        # Use simplified namespace
        namespace = f"documents:{user_id}"
        
        # Ingest document
        document_id = await agent.knowledge_base.ingest_document(
            content=content,
            namespace=namespace,
            doc_type=doc_type,
            source=title,
            title=title,
            metadata={
                "uploaded_by": user_id,
                "upload_session": session_id,
                "upload_date": datetime.now().isoformat()
            }
        )
        
        # Estimate chunk count
        chunk_size = agent.knowledge_base.chunk_size
        chunks_created = len(content) // chunk_size + (1 if len(content) % chunk_size else 0)
        
        return DocumentIngestionResponse(
            document_id=document_id,
            title=title,
            doc_type=doc_type,
            namespace=namespace,
            chunks_created=chunks_created
        )
    
    finally:
        await agent.cleanup()

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    agent_id: str,
    request: SearchRequest
):
    """Search the knowledge base"""
    agent = APIAgent(
        agent_id=agent_id, 
        user_id=request.user_id, 
        session_id=request.session_id
    )
    
    try:
        await agent.initialize()
        
        if not agent.knowledge_base:
            raise HTTPException(400, f"Agent {agent_id} does not have knowledge base configured")
        
        # Use simplified namespaces if not provided
        if not request.namespaces:
            request.namespaces = [f"documents:{request.user_id}", f"conversations:{request.user_id}"]
        
        # Perform search
        results = await agent.knowledge_base.search(
            query=request.query,
            namespaces=request.namespaces,
            doc_types=request.doc_types,
            limit=request.limit,
            use_reranking=request.use_reranking
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                SearchResultItem(
                    document_id=r.document.id if r.document else None,
                    title=r.document.title if r.document else None,
                    content=r.chunk.content,
                    score=r.score,
                    metadata=r.document.metadata if r.document else {}
                )
                for r in results
            ],
            total_results=len(results)
        )
    
    finally:
        await agent.cleanup()
```

### 2.2 Router Registration
**File:** `app/main.py`

Register document routes:
```python
from app.api.routes import documents

# Add after existing router registrations
app.include_router(documents.router)
```

## Phase 3: Testing

### 3.1 Integration Tests
**File:** `tests/test_integration/test_document_api_integration.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.resources.knowledge_base import DocumentType

client = TestClient(app)

@pytest.mark.asyncio
async def test_document_ingestion_api():
    """Test document ingestion via API"""
    response = client.post(
        "/agents/knowledge_agent/documents/",
        data={
            "content": "This is a test document about OAuth2 implementation",
            "doc_type": "text",
            "title": "OAuth2 Guide",
            "user_id": "test_user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["title"] == "OAuth2 Guide"
    assert data["namespace"] == "documents:test_user"

@pytest.mark.asyncio
async def test_document_search_api():
    """Test document search via API"""
    # First ingest a document
    ingest_response = client.post(
        "/agents/knowledge_agent/documents/",
        data={
            "content": "Advanced authentication using JWT tokens with refresh rotation",
            "doc_type": "text",
            "title": "JWT Guide",
            "user_id": "test_user"
        }
    )
    assert ingest_response.status_code == 200
    
    # Now search for it
    search_response = client.post(
        "/agents/knowledge_agent/documents/search",
        json={
            "query": "JWT refresh rotation",
            "user_id": "test_user",
            "limit": 5
        }
    )
    
    assert search_response.status_code == 200
    data = search_response.json()
    assert data["query"] == "JWT refresh rotation"
    assert len(data["results"]) > 0
    assert "JWT" in data["results"][0]["content"]

@pytest.mark.asyncio
async def test_file_upload_api():
    """Test file upload via API"""
    # Create a test file
    test_content = b"# Test Markdown\n\nThis is a test document"
    
    response = client.post(
        "/agents/knowledge_agent/documents/",
        files={"file": ("test.md", test_content, "text/markdown")},
        data={
            "doc_type": "markdown",
            "user_id": "test_user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "test.md"
    assert data["doc_type"] == "markdown"
```

## Usage Examples

### Document Upload API
```bash
# Upload a document via content
curl -X POST "http://localhost:8000/agents/knowledge_agent/documents" \
  -F "content=This is my document content" \
  -F "doc_type=text" \
  -F "title=My Document" \
  -F "user_id=user123"

# Upload a file
curl -X POST "http://localhost:8000/agents/knowledge_agent/documents" \
  -F "file=@guide.md" \
  -F "doc_type=markdown" \
  -F "user_id=user123"
```

### Document Search API
```bash
# Search across knowledge base
curl -X POST "http://localhost:8000/agents/knowledge_agent/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "refresh token implementation",
    "namespaces": ["documents:user123", "conversations:user123"],
    "use_reranking": true,
    "limit": 10
  }'

# Search only user documents
curl -X POST "http://localhost:8000/agents/knowledge_agent/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication guide",
    "namespaces": ["documents:user123"],
    "doc_types": ["markdown", "text"],
    "limit": 5
  }'
```

## Implementation Order

1. **Phase 1: Create API Models**
   - Create `app/models/documents.py` with request/response models
   - Define clear API contracts for document operations

2. **Phase 2: Implement Endpoints**
   - Create `app/api/routes/documents.py` with ingestion and search endpoints
   - Register router in `app/main.py`

3. **Phase 3: Testing**
   - Create comprehensive integration tests
   - Test file uploads, content uploads, and search functionality

## Key Features

1. **Flexible Document Ingestion**: Support both direct content and file uploads
2. **Namespace Isolation**: Documents are automatically namespaced by user
3. **Rich Search Capabilities**: Full-text search with reranking and filtering
4. **Metadata Tracking**: Automatic tracking of upload metadata
5. **Error Handling**: Graceful error responses for missing knowledge base

## Security Considerations

1. **User Isolation**: Documents are namespaced by user_id
2. **Agent Authorization**: Only agents with knowledge base can access documents
3. **Input Validation**: Pydantic models ensure type safety
4. **Size Limits**: Consider adding file size limits for uploads

## Future Enhancements

1. **Document Management**: Add endpoints for listing, updating, and deleting documents
2. **Bulk Operations**: Support batch document uploads
3. **Export Functionality**: Allow document export in various formats
4. **Access Control**: Add role-based access control for documents
5. **Versioning**: Implement document versioning support