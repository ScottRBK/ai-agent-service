# Knowledge Base Integration Implementation Specification

## Overview
Implement a comprehensive knowledge base integration that enables cross-session memory persistence and intelligent context retrieval for agents.

## Phase 1: Core Integration Components

### 1.1 Enhanced Memory Compression Agent
**File:** `app/core/agents/memory_compression_agent.py`

Add knowledge base archival after compression:
```python
async def compress_conversation(self, parent_agent_id: str, ...):
    # ... existing compression logic ...
    
    # After successful compression, archive to knowledge base
    if summary and self._should_archive_to_kb(parent_agent_id):
        await self._archive_compressed_session(
            summary=summary,
            older_messages=older_messages,
            parent_agent_id=parent_agent_id,
            user_id=user_id,
            session_id=session_id
        )

async def _archive_compressed_session(self, summary: str, older_messages: List, 
                                    parent_agent_id: str, user_id: str, session_id: str):
    # Extract metadata from conversation
    metadata = self._extract_conversation_metadata(older_messages)
    
    # Create structured document
    document_content = self._format_session_document(summary, metadata)
    
    # Store in knowledge base
    parent_agent = await self._get_parent_agent(parent_agent_id, user_id, session_id)
    if parent_agent and parent_agent.knowledge_base:
        await parent_agent.knowledge_base.ingest_document(
            content=document_content,
            namespace=f"user:{user_id}:conversations",
            doc_type=DocumentType.CONVERSATION,
            source=f"session:{session_id}",
            title=f"Session {metadata['start_date']} - {metadata['end_date']}",
            metadata={
                "session_id": session_id,
                "agent_id": parent_agent_id,
                "topics": metadata['topics'],
                "entities": metadata['entities'],
                "decisions": metadata['decisions'],
                "message_count": len(older_messages),
                "compression_date": datetime.now().isoformat(),
                "date_range": {
                    "start": metadata['start_date'],
                    "end": metadata['end_date']
                }
            }
        )
```

### 1.2 Knowledge Base Tools
**File:** `app/core/tools/function_calls/knowledge_base_tool.py`

```python
from typing import Dict, Any, List, Optional
from app.core.tools.function_calls.base import BaseFunctionTool

class SearchKnowledgeBaseTool(BaseFunctionTool):
    """Tool for searching the knowledge base"""
    
    def get_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search for information in the knowledge base including past conversations and documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["conversations", "documents", "all"],
                            "description": "Type of content to search",
                            "default": "all"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, agent_context: Any, **kwargs) -> str:
        query = kwargs.get("query")
        search_type = kwargs.get("search_type", "all")
        limit = kwargs.get("limit", 5)
        
        # Get knowledge base from agent
        kb = agent_context.get_resource("knowledge_base")
        if not kb:
            return "Knowledge base not available for this agent"
        
        # Determine namespaces based on search type
        namespaces = self._get_search_namespaces(agent_context, search_type)
        
        # Perform search
        results = await kb.search(
            query=query,
            namespaces=namespaces,
            limit=limit,
            use_reranking=True
        )
        
        # Format results
        return self._format_search_results(results)
```

### 1.3 Enhanced Base Agent
**File:** `app/core/agents/base_agent.py`

Add cross-session context methods:
```python
async def _should_search_cross_session(self, user_message: str) -> bool:
    """Determine if cross-session search is needed"""
    triggers = [
        "what did we discuss", "previous conversation", "last time",
        "remind me", "we mentioned", "earlier you said", "in the past",
        "have we talked about", "did I mention"
    ]
    return any(trigger in user_message.lower() for trigger in triggers)

async def _get_cross_session_context(self, query: str, current_session_id: str) -> Optional[str]:
    """Retrieve relevant context from other sessions"""
    if not self.knowledge_base:
        return None
    
    try:
        # Search for relevant past conversations
        results = await self.knowledge_base.search(
            query=query,
            namespaces=[f"user:{self.user_id}:conversations"],
            metadata_filters={
                "session_id": {"$ne": current_session_id}
            },
            limit=10,  # Get more initially
            use_reranking=True
        )
        
        # Filter by relevance threshold
        relevant_results = [r for r in results if r.score > 0.7][:3]
        
        if not relevant_results:
            return None
        
        # Format context
        context_parts = ["## Relevant Past Conversations\n"]
        for result in relevant_results:
            metadata = result.document.metadata
            context_parts.append(
                f"### Session from {metadata['date_range']['start']}\n"
                f"{result.chunk.content}\n"
            )
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error retrieving cross-session context: {e}")
        return None

async def prepare_messages_with_kb(self, user_message: str) -> List[Dict[str, str]]:
    """Prepare messages with intelligent knowledge base context"""
    messages = []
    
    # System prompt
    messages.append({"role": "system", "content": self.system_prompt})
    
    # Current session summary
    if self.summary:
        messages.append({
            "role": "system",
            "content": f"Current conversation summary:\n{self.summary}"
        })
    
    # Cross-session context (if relevant)
    if self._should_search_cross_session(user_message):
        cross_session_context = await self._get_cross_session_context(
            user_message, 
            self.session_id
        )
        if cross_session_context:
            messages.append({
                "role": "system",
                "content": cross_session_context
            })
    
    # Recent conversation history
    messages.extend(self.conversation_history[-10:])
    
    # Current user message
    messages.append({"role": "user", "content": user_message})
    
    return messages
```

### 1.4 Document Ingestion API
**File:** `app/api/routes/documents.py`

```python
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
from app.models.resources.knowledge_base import DocumentType
from app.core.agents.api_agent import APIAgent

router = APIRouter(prefix="/agents/{agent_id}/documents", tags=["documents"])

@router.post("/", response_model=DocumentIngestionResponse)
async def ingest_document(
    agent_id: str,
    file: Optional[UploadFile] = File(None),
    content: Optional[str] = Form(None),
    doc_type: DocumentType = Form(DocumentType.TEXT),
    namespace: Optional[str] = Form(None),
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
    await agent.initialize()
    
    if not agent.knowledge_base:
        raise HTTPException(400, f"Agent {agent_id} does not have knowledge base configured")
    
    # Default namespace if not provided
    if not namespace:
        namespace = f"user:{user_id}:documents"
    
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
    
    return DocumentIngestionResponse(
        document_id=document_id,
        title=title,
        doc_type=doc_type,
        namespace=namespace,
        chunks_created=len(content) // 1000 + 1  # Approximate
    )

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
    await agent.initialize()
    
    if not agent.knowledge_base:
        raise HTTPException(400, f"Agent {agent_id} does not have knowledge base configured")
    
    # Perform search
    results = await agent.knowledge_base.search(
        query=request.query,
        namespaces=request.namespaces or [f"user:{request.user_id}:documents"],
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
```

## Phase 2: Configuration

### 2.1 Agent Configuration Updates
**File:** `agent_config.json`

```json
{
  "agent_id": "knowledge_enabled_agent",
  "provider": "azure_openai_cc",
  "model": "gpt-4o-mini",
  "resources": ["memory", "knowledge_base"],
  "allowed_regular_tools": [
    "get_current_datetime",
    "search_knowledge_base",
    "list_documents"
  ],
  "memory": {
    "compression": {
      "enabled": true,
      "threshold_tokens": 8000,
      "recent_messages_to_keep": 10,
      "archive_to_knowledge_base": true
    }
  },
  "knowledge_base": {
    "vector_provider": "pgvector",
    "chunk_size": 800,
    "chunk_overlap": 100,
    "rerank_limit": 50,
    "auto_context": {
      "enabled": true,
      "mode": "selective",
      "max_tokens": 2000,
      "relevance_threshold": 0.7,
      "cross_session_search": true,
      "search_triggers": [
        "previous", "earlier", "discussed", 
        "mentioned", "last time", "remind"
      ]
    },
    "namespaces": {
      "conversations": "user:{user_id}:conversations",
      "documents": "user:{user_id}:documents",
      "shared": "organization:shared"
    }
  }
}
```

### 2.2 Tool Registry Updates
**File:** `app/core/tools/tool_registry.py`

Add knowledge base tools to registry:
```python
# In get_all_tools() method
regular_tools.extend([
    SearchKnowledgeBaseTool().get_definition(),
    ListDocumentsTool().get_definition(),
    GetDocumentTool().get_definition()
])
```

### 2.3 API Router Registration
**File:** `app/main.py`

```python
from app.api.routes import documents

# Add to router registration
app.include_router(documents.router)
```

## Phase 3: Models and Types

### 3.1 API Models
**File:** `app/models/documents.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.resources.knowledge_base import DocumentType

class DocumentIngestionResponse(BaseModel):
    document_id: str
    title: Optional[str]
    doc_type: DocumentType
    namespace: str
    chunks_created: int

class SearchRequest(BaseModel):
    query: str
    namespaces: Optional[List[str]] = None
    doc_types: Optional[List[DocumentType]] = None
    limit: int = Field(10, ge=1, le=50)
    use_reranking: bool = True
    user_id: str = "default_user"
    session_id: str = "default_session"

class SearchResultItem(BaseModel):
    document_id: Optional[str]
    title: Optional[str]
    content: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_results: int
```

## Phase 4: Database Schema Updates

### 4.1 Migration for Conversation Documents
**File:** `alembic/versions/xxx_add_conversation_documents.py`

```sql
-- Add index for conversation documents
CREATE INDEX idx_documents_conversation_metadata 
ON documents USING gin(metadata) 
WHERE doc_type = 'conversation';

-- Add index for session_id in metadata
CREATE INDEX idx_documents_session_id 
ON documents ((metadata->>'session_id'));
```

## Phase 5: Testing

### 5.1 Integration Tests
**File:** `tests/test_integration/test_knowledge_base_integration.py`

```python
async def test_memory_to_knowledge_base_flow():
    """Test automatic archival of compressed conversations"""
    agent = APIAgent("test_agent", "test_user", "test_session")
    await agent.initialize()
    
    # Simulate conversation that triggers compression
    for i in range(50):
        await agent.chat(f"Message {i}")
    
    # Verify compression occurred and was archived
    search_results = await agent.knowledge_base.search(
        "Message",
        namespaces=["user:test_user:conversations"]
    )
    
    assert len(search_results) > 0
    assert search_results[0].document.metadata["session_id"] == "test_session"

async def test_cross_session_context():
    """Test cross-session context retrieval"""
    # Session 1
    agent1 = APIAgent("test_agent", "test_user", "session1")
    await agent1.initialize()
    await agent1.chat("I'm building an authentication system with JWT tokens")
    
    # Session 2
    agent2 = APIAgent("test_agent", "test_user", "session2")
    await agent2.initialize()
    response = await agent2.chat("What did we discuss about authentication?")
    
    assert "JWT" in response
    assert "authentication" in response
```

## Usage Examples

### Basic Usage
```python
# Agent automatically has cross-session memory
agent = APIAgent("knowledge_agent", "user123", "session456")
response = await agent.chat("What did we discuss about authentication?")
# Agent searches past sessions and includes relevant context

# Direct document upload via API
POST /agents/knowledge_agent/documents
{
    "content": "OAuth2 Implementation Guide...",
    "doc_type": "markdown",
    "title": "Auth Documentation"
}

# Search across all knowledge
POST /agents/knowledge_agent/search
{
    "query": "refresh token implementation",
    "namespaces": ["user:user123:conversations", "user:user123:documents"]
}
```

### Configuration Example
```json
{
  "agent_id": "support_agent",
  "resources": ["memory", "knowledge_base"],
  "allowed_regular_tools": ["search_knowledge_base"],
  "memory": {
    "compression": {
      "enabled": true,
      "archive_to_knowledge_base": true
    }
  },
  "knowledge_base": {
    "auto_context": {
      "enabled": true,
      "cross_session_search": true
    }
  }
}
```

## Key Benefits

1. **Seamless Integration**: Memory compression automatically archives to knowledge base
2. **Cross-Session Intelligence**: Agents remember context from all past interactions
3. **Flexible Search**: Tools allow agents to search their knowledge proactively
4. **API Support**: HTTP endpoints for document management and search
5. **Performance Optimized**: Selective context injection prevents token bloat

This implementation provides a complete knowledge base integration with automatic conversation archival, cross-session context, and document management capabilities.