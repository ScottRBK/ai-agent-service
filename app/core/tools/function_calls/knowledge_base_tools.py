from typing import Any
from pydantic import BaseModel, Field
from app.core.tools.tool_registry import register_tool
from app.core.resources.knowledge_base import KnowledgeBaseResource

# Search Knowledge Base Tool
class SearchKnowledgeBaseParams(BaseModel):
    query: str = Field(description="The search query")
    search_type: str = Field(default="all", description="Type of content to search: conversations, documents, or all")
    limit: int = Field(default=5, description="Maximum number of results", ge=1, le=20)

@register_tool(
    name="search_knowledge_base",
    description="Search for information in the knowledge base including past conversations and documents",
    tool_type="function",
    examples=["Search for previous discussions about authentication"],
    params_model=SearchKnowledgeBaseParams
)
async def search_knowledge_base(agent_context: Any, query: str, 
                              search_type: str = "all", limit: int = 5) -> str:
    """Execute knowledge base search"""

    # Validate knowledge base access
    if not hasattr(agent_context, 'knowledge_base') or not agent_context.knowledge_base:
        return "Knowledge base not available for this agent"
    
    kb: KnowledgeBaseResource = agent_context.knowledge_base
    user_id = agent_context.user_id
    
    if search_type == "all": 
        namespace_types = ["conversations", "documents"]
    else:
        namespace_types = [f"{search_type}"]
        
    results = await kb.search(
        query=query,
        namespace_types=namespace_types,
        limit=limit,
        use_reranking=True
    )
    
    if not results:
        return f"No results found for '{query}' in {search_type}"
    
    # Format results
    formatted_results = []
    for i, result in enumerate(results, 1):
        doc = result.document
        content_preview = result.chunk.content
        formatted_results.append(
            f"{i}. [{doc.title or 'Untitled'}] (Score: {result.score:.2f})\n"
            f"   Type: {doc.doc_type}\n"
            f"   {content_preview}"
        )
    
    return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted_results)



# List Documents Tool
class ListDocumentsParams(BaseModel):
    doc_type: str = Field(default="all", description="Type of documents to list: conversations, documents, or all")
    limit: int = Field(default=10, description="Maximum number of documents", ge=1, le=50)

@register_tool(
    name="list_documents",
    description="List documents in the knowledge base",
    tool_type="function",
    examples=["List my recent conversations"],
    params_model=ListDocumentsParams
)
async def list_documents(agent_context: Any, doc_type: str = "all", limit: int = 10) -> str:
    """List documents in knowledge base"""

    if not hasattr(agent_context, 'knowledge_base') or not agent_context.knowledge_base:
        return "Knowledge base not available for this agent"
    
    kb = agent_context.knowledge_base
    user_id = agent_context.user_id
    
    # Determine namespace based on doc type
    if doc_type == "conversations":
        namespace = f"conversations:{user_id}"
    elif doc_type == "documents":
        namespace = f"documents:{user_id}"
    else:
        # For "all", we need to list from both namespaces
        docs_conv = await kb.list_documents(f"conversations:{user_id}")
        docs_user = await kb.list_documents(f"documents:{user_id}")
        documents = docs_conv + docs_user
        documents.sort(key=lambda x: x.created_at, reverse=True)
        documents = documents[:limit]
        
        if not documents:
            return "No documents found in knowledge base"
        
        # Format combined list
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(
                f"{i}. {doc.title or 'Untitled'} ({doc.doc_type})\n"
                f"   Created: {doc.created_at}\n"
                f"   Namespace: {doc.namespace}"
            )
        
        return f"All documents:\n\n" + "\n\n".join(formatted_docs)
    
    # Single namespace query
    documents = await kb.list_documents(namespace)
    
    if not documents:
        return f"No {doc_type} found"
    
    # Format document list
    formatted_docs = []
    for i, doc in enumerate(documents[:limit], 1):
        formatted_docs.append(
            f"{i}. {doc.title or 'Untitled'}\n"
            f"   Created: {doc.created_at}\n"
            f"   Type: {doc.doc_type}"
        )
    
    return f"{doc_type.capitalize()}:\n\n" + "\n\n".join(formatted_docs)
