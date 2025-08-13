from typing import List, Optional, Dict, Any
from app.core.resources.base import BaseResource, ResourceType, ResourceError
from app.core.resources.vector_providers.base import VectorStoreProvider
from app.core.resources.vector_providers.pgvector_provider import PGVectorProvider
from app.models.resources.knowledge_base import (
    Document, DocumentChunk, SearchResult, SearchFilters, DocumentType
)
from app.utils.logging import logger


class KnowledgeBaseResource(BaseResource):
    """Knolwedge Base resource with provider abstraction and namespacing"""

    def __init__(self, resource_id: str, config: dict):
        super().__init__(resource_id, config)
        self.vector_provider = self._create_vector_provider(config)
        self.chat_provider = None  
        self.embedding_provider = None  
        self.embedding_model = None  
        self.rerank_provider = None  
        self.rerank_model = None  
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.chunking_config = config.get("chunking", {})
        self.rerank_limit = config.get("rerank_limit", 50)

    def _get_resource_type(self) -> ResourceType:
        return ResourceType.KNOWLEDGE_BASE

    def _create_vector_provider(self, config: dict) -> VectorStoreProvider:
        """Create the appropriate vector store provider."""
        provider_type = config.get("vector_provider", "pgvector")
        
        if provider_type == "pgvector":
            return PGVectorProvider(config)
        else:
            raise ResourceError(f"Unknown vector provider: {provider_type}", self.resource_id)
        
    async def initialize(self) -> None:
        """Initialize the knowledge base resource."""
        try:
            await self.vector_provider.initialize()
            self.initialized = True
            logger.info(f"Knowledge Base Resource {self.resource_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base Resource {self.resource_id}: {e}")
            raise ResourceError(f"Failed to initialize knowledge base: {e}", self.resource_id)
        
    async def cleanup(self) -> None:
        """Cleanup the knowledge base resource."""
        try:
            await self.vector_provider.cleanup()
            logger.info(f"Knowledge Base Resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup of Knowledge Base Resource {self.resource_id}: {e}")

    async def health_check(self) -> bool:
        """Check if the knowledge base is healthy."""
        try:
            return await self.vector_provider.health_check()
        except Exception as e:
            logger.error(f"Health check failed for Knowledge Base Resource {self.resource_id}: {e}")
            return False
        
    def set_chat_provider(self, chat_provider):
        """Inject chat provider from agent."""
        self.chat_provider = chat_provider
    
    def set_embedding_provider(self, embedding_provider, embedding_model: str):
        """Inject embedding provider and model from agent."""
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
    
    def set_rerank_provider(self, rerank_provider, rerank_model: str):
        """Inject rerank provider and model from agent (optional)."""
        self.rerank_provider = rerank_provider
        self.rerank_model = rerank_model

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured embedding provider."""
        if not self.embedding_provider:
            raise ResourceError("No embedding provider configured", self.resource_id)
        if not self.embedding_model:
            raise ResourceError("No embedding model configured", self.resource_id)
        
        # Use the embed method from the dedicated embedding provider
        return await self.embedding_provider.embed(model=self.embedding_model, text=text)
    
    def _chunk_text(self, text: str, doc_type: DocumentType) -> List[str]:
        """Split text into chunks using appropriate strategy for document type."""
        chunking_strategy = self._get_chunking_strategy(doc_type)
        return chunking_strategy.chunk(text, self.chunk_size, self.chunk_overlap)
    
    def _get_chunking_strategy(self, doc_type: DocumentType):
        """Get the appropriate chunking strategy for document type."""
        # Check for custom strategy in config first
        strategy_name = self.chunking_config.get(doc_type.value, "simple")
        
        if strategy_name == "simple":
            from app.core.resources.chunking.simple import SimpleChunkingStrategy
            return SimpleChunkingStrategy()
        # Future strategies can be added here:
        # elif strategy_name == "semantic":
        #     from app.core.resources.chunking.semantic import SemanticChunkingStrategy
        #     return SemanticChunkingStrategy()
        # elif strategy_name == "token_aware":
        #     from app.core.resources.chunking.token_aware import TokenAwareChunkingStrategy
        #     return TokenAwareChunkingStrategy()
        # elif strategy_name == "markdown":
        #     from app.core.resources.chunking.document_specific import MarkdownChunkingStrategy
        #     return MarkdownChunkingStrategy()
        else:
            logger.warning(f"Unknown chunking strategy '{strategy_name}' for {doc_type.value}, using simple")
            from app.core.resources.chunking.simple import SimpleChunkingStrategy
            return SimpleChunkingStrategy()
        
    async def ingest_document(self, 
                            content: str,
                            user_id: str,
                            namespace_type: str,
                            doc_type: DocumentType,
                            source: Optional[str] = None,
                            title: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None,
                            namespace_qualifier: Optional[str] = None) -> str:
        """Ingest a document into the knowledge base."""
        try:
            import uuid
            document_id = str(uuid.uuid4())
            
            if not self.embedding_model:
                raise ResourceError("No embedding model configured", self.resource_id)

            document = Document(
                id=document_id,
                user_id=user_id,
                namespace_type=namespace_type,
                embedding_model=self.embedding_model,
                namespace_qualifier=namespace_qualifier,
                doc_type=doc_type,
                source=source,
                title=title,
                content=content,
                metadata=metadata
            )
            
            await self.vector_provider.store_document(document)
            
            text_chunks = self._chunk_text(content, doc_type)

            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                embedding = await self._generate_embedding(chunk_text)
                
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()), 
                    document_id=document_id,
                    user_id=user_id,
                    namespace_type=namespace_type,
                    embedding_model=self.embedding_model,
                    namespace_qualifier=namespace_qualifier,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=embedding,
                    metadata={"chunk_count": len(text_chunks)}
                )
                chunks.append(chunk)
            
            await self.vector_provider.store_chunks(chunks)
            
            await self.record_successful_call()
            logger.info(f"Ingested document {document_id} with {len(chunks)} chunks for user {user_id} in {namespace_type}")
            return document_id
            
        except Exception as e:
            await self.record_failed_call(ResourceError(f"Ingestion failed: {e}", self.resource_id))
            raise ResourceError(f"Failed to ingest document: {e}", self.resource_id)
        
    async def search(self, 
                    query: str,
                    user_id: Optional[str] = None,
                    namespace_types: Optional[List[str]] = None,
                    doc_types: Optional[List[DocumentType]] = None,
                    limit: int = 15,
                    use_reranking: bool = True) -> List[SearchResult]:
        """Search for relevant content with optional two-stage retrieval."""
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Ensure embedding model is set for filtering
        if not self.embedding_model:
            raise ResourceError("No embedding model configured", self.resource_id)
        
        # Determine search limit based on reranking
        search_limit = limit
        if use_reranking and self.rerank_provider and self.rerank_model:
            # Use larger limit for first stage, then rerank to final limit
            search_limit = max(limit, self.rerank_limit)
        
        # Create search filters with structured namespace
        filters = SearchFilters(
            user_id=user_id,
            namespace_types=namespace_types,
            doc_types=doc_types,
            embedding_model=self.embedding_model,
            limit=search_limit
        )
        logger.debug(f"KnowledgeBase - search - filters: {filters}")
        # Stage 1: Vector similarity search
        results = await self.vector_provider.search_similar(query_embedding, filters)
        logger.debug(f"KnowledgeBase - search - initial results: {len(results)} found")

        
        # Stage 2: Optional reranking
        if use_reranking and self.rerank_provider and self.rerank_model and len(results) > limit:
            results = await self._rerank_results(query, results, limit)
        
        await self.record_successful_call()
        return results  

    async def _rerank_results(self, query: str, results: List[SearchResult], final_limit: int) -> List[SearchResult]:
        """Rerank search results using the configured rerank provider."""
        try:
            # Extract candidate texts from results
            candidates = [result.chunk.content for result in results]
            
            # Get reranking scores
            rerank_scores = await self.rerank_provider.rerank(self.rerank_model, query, candidates)
            
            # Combine original results with rerank scores
            reranked_results = []
            for i, result in enumerate(results):
                # Update the score with rerank score (you could also combine with original score)
                result.score = rerank_scores[i] if i < len(rerank_scores) else result.score
                reranked_results.append(result)
            
            # Sort by rerank score and return top results
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results[:final_limit]
            
        except Exception as e:
            logger.warning(f"Reranking failed, returning original results: {e}")
            return results[:final_limit]
        
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        try:
            result = await self.vector_provider.get_document(document_id)
            await self.record_successful_call()
            return result
        except Exception as e:
            await self.record_failed_call(ResourceError(f"Get document failed: {e}", self.resource_id))
            raise ResourceError(f"Failed to get document: {e}", self.resource_id)
        
    async def list_documents(self, user_id: str, namespace_type: str) -> List[Document]:
        """List documents in a namespace."""
        try:
            if not self.embedding_model:
                raise ResourceError("No embedding model configured", self.resource_id)
            
            result = await self.vector_provider.list_documents(user_id, namespace_type, self.embedding_model)
            await self.record_successful_call()
            return result
        except Exception as e:
            await self.record_failed_call(ResourceError(f"List documents failed: {e}", self.resource_id))
            raise ResourceError(f"Failed to list documents: {e}", self.resource_id)
        
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            result = await self.vector_provider.delete_document(document_id)
            await self.record_successful_call()
            return result
        except Exception as e:
            await self.record_failed_call(ResourceError(f"Delete document failed: {e}", self.resource_id))
            raise ResourceError(f"Failed to delete document: {e}", self.resource_id)
        
    async def delete_all_documents_for_user(self, user_id: str) -> bool:
        """Delete all documents for a specific user."""
        try:
            result = await self.vector_provider.delete_all_documents_for_user(user_id)
            await self.record_successful_call()
            return result
        except Exception as e:
            await self.record_failed_call(ResourceError(f"Deleted all documents for user failed: {e}", self.resource_id))
            raise ResourceError(f"Failed to delete all documents for user: {e}", self.resource_id)