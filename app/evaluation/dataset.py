from typing import List, Dict, Any, Optional, Union
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden
import pickle
import pandas as pd
from deepeval.test_case import ToolCall
from app.evaluation.config import ContextWithMetadata
from app.evaluation.document_processor import DocumentProcessor
from app.core.resources.knowledge_base import KnowledgeBaseResource
from app.models.resources.knowledge_base import DocumentType
from pathlib import Path
from app.utils.logging import logger

class GoldenDataset:
    """Manages golden test cases"""
    
    def __init__(self, name: str):
        self.name = name
        self.goldens: List[Golden] = []
        
    async def generate_from_contexts(self, contexts_with_metadata: Union[List[Dict], List[ContextWithMetadata]], 
                                   synthesizer_config: Dict,
                                   max_goldens_per_context: int = 2) -> None:
        """Generate goldens from contexts"""
        synthesizer = Synthesizer(**synthesizer_config)
        print(f"\nGenerating goldens from {len(contexts_with_metadata)} contexts")
        
        for context_data in contexts_with_metadata:
            # Handle both dict and ContextWithMetadata formats
            if isinstance(context_data, ContextWithMetadata):
                context = context_data.context
                tools = context_data.tools
                retrieval_context = context_data.retrieval_context
                expected_output = context_data.expected_output
                user_id = context_data.user_id
                session_id = context_data.session_id
            else:
                context = context_data["context"]
                tools = context_data["tools"]
                retrieval_context = context_data.get("retrieval_context")
                expected_output = context_data.get("expected_output")
                user_id = context_data.get("user_id")
                session_id = context_data.get("session_id")
            
            goldens = await synthesizer.a_generate_goldens_from_contexts(
                contexts=[context],
                include_expected_output=(expected_output is None),
                max_goldens_per_context=max_goldens_per_context
            )
            
            # Add expected tools, user/session IDs, and RAG-specific fields to each golden
            for i, golden in enumerate(goldens):
                print(f"  Golden {i+1}: {golden.input[:50]}...")
                golden.expected_tools = [
                    ToolCall(name=tool) for tool in tools
                ]
                # Preserve user and session IDs in additional_metadata if provided
                if user_id or session_id:
                    if not hasattr(golden, 'additional_metadata') or golden.additional_metadata is None:
                        golden.additional_metadata = {}
                    if user_id:
                        golden.additional_metadata['user_id'] = user_id
                    if session_id:
                        golden.additional_metadata['session_id'] = session_id
                # Preserve RAG-specific fields if provided
                if retrieval_context:
                    golden.retrieval_context = retrieval_context
                if expected_output:
                    # Override synthesized expected_output with our specific one
                    golden.expected_output = expected_output
            
            self.goldens.extend(goldens)
            print(f"\nTotal goldens generated: {len(self.goldens)}")
    
    async def generate_from_documents(
        self,
        document_paths: List[str],
        synthesizer_config: Dict,
        max_goldens_per_context: int = 2,
        document_metadata: Optional[Dict[str, Dict]] = None,
        default_tools: Optional[List[str]] = None,
        parse_frontmatter: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,

    ) -> None:
        """Generate goldens directly from document files with metadata support
        
        Args:
            document_paths: List of document file paths
            synthesizer_config: Configuration for synthesizer
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            max_contexts_per_doc: Maximum contexts per document
            max_goldens_per_context: Maximum goldens per context
            document_metadata: Dict mapping document paths to their metadata
            default_tools: Default tools if not specified per document
            use_document_as_retrieval: Use document chunks as retrieval_context
            parse_frontmatter: Extract metadata from document frontmatter
        """
        synthesizer = Synthesizer(**synthesizer_config)
        logger.info(f"Generating goldens from {len(document_paths)} documents")
        
        for doc_path in document_paths:
            logger.info(f"Processing document: {doc_path}")
            
            # 1. Load document
            try:
                content = DocumentProcessor.load_document(doc_path)
            except Exception as e:
                logger.error(f"Failed to load document {doc_path}: {e}")
                continue
            
            # 2. Parse frontmatter if present
            metadata = {}
            if parse_frontmatter:
                metadata, content = DocumentProcessor.parse_frontmatter(content)
                logger.info(f"""Evaluation - Dataset - Metadata extracted from frontmatter: \nmetdata: {metadata}""")
                if metadata:
                    logger.info(f"  Found frontmatter metadata: {list(metadata.keys())}")
            
            # 3. Merge metadata sources (frontmatter > config > defaults)
            doc_metadata = document_metadata.get(doc_path, {}) if document_metadata else {}
            
            final_metadata = {
                "tools": (metadata.get("tools") or 
                         doc_metadata.get("tools") or 
                         default_tools or []),
                "expected_output": (metadata.get("expected_output") or 
                                   doc_metadata.get("expected_output"))
            }
            
            logger.info(f"  Final tools: {final_metadata['tools']}")
            
            # 4. Check for complete ContextWithMetadata objects in frontmatter
            if metadata.get("contexts_with_metadata"):
                logger.info(f"  Found {len(metadata['contexts_with_metadata'])} predefined contexts with metadata")
                
                # Process each ContextWithMetadata object
                for context_idx, ctx_metadata in enumerate(metadata["contexts_with_metadata"]):
                    logger.info(f"  Generating goldens for predefined context {context_idx + 1}/{len(metadata['contexts_with_metadata'])}")
                    
                    # Extract context and metadata
                    context = ctx_metadata.get("context", [])
                    if isinstance(context, str):
                        context = [context]
                    
                    tools = ctx_metadata.get("tools", final_metadata["tools"])
                    expected_output = ctx_metadata.get("expected_output", final_metadata["expected_output"])
                    
                    # Generate goldens from this specific context
                    goldens = await synthesizer.a_generate_goldens_from_contexts(
                        contexts=[context],
                        include_expected_output=(expected_output is None),
                        max_goldens_per_context=max_goldens_per_context
                    )
                    
                    # Apply metadata from ContextWithMetadata
                    for golden in goldens:
                        golden.expected_tools = [ToolCall(name=tool) for tool in tools]
                        if expected_output:
                            golden.expected_output = expected_output
                        
                        # Add source document to metadata
                        if not hasattr(golden, 'additional_metadata') or golden.additional_metadata is None:
                            golden.additional_metadata = {}
                        golden.additional_metadata["source_document"] = doc_path
                        golden.additional_metadata["context_index"] = context_idx
                        golden.additional_metadata["context_type"] = "predefined"
                        if user_id:
                            golden.additional_metadata["user_id"] = user_id 
                    
                    self.goldens.extend(goldens)
                    logger.info(f"    Generated {len(goldens)} goldens")
            
            else:
                logger.warning("Evaluation Dataset - No context provided")
        
        logger.info(f"Total goldens generated from documents: {len(self.goldens)}")
    
    async def generate_from_knowledge_base(
        self,
        knowledge_base: KnowledgeBaseResource,
        user_id: str,
        namespace_types: List[str],
        synthesizer_config: Dict,
        max_documents: int = 10,
        max_goldens_per_document: int = 5,
        tools: Optional[List[str]] = None
    ) -> None:
        """Generate goldens from documents in knowledge base
        
        Args:
            knowledge_base: KnowledgeBase resource instance
            user_id: User ID to retrieve documents for
            namespace_types: Namespace types to search
            synthesizer_config: Configuration for synthesizer
            max_documents: Maximum documents to process
            max_goldens_per_document: Maximum goldens per document
            tools: Expected tools for all goldens
        """
        synthesizer = Synthesizer(**synthesizer_config)
        logger.info(f"Generating goldens from knowledge base (user: {user_id}, namespaces: {namespace_types})")
        
        # Retrieve documents from knowledge base
        all_documents = []
        for namespace_type in namespace_types:
            docs = await knowledge_base.list_documents(
                user_id=user_id,
                namespace_type=namespace_type,
                embedding_model=knowledge_base.embedding_model
            )
            all_documents.extend(docs[:max_documents])
            logger.info(f"  Found {len(docs)} documents in namespace '{namespace_type}'")
        
        # Limit total documents
        all_documents = all_documents[:max_documents]
        logger.info(f"Processing {len(all_documents)} documents total")
        
        for doc in all_documents:
            logger.info(f"Processing document: {doc.title or doc.id}")
            
            # Get document chunks for context
            chunks = await knowledge_base.vector_provider.get_document_chunks(doc.id)
            if not chunks:
                logger.warning(f"  No chunks found for document {doc.id}")
                continue
            
            # Extract chunk content
            chunk_texts = [chunk.content for chunk in chunks]
            
            # Create contexts from chunks
            contexts = DocumentProcessor.create_contexts_from_chunks(
                chunk_texts, 
                context_size=3,
                max_contexts=max(1, max_goldens_per_document // 2)
            )
            
            logger.info(f"  Created {len(contexts)} contexts from {len(chunks)} chunks")
            
            # Generate goldens for each context
            for context_idx, context in enumerate(contexts):
                goldens = await synthesizer.a_generate_goldens_from_contexts(
                    contexts=[context],
                    include_expected_output=True,
                    max_goldens_per_context=2
                )
                
                # Add metadata to goldens
                for golden in goldens:
                    if tools:
                        golden.expected_tools = [ToolCall(name=tool) for tool in tools]
                    
                    # Use chunks as retrieval context for RAG evaluation
                    golden.retrieval_context = context
                    
                    # Add document metadata
                    if not hasattr(golden, 'additional_metadata') or golden.additional_metadata is None:
                        golden.additional_metadata = {}
                    golden.additional_metadata.update({
                        "source_document_id": doc.id,
                        "source_document_title": doc.title,
                        "namespace_type": doc.namespace_type,
                        "user_id": user_id,
                        "context_index": context_idx
                    })
                
                self.goldens.extend(goldens)
                logger.info(f"    Generated {len(goldens)} goldens")
        
        logger.info(f"Total goldens generated from knowledge base: {len(self.goldens)}")
    
    def save(self, filepath: str):
        """Save dataset"""
        # Ensure parent directory exists
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.goldens, f)
    
    def load(self, filepath: str):
        """Load dataset"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            self.goldens = pickle.load(f)

    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis"""
        data = []
        for golden in self.goldens:
            data.append({
                'input': golden.input,
                'expected_output': golden.expected_output,
                'context': str(golden.context) if hasattr(golden, 'context') else '',
                'expected_tools': [t.name for t in golden.expected_tools] if hasattr(golden, 'expected_tools') else []
            })
        return pd.DataFrame(data)