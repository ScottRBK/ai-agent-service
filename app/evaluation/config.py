from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import BaseMetric

from enum import Enum


class ContextWithMetadata(BaseModel):
    """Context with associated tool metadata"""
    context: List[str]
    tools: List[str]
    expected_output: Optional[str] = None  # For contextual recall metric
    retrieval_context: Optional[List[str]] = None  # For RAG metrics
    user_id: Optional[str] = None  # User ID for this context's test case
    session_id: Optional[str] = None  # Session ID for this context's test case


class SynthesizerConfig(BaseModel):
    """Configuration for the DeepEval synthesizer"""
    model: Any  # DeepEval model instance
    styling_config: Optional[StylingConfig] = None
    max_goldens_per_context: int = 2
    
    class Config:
        arbitrary_types_allowed = True

class GoldenGenerationType(str, Enum):
    """Type of golden generation"""
    DOCUMENT = "document"  
    CONTEXT = "context"  
    SCRATCH = "scratch"
    KNOWLEDGE_BASE = "knowledge_base"

class EvaluationConfig(BaseModel):
    """Complete configuration for an agent evaluation"""
    agent_id: str = Field(..., description="ID of the agent to evaluate")
    synthesizer_config: SynthesizerConfig = Field(..., description="Synthesizer configuration for golden generation")
    metrics: List[BaseMetric] = Field(..., description="List of metrics to evaluate")
    contexts: List[ContextWithMetadata] = Field(default_factory=list, description="Context data with expected tools")
    dataset_name: str = Field(..., description="Name for the evaluation dataset")
    dataset_file: str = Field(..., description="File path to save/load golden dataset")
    results_file: str = Field(..., description="File path to save evaluation results")
    golden_generation_type: GoldenGenerationType = Field(GoldenGenerationType.CONTEXT, description="Type of golden generation")
    
    # Document generation settings
    document_paths: Optional[List[str]] = Field(None, description="List of document file paths to generate goldens from")
    document_metadata: Optional[Dict[str, Dict]] = Field(None, description="Metadata per document (tools, expected_output, retrieval_context)")
    default_tools: Optional[List[str]] = Field(None, description="Default tools for all documents if not specified per document")
    use_document_as_retrieval: bool = Field(False, description="Use document chunks as retrieval_context for RAG metrics")
    parse_frontmatter: bool = Field(True, description="Extract metadata from document YAML frontmatter")
    chunk_size: Optional[int] = Field(1000, description="Size of document chunks")
    chunk_overlap: Optional[int] = Field(200, description="Overlap between document chunks")
    max_contexts_per_document: int = Field(3, description="Maximum contexts to create per document")
    max_goldens_per_context: int = Field(2, description="Maximum goldens to generate per context")
    use_knowledge_base: bool = Field(False, description="Use knowledge base for document storage/retrieval")
    persist_to_kb: bool = Field(False, description="Persist documents to knowledge base before generation")
    
    class Config:
        arbitrary_types_allowed = True