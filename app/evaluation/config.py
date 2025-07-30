from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import BaseMetric


class ContextWithMetadata(BaseModel):
    """Context with associated tool metadata"""
    context: List[str]
    tools: List[str]


class SynthesizerConfig(BaseModel):
    """Configuration for the DeepEval synthesizer"""
    model: Any  # DeepEval model instance
    styling_config: StylingConfig
    max_goldens_per_context: int = 2
    
    class Config:
        arbitrary_types_allowed = True


class EvaluationConfig(BaseModel):
    """Complete configuration for an agent evaluation"""
    agent_id: str = Field(..., description="ID of the agent to evaluate")
    synthesizer_config: SynthesizerConfig = Field(..., description="Synthesizer configuration for golden generation")
    metrics: List[BaseMetric] = Field(..., description="List of metrics to evaluate")
    contexts: List[ContextWithMetadata] = Field(..., description="Context data with expected tools")
    dataset_name: str = Field(..., description="Name for the evaluation dataset")
    dataset_file: str = Field(..., description="File path to save/load golden dataset")
    results_file: str = Field(..., description="File path to save evaluation results")
    
    class Config:
        arbitrary_types_allowed = True