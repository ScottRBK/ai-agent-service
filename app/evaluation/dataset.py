from typing import List, Dict, Any, Optional, Union
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden
import pickle
import pandas as pd
from deepeval.test_case import ToolCall
from app.evaluation.config import ContextWithMetadata
from pathlib import Path

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