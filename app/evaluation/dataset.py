from typing import List, Dict, Any, Optional
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden
import pickle
import pandas as pd
from deepeval.test_case import ToolCall

class GoldenDataset:
    """Manages golden test cases"""
    
    def __init__(self, name: str):
        self.name = name
        self.goldens: List[Golden] = []
        
    async def generate_from_contexts(self, contexts_with_metadata: List[Dict], 
                                   synthesizer_config: Dict,
                                   max_goldens_per_context: int = 2) -> None:
        """Generate goldens from contexts"""
        synthesizer = Synthesizer(**synthesizer_config)
        print(f"\nGenerating goldens from {len(contexts_with_metadata)} contexts")
        
        for context_data in contexts_with_metadata:
            goldens = await synthesizer.a_generate_goldens_from_contexts(
                contexts=[context_data["context"]],
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context
            )
            
            # Add expected tools to each golden
            for i, golden in enumerate(goldens):
                print(f"  Golden {i+1}: {golden.input[:50]}...")
                golden.expected_tools = [
                    ToolCall(name=tool) for tool in context_data["tools"]
                ]
            
            self.goldens.extend(goldens)
            print(f"\nTotal goldens generated: {len(self.goldens)}")
    
    def save(self, filepath: str):
        """Save dataset"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.goldens, f)
    
    def load(self, filepath: str):
        """Load dataset"""
        with open(filepath, 'rb') as f:
            self.goldens = pickle.load(f)

    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis"""
        data = []
        for golden in self.goldens:
            data.append({
                'input': golden.input,
                'expected_output': golden.expected_output,
                'context': str(golden.context),
                'expected_tools': [t.name for t in golden.expected_tools]
            })
        return pd.DataFrame(data)