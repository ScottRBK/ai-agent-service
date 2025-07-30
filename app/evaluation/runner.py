from typing import List, Dict, Any, Optional
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from app.core.agents.cli_agent import CLIAgent
import pandas as pd
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import uuid
from pathlib import Path
from datetime import datetime

from .dataset import GoldenDataset
from .config import EvaluationConfig
from .evaluation_utils import EvaluationUtils
from app.config.settings import settings

class EvaluationRunner:
    """Runs evaluations for agents"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results_cache = {}
        self.dataset = GoldenDataset(config.dataset_name)
        self._setup_output_directories()
    
    async def generate_goldens(self) -> None:
        """Generate golden test cases from contexts"""
        await self.dataset.generate_from_contexts(
            self.config.contexts,
            {
                "model": self.config.synthesizer_config.model,
                "styling_config": self.config.synthesizer_config.styling_config
            },
            self.config.synthesizer_config.max_goldens_per_context
        )
        golden_path = self._get_golden_path(self.config.dataset_file)
        self.dataset.save(str(golden_path))
        print(f"Saved {len(self.dataset.goldens)} goldens to {golden_path}")
    
    def load_goldens(self) -> None:
        """Load golden test cases from file"""
        golden_path = self._get_golden_path(self.config.dataset_file)
        self.dataset.load(str(golden_path))
        print(f"Loaded {len(self.dataset.goldens)} goldens from {golden_path}")
    
    async def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation for the configured agent"""
        
        
        # Create test cases
        test_cases = []
        for i, golden in enumerate(self.dataset.goldens):
            print(f"Running test {i+1}/{len(self.dataset.goldens)}: {golden.input[:60]}...")
            test_case = await self._create_test_case(golden)
            test_cases.append(test_case)
        
        # Run evaluation (suppress output)
        print("\nEvaluating...")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            results = evaluate(test_cases=test_cases, metrics=self.config.metrics)
        print("Evaluation complete")
        
        # Convert to useful format
        results_dict = json.loads(results.model_dump_json())
        df = self._create_results_dataframe(results_dict)
        
        # Save results with timestamp
        results_path = self._get_results_path(self.config.results_file)
        df.to_pickle(str(results_path))
        print(f"Saved results to {results_path}")
        
        return {
            'raw_results': results_dict,
            'dataframe': df,
            'summary': self._create_summary(df)
        }
    
    async def _create_test_case(self, golden) -> LLMTestCase:
        """Create a test case by running the agent"""
        #TODO: Use the agent factory to get the agent
        agent = CLIAgent(self.config.agent_id)
        # Initialize agent and create a new session to isolate evaluation
        agent.session_id = str(uuid.uuid4())
        agent.user_id = f"test_user_{uuid.uuid4()}"
        await agent.initialize()    
        agent.provider.config.track_tool_calls = True
        
        # Run agent
        response = await agent.chat(golden.input)
        
        # Get tool calls
        tool_calls = [ToolCall(name=tool["tool_name"]) for tool in agent.provider.get_tool_calls_made()]
        
        # Create test case
        return LLMTestCase(
            input=golden.input,
            actual_output=response,
            expected_output=golden.expected_output,
            context=golden.context,
            expected_tools=golden.expected_tools,
            tools_called=tool_calls,
            additional_metadata={
                'agent_id': self.config.agent_id,
                'expected_tool_names': [t.name for t in golden.expected_tools],
                'actual_tool_names': [t.name for t in tool_calls]
            }
        )
    
    def _create_results_dataframe(self, results_json: Dict) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []
        for test_result in results_json.get('test_results', []):
            metadata = test_result.get('additional_metadata', {})
            base_info = {
                'test_name': test_result['name'],
                'context': str(test_result.get('context', [])),
                'overall_success': test_result['success'],
                'input': test_result['input'],
                'actual_output': test_result.get('actual_output', ''),
                'expected_tools': metadata.get('expected_tool_names', []),
                'actual_tools': metadata.get('actual_tool_names', []),
            }
            
            for metric in test_result['metrics_data']:
                row = base_info.copy()
                row.update({
                    'metric_name': metric['name'],
                    'metric_success': metric['success'],
                    'metric_score': metric['score'],
                    'metric_reason': metric['reason'],
                    'metric_threshold': metric['threshold'],
                    'metric_evaluation_model': metric.get('evaluation_model', 'N/A'),
                    'metric_error': metric.get('error', None),
                    'verbose_logs': metric.get('verbose_logs', '')
                })
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create evaluation summary"""
        return {
            'total_tests': df['test_name'].nunique(),
            'passed_tests': df.groupby('test_name')['overall_success'].first().sum(),
            'metrics_summary': df.groupby('metric_name').agg({
                'metric_score': ['mean', 'min', 'max'],
                'metric_success': 'mean'
            }).to_dict()
        }
    
    def print_summary(self, verbose: bool = False) -> None:
        """Print evaluation summary"""
        if not hasattr(self, '_last_results'):
            print("No evaluation results to display. Run evaluation first.")
            return
            
        df = self._last_results['dataframe']
        raw_results = self._last_results['raw_results']
        
        if verbose:
            EvaluationUtils.print_evaluation_summary_verbose(df)
        else:
            EvaluationUtils.print_evaluation_summary(df, raw_results)
    
    async def run(self, generate: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Complete evaluation workflow"""
        # Generate or load goldens
        if generate:
            await self.generate_goldens()
        else:
            self.load_goldens()
        
        # Run evaluation
        results = await self.run_evaluation()
        self._last_results = results
        
        # Print summary
        self.print_summary(verbose)
        
        return results
    
    def _setup_output_directories(self) -> None:
        """Create output directories if they don't exist"""
        base_dir = Path(settings.EVALUATION_OUTPUT_DIR)
        goldens_dir = base_dir / "goldens"
        results_dir = base_dir / "results"
        
        goldens_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_golden_path(self, filename: str) -> Path:
        """Get full path for golden dataset file"""
        # Remove any directory components from filename, keep only the base name
        base_name = Path(filename).name
        return Path(settings.EVALUATION_OUTPUT_DIR) / "goldens" / base_name
    
    def _get_results_path(self, filename: str) -> Path:
        """Get full path for results file with timestamp"""
        # Remove any directory components and extension from filename
        base_name = Path(filename).stem
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_with_timestamp = f"{base_name}-{timestamp}.pkl"
        return Path(settings.EVALUATION_OUTPUT_DIR) / "results" / filename_with_timestamp