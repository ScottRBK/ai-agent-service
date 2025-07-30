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

from .dataset import GoldenDataset

class EvaluationRunner:
    """Runs evaluations for agents"""
    
    def __init__(self):
        self.results_cache = {}
    
    async def run_evaluation(self, 
                           agent_id: str,
                           dataset: GoldenDataset,
                           metrics: List) -> Dict[str, Any]:
        """Run evaluation for an agent"""
        
        
        # Create test cases
        test_cases = []
        for golden in dataset.goldens:
            test_case = await self._create_test_case(agent_id, golden)
            test_cases.append(test_case)
        
        # Run evaluation (suppress output)
        print("\nEvaluating...")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            results = evaluate(test_cases=test_cases, metrics=metrics)
        print("Evaluation complete")
        
        # Convert to useful format
        results_dict = json.loads(results.model_dump_json())
        df = self._create_results_dataframe(results_dict)
        
        return {
            'raw_results': results_dict,
            'dataframe': df,
            'summary': self._create_summary(df)
        }
    
    async def _create_test_case(self, agent_id: str, golden) -> LLMTestCase:
        """Create a test case by running the agent"""
        #TODO: Use the agent factory to get the agent
        agent = CLIAgent(agent_id)
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
                'agent_id': agent.agent_id,
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