from typing import List, Dict, Any, Optional
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from app.core.agents.cli_agent import CLIAgent
from app.utils.chat_utils import separate_chain_of_thought
import pandas as pd
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import uuid
from pathlib import Path
from datetime import datetime

from .dataset import GoldenDataset
from .config import EvaluationConfig, GoldenGenerationType
from .evaluation_utils import EvaluationUtils
from .document_processor import DocumentProcessor
from app.config.settings import settings
from app.utils.logging import logger

class EvaluationRunner:
    """Runs evaluations for agents"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results_cache = {}
        self.dataset = GoldenDataset(config.dataset_name)
        self._setup_output_directories()
        self.generated_user_id = f"test_user_{uuid.uuid4()}"
    
    async def generate_goldens(self) -> str:
        """Generate golden test cases from contexts"""

        if self.config.golden_generation_type == GoldenGenerationType.CONTEXT:

            await self.dataset.generate_from_contexts(
                self.config.contexts,
                {
                    "model": self.config.synthesizer_config.model,
                    "styling_config": self.config.synthesizer_config.styling_config
                },
                self.config.synthesizer_config.max_goldens_per_context
            )
        elif self.config.golden_generation_type == GoldenGenerationType.DOCUMENT:

            if self.config.persist_to_kb:
                test_user_id = await self.persist_documents_to_knowledge_base(
                    document_paths=self.config.document_paths)
            
            await self.dataset.generate_from_documents(
            document_paths=self.config.document_paths,
            synthesizer_config={
                    "model": self.config.synthesizer_config.model,
                    "styling_config": self.config.synthesizer_config.styling_config
                },
            max_goldens_per_context=self.config.max_goldens_per_context,
            document_metadata=self.config.document_metadata,
            default_tools=self.config.default_tools,
            parse_frontmatter=self.config.parse_frontmatter,
            user_id=self.generated_user_id
        )
            
        golden_path = self._get_golden_path(self.config.dataset_file)
        self.dataset.save(str(golden_path))
        logger.info(f"Saved {len(self.dataset.goldens)} goldens to {golden_path}")
    
    async def persist_documents_to_knowledge_base(self, document_paths: Optional[List[str]] = None, 
                                                  store_metadata: bool = False) -> str:
        """Persist documents to knowledge base"""

        agent = CLIAgent(self.config.agent_id)
        agent.user_id = self.generated_user_id
        agent.session_id = f"test_session_{uuid.uuid4()}"
        await agent.initialize()
        
        if agent.knowledge_base:
            logger.info("Evaluation Runner - Persisting documents to knowledge base")
            for path in document_paths:
                content = DocumentProcessor.load_document(path)
                doc_type = DocumentProcessor.detect_type(path)

                metadata, cleaned_content = DocumentProcessor.parse_metadata(content, doc_type)
                
                await agent.knowledge_base.ingest_document(
                    content=cleaned_content,
                    user_id=agent.user_id,
                    namespace_type="documents",
                    doc_type=doc_type,
                    source=path,
                    title=Path(path).name,
                    metadata=metadata if store_metadata else {}
                )
                logger.info(f"  Ingested: {path}")
        else: 
            logger.warning("""Evaluation Runner - No knowledge base available for this agent, 
                           skipping document persistence""")
           
   
    def _get_synthesizer_config(self) -> Dict:
        """Get synthesizer configuration from evaluation config"""
        return {
            "model": self.config.synthesizer_config.model,
            "styling_config": self.config.synthesizer_config.styling_config
        }
    
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
        
        if not test_cases:
            print("""No test cases to evaluate. 
                  Please generate golden test cases first using --generate flag.""")
            return {
                "dataframe": None,
                "raw_results": None,
                "summary": {"error": """No test cases available. 
                            Please generate golden test cases first using --generate flag."""}
            }
        
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

    async def _process_tool_calls(self, tool_calls) -> List[ToolCall]:
        """Process tool calls to generate a list of ToolCall objects"""
        evaluation_tool_calls = []
        retrieval_context = []
        for tool_call in tool_calls:

            tc = ToolCall(name=tool_call["tool_name"], 
                            input_parameters=tool_call.get("arguments", {}), 
                            output=tool_call.get("results", None))
            
            evaluation_tool_calls.append(tc)
            if tc.name in ["search_knowledge_base", 
                           "list_documents"] and tc.output is not None:
                retrieval_context.append(tc.output)
       
        logger.debug(f"""Evaluation Runner - process tool calls:
                    \n tool_calls: {evaluation_tool_calls} 
                    \n retrieval_context: {retrieval_context} """)
            
        return evaluation_tool_calls, retrieval_context

    async def _create_test_case(self, golden) -> LLMTestCase:
        """Create a test case by running the agent"""
        #TODO: Use the agent factory to get the agent
        agent = CLIAgent(self.config.agent_id)
        
        # Use context-specific IDs from additional_metadata if available, otherwise generate new ones
        if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
            agent.user_id = golden.additional_metadata.get('user_id', self.generated_user_id)
            agent.session_id = golden.additional_metadata.get('session_id', str(uuid.uuid4()))
        else:
            agent.user_id = f"test_user_{uuid.uuid4()}"
            agent.session_id = str(uuid.uuid4())
        
        await agent.initialize()    
        agent.provider.config.track_tool_calls = True
        
        response = await agent.chat(golden.input)
        chain_of_thought, cleaned_response = separate_chain_of_thought(response)
        
        agent_tool_calls = agent.provider.get_tool_calls_made()
        tool_calls, retrieval_context_from_tools = await self._process_tool_calls(agent_tool_calls)

        # Use golden's retrieval_context if it exists, otherwise use from tool calls
        logger.info(f"Evaluation Runner - Create Test Case - {golden.retrieval_context}")
        if hasattr(golden, 'retrieval_context') and golden.retrieval_context:
            retrieval_context = golden.retrieval_context
        else:
            retrieval_context = retrieval_context_from_tools

        logger.debug(f"Evaluation Runner - Create Test Case - Retrieval context: {retrieval_context}")

        test_case_params = {
            'input': golden.input,
            'actual_output': cleaned_response,
            'expected_output': golden.expected_output,
            'context': golden.context,
            'expected_tools': golden.expected_tools,
            'tools_called': tool_calls,
            'retrieval_context': retrieval_context,
            'additional_metadata': {
                'agent_id': self.config.agent_id,
                'expected_tool_calls': [
                    {
                        'name': tc.name,
                        'input_parameters': tc.input_parameters,
                        'output': tc.output
                    } for tc in golden.expected_tools
                ],
                'actual_tool_calls': [
                    {
                        'name': tc.name,
                        'input_parameters': tc.input_parameters,
                        'output': tc.output
                    } for tc in tool_calls
                ],
                'chain_of_thought': chain_of_thought,
            }
        }
        
        
        return LLMTestCase(**test_case_params)
    
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
                'expected_tool_calls': metadata.get('expected_tool_calls', []),
                'actual_tool_calls': metadata.get('actual_tool_calls', []),
                'chain_of_thought': metadata.get('chain_of_thought', ''),
                'agent_id': metadata.get('agent_id', self.config.agent_id),
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
            
        df = self._last_results.get('dataframe')
        raw_results = self._last_results.get('raw_results')
        
        # Handle case where no test cases were available
        if df is None or raw_results is None:
            if self._last_results.get('summary', {}).get('error'):
                print(f"\n{self._last_results['summary']['error']}")
            else:
                print("\nNo evaluation results available.")
            return
        
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