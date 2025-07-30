"""Unit tests for EvaluationRunner class"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import json
import uuid
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset import Golden

from app.evaluation.runner import EvaluationRunner
from app.evaluation.dataset import GoldenDataset


class TestEvaluationRunner:
    """Test suite for EvaluationRunner"""
    
    @pytest.fixture
    def runner(self):
        """Create an EvaluationRunner instance"""
        return EvaluationRunner()
    
    @pytest.fixture
    def mock_golden(self):
        """Create a mock Golden object"""
        golden = Mock(spec=Golden)
        golden.input = "What's the latest news about AI?"
        golden.expected_output = "Here are the latest AI news..."
        golden.context = ["AI news context"]
        golden.expected_tools = [ToolCall(name="searxng__searxng_web_search")]
        return golden
    
    @pytest.fixture
    def mock_dataset(self, mock_golden):
        """Create a mock GoldenDataset"""
        dataset = Mock(spec=GoldenDataset)
        dataset.goldens = [mock_golden]
        dataset.name = "test_dataset"
        return dataset
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics"""
        metric1 = Mock()
        metric1.name = "tool_correctness"
        metric2 = Mock()
        metric2.name = "coherence"
        return [metric1, metric2]
    
    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results"""
        return {
            "test_results": [
                {
                    "name": "test_1",
                    "success": True,
                    "input": "What's the latest news about AI?",
                    "actual_output": "Here are the latest AI news...",
                    "context": [],  # Add context field
                    "additional_metadata": {
                        "expected_tool_names": ["searxng__searxng_web_search"],
                        "actual_tool_names": ["searxng__searxng_web_search"]
                    },
                    "metrics_data": [
                        {
                            "name": "tool_correctness",
                            "success": True,
                            "score": 1.0,
                            "threshold": 0.5,
                            "reason": "Tools match",
                            "evaluation_model": "gpt-4",
                            "error": None,
                            "verbose_logs": ""
                        },
                        {
                            "name": "coherence",
                            "success": True,
                            "score": 0.9,
                            "threshold": 0.7,
                            "reason": "Response is coherent",
                            "evaluation_model": "gpt-4",
                            "error": None,
                            "verbose_logs": ""
                        }
                    ]
                }
            ]
        }
    
    def test_init(self, runner):
        """Test EvaluationRunner initialization"""
        assert hasattr(runner, 'results_cache')
        assert isinstance(runner.results_cache, dict)
        assert len(runner.results_cache) == 0
    
    @pytest.mark.asyncio
    async def test_run_evaluation_success(self, runner, mock_dataset, mock_metrics, mock_evaluation_results):
        """Test successful evaluation run"""
        # Mock the evaluate function
        mock_evaluate_result = Mock()
        mock_evaluate_result.model_dump_json.return_value = json.dumps(mock_evaluation_results)
        
        with patch('app.evaluation.runner.evaluate', return_value=mock_evaluate_result) as mock_evaluate:
            with patch.object(runner, '_create_test_case', new_callable=AsyncMock) as mock_create_test_case:
                # Setup mock test case
                mock_test_case = Mock(spec=LLMTestCase)
                mock_create_test_case.return_value = mock_test_case
                
                # Run evaluation
                results = await runner.run_evaluation("test_agent", mock_dataset, mock_metrics)
                
                # Verify test case creation
                mock_create_test_case.assert_called_once_with("test_agent", mock_dataset.goldens[0])
                
                # Verify evaluate was called
                mock_evaluate.assert_called_once()
                test_cases_arg = mock_evaluate.call_args[1]['test_cases']
                assert len(test_cases_arg) == 1
                assert test_cases_arg[0] == mock_test_case
                
                # Verify results structure
                assert 'raw_results' in results
                assert 'dataframe' in results
                assert 'summary' in results
                assert isinstance(results['dataframe'], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_run_evaluation_multiple_goldens(self, runner, mock_metrics, mock_evaluation_results):
        """Test evaluation with multiple golden test cases"""
        # Create multiple goldens
        goldens = []
        for i in range(3):
            golden = Mock(spec=Golden)
            golden.input = f"Test input {i}"
            golden.expected_output = f"Expected output {i}"
            golden.context = [f"Context {i}"]
            golden.expected_tools = [ToolCall(name=f"tool_{i}")]
            goldens.append(golden)
        
        dataset = Mock(spec=GoldenDataset)
        dataset.goldens = goldens
        
        mock_evaluate_result = Mock()
        mock_evaluate_result.model_dump_json.return_value = json.dumps(mock_evaluation_results)
        
        with patch('app.evaluation.runner.evaluate', return_value=mock_evaluate_result):
            with patch.object(runner, '_create_test_case', new_callable=AsyncMock) as mock_create_test_case:
                mock_create_test_case.return_value = Mock(spec=LLMTestCase)
                
                results = await runner.run_evaluation("test_agent", dataset, mock_metrics)
                
                # Verify create_test_case was called for each golden
                assert mock_create_test_case.call_count == 3
                for i, golden in enumerate(goldens):
                    mock_create_test_case.assert_any_call("test_agent", golden)
    
    @pytest.mark.asyncio
    async def test_create_test_case(self, runner):
        """Test _create_test_case method"""
        # Create mock golden
        golden = Mock(spec=Golden)
        golden.input = "Test input"
        golden.expected_output = "Expected output"
        golden.context = ["Test context"]
        golden.expected_tools = [ToolCall(name="test_tool")]
        
        # Mock CLIAgent
        mock_agent = AsyncMock()
        mock_agent.agent_id = "test_agent"
        mock_agent.session_id = None
        mock_agent.user_id = None
        mock_agent.initialize = AsyncMock()
        mock_agent.chat = AsyncMock(return_value="Agent response")
        mock_agent.provider = Mock()
        mock_agent.provider.config = Mock()
        mock_agent.provider.get_tool_calls_made = Mock(return_value=[{"tool_name": "test_tool"}])
        
        with patch('app.evaluation.runner.CLIAgent', return_value=mock_agent):
            test_case = await runner._create_test_case("test_agent", golden)
            
            # Verify agent initialization
            assert mock_agent.session_id is not None
            assert mock_agent.user_id.startswith("test_user_")
            mock_agent.initialize.assert_called_once()
            assert mock_agent.provider.config.track_tool_calls is True
            
            # Verify chat was called
            mock_agent.chat.assert_called_once_with("Test input")
            
            # Verify test case structure
            assert isinstance(test_case, LLMTestCase)
            assert test_case.input == "Test input"
            assert test_case.actual_output == "Agent response"
            assert test_case.expected_output == "Expected output"
            assert test_case.context == ["Test context"]
            assert len(test_case.expected_tools) == 1
            assert test_case.expected_tools[0].name == "test_tool"
            assert len(test_case.tools_called) == 1
            assert test_case.tools_called[0].name == "test_tool"
            assert test_case.additional_metadata['agent_id'] == "test_agent"
    
    @pytest.mark.asyncio
    async def test_create_test_case_no_tools(self, runner):
        """Test _create_test_case when no tools are called"""
        golden = Mock(spec=Golden)
        golden.input = "Simple question"
        golden.expected_output = "Simple answer"
        golden.context = []
        golden.expected_tools = []
        
        mock_agent = AsyncMock()
        mock_agent.agent_id = "test_agent"
        mock_agent.initialize = AsyncMock()
        mock_agent.chat = AsyncMock(return_value="Simple response")
        mock_agent.provider = Mock()
        mock_agent.provider.config = Mock()
        mock_agent.provider.get_tool_calls_made = Mock(return_value=[])
        
        with patch('app.evaluation.runner.CLIAgent', return_value=mock_agent):
            test_case = await runner._create_test_case("test_agent", golden)
            
            assert len(test_case.tools_called) == 0
            assert test_case.additional_metadata['actual_tool_names'] == []
    
    def test_create_results_dataframe(self, runner, mock_evaluation_results):
        """Test _create_results_dataframe method"""
        df = runner._create_results_dataframe(mock_evaluation_results)
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two metrics for one test
        
        # Verify columns
        expected_columns = [
            'test_name', 'context', 'overall_success', 'input', 'actual_output',
            'expected_tools', 'actual_tools', 'metric_name', 
            'metric_success', 'metric_score', 'metric_reason', 'metric_threshold',
            'metric_evaluation_model', 'metric_error', 'verbose_logs'
        ]
        for col in expected_columns:
            assert col in df.columns
        
        # Verify data
        assert df.iloc[0]['test_name'] == 'test_1'
        assert df.iloc[0]['context'] == '[]'  # Empty context becomes string '[]'
        assert df.iloc[0]['overall_success'] == True
        assert df.iloc[0]['metric_name'] == 'tool_correctness'
        assert df.iloc[0]['metric_score'] == 1.0
        assert df.iloc[0]['metric_reason'] == 'Tools match'
        assert df.iloc[0]['metric_threshold'] == 0.5
        assert df.iloc[0]['metric_evaluation_model'] == 'gpt-4'
        assert df.iloc[0]['metric_error'] is None
        assert df.iloc[0]['verbose_logs'] == ''
        assert df.iloc[1]['metric_name'] == 'coherence'
        assert df.iloc[1]['metric_score'] == 0.9
        assert df.iloc[1]['metric_reason'] == 'Response is coherent'
        assert df.iloc[1]['metric_threshold'] == 0.7
    
    def test_create_results_dataframe_empty(self, runner):
        """Test _create_results_dataframe with empty results"""
        empty_results = {"test_results": []}
        df = runner._create_results_dataframe(empty_results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_create_results_dataframe_missing_fields(self, runner):
        """Test _create_results_dataframe with missing optional fields"""
        results = {
            "test_results": [
                {
                    "name": "test_1",
                    "success": True,
                    "input": "Test input",
                    "metrics_data": [
                        {
                            "name": "metric_1",
                            "success": True,
                            "score": 0.8,
                            "threshold": 0.5,
                            "reason": "Good"
                        }
                    ]
                }
            ]
        }
        
        df = runner._create_results_dataframe(results)
        
        assert len(df) == 1
        assert df.iloc[0]['actual_output'] == ''
        assert df.iloc[0]['expected_tools'] == []
        assert df.iloc[0]['actual_tools'] == []
        assert df.iloc[0]['context'] == '[]'  # Default empty context
        assert df.iloc[0]['metric_evaluation_model'] == 'N/A'  # Default value
        assert df.iloc[0]['metric_error'] is None  # Default value
        assert df.iloc[0]['verbose_logs'] == ''  # Default value
    
    def test_create_summary(self, runner):
        """Test _create_summary method"""
        # Create test DataFrame
        data = {
            'test_name': ['test_1', 'test_1', 'test_2', 'test_2'],
            'overall_success': [True, True, False, False],
            'metric_name': ['tool_correctness', 'coherence', 'tool_correctness', 'coherence'],
            'metric_score': [1.0, 0.9, 0.3, 0.8],
            'metric_success': [True, True, False, True]
        }
        df = pd.DataFrame(data)
        
        summary = runner._create_summary(df)
        
        # Verify summary structure
        assert summary['total_tests'] == 2
        assert summary['passed_tests'] == 1
        
        # Verify metrics summary
        metrics_summary = summary['metrics_summary']
        assert ('metric_score', 'mean') in metrics_summary
        assert ('metric_score', 'min') in metrics_summary
        assert ('metric_score', 'max') in metrics_summary
        assert ('metric_success', 'mean') in metrics_summary
        
        # Verify values for tool_correctness metric
        tool_correctness_mean = metrics_summary[('metric_score', 'mean')]['tool_correctness']
        assert tool_correctness_mean == 0.65  # (1.0 + 0.3) / 2
        
        tool_correctness_min = metrics_summary[('metric_score', 'min')]['tool_correctness']
        assert tool_correctness_min == 0.3
        
        tool_correctness_max = metrics_summary[('metric_score', 'max')]['tool_correctness']
        assert tool_correctness_max == 1.0
    
    def test_create_summary_empty_dataframe(self, runner):
        """Test _create_summary with empty DataFrame"""
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['test_name', 'overall_success', 'metric_name', 'metric_score', 'metric_success'])
        summary = runner._create_summary(df)
        
        assert summary['total_tests'] == 0
        assert summary['passed_tests'] == 0
        # For empty dataframe, metrics_summary will be an empty dict structure
        assert isinstance(summary['metrics_summary'], dict)
    
    @pytest.mark.asyncio
    async def test_run_evaluation_with_errors(self, runner, mock_dataset, mock_metrics):
        """Test evaluation handling when errors occur"""
        with patch.object(runner, '_create_test_case', new_callable=AsyncMock) as mock_create_test_case:
            mock_create_test_case.side_effect = Exception("Agent initialization failed")
            
            with pytest.raises(Exception) as exc_info:
                await runner.run_evaluation("test_agent", mock_dataset, mock_metrics)
            
            assert "Agent initialization failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_test_case_with_unique_ids(self, runner):
        """Test that _create_test_case generates unique session and user IDs"""
        golden = Mock(spec=Golden)
        golden.input = "Test"
        golden.expected_output = "Output"
        golden.context = []
        golden.expected_tools = []
        
        captured_agents = []
        
        def capture_agent(*args, **kwargs):
            agent = AsyncMock()
            agent.agent_id = args[0]
            agent.initialize = AsyncMock()
            agent.chat = AsyncMock(return_value="Response")
            agent.provider = Mock()
            agent.provider.config = Mock()
            agent.provider.get_tool_calls_made = Mock(return_value=[])
            captured_agents.append(agent)
            return agent
        
        with patch('app.evaluation.runner.CLIAgent', side_effect=capture_agent):
            # Create multiple test cases
            test_case1 = await runner._create_test_case("agent1", golden)
            test_case2 = await runner._create_test_case("agent2", golden)
            
            # Verify unique IDs
            assert len(captured_agents) == 2
            assert captured_agents[0].session_id != captured_agents[1].session_id
            assert captured_agents[0].user_id != captured_agents[1].user_id
            assert captured_agents[0].user_id.startswith("test_user_")
            assert captured_agents[1].user_id.startswith("test_user_")
    
    @pytest.mark.asyncio
    async def test_run_evaluation_output_suppression(self, runner, mock_dataset, mock_metrics):
        """Test that evaluation output is properly suppressed"""
        mock_evaluate_result = Mock()
        mock_evaluate_result.model_dump_json.return_value = json.dumps({
            "test_results": [{
                "name": "test_1",
                "success": True,
                "input": "test input",
                "context": [],
                "metrics_data": [{
                    "name": "test_metric",
                    "success": True,
                    "score": 1.0,
                    "threshold": 0.5,
                    "reason": "Test passed",
                    "evaluation_model": "gpt-4",
                    "error": None,
                    "verbose_logs": ""
                }]
            }]
        })
        
        # Capture what would be printed
        import sys
        from io import StringIO
        
        captured_output = StringIO()
        original_stdout = sys.stdout
        
        try:
            sys.stdout = captured_output
            
            with patch('app.evaluation.runner.evaluate', return_value=mock_evaluate_result):
                with patch.object(runner, '_create_test_case', new_callable=AsyncMock) as mock_create:
                    mock_create.return_value = Mock(spec=LLMTestCase)
                    
                    await runner.run_evaluation("test_agent", mock_dataset, mock_metrics)
            
            output = captured_output.getvalue()
            
            # Verify expected output messages
            assert "Evaluating..." in output
            assert "Evaluation complete" in output
            
        finally:
            sys.stdout = original_stdout
    
    def test_create_results_dataframe_with_complex_metadata(self, runner):
        """Test handling of complex additional metadata"""
        results = {
            "test_results": [
                {
                    "name": "complex_test",
                    "success": False,
                    "input": "Complex input",
                    "actual_output": "Complex output",
                    "expected_output": "Expected complex output",
                    "context": ["Some context"],
                    "additional_metadata": {
                        "expected_tool_names": ["tool1", "tool2", "tool3"],
                        "actual_tool_names": ["tool1", "tool4"],
                        "extra_field": "extra_value",
                        "nested": {"key": "value"}
                    },
                    "metrics_data": [
                        {
                            "name": "tool_correctness",
                            "success": False,
                            "score": 0.33,
                            "threshold": 0.5,
                            "reason": "Only 1 of 3 expected tools used",
                            "evaluation_model": "gpt-4",
                            "error": None,
                            "verbose_logs": ""
                        }
                    ]
                }
            ]
        }
        
        df = runner._create_results_dataframe(results)
        
        assert len(df) == 1
        assert df.iloc[0]['test_name'] == 'complex_test'
        assert df.iloc[0]['overall_success'] == False
        assert len(df.iloc[0]['expected_tools']) == 3
        assert len(df.iloc[0]['actual_tools']) == 2
    
    @pytest.mark.asyncio
    async def test_create_test_case_with_multiple_tool_calls(self, runner):
        """Test handling of multiple tool calls in test case creation"""
        golden = Mock(spec=Golden)
        golden.input = "Multi-tool query"
        golden.expected_output = "Multi-tool response"
        golden.context = ["Context 1", "Context 2"]
        golden.expected_tools = [
            ToolCall(name="tool1"),
            ToolCall(name="tool2"),
            ToolCall(name="tool3")
        ]
        
        mock_agent = AsyncMock()
        mock_agent.agent_id = "multi_tool_agent"
        mock_agent.initialize = AsyncMock()
        mock_agent.chat = AsyncMock(return_value="Response using multiple tools")
        mock_agent.provider = Mock()
        mock_agent.provider.config = Mock()
        mock_agent.provider.get_tool_calls_made = Mock(return_value=[
            {"tool_name": "tool1"},
            {"tool_name": "tool2"},
            {"tool_name": "tool4"}  # Different from expected
        ])
        
        with patch('app.evaluation.runner.CLIAgent', return_value=mock_agent):
            test_case = await runner._create_test_case("multi_tool_agent", golden)
            
            # Verify multiple tools handling
            assert len(test_case.tools_called) == 3
            assert test_case.tools_called[0].name == "tool1"
            assert test_case.tools_called[1].name == "tool2"
            assert test_case.tools_called[2].name == "tool4"
            assert test_case.additional_metadata['expected_tool_names'] == ["tool1", "tool2", "tool3"]
            assert test_case.additional_metadata['actual_tool_names'] == ["tool1", "tool2", "tool4"]