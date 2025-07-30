import pytest
import pandas as pd
from unittest.mock import patch, call
from app.evaluation.evaluation_utils import EvaluationUtils


class TestEvaluationUtils:
    """Test suite for EvaluationUtils class"""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with typical evaluation data"""
        return pd.DataFrame([
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'What is the current weather in New York?',
                'actual_tools': ['get_weather'],
                'expected_tools': ['get_weather'],
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Tools match correctly',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'The weather in New York is sunny.'
            },
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'What is the current weather in New York?',
                'actual_tools': ['get_weather'],
                'expected_tools': ['get_weather'],
                'metric_name': 'Hallucination',
                'metric_success': True,
                'metric_score': 0.9,
                'metric_threshold': 0.5,
                'context': '[]',
                'metric_reason': 'No hallucinations detected',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'The weather in New York is sunny.'
            },
            {
                'test_name': 'test_1',
                'overall_success': False,
                'input': 'Search for information about Python programming',
                'actual_tools': ['get_weather'],
                'expected_tools': ['search_web'],
                'metric_name': 'Tool Correctness',
                'metric_success': False,
                'metric_score': 0.2,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Wrong tool used',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'I cannot search for that information.'
            },
            {
                'test_name': 'test_1',
                'overall_success': False,
                'input': 'Search for information about Python programming',
                'actual_tools': ['get_weather'],
                'expected_tools': ['search_web'],
                'metric_name': 'Hallucination',
                'metric_success': True,
                'metric_score': 0.8,
                'metric_threshold': 0.5,
                'context': '[]',
                'metric_reason': 'Response is factual',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'I cannot search for that information.'
            }
        ])

    @pytest.fixture
    def sample_results_json(self):
        """Create sample results JSON data"""
        return {
            'test_results': [
                {
                    'success': True,
                    'name': 'test_0',
                    'input': 'What is the current weather in New York?'
                },
                {
                    'success': False,
                    'name': 'test_1',
                    'input': 'Search for information about Python programming'
                }
            ]
        }

    @pytest.fixture
    def empty_df(self):
        """Create an empty DataFrame"""
        return pd.DataFrame()

    @pytest.fixture
    def empty_results_json(self):
        """Create empty results JSON"""
        return {'test_results': []}

    @patch('builtins.print')
    def test_print_evaluation_summary_normal_operation(self, mock_print, sample_df, sample_results_json):
        """Test normal operation with valid data"""
        EvaluationUtils.print_evaluation_summary(sample_df, sample_results_json)
        
        # Verify key printed elements
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Check overall results
        assert 'EVALUATION SUMMARY' in printed_text
        assert 'Total tests: 2' in printed_text
        assert 'Passed: 1 (50.0%)' in printed_text
        assert 'Failed: 1 (50.0%)' in printed_text
        
        # Check metric breakdown
        assert 'METRIC BREAKDOWN:' in printed_text
        assert 'Tool Correctness:' in printed_text
        assert 'Hallucination:' in printed_text
        assert 'Pass rate: 1/2 (50.0%)' in printed_text
        assert 'Pass rate: 2/2 (100.0%)' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_with_confident_link(self, mock_print, sample_df):
        """Test with Confident AI link present"""
        results_json = {
            'test_results': [{'success': True}],
            'confident_link': 'https://confident.ai/test123'
        }
        
        EvaluationUtils.print_evaluation_summary(sample_df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        assert 'View in Confident AI: https://confident.ai/test123' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_without_confident_link(self, mock_print, sample_df, sample_results_json):
        """Test without Confident AI link"""
        EvaluationUtils.print_evaluation_summary(sample_df, sample_results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        assert 'View in Confident AI:' not in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_empty_results(self, mock_print, empty_df, empty_results_json):
        """Test with empty test results - should handle gracefully"""
        EvaluationUtils.print_evaluation_summary(empty_df, empty_results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should show no tests found message
        assert 'Total tests: 0' in printed_text
        assert 'NO TESTS FOUND' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_missing_test_results_key(self, mock_print, empty_df):
        """Test with missing test_results key in JSON - should handle gracefully"""
        results_json = {}  # Missing test_results key
        
        EvaluationUtils.print_evaluation_summary(empty_df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should treat missing key as empty list
        assert 'Total tests: 0' in printed_text
        assert 'NO TESTS FOUND' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_single_metric(self, mock_print):
        """Test with single metric to ensure groupby works correctly"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            }
        ])
        
        results_json = {'test_results': [{'success': True}]}
        
        EvaluationUtils.print_evaluation_summary(df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        assert 'Tool Correctness:' in printed_text
        assert 'Pass rate: 1/1 (100.0%)' in printed_text
        assert 'Avg score: 1.000' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_normal_operation(self, mock_print, sample_df):
        """Test verbose summary with normal data"""
        EvaluationUtils.print_evaluation_summary_verbose(sample_df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Check headers
        assert 'EVALUATION RESULTS' in printed_text
        
        # Check test details
        assert 'Test 1: What is the current weather in New York?...' in printed_text
        assert 'Test 2: Search for information about Python programming...' in printed_text
        
        # Check expected and actual tools
        assert "Expected tools: ['get_weather']" in printed_text
        assert "Actual tools:   ['get_weather']" in printed_text
        assert "Expected tools: ['search_web']" in printed_text
        
        # Check metrics with emoji status
        assert '✅Tool:1.0' in printed_text
        assert '✅Hallucination:0.9' in printed_text
        assert '❌Tool:0.2' in printed_text
        
        # Check overall status
        assert '✅ PASSED' in printed_text
        assert '❌ FAILED' in printed_text
        
        # Check summary
        assert 'TOTAL: 1/2 passed (50%)' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_empty_df(self, mock_print, empty_df):
        """Test verbose summary with empty DataFrame - should handle gracefully"""
        EvaluationUtils.print_evaluation_summary_verbose(empty_df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should show no test results message
        assert 'No test results to display.' in printed_text
        assert 'TOTAL: 0/0 passed (0%)' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_single_test(self, mock_print):
        """Test verbose summary with single test"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'Short input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Tool Correctness Score',
                'metric_success': True,
                'metric_score': 1.0,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Tool executed successfully'
            }
        ])
        
        EvaluationUtils.print_evaluation_summary_verbose(df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        assert 'Test 1: Short input...' in printed_text
        assert '✅Tool:1.0' in printed_text  # Score should be rounded properly
        assert 'TOTAL: 1/1 passed (100%)' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_long_input_truncation(self, mock_print):
        """Test that long inputs are properly truncated"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'This is a very long input that should be truncated after sixty characters to ensure readability',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Tool executed successfully'
            }
        ])
        
        EvaluationUtils.print_evaluation_summary_verbose(df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should truncate at 60 characters plus "..."
        assert 'Test 1: This is a very long input that should be truncated after six...' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_multiple_metrics_per_test(self, mock_print):
        """Test with multiple metrics per test"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            },
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Hallucination Detection',
                'metric_success': False,
                'metric_score': 0.3,
                'context': '[]',
                'metric_reason': 'Hallucinations detected',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            },
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Answer Relevancy',
                'metric_success': True,
                'metric_score': 0.8,
                'context': '[]',
                'metric_reason': 'Answer is relevant',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            }
        ])
        
        EvaluationUtils.print_evaluation_summary_verbose(df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should show all metrics separated by pipes
        assert '✅Tool:1.0 | ❌Hallucination:0.3 | ✅Answer:0.8' in printed_text

    def test_evaluation_utils_init(self):
        """Test EvaluationUtils initialization"""
        utils = EvaluationUtils()
        assert utils is not None

    @patch('builtins.print')
    def test_print_evaluation_summary_zero_division_handling(self, mock_print):
        """Test division by zero handling when total_tests is 0 - should handle gracefully"""
        empty_df = pd.DataFrame()
        results_json = {'test_results': []}
        
        EvaluationUtils.print_evaluation_summary(empty_df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should handle zero division gracefully
        assert 'Total tests: 0' in printed_text
        assert 'NO TESTS FOUND' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_metric_aggregation(self, mock_print):
        """Test that metric aggregation works correctly"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            },
            {
                'test_name': 'test_1',
                'metric_name': 'Tool Correctness',
                'metric_success': False,
                'metric_score': 0.2,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Wrong tool used',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            },
            {
                'test_name': 'test_2',
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 0.9,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Good match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            }
        ])
        
        results_json = {'test_results': [{'success': True}, {'success': False}, {'success': True}]}
        
        EvaluationUtils.print_evaluation_summary(df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should aggregate correctly: 2 passed out of 3, average score (1.0 + 0.2 + 0.9) / 3 = 0.7
        assert 'Pass rate: 2/3 (66.7%)' in printed_text
        assert 'Avg score: 0.700' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_no_metrics(self, mock_print):
        """Test DataFrame with no metrics (empty DataFrame after groupby)"""
        # DataFrame with columns but no rows will still work
        df = pd.DataFrame(columns=['test_name', 'metric_name', 'metric_success', 'metric_score', 'metric_threshold'])
        results_json = {'test_results': [{'success': True}]}
        
        EvaluationUtils.print_evaluation_summary(df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should show overall results but no metric breakdown
        assert 'Total tests: 1' in printed_text
        assert 'METRIC BREAKDOWN:' in printed_text
        # No specific metrics should be shown

    @patch('builtins.print')
    def test_print_evaluation_summary_different_thresholds_same_metric(self, mock_print):
        """Test handling of same metric with different thresholds"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 0.9,
                'metric_threshold': 0.8,
                'context': '[]',
                'metric_reason': 'Good match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            },
            {
                'test_name': 'test_1',
                'metric_name': 'Tool Correctness',
                'metric_success': False,
                'metric_score': 0.7,
                'metric_threshold': 0.9,  # Different threshold
                'context': '[]',
                'metric_reason': 'Below threshold',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': ''
            }
        ])
        
        results_json = {'test_results': [{'success': True}, {'success': False}]}
        
        EvaluationUtils.print_evaluation_summary(df, results_json)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should use the first threshold encountered (0.8)
        assert 'Threshold: 0.8' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_missing_columns(self, mock_print):
        """Test verbose summary with DataFrame missing required columns"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'metric_name': 'Tool Correctness',
                'metric_score': 1.0
                # Missing: overall_success, input, actual_tools, expected_tools, metric_success, actual_output
            }
        ])
        
        with pytest.raises(KeyError):
            EvaluationUtils.print_evaluation_summary_verbose(df)

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_non_standard_test_names(self, mock_print):
        """Test verbose summary with non-standard test names"""
        df = pd.DataFrame([
            {
                'test_name': 'custom_test_name',  # Not in format test_X
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            }
        ])
        
        # Should handle ValueError when parsing test name
        with pytest.raises(ValueError):
            EvaluationUtils.print_evaluation_summary_verbose(df)

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_test_name_no_underscore(self, mock_print):
        """Test verbose summary with test names without underscore"""
        df = pd.DataFrame([
            {
                'test_name': 'test0',  # No underscore
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Tool Correctness',
                'metric_success': True,
                'metric_score': 1.0,
                'context': '[]',
                'metric_reason': 'Perfect match',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            }
        ])
        
        # Should raise ValueError when trying to convert 'test0' to int
        with pytest.raises(ValueError, match="invalid literal for int"):
            EvaluationUtils.print_evaluation_summary_verbose(df)

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_metric_name_single_word(self, mock_print):
        """Test verbose summary with single-word metric names"""
        df = pd.DataFrame([
            {
                'test_name': 'test_0',
                'overall_success': True,
                'input': 'Test input',
                'actual_tools': ['tool1'],
                'expected_tools': ['tool1'],
                'metric_name': 'Accuracy',  # Single word
                'metric_success': True,
                'metric_score': 0.95,
                'context': '[]',
                'metric_reason': 'High accuracy',
                'metric_evaluation_model': 'gpt-4',
                'metric_error': None,
                'verbose_logs': '',
                'actual_output': 'Test output'
            }
        ])
        
        EvaluationUtils.print_evaluation_summary_verbose(df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should handle single-word metric names correctly
        assert '✅Accuracy:0.9' in printed_text  # Rounded to 1 decimal

    @patch('builtins.print')
    def test_print_evaluation_summary_verbose_zero_tests(self, mock_print):
        """Test verbose summary division by zero when unique_tests is 0"""
        df = pd.DataFrame(columns=['test_name', 'overall_success', 'input', 'actual_tools', 
                                  'expected_tools', 'metric_name', 'metric_success', 'metric_score',
                                  'context', 'metric_reason', 'metric_evaluation_model', 'metric_error',
                                  'verbose_logs', 'actual_output'])
        
        EvaluationUtils.print_evaluation_summary_verbose(df)
        
        printed_calls = [call.args[0] for call in mock_print.call_args_list]
        printed_text = '\n'.join(printed_calls)
        
        # Should handle zero tests gracefully
        assert 'TOTAL: 0/0 passed (0%)' in printed_text

    @patch('builtins.print')
    def test_print_evaluation_summary_none_test_results(self, mock_print, empty_df):
        """Test with None test_results value"""
        results_json = {'test_results': None}
        
        # Should treat None as empty list
        with pytest.raises(TypeError):  # Can't iterate over None
            EvaluationUtils.print_evaluation_summary(empty_df, results_json)