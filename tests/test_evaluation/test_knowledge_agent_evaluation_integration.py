"""Integration tests for knowledge agent evaluation workflow with RAG metrics"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric

from app.evaluation.config import EvaluationConfig, SynthesizerConfig, ContextWithMetadata
from app.evaluation.runner import EvaluationRunner
from app.evaluation.dataset import GoldenDataset


class TestKnowledgeAgentEvaluationIntegration:
    """Integration tests for knowledge agent evaluation with RAG metrics"""

    @pytest.fixture
    def rag_contexts(self):
        """Create contexts with retrieval_context for RAG evaluation"""
        return [
            ContextWithMetadata(
                context=["User previously discussed JWT authentication implementation"],
                tools=["search_knowledge_base"],
                expected_output="Based on our previous discussion, use JWT tokens with 1-hour expiration",
                retrieval_context=[
                    "Previous conversation: JWT authentication with 1-hour token expiration",
                    "Security context: Use refresh tokens for extended sessions"
                ]
            ),
            ContextWithMetadata(
                context=["Multiple conversations about microservices architecture"],
                tools=["search_knowledge_base", "list_documents"],
                expected_output="Found discussions about service discovery, API gateways, and Kubernetes",
                retrieval_context=[
                    "Conversation 1: Service discovery using Consul",
                    "Conversation 2: API gateway patterns with Kong",
                    "Conversation 3: Kubernetes deployment strategies"
                ]
            )
        ]

    @pytest.fixture
    def mock_rag_metrics(self):
        """Create mock RAG-specific metrics"""
        faithfulness_metric = Mock(spec=BaseMetric)
        faithfulness_metric.name = "faithfulness"
        faithfulness_metric.threshold = 0.7
        
        contextual_relevancy_metric = Mock(spec=BaseMetric)
        contextual_relevancy_metric.name = "contextual_relevancy"
        contextual_relevancy_metric.threshold = 0.7
        
        tool_correctness_metric = Mock(spec=BaseMetric)
        tool_correctness_metric.name = "tool_correctness"
        tool_correctness_metric.threshold = 0.9
        
        return [faithfulness_metric, contextual_relevancy_metric, tool_correctness_metric]

    @pytest.fixture
    def knowledge_agent_config(self, rag_contexts, mock_rag_metrics):
        """Create evaluation config for knowledge agent with RAG metrics"""
        from deepeval.synthesizer.config import StylingConfig
        
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Knowledge base evaluation",
            task="Generate queries for knowledge base testing"
        )
        
        return EvaluationConfig(
            agent_id="knowledge_agent",
            synthesizer_config=SynthesizerConfig(
                model=mock_model,
                styling_config=styling_config,
                max_goldens_per_context=2
            ),
            metrics=mock_rag_metrics,
            contexts=rag_contexts,
            dataset_name="knowledge_agent_test",
            dataset_file="knowledge_agent_test.pkl",
            results_file="knowledge_agent_results"
        )

    @pytest.mark.asyncio
    async def test_golden_generation_with_rag_context(self, knowledge_agent_config):
        """Test that golden generation preserves RAG context metadata"""
        runner = EvaluationRunner(knowledge_agent_config)
        
        # Mock synthesizer to return goldens
        mock_golden_1 = Mock(spec=Golden)
        mock_golden_1.input = "How should we implement authentication?"
        mock_golden_1.expected_output = "Synthesized output 1"
        mock_golden_1.expected_tools = []
        
        mock_golden_2 = Mock(spec=Golden)
        mock_golden_2.input = "What microservices patterns should we use?"
        mock_golden_2.expected_output = "Synthesized output 2"
        mock_golden_2.expected_tools = []
        
        mock_synthesizer = AsyncMock()
        mock_synthesizer.a_generate_goldens_from_contexts.side_effect = [
            [mock_golden_1],  # First context
            [mock_golden_2]   # Second context
        ]
        
        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'), \
             patch.object(runner, '_get_golden_path') as mock_path, \
             patch('pathlib.Path.mkdir'), \
             patch('pickle.dump'):
            
            mock_path.return_value = "/tmp/test.pkl"
            
            await runner.generate_goldens()
            
            # Verify both contexts were processed
            assert mock_synthesizer.a_generate_goldens_from_contexts.call_count == 2
            
            # Verify goldens have RAG metadata
            assert len(runner.dataset.goldens) == 2
            
            # Check first golden
            golden_1 = runner.dataset.goldens[0]
            assert hasattr(golden_1, 'retrieval_context')
            assert golden_1.retrieval_context == [
                "Previous conversation: JWT authentication with 1-hour token expiration",
                "Security context: Use refresh tokens for extended sessions"
            ]
            assert golden_1.expected_output == "Based on our previous discussion, use JWT tokens with 1-hour expiration"
            assert len(golden_1.expected_tools) == 1
            assert golden_1.expected_tools[0].name == "search_knowledge_base"
            
            # Check second golden
            golden_2 = runner.dataset.goldens[1]
            assert hasattr(golden_2, 'retrieval_context')
            assert golden_2.retrieval_context == [
                "Conversation 1: Service discovery using Consul",
                "Conversation 2: API gateway patterns with Kong",
                "Conversation 3: Kubernetes deployment strategies"
            ]
            assert golden_2.expected_output == "Found discussions about service discovery, API gateways, and Kubernetes"
            assert len(golden_2.expected_tools) == 2

    @pytest.mark.asyncio
    async def test_rag_test_case_creation(self, knowledge_agent_config):
        """Test that test cases include retrieval_context for RAG metrics"""
        runner = EvaluationRunner(knowledge_agent_config)
        
        # Create a golden with retrieval_context
        golden = Mock(spec=Golden)
        golden.input = "What did we discuss about authentication?"
        golden.expected_output = "Based on our previous discussion, use JWT tokens"
        golden.context = ["Knowledge base context"]
        golden.expected_tools = [ToolCall(name="search_knowledge_base")]
        golden.retrieval_context = [
            "Previous conversation: JWT authentication implementation",
            "Security considerations: Token expiration and refresh"
        ]
        
        # Mock agent response
        mock_agent = AsyncMock()
        mock_agent.agent_id = "knowledge_agent"
        mock_agent.initialize = AsyncMock()
        mock_agent.chat = AsyncMock(return_value="Found our previous authentication discussion")
        mock_agent.provider = Mock()
        mock_agent.provider.config = Mock()
        mock_agent.provider.get_tool_calls_made = Mock(return_value=[
            {"tool_name": "search_knowledge_base"}
        ])
        
        with patch('app.evaluation.runner.CLIAgent', return_value=mock_agent):
            test_case = await runner._create_test_case(golden)
            
            # Verify test case has all RAG fields
            assert isinstance(test_case, LLMTestCase)
            assert test_case.input == "What did we discuss about authentication?"
            assert test_case.actual_output == "Found our previous authentication discussion"
            assert test_case.expected_output == "Based on our previous discussion, use JWT tokens"
            assert test_case.context == ["Knowledge base context"]
            
            # Verify retrieval_context is included
            assert hasattr(test_case, 'retrieval_context')
            assert test_case.retrieval_context == [
                "Previous conversation: JWT authentication implementation",
                "Security considerations: Token expiration and refresh"
            ]
            
            # Verify tools
            assert len(test_case.expected_tools) == 1
            assert test_case.expected_tools[0].name == "search_knowledge_base"
            assert len(test_case.tools_called) == 1
            assert test_case.tools_called[0].name == "search_knowledge_base"

    @pytest.mark.asyncio
    async def test_full_rag_evaluation_workflow(self, knowledge_agent_config):
        """Test complete RAG evaluation workflow from golden generation to results"""
        runner = EvaluationRunner(knowledge_agent_config)
        
        # Setup mock golden with RAG context
        mock_golden = Mock(spec=Golden)
        mock_golden.input = "What patterns did we discuss for microservices?"
        mock_golden.expected_output = "Service discovery and API gateway patterns"
        mock_golden.context = ["Microservices architecture context"]
        mock_golden.expected_tools = [ToolCall(name="search_knowledge_base")]
        mock_golden.retrieval_context = [
            "Architecture discussion: Service discovery patterns",
            "Implementation notes: API gateway configuration"
        ]
        
        runner.dataset.goldens = [mock_golden]
        
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.agent_id = "knowledge_agent"
        mock_agent.initialize = AsyncMock()
        mock_agent.chat = AsyncMock(return_value="Based on our architecture discussions, we covered service discovery with Consul and API gateway patterns")
        mock_agent.provider = Mock()
        mock_agent.provider.config = Mock()
        mock_agent.provider.get_tool_calls_made = Mock(return_value=[
            {"tool_name": "search_knowledge_base"}
        ])
        
        # Mock evaluation results
        mock_evaluation_results = {
            "test_results": [
                {
                    "name": "test_0",
                    "success": True,
                    "input": "What patterns did we discuss for microservices?",
                    "actual_output": "Based on our architecture discussions, we covered service discovery with Consul and API gateway patterns",
                    "context": ["Microservices architecture context"],
                    "retrieval_context": [
                        "Architecture discussion: Service discovery patterns",
                        "Implementation notes: API gateway configuration"
                    ],
                    "additional_metadata": {
                        "expected_tool_names": ["search_knowledge_base"],
                        "actual_tool_names": ["search_knowledge_base"],
                        "agent_id": "knowledge_agent"
                    },
                    "metrics_data": [
                        {
                            "name": "faithfulness",
                            "success": True,
                            "score": 0.9,
                            "threshold": 0.7,
                            "reason": "Response is faithful to retrieved context",
                            "evaluation_model": "mistral:7b",
                            "error": None,
                            "verbose_logs": ""
                        },
                        {
                            "name": "contextual_relevancy",
                            "success": True,
                            "score": 0.85,
                            "threshold": 0.7,
                            "reason": "Retrieved context is relevant to query",
                            "evaluation_model": "mistral:7b",
                            "error": None,
                            "verbose_logs": ""
                        },
                        {
                            "name": "tool_correctness",
                            "success": True,
                            "score": 1.0,
                            "threshold": 0.9,
                            "reason": "Correct tool used for knowledge base search",
                            "evaluation_model": "mistral:7b",
                            "error": None,
                            "verbose_logs": ""
                        }
                    ]
                }
            ]
        }
        
        mock_evaluate_result = Mock()
        mock_evaluate_result.model_dump_json.return_value = json.dumps(mock_evaluation_results)
        
        with patch('app.evaluation.runner.CLIAgent', return_value=mock_agent), \
             patch('app.evaluation.runner.evaluate', return_value=mock_evaluate_result), \
             patch.object(runner, '_get_results_path') as mock_results_path, \
             patch('pandas.DataFrame.to_pickle') as mock_to_pickle, \
             patch('builtins.print'):
            
            mock_results_path.return_value = "/tmp/results.pkl"
            
            results = await runner.run_evaluation()
            
            # Verify results structure
            assert 'raw_results' in results
            assert 'dataframe' in results
            assert 'summary' in results
            
            # Verify RAG metrics in results
            df = results['dataframe']
            assert len(df) == 3  # Three metrics
            
            metric_names = df['metric_name'].tolist()
            assert 'faithfulness' in metric_names
            assert 'contextual_relevancy' in metric_names
            assert 'tool_correctness' in metric_names
            
            # Verify RAG-specific scores
            faithfulness_row = df[df['metric_name'] == 'faithfulness'].iloc[0]
            assert faithfulness_row['metric_score'] == 0.9
            assert faithfulness_row['metric_success'] == True
            
            # Verify summary includes all tests passed
            summary = results['summary']
            assert summary['total_tests'] == 1
            assert summary['passed_tests'] == 1

    @pytest.mark.asyncio
    async def test_mixed_rag_and_regular_contexts(self, mock_rag_metrics):
        """Test evaluation with mix of RAG and regular contexts"""
        mixed_contexts = [
            ContextWithMetadata(
                context=["RAG context with retrieval"],
                tools=["search_knowledge_base"],
                expected_output="Retrieved response",
                retrieval_context=["Retrieved document content"]
            ),
            ContextWithMetadata(
                context=["Regular context without retrieval"],
                tools=["regular_tool"],
                expected_output="Regular response"
                # No retrieval_context
            )
        ]
        
        from deepeval.synthesizer.config import StylingConfig
        
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Mixed evaluation",
            task="Generate mixed RAG and regular queries"
        )
        
        config = EvaluationConfig(
            agent_id="mixed_agent",
            synthesizer_config=SynthesizerConfig(
                model=mock_model,
                styling_config=styling_config,
                max_goldens_per_context=1
            ),
            metrics=mock_rag_metrics,
            contexts=mixed_contexts,
            dataset_name="mixed_test",
            dataset_file="mixed_test.pkl",
            results_file="mixed_results"
        )
        
        runner = EvaluationRunner(config)
        
        # Mock goldens
        mock_rag_golden = Mock(spec=Golden)
        mock_rag_golden.input = "RAG query"
        mock_rag_golden.expected_output = "RAG response"
        mock_rag_golden.expected_tools = []
        
        mock_regular_golden = Mock(spec=Golden)
        mock_regular_golden.input = "Regular query"
        mock_regular_golden.expected_output = "Regular response"
        mock_regular_golden.expected_tools = []
        
        mock_synthesizer = AsyncMock()
        mock_synthesizer.a_generate_goldens_from_contexts.side_effect = [
            [mock_rag_golden],
            [mock_regular_golden]
        ]
        
        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'), \
             patch.object(runner, '_get_golden_path') as mock_path, \
             patch('pathlib.Path.mkdir'), \
             patch('pickle.dump'):
            
            mock_path.return_value = "/tmp/mixed_test.pkl"
            
            await runner.generate_goldens()
            
            # Verify goldens have appropriate metadata
            assert len(runner.dataset.goldens) == 2
            
            # RAG golden should have retrieval_context
            rag_golden = runner.dataset.goldens[0]
            assert hasattr(rag_golden, 'retrieval_context')
            assert rag_golden.retrieval_context == ["Retrieved document content"]
            
            # Regular golden should not have retrieval_context
            regular_golden = runner.dataset.goldens[1]
            assert not hasattr(regular_golden, 'retrieval_context') or regular_golden.retrieval_context is None

    def test_rag_dataframe_creation(self, knowledge_agent_config):
        """Test that DataFrame creation handles RAG evaluation results correctly"""
        runner = EvaluationRunner(knowledge_agent_config)
        
        rag_results = {
            "test_results": [
                {
                    "name": "rag_test_1",
                    "success": True,
                    "input": "Knowledge base query",
                    "actual_output": "Retrieved information from knowledge base",
                    "context": ["KB context"],
                    "additional_metadata": {
                        "expected_tool_names": ["search_knowledge_base"],
                        "actual_tool_names": ["search_knowledge_base"]
                    },
                    "metrics_data": [
                        {
                            "name": "faithfulness",
                            "success": True,
                            "score": 0.95,
                            "threshold": 0.7,
                            "reason": "Response is faithful to retrieved context",
                            "evaluation_model": "mistral:7b",
                            "error": None,
                            "verbose_logs": ""
                        },
                        {
                            "name": "contextual_precision",
                            "success": False,
                            "score": 0.6,
                            "threshold": 0.7,
                            "reason": "Some irrelevant context included",
                            "evaluation_model": "mistral:7b",
                            "error": None,
                            "verbose_logs": ""
                        }
                    ]
                }
            ]
        }
        
        df = runner._create_results_dataframe(rag_results)
        
        # Verify DataFrame structure
        assert len(df) == 2  # Two metrics
        
        # Verify both RAG metrics are present
        metric_names = df['metric_name'].tolist()
        assert 'faithfulness' in metric_names
        assert 'contextual_precision' in metric_names
        
        # Verify success/failure handling
        faithfulness_row = df[df['metric_name'] == 'faithfulness'].iloc[0]
        assert faithfulness_row['metric_success'] == True
        assert faithfulness_row['metric_score'] == 0.95
        
        precision_row = df[df['metric_name'] == 'contextual_precision'].iloc[0]
        assert precision_row['metric_success'] == False
        assert precision_row['metric_score'] == 0.6
        
        # Verify summary creation handles mixed results
        summary = runner._create_summary(df)
        assert summary['total_tests'] == 1
        assert summary['passed_tests'] == 1  # Overall test passed despite one metric failing