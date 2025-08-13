"""Tests for evaluation configuration models"""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock
from deepeval.synthesizer.config import StylingConfig
from deepeval.metrics import BaseMetric

from app.evaluation.config import ContextWithMetadata, SynthesizerConfig, EvaluationConfig, GoldenGenerationType


class TestContextWithMetadata:
    """Tests for ContextWithMetadata model"""

    def test_context_with_metadata_basic_creation(self):
        """Test basic ContextWithMetadata creation"""
        context = ContextWithMetadata(
            context=["Basic context information"],
            tools=["basic_tool"]
        )
        
        assert context.context == ["Basic context information"]
        assert context.tools == ["basic_tool"]
        assert context.expected_output is None
        assert context.retrieval_context is None

    def test_context_with_metadata_with_all_fields(self):
        """Test ContextWithMetadata with all optional fields"""
        context = ContextWithMetadata(
            context=["Knowledge base context"],
            tools=["search_knowledge_base", "list_documents"],
            expected_output="Found relevant information from knowledge base",
            retrieval_context=[
                "Document 1: API documentation",
                "Document 2: Implementation guide"
            ]
        )
        
        assert context.context == ["Knowledge base context"]
        assert context.tools == ["search_knowledge_base", "list_documents"]
        assert context.expected_output == "Found relevant information from knowledge base"
        assert context.retrieval_context == [
            "Document 1: API documentation",
            "Document 2: Implementation guide"
        ]

    def test_context_with_metadata_empty_lists(self):
        """Test ContextWithMetadata with empty lists"""
        context = ContextWithMetadata(
            context=[],
            tools=[],
            expected_output="",
            retrieval_context=[]
        )
        
        assert context.context == []
        assert context.tools == []
        assert context.expected_output == ""
        assert context.retrieval_context == []

    def test_context_with_metadata_validation_missing_required_fields(self):
        """Test validation fails when required fields are missing"""
        with pytest.raises(ValidationError) as exc_info:
            ContextWithMetadata()
        
        # Should require both context and tools
        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'context' in error_fields
        assert 'tools' in error_fields

    def test_context_with_metadata_validation_wrong_types(self):
        """Test validation fails with wrong field types"""
        with pytest.raises(ValidationError):
            ContextWithMetadata(
                context="should be list",  # Wrong type
                tools=["valid_tool"]
            )
        
        with pytest.raises(ValidationError):
            ContextWithMetadata(
                context=["valid context"],
                tools="should be list"  # Wrong type
            )

    def test_context_with_metadata_serialization(self):
        """Test ContextWithMetadata can be serialized and deserialized"""
        original = ContextWithMetadata(
            context=["Serialization test context"],
            tools=["test_tool"],
            expected_output="Test output",
            retrieval_context=["Retrieved content"]
        )
        
        # Test dict conversion
        data = original.model_dump()
        reconstructed = ContextWithMetadata(**data)
        
        assert reconstructed.context == original.context
        assert reconstructed.tools == original.tools
        assert reconstructed.expected_output == original.expected_output
        assert reconstructed.retrieval_context == original.retrieval_context

    def test_context_with_metadata_json_serialization(self):
        """Test ContextWithMetadata JSON serialization"""
        context = ContextWithMetadata(
            context=["JSON test context"],
            tools=["json_tool"],
            expected_output="JSON output",
            retrieval_context=["JSON retrieved content"]
        )
        
        json_str = context.model_dump_json()
        reconstructed = ContextWithMetadata.model_validate_json(json_str)
        
        assert reconstructed.context == context.context
        assert reconstructed.tools == context.tools
        assert reconstructed.expected_output == context.expected_output
        assert reconstructed.retrieval_context == context.retrieval_context

    def test_context_with_metadata_immutability(self):
        """Test that ContextWithMetadata behaves as expected with list fields"""
        context = ContextWithMetadata(
            context=["Original context"],
            tools=["original_tool"],
            retrieval_context=["Original retrieval"]
        )
        
        # Modifying the original lists should not affect the model
        original_context_list = ["Original context"]
        original_tools_list = ["original_tool"]
        original_retrieval_list = ["Original retrieval"]
        
        context = ContextWithMetadata(
            context=original_context_list,
            tools=original_tools_list,
            retrieval_context=original_retrieval_list
        )
        
        # Modify original lists
        original_context_list.append("Modified")
        original_tools_list.append("modified_tool")
        original_retrieval_list.append("Modified retrieval")
        
        # Context should still have original values
        assert context.context == ["Original context"]
        assert context.tools == ["original_tool"]
        assert context.retrieval_context == ["Original retrieval"]


class TestSynthesizerConfig:
    """Tests for SynthesizerConfig model"""

    def test_synthesizer_config_creation(self):
        """Test SynthesizerConfig creation"""
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Test scenario",
            task="Test task"
        )
        
        config = SynthesizerConfig(
            model=mock_model,
            styling_config=styling_config,
            max_goldens_per_context=3
        )
        
        assert config.model == mock_model
        assert config.styling_config == styling_config
        assert config.max_goldens_per_context == 3

    def test_synthesizer_config_default_max_goldens(self):
        """Test SynthesizerConfig uses default max_goldens_per_context"""
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Default test",
            task="Default task"
        )
        
        config = SynthesizerConfig(
            model=mock_model,
            styling_config=styling_config
        )
        
        assert config.max_goldens_per_context == 2  # Default value

    def test_synthesizer_config_validation_missing_fields(self):
        """Test SynthesizerConfig validation with missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            SynthesizerConfig()
        
        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        assert 'model' in error_fields

    def test_synthesizer_config_arbitrary_types(self):
        """Test SynthesizerConfig allows arbitrary types for model"""
        # This is needed because DeepEval models are not Pydantic models
        class CustomModel:
            def __init__(self, name):
                self.name = name
        
        custom_model = CustomModel("custom")
        styling_config = StylingConfig(
            scenario="Custom model test",
            task="Custom model task"
        )
        
        config = SynthesizerConfig(
            model=custom_model,
            styling_config=styling_config
        )
        
        assert config.model.name == "custom"


class TestEvaluationConfig:
    """Tests for EvaluationConfig model"""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for EvaluationConfig"""
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Evaluation test",
            task="Evaluation task"
        )
        synthesizer_config = SynthesizerConfig(
            model=mock_model,
            styling_config=styling_config
        )
        
        mock_metric = Mock(spec=BaseMetric)
        mock_metric.name = "test_metric"
        
        context = ContextWithMetadata(
            context=["Test context"],
            tools=["test_tool"]
        )
        
        return {
            'synthesizer_config': synthesizer_config,
            'metrics': [mock_metric],
            'contexts': [context]
        }

    def test_evaluation_config_creation(self, mock_components):
        """Test EvaluationConfig creation with all fields"""
        config = EvaluationConfig(
            agent_id="test_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="test_dataset",
            dataset_file="test_dataset.pkl",
            results_file="test_results"
        )
        
        assert config.agent_id == "test_agent"
        assert config.synthesizer_config == mock_components['synthesizer_config']
        assert config.metrics == mock_components['metrics']
        assert config.contexts == mock_components['contexts']
        assert config.dataset_name == "test_dataset"
        assert config.dataset_file == "test_dataset.pkl"
        assert config.results_file == "test_results"

    def test_evaluation_config_validation_missing_fields(self):
        """Test EvaluationConfig validation with missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig()
        
        errors = exc_info.value.errors()
        error_fields = [error['loc'][0] for error in errors]
        
        required_fields = [
            'agent_id', 'synthesizer_config', 'metrics', 
            'dataset_name', 'dataset_file', 'results_file'
        ]
        
        for field in required_fields:
            assert field in error_fields

    def test_evaluation_config_with_rag_contexts(self, mock_components):
        """Test EvaluationConfig with RAG-specific contexts"""
        rag_contexts = [
            ContextWithMetadata(
                context=["RAG context 1"],
                tools=["search_knowledge_base"],
                expected_output="RAG output 1",
                retrieval_context=["Retrieved doc 1", "Retrieved doc 2"]
            ),
            ContextWithMetadata(
                context=["RAG context 2"],
                tools=["search_docs", "list_documents"],
                expected_output="RAG output 2",
                retrieval_context=["Retrieved doc 3"]
            )
        ]
        
        config = EvaluationConfig(
            agent_id="knowledge_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=rag_contexts,
            dataset_name="rag_dataset",
            dataset_file="rag_dataset.pkl",
            results_file="rag_results"
        )
        
        assert len(config.contexts) == 2
        assert all(hasattr(ctx, 'retrieval_context') for ctx in config.contexts)
        assert all(ctx.retrieval_context is not None for ctx in config.contexts)

    def test_evaluation_config_empty_contexts_list(self, mock_components):
        """Test EvaluationConfig with empty contexts list"""
        config = EvaluationConfig(
            agent_id="empty_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=[],  # Empty list
            dataset_name="empty_dataset",
            dataset_file="empty_dataset.pkl",
            results_file="empty_results"
        )
        
        assert config.contexts == []

    def test_evaluation_config_empty_metrics_list(self, mock_components):
        """Test EvaluationConfig with empty metrics list"""
        config = EvaluationConfig(
            agent_id="no_metrics_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=[],  # Empty list
            contexts=mock_components['contexts'],
            dataset_name="no_metrics_dataset",
            dataset_file="no_metrics_dataset.pkl",
            results_file="no_metrics_results"
        )
        
        assert config.metrics == []

    def test_evaluation_config_field_descriptions(self, mock_components):
        """Test that EvaluationConfig field descriptions are accessible"""
        config = EvaluationConfig(
            agent_id="desc_test_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="desc_test_dataset",
            dataset_file="desc_test_dataset.pkl",
            results_file="desc_test_results"
        )
        
        # Access field info to verify descriptions exist using Pydantic v2 syntax
        fields = config.model_fields
        assert fields['agent_id'].description is not None
        assert fields['synthesizer_config'].description is not None
        assert fields['metrics'].description is not None
        assert fields['contexts'].description is not None
        assert fields['dataset_name'].description is not None
        assert fields['dataset_file'].description is not None
        assert fields['results_file'].description is not None

    def test_evaluation_config_serialization(self, mock_components):
        """Test EvaluationConfig serialization and deserialization"""
        original = EvaluationConfig(
            agent_id="serialization_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="serialization_dataset",
            dataset_file="serialization_dataset.pkl",
            results_file="serialization_results"
        )
        
        # Test dict conversion (note: model and metrics won't serialize fully due to arbitrary types)
        data = original.model_dump()
        assert data['agent_id'] == "serialization_agent"
        assert data['dataset_name'] == "serialization_dataset"
        assert data['dataset_file'] == "serialization_dataset.pkl"
        assert data['results_file'] == "serialization_results"


class TestGoldenGenerationType:
    """Tests for GoldenGenerationType enum"""

    def test_golden_generation_type_enum_values(self):
        """Test GoldenGenerationType enum has expected values"""
        assert GoldenGenerationType.DOCUMENT == "document"
        assert GoldenGenerationType.CONTEXT == "context"
        assert GoldenGenerationType.SCRATCH == "scratch"
        assert GoldenGenerationType.KNOWLEDGE_BASE == "knowledge_base"

    def test_golden_generation_type_enum_membership(self):
        """Test GoldenGenerationType enum membership"""
        assert "document" in GoldenGenerationType
        assert "context" in GoldenGenerationType
        assert "scratch" in GoldenGenerationType
        assert "knowledge_base" in GoldenGenerationType
        assert "invalid_type" not in GoldenGenerationType

    def test_golden_generation_type_enum_iteration(self):
        """Test GoldenGenerationType enum can be iterated"""
        values = [gen_type.value for gen_type in GoldenGenerationType]
        expected_values = ["document", "context", "scratch", "knowledge_base"]
        assert values == expected_values


class TestEvaluationConfigDocumentFields:
    """Tests for EvaluationConfig document-related fields"""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for EvaluationConfig"""
        mock_model = Mock()
        styling_config = StylingConfig(
            scenario="Document evaluation test",
            task="Document evaluation task"
        )
        synthesizer_config = SynthesizerConfig(
            model=mock_model,
            styling_config=styling_config
        )
        
        mock_metric = Mock(spec=BaseMetric)
        mock_metric.name = "test_metric"
        
        context = ContextWithMetadata(
            context=["Test context"],
            tools=["test_tool"]
        )
        
        return {
            'synthesizer_config': synthesizer_config,
            'metrics': [mock_metric],
            'contexts': [context]
        }

    def test_evaluation_config_with_document_generation_type(self, mock_components):
        """Test EvaluationConfig with document generation type"""
        config = EvaluationConfig(
            agent_id="document_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="document_dataset",
            dataset_file="document_dataset.pkl",
            results_file="document_results",
            golden_generation_type=GoldenGenerationType.DOCUMENT
        )
        
        assert config.golden_generation_type == GoldenGenerationType.DOCUMENT
        assert config.golden_generation_type == "document"

    def test_evaluation_config_with_document_paths_and_metadata(self, mock_components):
        """Test EvaluationConfig with document paths and metadata"""
        document_paths = ["/path/to/doc1.md", "/path/to/doc2.txt"]
        document_metadata = {
            "doc1.md": {
                "tools": ["search_tool", "analysis_tool"],
                "expected_output": "Analysis complete",
                "retrieval_context": ["Context from doc1"]
            },
            "doc2.txt": {
                "tools": ["text_tool"],
                "expected_output": "Text processed"
            }
        }
        
        config = EvaluationConfig(
            agent_id="document_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="document_dataset",
            dataset_file="document_dataset.pkl",
            results_file="document_results",
            golden_generation_type=GoldenGenerationType.DOCUMENT,
            document_paths=document_paths,
            document_metadata=document_metadata
        )
        
        assert config.document_paths == document_paths
        assert config.document_metadata == document_metadata
        assert config.document_metadata["doc1.md"]["tools"] == ["search_tool", "analysis_tool"]
        assert config.document_metadata["doc2.txt"]["expected_output"] == "Text processed"

    def test_evaluation_config_default_values_for_new_fields(self, mock_components):
        """Test EvaluationConfig default values for new document-related fields"""
        config = EvaluationConfig(
            agent_id="default_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="default_dataset",
            dataset_file="default_dataset.pkl",
            results_file="default_results"
        )
        
        # Test default values
        assert config.golden_generation_type == GoldenGenerationType.CONTEXT
        assert config.document_paths is None
        assert config.document_metadata is None
        assert config.default_tools is None
        assert config.use_document_as_retrieval is False
        assert config.parse_frontmatter is True
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_contexts_per_document == 3
        assert config.max_goldens_per_context == 2
        assert config.use_knowledge_base is False
        assert config.persist_to_kb is False

    def test_evaluation_config_with_knowledge_base_settings(self, mock_components):
        """Test EvaluationConfig with knowledge base settings"""
        config = EvaluationConfig(
            agent_id="kb_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="kb_dataset",
            dataset_file="kb_dataset.pkl",
            results_file="kb_results",
            golden_generation_type=GoldenGenerationType.KNOWLEDGE_BASE,
            use_knowledge_base=True,
            persist_to_kb=True,
            chunk_size=500,
            chunk_overlap=100
        )
        
        assert config.golden_generation_type == GoldenGenerationType.KNOWLEDGE_BASE
        assert config.use_knowledge_base is True
        assert config.persist_to_kb is True
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100

    def test_evaluation_config_with_document_processing_settings(self, mock_components):
        """Test EvaluationConfig with document processing settings"""
        default_tools = ["default_search", "default_analysis"]
        
        config = EvaluationConfig(
            agent_id="processing_agent",
            synthesizer_config=mock_components['synthesizer_config'],
            metrics=mock_components['metrics'],
            contexts=mock_components['contexts'],
            dataset_name="processing_dataset",
            dataset_file="processing_dataset.pkl",
            results_file="processing_results",
            golden_generation_type=GoldenGenerationType.DOCUMENT,
            default_tools=default_tools,
            use_document_as_retrieval=True,
            parse_frontmatter=False,
            max_contexts_per_document=5,
            max_goldens_per_context=3
        )
        
        assert config.default_tools == default_tools
        assert config.use_document_as_retrieval is True
        assert config.parse_frontmatter is False
        assert config.max_contexts_per_document == 5
        assert config.max_goldens_per_context == 3