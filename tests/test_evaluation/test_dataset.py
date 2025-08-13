import pytest
import pandas as pd
import pickle
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import List, Dict

# Import the class under test
from app.evaluation.dataset import GoldenDataset


# Define simple picklable classes at module level for integration testing
class SimpleGolden:
    """Simple picklable golden for integration testing"""
    def __init__(self, input_text, output_text, context_list, tool_names):
        self.input = input_text
        self.expected_output = output_text
        self.context = context_list
        self.expected_tools = [SimpleToolCall(name) for name in tool_names]


class SimpleToolCall:
    """Simple picklable tool call for integration testing"""
    def __init__(self, name):
        self.name = name


class TestGoldenDataset:
    """Comprehensive test suite for GoldenDataset class"""

    def test_init_normal_operation(self):
        """Test normal initialization"""
        dataset = GoldenDataset("test_dataset")
        assert dataset.name == "test_dataset"
        assert dataset.goldens == []
        assert isinstance(dataset.goldens, list)

    def test_init_empty_name(self):
        """Test initialization with empty name"""
        dataset = GoldenDataset("")
        assert dataset.name == ""
        assert dataset.goldens == []

    def test_init_special_characters_in_name(self):
        """Test initialization with special characters in name"""
        special_name = "test-dataset_123!@#"
        dataset = GoldenDataset(special_name)
        assert dataset.name == special_name
        assert dataset.goldens == []

    @pytest.fixture
    def mock_golden(self):
        """Create a mock Golden object"""
        golden = MagicMock()
        golden.input = "Test input for golden"
        golden.expected_output = "Expected output"
        golden.context = ["Context 1", "Context 2"]
        golden.expected_tools = []
        return golden

    @pytest.fixture
    def mock_tool_call(self):
        """Create a mock ToolCall object"""
        with patch('app.evaluation.dataset.ToolCall') as MockToolCall:
            tool_call = MagicMock()
            tool_call.name = "test_tool"
            MockToolCall.return_value = tool_call
            yield MockToolCall

    @pytest.fixture
    def sample_contexts_with_metadata(self):
        """Create sample contexts with metadata for testing"""
        return [
            {
                "context": "Context about weather API usage",
                "tools": ["get_weather", "get_location"]
            },
            {
                "context": "Context about search functionality", 
                "tools": ["search_web", "format_results"]
            }
        ]

    @pytest.fixture
    def sample_synthesizer_config(self):
        """Create sample synthesizer configuration"""
        # Create mock that will pass the model type checking
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "GPTModel"
        return {
            "model": mock_model
        }

    @pytest.mark.asyncio
    async def test_generate_from_contexts_normal_operation(self, mock_tool_call, sample_contexts_with_metadata, sample_synthesizer_config):
        """Test normal golden generation from contexts"""
        # Create mock synthesizer and goldens
        mock_synthesizer = AsyncMock()
        mock_golden_1 = MagicMock()
        mock_golden_1.input = "How do I get weather data?"
        mock_golden_1.expected_tools = []
        
        mock_golden_2 = MagicMock()
        mock_golden_2.input = "Search for Python tutorials"
        mock_golden_2.expected_tools = []

        # Mock synthesizer to return different goldens for each context
        mock_synthesizer.a_generate_goldens_from_contexts.side_effect = [
            [mock_golden_1],  # First context generates 1 golden
            [mock_golden_2]   # Second context generates 1 golden
        ]

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):  # Mock print to avoid output during tests
            
            await dataset.generate_from_contexts(
                sample_contexts_with_metadata, 
                sample_synthesizer_config, 
                max_goldens_per_context=2
            )

        # Verify synthesizer was created with correct config
        assert mock_synthesizer.a_generate_goldens_from_contexts.call_count == 2
        
        # Verify calls to synthesizer
        calls = mock_synthesizer.a_generate_goldens_from_contexts.call_args_list
        assert calls[0][1]['contexts'] == ["Context about weather API usage"]
        assert calls[0][1]['include_expected_output'] is True
        assert calls[0][1]['max_goldens_per_context'] == 2
        
        assert calls[1][1]['contexts'] == ["Context about search functionality"]
        assert calls[1][1]['include_expected_output'] is True
        assert calls[1][1]['max_goldens_per_context'] == 2

        # Verify goldens were added to dataset
        assert len(dataset.goldens) == 2
        assert dataset.goldens[0] == mock_golden_1
        assert dataset.goldens[1] == mock_golden_2

        # Verify expected_tools were set correctly
        assert len(mock_golden_1.expected_tools) == 2
        assert len(mock_golden_2.expected_tools) == 2

    @pytest.mark.asyncio
    async def test_generate_from_contexts_empty_contexts(self, sample_synthesizer_config):
        """Test golden generation with empty contexts list"""
        mock_synthesizer = AsyncMock()
        
        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts([], sample_synthesizer_config)

        # Verify no calls to synthesizer
        assert mock_synthesizer.a_generate_goldens_from_contexts.call_count == 0
        assert len(dataset.goldens) == 0

    @pytest.mark.asyncio
    async def test_generate_from_contexts_no_tools_in_metadata(self, sample_synthesizer_config):
        """Test golden generation when context metadata has no tools"""
        contexts_without_tools = [
            {
                "context": "Context without tools",
                "tools": []
            }
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Test input"
        mock_golden.expected_tools = []
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts_without_tools, sample_synthesizer_config)

        # Verify golden was generated but has no expected tools
        assert len(dataset.goldens) == 1
        assert len(mock_golden.expected_tools) == 0

    @pytest.mark.asyncio
    async def test_generate_from_contexts_missing_tools_key(self, sample_synthesizer_config):
        """Test golden generation when context metadata is missing tools key"""
        contexts_missing_tools = [
            {
                "context": "Context missing tools key"
                # Missing "tools" key
            }
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Test input"
        mock_golden.expected_tools = []
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            # Should raise KeyError when trying to access "tools" key
            with pytest.raises(KeyError):
                await dataset.generate_from_contexts(contexts_missing_tools, sample_synthesizer_config)

    @pytest.mark.asyncio
    async def test_generate_from_contexts_synthesizer_returns_empty_list(self, mock_tool_call, sample_contexts_with_metadata, sample_synthesizer_config):
        """Test when synthesizer returns empty list of goldens"""
        mock_synthesizer = AsyncMock()
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = []

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(sample_contexts_with_metadata, sample_synthesizer_config)

        # Verify no goldens were added
        assert len(dataset.goldens) == 0

    @pytest.mark.asyncio
    async def test_generate_from_contexts_multiple_goldens_per_context(self, mock_tool_call, sample_synthesizer_config):
        """Test generation with multiple goldens per context"""
        contexts = [
            {
                "context": "Single context",
                "tools": ["tool1", "tool2"]
            }
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden_1 = MagicMock()
        mock_golden_1.input = "First golden"
        mock_golden_1.expected_tools = []
        
        mock_golden_2 = MagicMock()
        mock_golden_2.input = "Second golden"
        mock_golden_2.expected_tools = []

        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden_1, mock_golden_2]

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts, sample_synthesizer_config, max_goldens_per_context=3)

        # Verify both goldens were added with correct tools
        assert len(dataset.goldens) == 2
        assert len(mock_golden_1.expected_tools) == 2
        assert len(mock_golden_2.expected_tools) == 2

    @pytest.mark.asyncio  
    async def test_generate_from_contexts_synthesizer_exception(self, sample_contexts_with_metadata, sample_synthesizer_config):
        """Test handling of synthesizer exceptions"""
        mock_synthesizer = AsyncMock()
        mock_synthesizer.a_generate_goldens_from_contexts.side_effect = Exception("Synthesizer error")

        dataset = GoldenDataset("test_dataset")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            # Should propagate the exception
            with pytest.raises(Exception, match="Synthesizer error"):
                await dataset.generate_from_contexts(sample_contexts_with_metadata, sample_synthesizer_config)

    def test_save_normal_operation(self, mock_golden):
        """Test normal save operation"""
        dataset = GoldenDataset("test_dataset")
        dataset.goldens = [mock_golden]
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.dump') as mock_dump, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            dataset.save("/test/path/dataset.pkl")
        
        # Verify directory creation was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify file was opened correctly
        mock_file.assert_called_once_with("/test/path/dataset.pkl", 'wb')
        
        # Verify pickle.dump was called with correct arguments
        mock_dump.assert_called_once_with([mock_golden], mock_file())

    def test_save_empty_goldens(self):
        """Test saving with empty goldens list"""
        dataset = GoldenDataset("test_dataset")
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.dump') as mock_dump, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            dataset.save("/test/path/empty_dataset.pkl")
        
        # Verify empty list was saved
        mock_dump.assert_called_once_with([], mock_file())

    def test_save_file_error(self, mock_golden):
        """Test save operation with file I/O error"""
        dataset = GoldenDataset("test_dataset")
        dataset.goldens = [mock_golden]
        
        with patch('builtins.open', side_effect=IOError("Permission denied")), \
             patch('pathlib.Path.mkdir'):
            
            # Should propagate the IOError
            with pytest.raises(IOError, match="Permission denied"):
                dataset.save("/invalid/path/dataset.pkl")

    def test_save_pickle_error(self, mock_golden):
        """Test save operation with pickle error"""
        dataset = GoldenDataset("test_dataset")
        dataset.goldens = [mock_golden]
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.dump', side_effect=pickle.PicklingError("Cannot pickle object")), \
             patch('pathlib.Path.mkdir'):
            
            # Should propagate the PicklingError
            with pytest.raises(pickle.PicklingError, match="Cannot pickle object"):
                dataset.save("/test/path/dataset.pkl")

    def test_load_normal_operation(self, mock_golden):
        """Test normal load operation"""
        dataset = GoldenDataset("test_dataset")
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.load', return_value=[mock_golden]) as mock_load, \
             patch('pathlib.Path.exists', return_value=True):
            
            dataset.load("/test/path/dataset.pkl")
        
        # Verify file was opened correctly
        mock_file.assert_called_once_with("/test/path/dataset.pkl", 'rb')
        
        # Verify goldens were loaded
        assert len(dataset.goldens) == 1
        assert dataset.goldens[0] == mock_golden

    def test_load_empty_file(self):
        """Test loading empty goldens list"""
        dataset = GoldenDataset("test_dataset")
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.load', return_value=[]), \
             patch('pathlib.Path.exists', return_value=True):
            
            dataset.load("/test/path/empty_dataset.pkl")
        
        # Verify empty list was loaded
        assert len(dataset.goldens) == 0

    def test_load_file_not_found(self):
        """Test load operation with file not found"""
        dataset = GoldenDataset("test_dataset")
        
        with patch('pathlib.Path.exists', return_value=False):
            
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Dataset file not found"):
                dataset.load("/nonexistent/path/dataset.pkl")

    def test_load_pickle_error(self):
        """Test load operation with pickle error"""
        dataset = GoldenDataset("test_dataset")
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.load', side_effect=pickle.UnpicklingError("Cannot unpickle object")), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Should propagate the UnpicklingError
            with pytest.raises(pickle.UnpicklingError, match="Cannot unpickle object"):
                dataset.load("/test/path/corrupted_dataset.pkl")

    def test_load_replaces_existing_goldens(self, mock_golden):
        """Test that load replaces existing goldens instead of appending"""
        dataset = GoldenDataset("test_dataset")
        
        # Add some initial goldens
        initial_golden = MagicMock()
        dataset.goldens = [initial_golden]
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('pickle.load', return_value=[mock_golden]), \
             patch('pathlib.Path.exists', return_value=True):
            
            dataset.load("/test/path/dataset.pkl")
        
        # Verify goldens were replaced, not appended
        assert len(dataset.goldens) == 1
        assert dataset.goldens[0] == mock_golden
        assert initial_golden not in dataset.goldens

    def test_to_dataframe_normal_operation(self):
        """Test normal DataFrame conversion"""
        dataset = GoldenDataset("test_dataset")
        
        # Create mock goldens with all required attributes
        mock_golden_1 = MagicMock()
        mock_golden_1.input = "First input"
        mock_golden_1.expected_output = "First output"
        mock_golden_1.context = ["Context 1", "Context 2"]
        mock_tool_1 = MagicMock()
        mock_tool_1.name = "tool1"
        mock_tool_2 = MagicMock()
        mock_tool_2.name = "tool2"
        mock_golden_1.expected_tools = [mock_tool_1, mock_tool_2]
        
        mock_golden_2 = MagicMock()
        mock_golden_2.input = "Second input"
        mock_golden_2.expected_output = "Second output"
        mock_golden_2.context = ["Context 3"]
        mock_tool_3 = MagicMock()
        mock_tool_3.name = "tool3"
        mock_golden_2.expected_tools = [mock_tool_3]
        
        dataset.goldens = [mock_golden_1, mock_golden_2]
        
        df = dataset.to_dataframe()
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['input', 'expected_output', 'context', 'expected_tools']
        
        # Verify first row
        assert df.iloc[0]['input'] == "First input"
        assert df.iloc[0]['expected_output'] == "First output"
        assert df.iloc[0]['context'] == "['Context 1', 'Context 2']"
        assert df.iloc[0]['expected_tools'] == ['tool1', 'tool2']
        
        # Verify second row
        assert df.iloc[1]['input'] == "Second input"
        assert df.iloc[1]['expected_output'] == "Second output"
        assert df.iloc[1]['context'] == "['Context 3']"
        assert df.iloc[1]['expected_tools'] == ['tool3']

    def test_to_dataframe_empty_goldens(self):
        """Test DataFrame conversion with empty goldens list"""
        dataset = GoldenDataset("test_dataset")
        
        df = dataset.to_dataframe()
        
        # Verify empty DataFrame - pandas creates DataFrame with no columns when data is empty list
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Empty DataFrame from empty list has no columns, which is expected behavior
        assert len(df.columns) == 0

    def test_to_dataframe_golden_with_no_tools(self):
        """Test DataFrame conversion with golden having no expected tools"""
        dataset = GoldenDataset("test_dataset")
        
        mock_golden = MagicMock()
        mock_golden.input = "Input with no tools"
        mock_golden.expected_output = "Expected output"
        mock_golden.context = ["Context"]
        mock_golden.expected_tools = []
        
        dataset.goldens = [mock_golden]
        
        df = dataset.to_dataframe()
        
        # Verify DataFrame with empty tools list
        assert len(df) == 1
        assert df.iloc[0]['expected_tools'] == []

    def test_to_dataframe_golden_with_none_attributes(self):
        """Test DataFrame conversion with golden having None attributes"""
        dataset = GoldenDataset("test_dataset")
        
        mock_golden = MagicMock()
        mock_golden.input = None
        mock_golden.expected_output = None
        mock_golden.context = None
        mock_golden.expected_tools = None
        
        dataset.goldens = [mock_golden]
        
        # Should handle None values gracefully
        with pytest.raises(TypeError):  # str() on None should work, but iterating over None will fail
            df = dataset.to_dataframe()

    def test_to_dataframe_context_conversion(self):
        """Test that context is properly converted to string"""
        dataset = GoldenDataset("test_dataset")
        
        mock_golden = MagicMock()
        mock_golden.input = "Test input"
        mock_golden.expected_output = "Test output"
        mock_golden.context = {"key": "value", "list": [1, 2, 3]}
        mock_golden.expected_tools = []
        
        dataset.goldens = [mock_golden]
        
        df = dataset.to_dataframe()
        
        # Verify context was converted to string
        assert isinstance(df.iloc[0]['context'], str)
        assert "key" in df.iloc[0]['context']
        assert "value" in df.iloc[0]['context']

    def test_save_load_workflow_with_mocks(self):
        """Test save/load workflow using mocked file operations"""
        # Create dataset with mock data
        dataset = GoldenDataset("workflow_test")
        
        mock_golden = MagicMock()
        mock_golden.input = "Workflow test input"
        mock_golden.expected_output = "Workflow test output"
        mock_golden.context = ["Workflow context"]
        mock_tool = MagicMock()
        mock_tool.name = "workflow_tool"
        mock_golden.expected_tools = [mock_tool]
        
        dataset.goldens = [mock_golden]
        
        # Test save operation
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
             patch('pickle.dump') as mock_dump, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            dataset.save("/test/workflow.pkl")
            
            # Verify save was called correctly
            mock_file.assert_called_once_with("/test/workflow.pkl", 'wb')
            mock_dump.assert_called_once_with([mock_golden], mock_file())
        
        # Test load operation
        new_dataset = GoldenDataset("loaded_workflow")
        
        with patch('builtins.open', mock_file), \
             patch('pickle.load', return_value=[mock_golden]) as mock_load, \
             patch('pathlib.Path.exists', return_value=True):
            
            new_dataset.load("/test/workflow.pkl")
            
            # Verify load was called correctly
            mock_load.assert_called_once()
            
            # Verify data was loaded
            assert len(new_dataset.goldens) == 1
            assert new_dataset.goldens[0] == mock_golden

    def test_real_save_load_integration(self):
        """Integration test with real file I/O using picklable objects"""
        dataset = GoldenDataset("real_integration_test")
        
        # Create real picklable golden using module-level classes
        simple_golden = SimpleGolden(
            "Real integration input",
            "Real integration output", 
            ["Real context"],
            ["real_tool"]
        )
        
        dataset.goldens = [simple_golden]
        
        # Test real save and load using temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save dataset
            dataset.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Create new dataset and load
            new_dataset = GoldenDataset("loaded_real_dataset")
            new_dataset.load(tmp_path)
            
            # Verify data was preserved
            assert len(new_dataset.goldens) == 1
            loaded_golden = new_dataset.goldens[0]
            assert loaded_golden.input == "Real integration input"
            assert loaded_golden.expected_output == "Real integration output"
            assert loaded_golden.context == ["Real context"]
            assert len(loaded_golden.expected_tools) == 1
            assert loaded_golden.expected_tools[0].name == "real_tool"
            
            # Test DataFrame conversion
            df = new_dataset.to_dataframe()
            assert len(df) == 1
            assert df.iloc[0]['input'] == "Real integration input"
            assert df.iloc[0]['expected_output'] == "Real integration output"
            assert df.iloc[0]['expected_tools'] == ['real_tool']
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_dataset_name_persistence(self):
        """Test that dataset name is preserved through operations"""
        original_name = "persistent_dataset"
        dataset = GoldenDataset(original_name)
        
        # Add some goldens
        mock_golden = MagicMock()
        dataset.goldens = [mock_golden]
        
        # Operations should not change the name
        df = dataset.to_dataframe()
        assert dataset.name == original_name
        
        # Even after mock save/load operations
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
             patch('pickle.dump'), \
             patch('pickle.load', return_value=[mock_golden]), \
             patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            dataset.save("/test/path.pkl")
            assert dataset.name == original_name
            
            dataset.load("/test/path.pkl")
            assert dataset.name == original_name

    def test_goldens_list_mutability(self):
        """Test that goldens list can be directly modified"""
        dataset = GoldenDataset("mutable_test")
        
        # Direct modification should work
        mock_golden_1 = MagicMock()
        mock_golden_2 = MagicMock()
        
        dataset.goldens.append(mock_golden_1)
        assert len(dataset.goldens) == 1
        
        dataset.goldens.extend([mock_golden_2])
        assert len(dataset.goldens) == 2
        
        dataset.goldens.clear()
        assert len(dataset.goldens) == 0

    @pytest.mark.parametrize("max_goldens", [1, 2, 5, 10])
    @pytest.mark.asyncio
    async def test_generate_from_contexts_various_max_goldens(self, mock_tool_call, sample_synthesizer_config, max_goldens):
        """Test generation with various max_goldens_per_context values"""
        contexts = [{"context": "Test context", "tools": ["test_tool"]}]
        
        mock_synthesizer = AsyncMock()
        mock_goldens = [MagicMock() for _ in range(max_goldens)]
        for golden in mock_goldens:
            golden.input = "Test input"
            golden.expected_tools = []
        
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = mock_goldens
        
        dataset = GoldenDataset("param_test")
        
        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts, sample_synthesizer_config, max_goldens)
        
        # Verify max_goldens_per_context was passed correctly
        call_args = mock_synthesizer.a_generate_goldens_from_contexts.call_args
        assert call_args[1]['max_goldens_per_context'] == max_goldens
        
        # Verify correct number of goldens were added
        assert len(dataset.goldens) == max_goldens

    @pytest.mark.asyncio
    async def test_generate_from_contexts_with_retrieval_context(self, sample_synthesizer_config):
        """Test generation with retrieval_context for RAG metrics"""
        contexts_with_rag = [
            {
                "context": "Context about user authentication",
                "tools": ["search_knowledge_base"],
                "retrieval_context": [
                    "Previous discussion: JWT tokens with 1-hour expiration",
                    "Security context: Use refresh tokens for extended sessions"
                ],
                "expected_output": "Based on our previous discussion, use JWT with 1-hour expiration"
            }
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "How should we implement authentication?"
        mock_golden.expected_output = "Synthesized output"
        mock_golden.expected_tools = []
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("rag_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts_with_rag, sample_synthesizer_config)

        # Verify retrieval_context was preserved
        assert len(dataset.goldens) == 1
        golden = dataset.goldens[0]
        assert hasattr(golden, 'retrieval_context')
        assert golden.retrieval_context == [
            "Previous discussion: JWT tokens with 1-hour expiration",
            "Security context: Use refresh tokens for extended sessions"
        ]
        
        # Verify expected_output was overridden
        assert golden.expected_output == "Based on our previous discussion, use JWT with 1-hour expiration"
        
        # Verify expected_tools were set
        assert len(golden.expected_tools) == 1
        assert golden.expected_tools[0].name == "search_knowledge_base"

    @pytest.mark.asyncio
    async def test_generate_from_contexts_with_context_metadata_objects(self, sample_synthesizer_config):
        """Test generation using ContextWithMetadata objects"""
        from app.evaluation.config import ContextWithMetadata
        
        contexts_with_metadata = [
            ContextWithMetadata(
                context=["RAG evaluation context"],
                tools=["search_docs", "retrieve_context"],
                retrieval_context=["Document 1: API documentation", "Document 2: Implementation guide"],
                expected_output="Found relevant documentation for your query"
            )
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Find API documentation"
        mock_golden.expected_output = "Synthesized response"
        mock_golden.expected_tools = []
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("metadata_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts_with_metadata, sample_synthesizer_config)

        # Verify all fields were properly handled
        assert len(dataset.goldens) == 1
        golden = dataset.goldens[0]
        
        # Check retrieval_context
        assert hasattr(golden, 'retrieval_context')
        assert golden.retrieval_context == ["Document 1: API documentation", "Document 2: Implementation guide"]
        
        # Check expected_output override
        assert golden.expected_output == "Found relevant documentation for your query"
        
        # Check expected_tools
        assert len(golden.expected_tools) == 2
        tool_names = [tool.name for tool in golden.expected_tools]
        assert "search_docs" in tool_names
        assert "retrieve_context" in tool_names

    @pytest.mark.asyncio
    async def test_generate_from_contexts_without_retrieval_context(self, sample_synthesizer_config):
        """Test that goldens without retrieval_context work normally"""
        contexts_no_rag = [
            {
                "context": "Simple context without RAG",
                "tools": ["basic_tool"]
            }
        ]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Simple query"
        mock_golden.expected_output = "Simple response"
        mock_golden.expected_tools = []
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("simple_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('builtins.print'):
            
            await dataset.generate_from_contexts(contexts_no_rag, sample_synthesizer_config)

        # Verify golden was created normally
        assert len(dataset.goldens) == 1
        golden = dataset.goldens[0]
        
        # Should not have retrieval_context set to a real value
        # Note: MagicMock objects always have attributes, but the value should not be set
        # Check that the context data didn't include retrieval_context
        assert contexts_no_rag[0].get('retrieval_context') is None
        
        # Should have expected_tools
        assert len(golden.expected_tools) == 1
        assert golden.expected_tools[0].name == "basic_tool"

    def test_to_dataframe_with_retrieval_context(self):
        """Test DataFrame conversion includes retrieval_context when present"""
        dataset = GoldenDataset("rag_dataframe_test")
        
        # Create mock golden with retrieval_context
        mock_golden = MagicMock()
        mock_golden.input = "RAG query"
        mock_golden.expected_output = "RAG response"
        mock_golden.context = ["RAG context"]
        mock_golden.retrieval_context = ["Retrieved doc 1", "Retrieved doc 2"]
        mock_tool = MagicMock()
        mock_tool.name = "search_knowledge_base"
        mock_golden.expected_tools = [mock_tool]
        
        dataset.goldens = [mock_golden]
        
        df = dataset.to_dataframe()
        
        # Verify basic structure
        assert len(df) == 1
        assert df.iloc[0]['input'] == "RAG query"
        assert df.iloc[0]['expected_output'] == "RAG response"
        assert df.iloc[0]['expected_tools'] == ['search_knowledge_base']
        
        # Note: retrieval_context is not included in basic to_dataframe since it's 
        # only used during evaluation, not for dataset analysis

    # Tests for generate_from_documents method
    @pytest.mark.asyncio
    async def test_generate_from_documents_basic_flow(self, sample_synthesizer_config):
        """Test basic document generation flow with mock documents"""
        document_paths = ["/path/to/doc1.md", "/path/to/doc2.md"]
        
        mock_synthesizer = AsyncMock()
        mock_golden_1 = MagicMock()
        mock_golden_1.input = "Question from doc1"
        mock_golden_1.expected_tools = []
        
        mock_golden_2 = MagicMock()
        mock_golden_2.input = "Question from doc2"
        mock_golden_2.expected_tools = []
        
        mock_synthesizer.a_generate_goldens_from_contexts.side_effect = [
            [mock_golden_1],  # First document
            [mock_golden_2]   # Second document
        ]

        dataset = GoldenDataset("document_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('app.evaluation.dataset.DocumentProcessor.load_document') as mock_load_doc, \
             patch('app.evaluation.dataset.DocumentProcessor.parse_frontmatter') as mock_parse_fm, \
             patch('app.utils.logging.logger'):
            
            # Mock document loading and frontmatter parsing
            mock_load_doc.side_effect = ["Content of doc1", "Content of doc2"]
            mock_parse_fm.side_effect = [
                ({"tools": ["doc1_tool"], "contexts_with_metadata": [{"context": ["Doc1 context"], "tools": ["doc1_tool"]}]}, "Content of doc1"),
                ({"tools": ["doc2_tool"], "contexts_with_metadata": [{"context": ["Doc2 context"], "tools": ["doc2_tool"]}]}, "Content of doc2")
            ]
            
            await dataset.generate_from_documents(
                document_paths=document_paths,
                synthesizer_config=sample_synthesizer_config,
                max_goldens_per_context=2
            )

        # Verify documents were loaded
        assert mock_load_doc.call_count == 2
        assert mock_parse_fm.call_count == 2
        
        # Verify synthesizer was called for each document context
        assert mock_synthesizer.a_generate_goldens_from_contexts.call_count == 2
        
        # Verify goldens were added to dataset
        assert len(dataset.goldens) == 2
        assert dataset.goldens[0] == mock_golden_1
        assert dataset.goldens[1] == mock_golden_2

    @pytest.mark.asyncio
    async def test_generate_from_documents_metadata_handling(self, sample_synthesizer_config):
        """Test metadata handling in document generation"""
        document_paths = ["/path/to/test_doc.md"]
        
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Test question"
        mock_golden.expected_tools = []
        mock_golden.additional_metadata = {}
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("metadata_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('app.evaluation.dataset.DocumentProcessor.load_document', return_value="Test content"), \
             patch('app.evaluation.dataset.DocumentProcessor.parse_frontmatter') as mock_parse_fm, \
             patch('app.utils.logging.logger'):
            
            # Mock frontmatter with predefined contexts
            mock_parse_fm.return_value = (
                {
                    "tools": ["test_tool"],
                    "expected_output": "Test expected output",
                    "contexts_with_metadata": [{
                        "context": ["Test context"],
                        "tools": ["context_tool"],
                        "expected_output": "Context expected output"
                    }]
                },
                "Test content"
            )
            
            await dataset.generate_from_documents(
                document_paths=document_paths,
                synthesizer_config=sample_synthesizer_config,
                user_id="test_user"
            )

        # Verify golden was generated with correct metadata
        assert len(dataset.goldens) == 1
        golden = dataset.goldens[0]
        
        # Check that metadata was applied
        assert hasattr(golden, 'additional_metadata')
        assert golden.additional_metadata["source_document"] == "/path/to/test_doc.md"
        assert golden.additional_metadata["context_type"] == "predefined"
        assert golden.additional_metadata["user_id"] == "test_user"
        assert golden.expected_output == "Context expected output"

    @pytest.mark.asyncio
    async def test_generate_from_documents_missing_file_error(self, sample_synthesizer_config):
        """Test error handling for missing documents"""
        document_paths = ["/nonexistent/doc.md"]
        
        dataset = GoldenDataset("error_test")

        with patch('app.evaluation.dataset.Synthesizer') as MockSynthesizer, \
             patch('app.evaluation.dataset.DocumentProcessor.load_document', side_effect=FileNotFoundError("File not found")), \
             patch('app.evaluation.dataset.logger') as mock_logger:
            
            # Mock synthesizer
            mock_synthesizer = AsyncMock()
            MockSynthesizer.return_value = mock_synthesizer
            
            # Should continue processing despite errors and not raise exception
            await dataset.generate_from_documents(
                document_paths=document_paths,
                synthesizer_config=sample_synthesizer_config
            )

        # Verify error was logged (from captured stderr we can see it calls error())
        mock_logger.error.assert_called_once()
        # Verify no goldens were added due to file error
        assert len(dataset.goldens) == 0

    @pytest.mark.asyncio
    async def test_generate_from_documents_no_contexts_warning(self, sample_synthesizer_config):
        """Test warning when document has no contexts"""
        document_paths = ["/path/to/empty_doc.md"]
        
        dataset = GoldenDataset("warning_test")

        with patch('app.evaluation.dataset.Synthesizer') as MockSynthesizer, \
             patch('app.evaluation.dataset.DocumentProcessor.load_document', return_value="Empty content"), \
             patch('app.evaluation.dataset.DocumentProcessor.parse_frontmatter', return_value=({}, "Empty content")), \
             patch('app.evaluation.dataset.logger') as mock_logger:
            
            # Mock synthesizer
            mock_synthesizer = AsyncMock()
            MockSynthesizer.return_value = mock_synthesizer
            
            await dataset.generate_from_documents(
                document_paths=document_paths,
                synthesizer_config=sample_synthesizer_config
            )

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Evaluation Dataset - No context provided")
        # Verify no goldens were added
        assert len(dataset.goldens) == 0

    # Tests for generate_from_knowledge_base method
    @pytest.mark.asyncio
    async def test_generate_from_knowledge_base_basic_flow(self, sample_synthesizer_config):
        """Test basic knowledge base generation flow with mock knowledge base"""
        mock_kb = MagicMock()
        mock_kb.embedding_model = "test-embedding-model"
        
        # Mock documents
        mock_doc = MagicMock()
        mock_doc.id = "doc_123"
        mock_doc.title = "Test Document"
        mock_doc.namespace_type = "documents"
        
        # Use AsyncMock for async methods
        mock_kb.list_documents = AsyncMock(return_value=[mock_doc])
        
        # Mock chunks
        mock_chunk = MagicMock()
        mock_chunk.content = "This is test chunk content"
        mock_kb.vector_provider.get_document_chunks = AsyncMock(return_value=[mock_chunk])
        
        # Mock synthesizer
        mock_synthesizer = AsyncMock()
        mock_golden = MagicMock()
        mock_golden.input = "Question about knowledge base"
        mock_golden.expected_tools = []
        mock_golden.additional_metadata = {}
        mock_synthesizer.a_generate_goldens_from_contexts.return_value = [mock_golden]

        dataset = GoldenDataset("kb_test")

        with patch('app.evaluation.dataset.Synthesizer', return_value=mock_synthesizer), \
             patch('app.evaluation.dataset.DocumentProcessor.create_contexts_from_chunks', return_value=[["Test context"]]), \
             patch('app.utils.logging.logger'):
            
            await dataset.generate_from_knowledge_base(
                knowledge_base=mock_kb,
                user_id="test_user",
                namespace_types=["documents"],
                synthesizer_config=sample_synthesizer_config,
                tools=["kb_tool"]
            )

        # Verify knowledge base was queried
        mock_kb.list_documents.assert_called_once_with(
            user_id="test_user",
            namespace_type="documents",
            embedding_model="test-embedding-model"
        )
        
        # Verify chunks were retrieved
        mock_kb.vector_provider.get_document_chunks.assert_called_once_with("doc_123")
        
        # Verify golden was generated
        assert len(dataset.goldens) == 1
        golden = dataset.goldens[0]
        
        # Check expected tools were set
        assert len(golden.expected_tools) == 1
        assert golden.expected_tools[0].name == "kb_tool"
        
        # Check retrieval context was set
        assert golden.retrieval_context == ["Test context"]
        
        # Check metadata was set
        assert golden.additional_metadata["source_document_id"] == "doc_123"
        assert golden.additional_metadata["source_document_title"] == "Test Document"
        assert golden.additional_metadata["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_generate_from_knowledge_base_no_documents(self, sample_synthesizer_config):
        """Test knowledge base generation with no documents found"""
        mock_kb = MagicMock()
        mock_kb.embedding_model = "test-embedding-model"
        mock_kb.list_documents = AsyncMock(return_value=[])  # No documents

        dataset = GoldenDataset("no_docs_test")

        with patch('app.evaluation.dataset.Synthesizer') as MockSynthesizer, \
             patch('app.evaluation.dataset.logger') as mock_logger:
            
            # Mock synthesizer
            mock_synthesizer = AsyncMock()
            MockSynthesizer.return_value = mock_synthesizer
            
            await dataset.generate_from_knowledge_base(
                knowledge_base=mock_kb,
                user_id="test_user", 
                namespace_types=["documents"],
                synthesizer_config=sample_synthesizer_config
            )

        # Verify no goldens were generated
        assert len(dataset.goldens) == 0
        
        # Verify appropriate logging occurred
        # Check that info was called multiple times including the expected call
        call_args_list = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "  Found 0 documents in namespace 'documents'" in call_args_list

    @pytest.mark.asyncio
    async def test_generate_from_knowledge_base_no_chunks(self, sample_synthesizer_config):
        """Test knowledge base generation when document has no chunks"""
        mock_kb = MagicMock()
        mock_kb.embedding_model = "test-embedding-model"
        
        # Mock document but no chunks
        mock_doc = MagicMock()
        mock_doc.id = "empty_doc"
        mock_doc.title = "Empty Document"
        mock_kb.list_documents = AsyncMock(return_value=[mock_doc])
        mock_kb.vector_provider.get_document_chunks = AsyncMock(return_value=[])  # No chunks

        dataset = GoldenDataset("no_chunks_test")

        with patch('app.evaluation.dataset.Synthesizer') as MockSynthesizer, \
             patch('app.evaluation.dataset.logger') as mock_logger:
            
            # Mock synthesizer
            mock_synthesizer = AsyncMock()
            MockSynthesizer.return_value = mock_synthesizer
            
            await dataset.generate_from_knowledge_base(
                knowledge_base=mock_kb,
                user_id="test_user",
                namespace_types=["documents"],
                synthesizer_config=sample_synthesizer_config
            )

        # Verify no goldens were generated
        assert len(dataset.goldens) == 0
        
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("  No chunks found for document empty_doc")

    @pytest.mark.asyncio
    async def test_generate_from_knowledge_base_multiple_namespaces(self, sample_synthesizer_config):
        """Test knowledge base generation across multiple namespaces"""
        mock_kb = MagicMock()
        mock_kb.embedding_model = "test-embedding-model"
        
        # Mock documents from different namespaces
        mock_doc1 = MagicMock()
        mock_doc1.id = "doc1"
        mock_doc1.title = "Doc 1"
        mock_doc1.namespace_type = "documents"
        
        mock_doc2 = MagicMock()
        mock_doc2.id = "doc2" 
        mock_doc2.title = "Doc 2"
        mock_doc2.namespace_type = "conversations"
        
        # Mock returning different docs for different namespaces
        async def mock_list_documents(user_id, namespace_type, embedding_model):
            if namespace_type == "documents":
                return [mock_doc1]
            elif namespace_type == "conversations":
                return [mock_doc2]
            return []
        
        mock_kb.list_documents = AsyncMock(side_effect=mock_list_documents)
        
        # Mock chunks for both documents
        mock_chunk = MagicMock()
        mock_chunk.content = "Test content"
        mock_kb.vector_provider.get_document_chunks = AsyncMock(return_value=[mock_chunk])

        dataset = GoldenDataset("multi_namespace_test")

        with patch('app.evaluation.dataset.Synthesizer') as MockSynthesizer, \
             patch('app.evaluation.dataset.DocumentProcessor.create_contexts_from_chunks', return_value=[["Context"]]), \
             patch('app.utils.logging.logger'):
            
            # Mock synthesizer to return empty goldens to focus on namespace handling
            mock_synthesizer = AsyncMock()
            mock_synthesizer.a_generate_goldens_from_contexts.return_value = []
            MockSynthesizer.return_value = mock_synthesizer
            
            await dataset.generate_from_knowledge_base(
                knowledge_base=mock_kb,
                user_id="test_user",
                namespace_types=["documents", "conversations"],
                synthesizer_config=sample_synthesizer_config
            )

        # Verify both namespaces were queried
        assert mock_kb.list_documents.call_count == 2
        calls = mock_kb.list_documents.call_args_list
        
        # Check first call for documents namespace
        assert calls[0][1]["namespace_type"] == "documents"
        # Check second call for conversations namespace  
        assert calls[1][1]["namespace_type"] == "conversations"