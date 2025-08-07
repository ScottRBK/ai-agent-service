"""Tests for CustomOllamaModel class"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call, PropertyMock
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional

from app.evaluation.custom_ollama import CustomOllamaModel


class ResponseSchema(BaseModel):
    """Test Pydantic schema for structured response testing"""
    answer: str
    confidence: float
    sources: List[str]


class MinimalSchema(BaseModel):
    """Minimal test schema with required fields"""
    required_field: str


class ComplexSchema(BaseModel):
    """Complex test schema with various field types"""
    text_field: str
    number_field: int
    float_field: float
    bool_field: bool
    list_field: List[str]
    dict_field: Dict[str, Any]
    optional_field: Optional[str] = None


class TestCustomOllamaModelInitialization:
    """Tests for CustomOllamaModel initialization"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_init_basic_parameters(self, mock_openai, mock_instructor):
        """Test basic initialization with default parameters"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        assert model.model == "mistral:7b"
        assert model.temperature == 0.0
        assert model.client == mock_client
        
        # Verify OpenAI client creation
        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
        # Verify instructor enhancement
        mock_instructor.from_openai.assert_called_once()
        args, kwargs = mock_instructor.from_openai.call_args
        assert kwargs["mode"] == mock_instructor.Mode.JSON

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_init_custom_parameters(self, mock_openai, mock_instructor):
        """Test initialization with custom parameters"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel(
            model="qwen3:8b",
            base_url="http://192.168.1.100:11434",
            temperature=0.7
        )
        
        assert model.model == "qwen3:8b"
        assert model.temperature == 0.7
        
        # Verify URL conversion
        mock_openai.assert_called_once_with(
            base_url="http://192.168.1.100:11434/v1",
            api_key="ollama"
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_init_url_conversion_already_has_v1(self, mock_openai, mock_instructor):
        """Test URL conversion when URL already ends with /v1"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        CustomOllamaModel(
            model="test:model",
            base_url="http://localhost:11434/v1"
        )
        
        # Should not duplicate /v1
        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_init_url_conversion_with_trailing_slash(self, mock_openai, mock_instructor):
        """Test URL conversion with trailing slash"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        CustomOllamaModel(
            model="test:model",
            base_url="http://localhost:11434/"
        )
        
        # Should strip trailing slash and add /v1
        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_init_instructor_configuration(self, mock_openai, mock_instructor):
        """Test that instructor is configured with correct mode"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        mock_instructor.Mode.JSON = "JSON_MODE"
        
        CustomOllamaModel("test:model")
        
        mock_instructor.from_openai.assert_called_once()
        args, kwargs = mock_instructor.from_openai.call_args
        assert kwargs["mode"] == "JSON_MODE"


class TestCustomOllamaModelLoadModel:
    """Tests for load_model method"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_load_model_returns_model_name(self, mock_openai, mock_instructor):
        """Test load_model returns the model name"""
        mock_instructor.from_openai.return_value = Mock()
        
        model = CustomOllamaModel("mistral:7b")
        result = model.load_model()
        
        assert result == "mistral:7b"

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_load_model_different_names(self, mock_openai, mock_instructor):
        """Test load_model with different model names"""
        mock_instructor.from_openai.return_value = Mock()
        
        test_models = ["qwen3:8b", "llama3:instruct", "custom-model:latest"]
        
        for model_name in test_models:
            model = CustomOllamaModel(model_name)
            assert model.load_model() == model_name


class TestCustomOllamaModelGetModelName:
    """Tests for get_model_name method"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_get_model_name_returns_model_name(self, mock_openai, mock_instructor):
        """Test get_model_name returns the model name"""
        mock_instructor.from_openai.return_value = Mock()
        
        model = CustomOllamaModel("mistral:7b")
        result = model.get_model_name()
        
        assert result == "mistral:7b"

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_get_model_name_consistency_with_load_model(self, mock_openai, mock_instructor):
        """Test get_model_name returns same value as load_model"""
        mock_instructor.from_openai.return_value = Mock()
        
        model = CustomOllamaModel("qwen3:8b")
        
        assert model.get_model_name() == model.load_model()


class TestCustomOllamaModelGenerateWithoutSchema:
    """Tests for generate method without schema (plain text generation)"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_without_schema_success(self, mock_openai, mock_instructor):
        """Test generate without schema returns text content"""
        # Setup mocks
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response text"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        # Test generation
        model = CustomOllamaModel("mistral:7b", temperature=0.5)
        result = model.generate("Test prompt")
        
        assert result == "Generated response text"
        
        # Verify API call
        mock_base_client.chat.completions.create.assert_called_once_with(
            model="mistral:7b",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.5
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_without_schema_empty_choices(self, mock_openai, mock_instructor):
        """Test generate without schema when response has no choices"""
        mock_response = Mock()
        mock_response.choices = []
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate("Test prompt")
        
        # Should return string representation of response object
        assert result == str(mock_response)

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_without_schema_no_choices_attr(self, mock_openai, mock_instructor):
        """Test generate without schema when response has no choices attribute"""
        mock_response = Mock(spec=[])  # Mock without choices attribute
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate("Test prompt")
        
        assert result == str(mock_response)

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_without_schema_different_temperatures(self, mock_openai, mock_instructor):
        """Test generate without schema with different temperature values"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        temperatures = [0.0, 0.5, 1.0, 1.5]
        
        for temp in temperatures:
            model = CustomOllamaModel("mistral:7b", temperature=temp)
            model.generate("Test prompt")
            
            # Verify temperature is passed correctly
            last_call = mock_base_client.chat.completions.create.call_args
            assert last_call[1]['temperature'] == temp


class TestCustomOllamaModelGenerateWithSchema:
    """Tests for generate method with schema (structured generation)"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_schema_success(self, mock_openai, mock_instructor):
        """Test generate with schema returns structured response"""
        expected_response = ResponseSchema(
            answer="Test answer",
            confidence=0.95,
            sources=["source1", "source2"]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = expected_response
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b", temperature=0.3)
        result = model.generate("Test prompt", schema=ResponseSchema)
        
        assert result == expected_response
        
        # Verify instructor API call
        mock_client.chat.completions.create.assert_called_once_with(
            model="mistral:7b",
            messages=[{"role": "user", "content": "Test prompt"}],
            response_model=ResponseSchema,
            temperature=0.3,
            max_retries=3
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_schema_retry_failure_fallback(self, mock_openai, mock_instructor):
        """Test generate with schema fallback when retries fail"""
        # Setup client to raise exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        # Test with schema that has required fields
        result = model.generate("Test prompt", schema=MinimalSchema)
        
        # Should create minimal valid instance
        assert isinstance(result, MinimalSchema)
        assert result.required_field == ""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    @patch('builtins.print')
    def test_generate_with_schema_prints_warning_on_fallback(self, mock_print, mock_openai, mock_instructor):
        """Test that warning is printed when fallback is used"""
        mock_client = Mock()
        test_error = Exception("Test API Error")
        mock_client.chat.completions.create.side_effect = test_error
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        model.generate("Test prompt", schema=MinimalSchema)
        
        # Verify warning was printed
        mock_print.assert_called_once_with(f"Warning: Instructor failed after retries: {test_error}")

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_schema_complex_fallback(self, mock_openai, mock_instructor):
        """Test fallback with complex schema containing various field types"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate("Test prompt", schema=ComplexSchema)
        
        assert isinstance(result, ComplexSchema)
        assert result.text_field == ""
        assert result.number_field == 0
        assert result.float_field == 0
        assert result.bool_field == False
        assert result.list_field == []
        assert result.dict_field == {}
        assert result.optional_field is None  # Optional field should remain None

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_schema_fallback_creation_fails(self, mock_openai, mock_instructor):
        """Test that original exception is raised if fallback creation fails"""
        mock_client = Mock()
        original_error = Exception("Original API Error")
        mock_client.chat.completions.create.side_effect = original_error
        mock_instructor.from_openai.return_value = mock_client
        
        # Create a mock schema that will fail during fallback creation
        failing_schema = Mock(spec=BaseModel)
        # Remove model_fields attribute to trigger the fallback failure path
        type(failing_schema).model_fields = PropertyMock(side_effect=AttributeError("no model_fields"))
        
        model = CustomOllamaModel("mistral:7b")
        
        with pytest.raises(Exception) as exc_info:
            model.generate("Test prompt", schema=failing_schema)
        
        assert exc_info.value == original_error

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_schema_handles_list_and_dict_field_types(self, mock_openai, mock_instructor):
        """Test fallback correctly handles list and dict field types with __origin__"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_instructor.from_openai.return_value = mock_client
        
        # Create a schema with list and dict types that have __origin__
        from typing import List, Dict
        
        class TypedSchema(BaseModel):
            list_field: List[str]
            dict_field: Dict[str, int]
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate("Test prompt", schema=TypedSchema)
        
        assert isinstance(result, TypedSchema)
        assert result.list_field == []
        assert result.dict_field == {}


class TestCustomOllamaModelAsyncGenerate:
    """Tests for a_generate async method"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    @pytest.mark.asyncio
    async def test_a_generate_without_schema(self, mock_openai, mock_instructor):
        """Test async generate without schema"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async response"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        # Since a_generate is actually sync wrapped in async, we need to await it
        result = await model.a_generate("Async test prompt")
        
        assert result == "Async response"

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    @pytest.mark.asyncio
    async def test_a_generate_with_schema(self, mock_openai, mock_instructor):
        """Test async generate with schema"""
        expected_response = ResponseSchema(
            answer="Async answer",
            confidence=0.88,
            sources=["async_source"]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = expected_response
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = await model.a_generate("Async test prompt", schema=ResponseSchema)
        
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_a_generate_is_awaitable(self):
        """Test that a_generate can be awaited"""
        with patch('app.evaluation.custom_ollama.instructor') as mock_instructor, \
             patch('app.evaluation.custom_ollama.OpenAI') as mock_openai:
            
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Awaitable response"
            
            mock_base_client = Mock()
            mock_base_client.chat.completions.create.return_value = mock_response
            
            mock_client = Mock()
            mock_client.client = mock_base_client
            mock_instructor.from_openai.return_value = mock_client
            
            model = CustomOllamaModel("mistral:7b")
            result = await model.a_generate("Awaitable test prompt")
            
            assert result == "Awaitable response"

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    @pytest.mark.asyncio
    async def test_a_generate_calls_sync_generate(self, mock_openai, mock_instructor):
        """Test that a_generate calls the sync generate method"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        # Mock the generate method to track calls
        with patch.object(model, 'generate') as mock_generate:
            mock_generate.return_value = "Mocked response"
            
            result = await model.a_generate("Test prompt", schema=ResponseSchema)
            
            # Verify generate was called with correct parameters
            mock_generate.assert_called_once_with("Test prompt", ResponseSchema)
            assert result == "Mocked response"


class TestCustomOllamaModelEdgeCases:
    """Tests for edge cases and error conditions"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_none_prompt(self, mock_openai, mock_instructor):
        """Test generate with None prompt"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response to None"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate(None)
        
        # Should handle None prompt gracefully
        assert result == "Response to None"
        
        # Verify API call with None content
        mock_base_client.chat.completions.create.assert_called_once_with(
            model="mistral:7b",
            messages=[{"role": "user", "content": None}],
            temperature=0.0
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_empty_string_prompt(self, mock_openai, mock_instructor):
        """Test generate with empty string prompt"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response to empty"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate("")
        
        assert result == "Response to empty"

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_generate_with_very_long_prompt(self, mock_openai, mock_instructor):
        """Test generate with very long prompt"""
        long_prompt = "Very long prompt " * 1000  # 17,000 characters
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response to long prompt"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        result = model.generate(long_prompt)
        
        assert result == "Response to long prompt"
        
        # Verify the long prompt was passed correctly
        call_args = mock_base_client.chat.completions.create.call_args
        assert call_args[1]['messages'][0]['content'] == long_prompt

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_extreme_temperature_values(self, mock_openai, mock_instructor):
        """Test model initialization with extreme temperature values"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        # Test extreme values
        extreme_temps = [-1.0, 0.0, 2.0, 100.0]
        
        for temp in extreme_temps:
            model = CustomOllamaModel("mistral:7b", temperature=temp)
            assert model.temperature == temp

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_malformed_base_url(self, mock_openai, mock_instructor):
        """Test initialization with various malformed URLs"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        test_urls = [
            "localhost:11434",  # No protocol
            "http://",  # Incomplete URL
            "http://localhost",  # No port
            "http://localhost:11434/some/path",  # With path
            "https://remote-server.com:8080/v1/extra",  # HTTPS with extra path
        ]
        
        expected_conversions = [
            "localhost:11434/v1",
            "http:/v1",
            "http://localhost/v1",
            "http://localhost:11434/some/path/v1",
            "https://remote-server.com:8080/v1/extra/v1",
        ]
        
        for url, expected in zip(test_urls, expected_conversions):
            model = CustomOllamaModel("test:model", base_url=url)
            
            # Verify the URL conversion
            call_args = mock_openai.call_args
            assert call_args[1]['base_url'] == expected

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_model_name_with_special_characters(self, mock_openai, mock_instructor):
        """Test model names with special characters"""
        mock_client = Mock()
        mock_instructor.from_openai.return_value = mock_client
        
        special_model_names = [
            "model:latest",
            "model-with-dashes:v1.0",
            "model_with_underscores:latest",
            "model.with.dots:v2.0",
            "registry.example.com/model:latest"
        ]
        
        for model_name in special_model_names:
            model = CustomOllamaModel(model_name)
            assert model.model == model_name
            assert model.load_model() == model_name
            assert model.get_model_name() == model_name


class TestCustomOllamaModelIntegration:
    """Integration tests combining multiple components"""

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_full_workflow_without_schema(self, mock_openai, mock_instructor):
        """Test complete workflow without schema"""
        # Setup response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Integration test response"
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        # Test complete flow
        model = CustomOllamaModel("mistral:7b", base_url="http://test-server:11434", temperature=0.8)
        
        # Verify initialization
        assert model.get_model_name() == "mistral:7b"
        assert model.load_model() == "mistral:7b"
        
        # Test generation
        result = model.generate("Integration test prompt")
        assert result == "Integration test response"
        
        # Verify all components were called correctly
        mock_openai.assert_called_once_with(
            base_url="http://test-server:11434/v1",
            api_key="ollama"
        )
        mock_base_client.chat.completions.create.assert_called_once_with(
            model="mistral:7b",
            messages=[{"role": "user", "content": "Integration test prompt"}],
            temperature=0.8
        )

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    @pytest.mark.asyncio
    async def test_full_workflow_with_schema(self, mock_openai, mock_instructor):
        """Test complete workflow with schema"""
        expected_response = ResponseSchema(
            answer="Integration test answer",
            confidence=0.92,
            sources=["integration_source"]
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = expected_response
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("qwen3:8b", temperature=0.5)
        
        # Test both sync and async
        sync_result = model.generate("Test prompt", schema=ResponseSchema)
        async_result = await model.a_generate("Test prompt", schema=ResponseSchema)
        
        assert sync_result == expected_response
        assert async_result == expected_response
        
        # Should have been called twice (sync and async)
        assert mock_client.chat.completions.create.call_count == 2

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_multiple_generations_same_model(self, mock_openai, mock_instructor):
        """Test multiple generations with the same model instance"""
        # Setup different responses
        responses = ["Response 1", "Response 2", "Response 3"]
        mock_response_objects = []
        
        for resp_text in responses:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = resp_text
            mock_response_objects.append(mock_response)
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.side_effect = mock_response_objects
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        # Generate multiple responses
        results = []
        for i in range(3):
            result = model.generate(f"Prompt {i+1}")
            results.append(result)
        
        assert results == responses
        assert mock_base_client.chat.completions.create.call_count == 3

    @patch('app.evaluation.custom_ollama.instructor')
    @patch('app.evaluation.custom_ollama.OpenAI')
    def test_mixed_schema_and_non_schema_generations(self, mock_openai, mock_instructor):
        """Test mixing schema and non-schema generations"""
        # Setup for non-schema response
        mock_text_response = Mock()
        mock_text_response.choices = [Mock()]
        mock_text_response.choices[0].message.content = "Plain text response"
        
        # Setup for schema response
        schema_response = ResponseSchema(
            answer="Structured answer",
            confidence=0.85,
            sources=["mixed_source"]
        )
        
        mock_base_client = Mock()
        mock_base_client.chat.completions.create.return_value = mock_text_response
        
        mock_client = Mock()
        mock_client.client = mock_base_client
        mock_client.chat.completions.create.return_value = schema_response
        mock_instructor.from_openai.return_value = mock_client
        
        model = CustomOllamaModel("mistral:7b")
        
        # Test non-schema generation
        text_result = model.generate("Plain text prompt")
        assert text_result == "Plain text response"
        
        # Test schema generation
        schema_result = model.generate("Structured prompt", schema=ResponseSchema)
        assert schema_result == schema_response
        
        # Verify both clients were used appropriately
        mock_base_client.chat.completions.create.assert_called_once()  # Non-schema call
        mock_client.chat.completions.create.assert_called_once()  # Schema call