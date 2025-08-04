"""
Unit tests for Ollama provider re-ranking functionality.

This module tests the Ollama provider's re-ranking capabilities using specialized
re-ranking models like dengcao/Qwen3-Reranker-4B:Q8_0.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.providers.ollama import OllamaProvider
from app.models.providers import OllamaConfig
from app.core.providers.base import ProviderConnectionError


class TestOllamaReranking:
    """Test Ollama provider re-ranking functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock Ollama configuration."""
        return OllamaConfig(
            name="test-ollama",
            base_url="http://localhost:11434",
            model_list=["dengcao/Qwen3-Reranker-4B:Q8_0", "llama3.1:8b"]
        )
    
    @pytest.fixture
    def ollama_provider(self, mock_config):
        """Create Ollama provider with mocked client."""
        with patch('app.core.providers.ollama.AsyncClient'):
            provider = OllamaProvider(mock_config)
            provider.client = Mock()
            provider.record_successful_call = AsyncMock()
            return provider
    
    @pytest.mark.asyncio
    async def test_rerank_with_specialized_model_simple_scores(self, ollama_provider):
        """Test re-ranking with specialized model returning simple numeric scores."""
        # Mock responses for each candidate with simple numeric scores
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "0.95"},  # High relevance
            {"response": "0.15"},  # Low relevance
            {"response": "0.72"}   # Medium relevance
        ])
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "How to implement JWT authentication?",
            [
                "JWT tokens are a secure way to handle authentication in web applications...",
                "The weather today is sunny with some clouds in the afternoon...",
                "Authentication can be implemented using various methods including JWT..."
            ]
        )
        
        assert len(scores) == 3
        assert scores[0] == 0.95
        assert scores[1] == 0.15
        assert scores[2] == 0.72
        
        # Verify correct model and options were used
        calls = ollama_provider.client.generate.call_args_list
        assert len(calls) == 3
        
        for call in calls:
            assert call[1]['model'] == "dengcao/Qwen3-Reranker-4B:Q8_0"
            assert call[1]['options']['temperature'] == 0.0
            assert call[1]['options']['num_predict'] == 5
            assert call[1]['options']['stop'] == ["\n", " ", "<|"]
        
        # Verify record_successful_call was called
        ollama_provider.record_successful_call.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rerank_with_formatted_scores(self, ollama_provider):
        """Test re-ranking with various score output formats."""
        # Mock responses with different score formats
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "Score: 0.89"},        # Colon-separated format
            {"response": "Relevance: 0.23"},    # Different prefix
            {"response": "0.67"}                # Simple numeric
        ])
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "machine learning algorithms",
            ["neural networks", "weather forecast", "classification methods"]
        )
        
        assert len(scores) == 3
        assert scores[0] == 0.89
        assert scores[1] == 0.23
        assert scores[2] == 0.67
    
    @pytest.mark.asyncio
    async def test_rerank_with_percentage_scores(self, ollama_provider):
        """Test re-ranking with percentage format scores."""
        # Mock responses with percentage format - implementation extracts first numeric value
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "85%"},  # Will extract 85, then normalize to 1.0 (clamped)
            {"response": "Relevance: 12%"},  # Will extract 12, then normalize to 1.0 (clamped)
            {"response": "Score: 96%"}  # Will extract 96, then normalize to 1.0 (clamped)
        ])
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "python programming",
            ["python syntax guide", "java tutorial", "python frameworks"]
        )
        
        assert len(scores) == 3
        # Implementation extracts numeric values and clamps to 0-1, so percentages become 1.0
        assert scores[0] == 1.0  # 85 -> clamped to 1.0
        assert scores[1] == 1.0  # 12 -> clamped to 1.0 
        assert scores[2] == 1.0  # 96 -> clamped to 1.0
    
    @pytest.mark.asyncio
    async def test_rerank_score_normalization(self, ollama_provider):
        """Test that scores are normalized to 0-1 range."""
        # Mock responses with out-of-range scores
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "1.5"},   # Above 1.0
            {"response": "-0.2"},  # Below 0.0
            {"response": "0.5"}    # Normal range
        ])
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "test query",
            ["doc1", "doc2", "doc3"]
        )
        
        assert len(scores) == 3
        assert scores[0] == 1.0  # Clamped to 1.0
        assert scores[1] == 0.0  # Clamped to 0.0
        assert scores[2] == 0.5  # Unchanged
    
    @pytest.mark.asyncio
    async def test_rerank_malformed_response_handling(self, ollama_provider):
        """Test handling of malformed responses from re-ranking model."""
        # Mock responses with malformed/unparseable content
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "not a number"},
            {"response": "Score: invalid"},
            {"response": "0.75"}  # Valid response
        ])
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "test query",
            ["doc1", "doc2", "doc3"]
        )
        
        assert len(scores) == 3
        assert scores[0] == 0.5  # Default score for malformed response
        assert scores[1] == 0.5  # Default score for malformed response
        assert scores[2] == 0.75  # Parsed correctly
    
    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self, ollama_provider):
        """Test re-ranking with empty candidates list."""
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "test query",
            []
        )
        
        assert scores == []
        # Should not call the model for empty candidates
        ollama_provider.client.generate.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_rerank_single_candidate(self, ollama_provider):
        """Test re-ranking with single candidate."""
        ollama_provider.client.generate = AsyncMock(return_value={"response": "0.88"})
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "machine learning",
            ["A comprehensive guide to machine learning algorithms"]
        )
        
        assert len(scores) == 1
        assert scores[0] == 0.88
        
        # Verify the prompt format
        call_args = ollama_provider.client.generate.call_args
        prompt = call_args[1]['prompt']
        assert "Query: machine learning" in prompt
        assert "Document: A comprehensive guide to machine learning algorithms" in prompt
    
    @pytest.mark.asyncio
    async def test_rerank_model_failure(self, ollama_provider):
        """Test graceful handling when re-ranking model fails."""
        ollama_provider.client.generate = AsyncMock(side_effect=Exception("Model not available"))
        
        scores = await ollama_provider.rerank(
            "nonexistent-rerank-model",
            "test query",
            ["doc1", "doc2", "doc3"]
        )
        
        # Should return default scores for all candidates
        assert scores == [0.5, 0.5, 0.5]
        
        # Should not call record_successful_call on failure
        ollama_provider.record_successful_call.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_rerank_client_not_initialized(self, ollama_provider):
        """Test re-ranking when client is not initialized."""
        ollama_provider.client = None
        
        with pytest.raises(ProviderConnectionError) as exc_info:
            await ollama_provider.rerank(
                "dengcao/Qwen3-Reranker-4B:Q8_0",
                "test query",
                ["doc1", "doc2"]
            )
        
        assert "Ollama client not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_rerank_partial_failure(self, ollama_provider):
        """Test re-ranking when some candidates fail to score."""
        # All candidates fail due to exception in the main try block
        ollama_provider.client.generate = AsyncMock(side_effect=Exception("Timeout"))
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "test query",
            ["doc1", "doc2", "doc3"]
        )
        
        # When exception occurs in main try block, all candidates get 0.5
        assert len(scores) == 3
        assert scores[0] == 0.5  # Default for exception
        assert scores[1] == 0.5  # Default for exception
        assert scores[2] == 0.5  # Default for exception
    
    @pytest.mark.asyncio
    async def test_rerank_prompt_formatting(self, ollama_provider):
        """Test that re-ranking prompt is formatted correctly."""
        ollama_provider.client.generate = AsyncMock(return_value={"response": "0.8"})
        
        query = "How to deploy Docker containers?"
        candidate = "Docker containers can be deployed using various orchestration tools..."
        
        await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            query,
            [candidate]
        )
        
        # Verify the prompt format
        call_args = ollama_provider.client.generate.call_args
        prompt = call_args[1]['prompt']
        
        expected_prompt = (
            f"<|system|>You are a relevance scoring model. Output only a decimal number between 0.0 and 1.0 "
            f"where 0.0 means completely irrelevant and 1.0 means perfectly relevant."
            f"<|user|>Query: {query}\nDocument: {candidate}<|assistant|>"
        )
        assert prompt == expected_prompt
    
    @pytest.mark.asyncio
    async def test_rerank_with_special_characters(self, ollama_provider):
        """Test re-ranking with special characters in query and documents."""
        ollama_provider.client.generate = AsyncMock(side_effect=[
            {"response": "0.7"},
            {"response": "0.3"}
        ])
        
        query = "What is the @import syntax in CSS & how does it work?"
        candidates = [
            "The @import rule allows you to import a style sheet into another CSS file...",
            "JavaScript variables can be declared using var, let, or const keywords..."
        ]
        
        scores = await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            query,
            candidates
        )
        
        assert len(scores) == 2
        assert scores[0] == 0.7
        assert scores[1] == 0.3
        
        # Verify special characters are preserved in prompt
        calls = ollama_provider.client.generate.call_args_list
        first_prompt = calls[0][1]['prompt']
        assert "@import" in first_prompt
        assert "&" in first_prompt
    
    @pytest.mark.asyncio
    async def test_rerank_options_configuration(self, ollama_provider):
        """Test that re-ranking uses correct model options."""
        ollama_provider.client.generate = AsyncMock(return_value={"response": "0.6"})
        
        await ollama_provider.rerank(
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "test query",
            ["test document"]
        )
        
        call_args = ollama_provider.client.generate.call_args
        options = call_args[1]['options']
        
        # Verify deterministic scoring options
        assert options['temperature'] == 0.0
        assert options['num_predict'] == 5
        assert options['stop'] == ["\n", " ", "<|"]
    
    @pytest.mark.asyncio
    async def test_rerank_different_models(self, ollama_provider):
        """Test re-ranking with different model names."""
        ollama_provider.client.generate = AsyncMock(return_value={"response": "0.85"})
        
        models_to_test = [
            "dengcao/Qwen3-Reranker-4B:Q8_0",
            "custom-rerank-model:latest",
            "another-reranker:v1.0"
        ]
        
        for model in models_to_test:
            scores = await ollama_provider.rerank(
                model,
                "test query",
                ["test document"]
            )
            
            assert scores == [0.85]
            
            # Verify correct model was used
            call_args = ollama_provider.client.generate.call_args
            assert call_args[1]['model'] == model