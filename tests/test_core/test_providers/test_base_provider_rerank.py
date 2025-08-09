"""
Unit tests for BaseProvider default re-ranking functionality.

This module tests the default re-ranking implementation in BaseProvider,
which should return 0.5 scores for all candidates.
"""

import pytest
from unittest.mock import Mock
from app.core.providers.base import BaseProvider
from app.models.providers import ProviderConfig, ProviderType


class TestableBaseProvider(BaseProvider):
    """Test implementation of BaseProvider that uses default re-ranking"""
    
    async def health_check(self):
        return Mock(is_healthy=True)
    
    async def cleanup(self):
        pass
    
    async def get_model_list(self):
        return ["test-model"]
    
    async def send_chat(self, context, model, instructions, tools):
        return "test response"
    
    async def send_chat_with_streaming(self, context, model, instructions, tools):
        yield "test response"
    
    async def embed(self, text):
        return [0.1, 0.2, 0.3]
    
    # Inherits default rerank() implementation from BaseProvider


class TestBaseProviderDefaultReranking:
    """Test cases for BaseProvider default re-ranking implementation"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock provider configuration"""
        return ProviderConfig(
            name="test_base_provider",
            provider_type=ProviderType.AZURE_OPENAI_CC,
            api_key="test_key",
            endpoint="https://test.openai.azure.com/",
            default_model="gpt-4"
        )
    
    @pytest.mark.asyncio
    async def test_default_rerank_returns_equal_scores(self, mock_config):
        """Test that default re-ranking returns 0.5 for all candidates"""
        provider = TestableBaseProvider(mock_config)
        
        scores = await provider.rerank(
            "gpt-4",
            "What is machine learning?",
            [
                "Machine learning is a subset of artificial intelligence...",
                "The weather forecast for tomorrow shows rain...",
                "Deep learning models require large amounts of data..."
            ]
        )
        
        # Should return 0.5 for all candidates
        assert len(scores) == 3
        assert all(score == 0.5 for score in scores)
    
    @pytest.mark.asyncio
    async def test_default_rerank_empty_candidates(self, mock_config):
        """Test default re-ranking with empty candidates list"""
        provider = TestableBaseProvider(mock_config)
        
        scores = await provider.rerank(
            "gpt-4",
            "test query", 
            []
        )
        
        assert scores == []
    
    @pytest.mark.asyncio
    async def test_default_rerank_single_candidate(self, mock_config):
        """Test default re-ranking with single candidate"""
        provider = TestableBaseProvider(mock_config)
        
        scores = await provider.rerank(
            "gpt-4",
            "How to implement authentication?",
            ["JWT tokens provide a stateless authentication mechanism..."]
        )
        
        assert scores == [0.5]
    
    @pytest.mark.asyncio
    async def test_default_rerank_many_candidates(self, mock_config):
        """Test default re-ranking with many candidates"""
        provider = TestableBaseProvider(mock_config)
        
        candidates = [f"Document {i} content..." for i in range(10)]
        
        scores = await provider.rerank(
            "gpt-4",
            "test query",
            candidates
        )
        
        assert len(scores) == 10
        assert all(score == 0.5 for score in scores)
    
    @pytest.mark.asyncio
    async def test_default_rerank_with_different_models(self, mock_config):
        """Test default re-ranking with various model names"""
        provider = TestableBaseProvider(mock_config)
        
        models_to_test = [
            "gpt-4",
            "gpt-3.5-turbo",
            "custom-model:latest",
            "rerank-model-v1"
        ]
        
        candidates = ["doc1", "doc2", "doc3"]
        
        for model in models_to_test:
            scores = await provider.rerank(model, "test query", candidates)
            
            # Should return same default scores regardless of model
            assert scores == [0.5, 0.5, 0.5]
    
    @pytest.mark.asyncio
    async def test_default_rerank_with_special_characters(self, mock_config):
        """Test default re-ranking with special characters in query and candidates"""
        provider = TestableBaseProvider(mock_config)
        
        query = "What is the @import syntax in CSS & how does it work?"
        candidates = [
            "The @import rule allows you to import a style sheet...",
            "JavaScript variables can be declared using var, let, or const...",
            "HTML elements can have attributes with special characters like & and @..."
        ]
        
        scores = await provider.rerank("gpt-4", query, candidates)
        
        assert len(scores) == 3
        assert all(score == 0.5 for score in scores)
    
    @pytest.mark.asyncio
    async def test_default_rerank_with_long_content(self, mock_config):
        """Test default re-ranking with very long content"""
        provider = TestableBaseProvider(mock_config)
        
        # Create long candidates to test performance and behavior
        long_candidates = [
            "This is a very long document " * 100 + "about topic A.",
            "Another extremely long text " * 100 + "covering topic B.",
            "Yet another lengthy content " * 100 + "discussing topic C."
        ]
        
        scores = await provider.rerank(
            "gpt-4",
            "Find information about topic A",
            long_candidates
        )
        
        assert len(scores) == 3
        assert all(score == 0.5 for score in scores)
    
    @pytest.mark.asyncio
    async def test_default_rerank_preserves_candidate_order(self, mock_config):
        """Test that default re-ranking preserves the original candidate order"""
        provider = TestableBaseProvider(mock_config)
        
        candidates = [
            "First document with unique content A",
            "Second document with unique content B", 
            "Third document with unique content C",
            "Fourth document with unique content D"
        ]
        
        scores = await provider.rerank("gpt-4", "test query", candidates)
        
        # Should return scores in same order as candidates
        assert len(scores) == len(candidates)
        assert scores == [0.5, 0.5, 0.5, 0.5]
    
    def test_provider_class_name(self, mock_config):
        """Test different provider class names"""
        
        class CustomTestProvider(BaseProvider):
            async def health_check(self):
                return Mock()
            async def cleanup(self):
                pass
            async def get_model_list(self):
                return []
            async def send_chat(self, context, model, instructions, tools):
                return ""
            async def send_chat_with_streaming(self, context, model, instructions, tools):
                yield ""
            async def embed(self, text):
                return []
        
        provider = CustomTestProvider(mock_config)
        
        # Test that the class name is correctly extracted
        assert provider.__class__.__name__ == "CustomTestProvider"