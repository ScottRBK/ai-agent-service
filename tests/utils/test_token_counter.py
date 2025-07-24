import pytest
from unittest.mock import patch, MagicMock
from app.utils.token_counter import TokenCounter


class TestTokenCounter:
    """Test cases for TokenCounter class."""
    
    def test_init_with_valid_model(self):
        """Test initialization with a valid model."""
        # Arrange & Act
        counter = TokenCounter("gpt-4")
        
        # Assert
        assert counter.encoding is not None
    
    @patch('app.utils.token_counter.tiktoken.encoding_for_model')
    @patch('app.utils.token_counter.tiktoken.get_encoding')
    def test_init_with_invalid_model_falls_back_to_cl100k_base(self, mock_get_encoding, mock_encoding_for_model):
        """Test initialization falls back to cl100k_base when model is invalid."""
        # Arrange
        mock_encoding_for_model.side_effect = Exception("Model not found")
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        
        # Act
        counter = TokenCounter("invalid-model")
        
        # Assert
        mock_get_encoding.assert_called_once_with("cl100k_base")
        assert counter.encoding == mock_encoding
    
    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string returns 0."""
        # Arrange
        counter = TokenCounter()
        
        # Act
        result = counter.count_tokens("")
        
        # Assert
        assert result == 0
    
    def test_count_tokens_none_string(self):
        """Test counting tokens in None string returns 0."""
        # Arrange
        counter = TokenCounter()
        
        # Act
        result = counter.count_tokens(None)
        
        # Assert
        assert result == 0
    
    def test_count_tokens_simple_text(self):
        """Test counting tokens in simple text."""
        # Arrange
        counter = TokenCounter()
        text = "Hello world"
        
        # Act
        result = counter.count_tokens(text)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_tokens_complex_text(self):
        """Test counting tokens in complex text with special characters."""
        # Arrange
        counter = TokenCounter()
        text = "Hello world! This is a test with special characters: @#$%^&*()"
        
        # Act
        result = counter.count_tokens(text)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_conversation_tokens_empty_list(self):
        """Test counting tokens in empty conversation returns 0."""
        # Arrange
        counter = TokenCounter()
        conversation = []
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result == 0
    
    def test_count_conversation_tokens_single_message(self):
        """Test counting tokens in conversation with single message."""
        # Arrange
        counter = TokenCounter()
        conversation = [{"content": "Hello world"}]
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_conversation_tokens_multiple_messages(self):
        """Test counting tokens in conversation with multiple messages."""
        # Arrange
        counter = TokenCounter()
        conversation = [
            {"content": "Hello world"},
            {"content": "How are you?"},
            {"content": "I'm doing well, thank you!"}
        ]
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_conversation_tokens_message_without_content(self):
        """Test counting tokens in conversation with message missing content."""
        # Arrange
        counter = TokenCounter()
        conversation = [
            {"content": "Hello world"},
            {"role": "user"},  # Missing content
            {"content": "How are you?"}
        ]
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_conversation_tokens_non_string_content(self):
        """Test counting tokens in conversation with non-string content."""
        # Arrange
        counter = TokenCounter()
        conversation = [
            {"content": "Hello world"},
            {"content": 123},  # Non-string content
            {"content": ["list", "content"]}  # List content
        ]
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result > 0
        assert isinstance(result, int)
    
    def test_count_conversation_tokens_mixed_content_types(self):
        """Test counting tokens in conversation with mixed content types."""
        # Arrange
        counter = TokenCounter()
        conversation = [
            {"content": "Hello world"},
            {"content": ""},  # Empty string
            {"content": None},  # None content
            {"content": "Final message"}
        ]
        
        # Act
        result = counter.count_conversation_tokens(conversation)
        
        # Assert
        assert result > 0
        assert isinstance(result, int) 