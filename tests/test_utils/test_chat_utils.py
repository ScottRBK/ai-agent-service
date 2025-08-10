import pytest
from app.utils.chat_utils import clean_response_for_memory, separate_chain_of_thought


class TestChatUtils:
    """Test cases for chat_utils module."""
    
    def test_clean_response_for_memory_empty_string(self):
        """Test cleaning empty string returns empty string."""
        # Arrange
        response = ""
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == ""
    
    def test_clean_response_for_memory_none_input(self):
        """Test cleaning None input raises TypeError."""
        # Arrange
        response = None
        
        # Act & Assert
        with pytest.raises(TypeError):
            clean_response_for_memory(response)
    
    def test_clean_response_for_memory_simple_text(self):
        """Test cleaning simple text without special patterns."""
        # Arrange
        response = "Hello world"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello world"
    
    def test_clean_response_for_memory_removes_think_tags(self):
        """Test cleaning removes <think> tags and their content."""
        # Arrange
        response = "Hello <think>This is internal thinking</think> world"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello  world"
    
    def test_clean_response_for_memory_removes_multiple_think_tags(self):
        """Test cleaning removes multiple <think> tags."""
        # Arrange
        response = "Hello <think>First thought</think> world <think>Second thought</think>!"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello  world !"
    
    def test_clean_response_for_memory_removes_think_tags_with_newlines(self):
        """Test cleaning removes <think> tags that span multiple lines."""
        # Arrange
        response = "Hello <think>This is\ninternal thinking\nacross lines</think> world"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello  world"
    
    def test_clean_response_for_memory_replaces_escaped_newlines(self):
        """Test cleaning replaces \\n with actual newlines."""
        # Arrange
        response = "Hello\\nworld\\n!"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello\nworld\n!"
    
    def test_clean_response_for_memory_combines_think_removal_and_newline_replacement(self):
        """Test cleaning handles both think tag removal and newline replacement."""
        # Arrange
        response = "Hello <think>Internal</think>\\nworld\\n!"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello \nworld\n!"
    
    def test_clean_response_for_memory_strips_whitespace(self):
        """Test cleaning strips leading and trailing whitespace."""
        # Arrange
        response = "  Hello world  "
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello world"
    
    def test_clean_response_for_memory_complex_scenario(self):
        """Test cleaning handles complex scenario with all features."""
        # Arrange
        response = "  Hello <think>Internal thinking</think>\\nworld\\n!  "
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello \nworld\n!"
    
    def test_clean_response_for_memory_preserves_other_html_tags(self):
        """Test cleaning preserves other HTML tags that are not <think>."""
        # Arrange
        response = "Hello <b>bold</b> <think>internal</think> <i>italic</i>"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello <b>bold</b>  <i>italic</i>"
    
    def test_clean_response_for_memory_handles_malformed_think_tags(self):
        """Test cleaning handles malformed think tags gracefully."""
        # Arrange
        response = "Hello <think>Unclosed tag"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        # The regex only matches complete think tags, so malformed tags remain
        assert result == "Hello <think>Unclosed tag"
    
    def test_clean_response_for_memory_removes_thoughts_tags(self):
        """Test cleaning removes <thoughts> tags and their content."""
        # Arrange
        response = "Hello <thoughts>This is internal thinking</thoughts> world"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello  world"
    
    def test_clean_response_for_memory_removes_mixed_tags(self):
        """Test cleaning removes both <think> and <thoughts> tags."""
        # Arrange
        response = "Hello <think>First</think> and <thoughts>Second</thoughts> world"
        
        # Act
        result = clean_response_for_memory(response)
        
        # Assert
        assert result == "Hello  and  world"


class TestSeparateChainOfThought:
    """Test cases for separate_chain_of_thought function."""
    
    def test_separate_with_think_tags(self):
        """Test separating content with <think> tags."""
        # Arrange
        response = "Hello <think>This is my thinking</think> world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == "This is my thinking"
        assert cleaned == "Hello  world"
    
    def test_separate_with_thoughts_tags(self):
        """Test separating content with <thoughts> tags."""
        # Arrange
        response = "Hello <thoughts>This is my thinking</thoughts> world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == "This is my thinking"
        assert cleaned == "Hello  world"
    
    def test_separate_with_no_tags(self):
        """Test separating content without any tags."""
        # Arrange
        response = "Hello world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == ""
        assert cleaned == "Hello world"
    
    def test_separate_with_multiple_think_tags(self):
        """Test separating content with multiple <think> tags (returns first match)."""
        # Arrange
        response = "Hello <think>First thought</think> and <think>Second thought</think> world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == "First thought"
        assert cleaned == "Hello  and  world"
    
    def test_separate_with_newlines_in_tags(self):
        """Test separating content with newlines inside tags."""
        # Arrange
        response = "Hello <think>This is\nmultiline\nthinking</think> world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == "This is\nmultiline\nthinking"
        assert cleaned == "Hello  world"
    
    def test_separate_strips_whitespace(self):
        """Test that both outputs are stripped of whitespace."""
        # Arrange
        response = "  Hello <think>  Internal  </think> world  "
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == "Internal"
        assert cleaned == "Hello  world"
    
    def test_separate_empty_string(self):
        """Test separating empty string."""
        # Arrange
        response = ""
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == ""
        assert cleaned == ""
    
    def test_separate_with_empty_tags(self):
        """Test separating content with empty tags."""
        # Arrange
        response = "Hello <think></think> world"
        
        # Act
        cot_content, cleaned = separate_chain_of_thought(response)
        
        # Assert
        assert cot_content == ""
        assert cleaned == "Hello  world" 