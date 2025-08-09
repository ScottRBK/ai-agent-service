"""
Unit tests for PromptManager.
Tests prompt loading, file handling, and fallback behavior.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from app.core.agents.prompt_manager import PromptManager


class TestPromptManager:
    """Test cases for PromptManager."""
    
    def test_init_with_valid_agent_id(self):
        """Test initialization with a valid agent ID."""
        prompt_manager = PromptManager("research_agent")
        assert prompt_manager.agent_id == "research_agent"
        assert prompt_manager.agent_tool_manager is not None
    
    def test_get_default_prompt(self):
        """Test getting default prompt when no config is available."""
        prompt_manager = PromptManager("nonexistent_agent")
        default_prompt = prompt_manager.get_default_prompt()
        
        assert isinstance(default_prompt, str)
        assert len(default_prompt) > 0
        assert "helpful AI assistant" in default_prompt
    
    def test_get_system_prompt_with_file(self):
        """Test getting system prompt from external file."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("You are a test assistant with specific capabilities.")
            temp_file_path = f.name
        
        try:
            # Mock the agent config to include the file path
            mock_config = {
                "agent_id": "test_agent",
                "system_prompt_file": temp_file_path
            }
            
            # Create a mock tool manager
            mock_tool_manager = MagicMock()
            mock_tool_manager.config = mock_config
            
            # Patch the AgentToolManager constructor
            with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
                mock_agent_tool_manager_class.return_value = mock_tool_manager
                
                prompt_manager = PromptManager("test_agent")
                system_prompt = prompt_manager.get_system_prompt()
                
                assert system_prompt == "You are a test assistant with specific capabilities."
        
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_get_system_prompt_with_inline_prompt(self):
        """Test getting system prompt from inline config."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are an inline configured assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            system_prompt = prompt_manager.get_system_prompt()
            
            assert system_prompt == "You are an inline configured assistant."
    
    def test_get_system_prompt_fallback_to_default(self):
        """Test fallback to default prompt when no config is available."""
        # Create a mock tool manager with None config
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = None
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            system_prompt = prompt_manager.get_system_prompt()
            
            assert system_prompt == prompt_manager.get_default_prompt()
    
    def test_get_system_prompt_with_tools(self):
        """Test getting system prompt with tool information."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            available_tools = ["get_current_datetime", "add_two_numbers"]
            
            system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
            
            assert "You are a test assistant." in system_prompt
            # Updated to match new bulleted list format
            assert "Available tools:" in system_prompt
            assert "• get_current_datetime" in system_prompt
            assert "• add_two_numbers" in system_prompt
    
    def test_get_system_prompt_with_tools_empty(self):
        """Test getting system prompt with no tools."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            system_prompt = prompt_manager.get_system_prompt_with_tools([])
            
            # Check base prompt is included
            assert "You are a test assistant." in system_prompt
            # Check date is included
            assert "The Current Date is:" in system_prompt
            # No tools should be listed
            assert "Available tools:" not in system_prompt
    
    def test_get_system_prompt_with_tools_none(self):
        """Test getting system prompt with None tools."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            system_prompt = prompt_manager.get_system_prompt_with_tools(None)
            
            # Check base prompt and date are included
            assert "You are a test assistant." in system_prompt
            assert "The Current Date is:" in system_prompt
    
    def test_load_prompt_from_file_not_found(self):
        """Test loading prompt from non-existent file."""
        prompt_manager = PromptManager("test_agent")
        
        # Should return default prompt when file doesn't exist
        result = prompt_manager.load_prompt_from_file("nonexistent_file.txt")
        assert result == prompt_manager.get_default_prompt()
    
    def test_load_prompt_from_file_with_error(self):
        """Test loading prompt from file with read error."""
        prompt_manager = PromptManager("test_agent")
        
        # Mock file that raises an exception
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = prompt_manager.load_prompt_from_file("test_file.txt")
            assert result == prompt_manager.get_default_prompt()
    
    def test_load_prompt_from_file_success(self):
        """Test successfully loading prompt from file."""
        # Create a temporary file with content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Custom prompt content with\nmultiple lines.")
            temp_file_path = f.name
        
        try:
            prompt_manager = PromptManager("test_agent")
            result = prompt_manager.load_prompt_from_file(temp_file_path)
            
            assert result == "Custom prompt content with\nmultiple lines."
        
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_load_prompt_from_file_with_encoding(self):
        """Test loading prompt from file with special characters."""
        # Create a temporary file with special characters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Custom prompt with special chars: éñç")
            temp_file_path = f.name
        
        try:
            prompt_manager = PromptManager("test_agent")
            result = prompt_manager.load_prompt_from_file(temp_file_path)
            
            assert result == "Custom prompt with special chars: éñç"
        
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_preference_order_file_over_inline(self):
        """Test that file prompt takes precedence over inline prompt."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("File-based prompt content.")
            temp_file_path = f.name
        
        try:
            # Mock config with both file and inline prompts
            mock_config = {
                "agent_id": "test_agent",
                "system_prompt_file": temp_file_path,
                "system_prompt": "Inline prompt content."
            }
            
            # Create a mock tool manager
            mock_tool_manager = MagicMock()
            mock_tool_manager.config = mock_config
            
            # Patch the AgentToolManager constructor
            with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
                mock_agent_tool_manager_class.return_value = mock_tool_manager
                
                prompt_manager = PromptManager("test_agent")
                system_prompt = prompt_manager.get_system_prompt()
                
                # Should use file content, not inline content
                assert system_prompt == "File-based prompt content."
                assert "Inline prompt content." not in system_prompt
        
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_empty_file_returns_default(self):
        """Test that empty file returns default prompt."""
        # Create an empty temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_file_path = f.name
        
        try:
            mock_config = {
                "agent_id": "test_agent",
                "system_prompt_file": temp_file_path
            }
            
            # Create a mock tool manager
            mock_tool_manager = MagicMock()
            mock_tool_manager.config = mock_config
            
            # Patch the AgentToolManager constructor
            with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
                mock_agent_tool_manager_class.return_value = mock_tool_manager
                
                prompt_manager = PromptManager("test_agent")
                system_prompt = prompt_manager.get_system_prompt()
                
                # Should return default prompt for empty file
                assert system_prompt == prompt_manager.get_default_prompt()
        
        finally:
            # Clean up
            os.unlink(temp_file_path) 

    def test_get_system_prompt_with_tools_dict_format(self):
        """Test getting system prompt with tool dictionaries (new format)."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            # Simulate tool dictionaries from get_available_tools()
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "description": "Get current date and time"
                    }
                },
                {
                    "type": "function", 
                    "function": {
                        "name": "deepwiki__search_wiki",
                        "description": "Search wiki content"
                    }
                }
            ]
            
            system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
            
            assert "You are a test assistant." in system_prompt
            # Updated to match new bulleted list format with descriptions
            assert "Available tools:" in system_prompt
            assert "• get_current_datetime: Get current date and time" in system_prompt
            assert "• deepwiki__search_wiki: Search wiki content" in system_prompt
    
    def test_get_system_prompt_with_tools_mixed_formats(self):
        """Test that the method handles both formats correctly."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            
            # Test with tool names (existing format)
            tool_names = ["get_current_datetime", "add_two_numbers"]
            system_prompt_names = prompt_manager.get_system_prompt_with_tools(tool_names)
            
            # Test with tool dictionaries (new format)
            tool_dicts = [
                {"type": "function", "function": {"name": "get_current_datetime"}},
                {"type": "function", "function": {"name": "add_two_numbers"}}
            ]
            system_prompt_dicts = prompt_manager.get_system_prompt_with_tools(tool_dicts)
            
            # Both should produce the same result
            assert system_prompt_names == system_prompt_dicts
            # Updated to match new bulleted list format
            assert "Available tools:" in system_prompt_names
            assert "• get_current_datetime" in system_prompt_names
            assert "• add_two_numbers" in system_prompt_names
    
    def test_get_system_prompt_with_tools_empty_dict_list(self):
        """Test getting system prompt with empty list of tool dictionaries."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            system_prompt = prompt_manager.get_system_prompt_with_tools([])
            
            # Check base prompt is included
            assert "You are a test assistant." in system_prompt
            # Check date is included
            assert "The Current Date is:" in system_prompt
            # No tools should be listed
            assert "Available tools:" not in system_prompt
    
    def test_get_system_prompt_with_tools_malformed_dict(self):
        """Test handling of malformed tool dictionaries."""
        mock_config = {
            "agent_id": "test_agent",
            "system_prompt": "You are a test assistant."
        }
        
        # Create a mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.config = mock_config
        
        # Patch the AgentToolManager constructor
        with patch('app.core.agents.prompt_manager.AgentToolManager') as mock_agent_tool_manager_class:
            mock_agent_tool_manager_class.return_value = mock_tool_manager
            
            prompt_manager = PromptManager("test_agent")
            
            # Test with malformed dictionary - should handle gracefully
            malformed_tools = [
                {"type": "function", "name": "get_current_datetime"}  # Missing "function" key
            ]
            
            # Should handle malformed input gracefully and return base prompt
            system_prompt = prompt_manager.get_system_prompt_with_tools(malformed_tools)
            # Check base prompt is included
            assert "You are a test assistant." in system_prompt
            # Check date is included
            assert "The Current Date is:" in system_prompt
            # No tools should be listed since the input was malformed
            assert "Available tools:" not in system_prompt 