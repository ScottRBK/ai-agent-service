"""
Simple Prompt Manager that reads from agent configuration.
"""

import os
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.utils.logging import logger


class PromptManager:
    """
    Manages system prompts for agents using agent configuration.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent_tool_manager = AgentToolManager(agent_id)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent from agent configuration."""
        config = self.agent_tool_manager.config
        
        if not config:
            return self.get_default_prompt()
        
        # Check if using external prompt file
        prompt_file = config.get("system_prompt_file")
        if prompt_file:
            return self.load_prompt_from_file(prompt_file)
        
        # Use inline prompt
        return config.get("system_prompt", self.get_default_prompt())
    
    def load_prompt_from_file(self, file_path: str) -> str:
        """Load prompt from external file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # Return default prompt if file is empty
                    if not content:
                        return self.get_default_prompt()
                    return content
            else:
                logger.warning(f"Prompt file not found: {file_path}")
                return self.get_default_prompt()
        except Exception as e:
            logger.error(f"Error loading prompt file {file_path}: {e}")
            return self.get_default_prompt()
    
    def get_default_prompt(self) -> str:
        """Get default system prompt."""
        return "You are a helpful AI assistant. Use available tools when needed to provide accurate and helpful responses."
    
    def get_system_prompt_with_tools(self, available_tools: list = None) -> str:
        """
        Get system prompt with optional tool information.
        
        Args:
            available_tools: List of tool dictionaries from get_available_tools() 
                           or list of tool names (strings)
        """
        base_prompt = self.get_system_prompt()
        
        if not available_tools:
            return base_prompt
        
        # Extract tool names from dictionaries
        if available_tools and isinstance(available_tools[0], dict):
            tool_names = [tool["function"]["name"] for tool in available_tools]
        else:
            tool_names = available_tools
        
        tool_info = f"\n\nAvailable tools: {', '.join(tool_names)}"
        return base_prompt + tool_info 