"""
Simple Prompt Manager that reads from agent configuration.
"""

import os
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.utils.logging import logger


class PromptManager:
    """
    Manages system prompts for agents using agent configuration.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.agent_tool_manager = AgentToolManager(agent_id)
        self.agent_resource_manager = AgentResourceManager(agent_id)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent from agent configuration."""
        tool_config = self.agent_tool_manager.config
        resource_config = self.agent_resource_manager.config
        
        if not tool_config:
            return self.get_default_prompt()
        
        # Check if using external prompt file
        prompt_file = tool_config.get("system_prompt_file")
        if prompt_file:
            return self.load_prompt_from_file(prompt_file)
        
        # Use inline prompt
        return tool_config.get("system_prompt", self.get_default_prompt())
    
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
        Get system prompt with optional tool and resource information.
        
        Args:
            available_tools: List of tool dictionaries from get_available_tools() 
                           or list of tool names (strings)
        """
        base_prompt = self.get_system_prompt()
        
        # Add tool information
        tool_info = self._get_tool_info(available_tools)
        
        # Add resource information
        resource_info = self._get_resource_info()
        
        # Combine all information
        full_prompt = base_prompt
        if tool_info:
            full_prompt += tool_info
        if resource_info:
            full_prompt += resource_info
        
        return full_prompt
    
    def _get_tool_info(self, available_tools: list = None) -> str:
        """Get tool information for the prompt."""
        if not available_tools:
            return ""
        
        # Extract tool names from dictionaries
        if available_tools and isinstance(available_tools[0], dict):
            tool_names = [tool["function"]["name"] for tool in available_tools]
        else:
            tool_names = available_tools
        
        if tool_names:
            return f"\n\nAvailable tools: {', '.join(tool_names)}"
        return ""
    
    def _get_resource_info(self) -> str:
        """Get resource information for the prompt."""
        resource_config = self.agent_resource_manager.config
        available_resources = resource_config.get("resources", [])
        
        if not available_resources:
            return ""
        
        # Create human-readable resource descriptions
        resource_descriptions = []
        for resource in available_resources:
            if resource == "memory":
                resource_descriptions.append("conversation memory (can remember previous interactions) - so you can recall previous interactions and use them to help the user")
            elif resource == "knowledge_base":
                resource_descriptions.append("knowledge base access")
            elif resource == "cache":
                resource_descriptions.append("caching capabilities")
            else:
                resource_descriptions.append(resource)
        
        if resource_descriptions:
            return f"\n\nAvailable resources: {', '.join(resource_descriptions)}"
        return "" 