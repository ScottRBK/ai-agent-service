"""
Simple Prompt Manager that reads from agent configuration.
"""

import os
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.utils.logging import logger
from datetime import datetime 


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
            from app.config.settings import settings
            # If the file path is relative, prepend the prompts directory
            if not os.path.isabs(file_path):
                file_path = os.path.join(settings.PROMPTS_DIR_PATH, file_path)
            
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
            full_prompt += f"\n\n{tool_info}"
        if resource_info:
            full_prompt += f"\n\n{resource_info}"

        full_prompt += f"\n\nThe Current Date is: {datetime.now().strftime('%Y-%m-%d')}"
        
        return full_prompt
    
    def _get_tool_info(self, available_tools: list = None) -> str:
        """Get tool information for the prompt including names and descriptions."""
        if not available_tools:
            return ""
        
        tool_descriptions = []
        
        # Handle both dictionary format (from get_available_tools) and string list format
        if available_tools and isinstance(available_tools[0], dict):
            for tool in available_tools:
                if tool.get("type") == "function" and "function" in tool:
                    tool_name = tool["function"]["name"]
                    tool_description = tool["function"].get("description", "")
                    
                    # Format: "tool_name: description"
                    if tool_description:
                        tool_descriptions.append(f"{tool_name}: {tool_description}")
                    else:
                        tool_descriptions.append(tool_name)
        else:
            # Handle string list format
            tool_descriptions = available_tools
        
        if tool_descriptions:
            # Format as a bulleted list for better readability
            formatted_tools = "\n".join([f"• {tool}" for tool in tool_descriptions])
            return f"\n\nAvailable tools:\n{formatted_tools}"
        return ""
    
    def _get_resource_info(self) -> str:
        """Get resource information for the prompt including names and descriptions."""
        config = self.agent_tool_manager.config
        available_resources = config.get("resources", [])
        
        if not available_resources:
            return ""
        
        # Create detailed resource descriptions with names and explanations
        resource_descriptions = []
        for resource in available_resources:
            if resource == "memory":
                resource_descriptions.append("memory: conversation memory (can remember previous interactions) - so you can recall previous interactions and use them to help the user")
            elif resource == "knowledge_base":
                resource_descriptions.append("knowledge_base: access to structured knowledge base for enhanced information retrieval")
            elif resource == "cache":
                resource_descriptions.append("cache: caching capabilities for improved performance and reduced API calls")
            elif resource == "file_system":
                resource_descriptions.append("file_system: ability to read and write files for data persistence")
            elif resource == "database":
                resource_descriptions.append("database: direct database access for data storage and retrieval")
            elif resource == "api_access":
                resource_descriptions.append("api_access: ability to make external API calls for real-time data")
            elif resource == "web_search":
                resource_descriptions.append("web_search: internet search capabilities for current information")
            else:
                resource_descriptions.append(f"{resource}: custom resource for specialized functionality")
        
        if resource_descriptions:
            # Format as a bulleted list for better readability
            formatted_resources = "\n".join([f"• {resource}" for resource in resource_descriptions])
            return f"\n\nAvailable resources:\n{formatted_resources}"
        return "" 