"""
Agent Tool Manager is responsible for managing which tools are available to specific agents.
It provides agent-specific tool filtering to ensure agents only have access to appropriate tools.

Example agent_config.json:
{
  "agent_id": "research_agent",
  "allowed_regular_tools": ["get_current_datetime"],
  "allowed_mcp_servers": ["deepwiki", "fetch"],
  "allowed_mcp_tools": {
    "deepwiki": ["search", "get_article"],
    "fetch": ["fetch_url"]
  },
  "provider": "azure_openai_cc"
}
"""

import json
import os
from typing import List, Dict, Any, Optional
from app.core.tools.tool_registry import ToolRegistry
from app.utils.logging import logger
from fastmcp import Client
from mcp.types import Tool as MCPTool


class AgentToolManager:
    """
    Manages tool availability for specific agents.
    Provides filtering capabilities to ensure agents only have access to appropriate tools.
    
    Supports MCP servers like deepwiki and fetch, as well as regular registered tools.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.config = self.load_agent_config()
        self.mcp_servers_cache = None
        self.mcp_tools_cache = None
    
    def load_agent_config(self) -> Dict[str, Any]:
        """
        Load agent configuration from agent_config.json or return default config.
        
        Example config:
        {
          "agent_id": "research_agent",
          "allowed_regular_tools": ["get_current_datetime"],
          "allowed_mcp_servers": ["deepwiki", "fetch"],
          "allowed_mcp_tools": {
            "deepwiki": ["search", "get_article"],
            "fetch": ["fetch_url"]
          }
        }
        """
        try:
            from app.config.settings import settings
            config_path = settings.AGENT_CONFIG_PATH
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    all_configs = json.load(f)
                
                # Find config for this agent
                if isinstance(all_configs, dict):
                    # Single agent config
                    if all_configs.get("agent_id") == self.agent_id:
                        return all_configs
                elif isinstance(all_configs, list):
                    # Multiple agent configs
                    for config in all_configs:
                        if config.get("agent_id") == self.agent_id:
                            return config
                
                logger.warning(f"No config found for agent {self.agent_id}, using default")
            
        except Exception as e:
            logger.error(f"Error loading agent config for {self.agent_id}: {e}")
        
        # Return default config (all tools available)
        return {
            "agent_id": self.agent_id,
            "allowed_regular_tools": None,  # None means all tools
            "allowed_mcp_servers": None,    # None means all servers (deepwiki, fetch)
            "allowed_mcp_tools": {}         # Empty means all tools from allowed servers
        }
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools available to this specific agent.
        
        Returns:
            List of tools in chat completions format including:
            - Regular tools (e.g., get_current_datetime)
            - MCP tools from deepwiki and fetch servers
        """
        tools = []
        
        # Add regular tools
        regular_tools = await self.get_regular_tools()
        tools.extend(regular_tools)
        
        # Add MCP tools
        mcp_tools = await self.get_mcp_tools()
        tools.extend(mcp_tools)
        
        logger.debug(f"Agent {self.agent_id} has access to {len(tools)} tools")
        return tools
    
    async def get_regular_tools(self) -> List[Dict[str, Any]]:
        """
        Get regular tools filtered by agent permissions.
        
        Configuration behavior:
        - null/None: Use system default (all tools)
        - []: Explicitly no tools allowed
        - ["tool1", "tool2"]: Only specified tools allowed
        """
        all_regular_tools = ToolRegistry.convert_tool_registry_to_chat_completions_format()
        
        allowed_regular_tools = self.config.get("allowed_regular_tools")
        
        if allowed_regular_tools is None:
            # Use system default (all tools)
            return all_regular_tools
        elif allowed_regular_tools == []:
            # Explicitly no tools allowed
            return []
        else:
            # Filter to specified tools
            filtered_tools = [
                tool for tool in all_regular_tools 
                if tool["function"]["name"] in allowed_regular_tools
            ]
            return filtered_tools
    
    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tools filtered by agent permissions.
        
        Supports servers like deepwiki and fetch from mcp.json.
        """
        if self.mcp_tools_cache is not None:
            return self.mcp_tools_cache
        
        mcp_servers = ToolRegistry.load_mcp_servers()
        all_mcp_tools = []
        
        # Determine which servers this agent can access
        allowed_servers = self.config.get("allowed_mcp_servers")
        if allowed_servers is None:
            # Agent has access to all servers (deepwiki, fetch, etc.)
            accessible_servers = mcp_servers
        elif allowed_servers == []:
            # Agent has no access to MCP servers
            accessible_servers = []
        else:
            # Filter to allowed servers
            accessible_servers = [
                server for server in mcp_servers 
                if server.server_label in allowed_servers
            ]
        
        # Load tools from accessible servers
        for mcp_server in accessible_servers:
            server_tools = await self.load_server_tools(mcp_server)
            
            # Filter tools if specific tools are restricted
            allowed_mcp_tools = self.config.get("allowed_mcp_tools")
            if allowed_mcp_tools is None:
                # Allow all tools from this server (null = all tools)
                all_mcp_tools.extend(server_tools)
            elif allowed_mcp_tools == {}:
                # Allow all tools from this server (empty dict = all tools)
                all_mcp_tools.extend(server_tools)
            else:
                # Filter to specific tools
                allowed_tools_for_server = allowed_mcp_tools.get(mcp_server.server_label, [])
                if allowed_tools_for_server:
                    # Filter to specific tools
                    filtered_server_tools = [
                        tool for tool in server_tools
                        if tool["function"]["name"].split("__")[1] in allowed_tools_for_server
                    ]
                    all_mcp_tools.extend(filtered_server_tools)
                else:
                    # No specific tools allowed from this server
                    continue
        
        self.mcp_tools_cache = all_mcp_tools
        logger.debug(f"Agent {self.agent_id} has access to {len(all_mcp_tools)} MCP tools from {len(accessible_servers)} servers")
        return all_mcp_tools
    
    async def load_server_tools(self, mcp_server) -> List[Dict[str, Any]]:
        """
        Load tools from a specific MCP server (e.g., deepwiki, fetch, searxng).
        """
        try:
            if mcp_server.server_url:
                # HTTP-based MCP server (existing code)
                mcp_client = Client(mcp_server.server_url)
            elif mcp_server.command:
                # Command-based MCP server using fastmcp StdioTransport
                from fastmcp.client.transports import StdioTransport
                
                # Prepare environment variables
                env = os.environ.copy()  # Start with current environment
                if mcp_server.env:
                    env.update(mcp_server.env)  # Add MCP server specific env vars
                
                # Create StdioTransport for command-based MCP
                transport = StdioTransport(
                    command=mcp_server.command,
                    args=mcp_server.args or [],
                    env=env,
                    keep_alive=False
                )
                mcp_client = Client(transport)
            else:
                logger.error(f"MCP server {mcp_server.server_label} has no server_url or command")
                return []
                
            async with mcp_client:
                server_tools = await mcp_client.list_tools()
                formatted_tools = ToolRegistry.convert_mcp_tools_to_chatcompletions(
                    mcp_server.server_label, server_tools
                )
                await mcp_client.close()
                return formatted_tools
        except Exception as e:
            logger.error(f"Error loading tools from MCP server {mcp_server.server_label}: {e}")
            return []
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool (regular or MCP) for this agent.
        
        Args:
            tool_name: Name of the tool to execute (e.g., "get_current_datetime" or "deepwiki__search")
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        separator = "__"
        
        if separator in tool_name:
            # This is an MCP tool (e.g., "deepwiki__search", "fetch__fetch_url")
            return await self.execute_mcp_tool(tool_name, arguments)
        else:
            # This is a regular tool (e.g., "get_current_datetime")
            return await self.execute_regular_tool(tool_name, arguments)
    
    async def execute_regular_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a regular tool.
        """
        # Check if agent has access to this tool
        if self.config.get("allowed_regular_tools") is not None:
            if tool_name not in self.config.get("allowed_regular_tools", []):
                raise ValueError(f"Agent {self.agent_id} does not have access to tool {tool_name}")
        
        return ToolRegistry.execute_tool_call(tool_name, arguments)
    
    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute an MCP tool (e.g., from deepwiki, fetch, or searxng servers).
        """
        separator = "__"
        mcp_server_label, actual_tool_name = tool_name.split(separator, 1)
        
        # Check if agent has access to this server
        allowed_servers = self.config.get("allowed_mcp_servers") if self.config else None
        if allowed_servers is not None and mcp_server_label not in allowed_servers:
            raise ValueError(f"Agent {self.agent_id} does not have access to MCP server {mcp_server_label}")
        
        # Check if agent has access to this specific tool
        allowed_tools_for_server = []
        if self.config:
            allowed_mcp_tools = self.config.get("allowed_mcp_tools", {})
            if allowed_mcp_tools is not None:
                allowed_tools_for_server = allowed_mcp_tools.get(mcp_server_label, [])
        
        # If specific tools are allowed for this server, check if this tool is in the list
        if allowed_tools_for_server and actual_tool_name not in allowed_tools_for_server:
            raise ValueError(f"Agent {self.agent_id} does not have access to tool {actual_tool_name} from server {mcp_server_label}")
        
        # Execute the MCP tool
        mcp_servers = ToolRegistry.load_mcp_servers()
        # Create a mapping for easy lookup
        servers_by_label = {server.server_label: server for server in mcp_servers}
        
        if mcp_server_label not in servers_by_label:
            raise ValueError(f"MCP server {mcp_server_label} not found")
        
        server = servers_by_label[mcp_server_label]
        
        if server.server_url:
            # HTTP-based server (existing code)
            client = Client(server.server_url)
        elif server.command:
            # Command-based server using fastmcp StdioTransport
            from fastmcp.client.transports import StdioTransport
            
            # Prepare environment variables
            env = os.environ.copy()  # Start with current environment
            if server.env:
                env.update(server.env)  # Add MCP server specific env vars
            
            transport = StdioTransport(
                command=server.command,
                args=server.args or [],
                env=env,
                keep_alive=False
            )
            client = Client(transport)
        else:
            raise ValueError(f"MCP server {mcp_server_label} has no server_url or command")
        
        try:
            async with client:
                result = await client.call_tool(actual_tool_name, arguments)
                await client.close()
                return str(result)
        except Exception as e:
            # Return the error message as a string so the LLM can see it
            logger.error(f"Error executing MCP tool {tool_name}: {e}")
            return f"Tool execution failed: {str(e)}"
    
    def clear_cache(self):
        """Clear all cached data."""
        self.mcp_servers_cache = None
        self.mcp_tools_cache = None
        logger.debug(f"Cleared MCP tools cache for agent {self.agent_id}") 