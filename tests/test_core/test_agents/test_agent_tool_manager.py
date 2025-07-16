"""
Unit tests for AgentToolManager.
Tests agent-specific tool filtering and MCP tool integration.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.tools.tool_registry import TOOL_REGISTRY, ToolRegistry
from app.core.tools.function_calls.date_tool import DateTool
from app.core.tools.function_calls.arithmetic_tool import ArithmeticTool


class TestAgentToolManager:
    """Test cases for AgentToolManager."""
    
    def test_init_with_valid_agent_id(self):
        """Test initialization with a valid agent ID."""
        agent_manager = AgentToolManager("research_agent")
        assert agent_manager.agent_id == "research_agent"
        assert agent_manager.config is not None
    
    def test_init_with_invalid_agent_id(self):
        """Test initialization with an invalid agent ID (should use default config)."""
        agent_manager = AgentToolManager("nonexistent_agent")
        assert agent_manager.agent_id == "nonexistent_agent"
        # Should return default config, not None
        assert agent_manager.config is not None
        assert agent_manager.config["agent_id"] == "nonexistent_agent"
        assert agent_manager.config["allowed_regular_tools"] is None  # All tools allowed
        assert agent_manager.config["allowed_mcp_servers"] is None    # All servers allowed
        assert agent_manager.config["allowed_mcp_tools"] == {}        # All tools from allowed servers
    
    @pytest.mark.asyncio
    async def test_get_regular_tools_with_allowed_tools(self):
        """Test getting regular tools when agent has specific allowed tools."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config to return specific allowed tools
        agent_manager.config = {
            "allowed_regular_tools": ["get_current_datetime"]
        }
        
        # Debug: Check what's in the tool registry
        print(f"TOOL_REGISTRY keys: {list(TOOL_REGISTRY.keys())}")
        
        # Debug: Check what tools are being returned
        all_regular_tools = ToolRegistry.convert_tool_registry_to_chat_completions_format()
        print(f"All regular tools: {[tool['function']['name'] for tool in all_regular_tools]}")
        
        tools = await agent_manager.get_regular_tools()
        print(f"Filtered tools: {[tool['function']['name'] for tool in tools]}")
        
        # Should only return the allowed tool
        assert len(tools) == 1
        assert any("get_current_datetime" in tool["function"]["name"] for tool in tools)
    
    @pytest.mark.asyncio
    async def test_get_regular_tools_with_no_restrictions(self):
        """Test getting regular tools when agent has no restrictions."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config to allow all tools
        agent_manager.config = {
            "allowed_regular_tools": None
        }
        
        tools = await agent_manager.get_regular_tools()
        
        # Should return all available tools
        assert len(tools) >= 2  # At least get_current_datetime and add_two_numbers
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "get_current_datetime" in tool_names
        assert "add_two_numbers" in tool_names
    
    @pytest.mark.asyncio
    async def test_get_regular_tools_with_empty_list(self):
        """Test getting regular tools when agent has empty allowed list."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config to allow no tools
        agent_manager.config = {
            "allowed_regular_tools": []
        }
        
        tools = await agent_manager.get_regular_tools()
        
        # Should return empty list
        assert tools == []
    
    @pytest.mark.asyncio
    async def test_get_mcp_tools_with_allowed_servers(self):
        """Test getting MCP tools when agent has specific allowed servers."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config
        agent_manager.config = {
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Mock the MCP client
            mock_client = AsyncMock()
            mock_tool = MagicMock()
            mock_tool.name = "read_wiki_structure"
            mock_tool.description = "Read wiki structure"
            mock_tool.inputSchema = {"type": "object", "properties": {}}
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.list_tools.return_value = [mock_tool]
                
                tools = await agent_manager.get_mcp_tools()
                
                # Should return the allowed MCP tool
                assert len(tools) == 1
                assert "deepwiki__read_wiki_structure" in tools[0]["function"]["name"]
    
    @pytest.mark.asyncio
    async def test_get_mcp_tools_with_no_restrictions(self):
        """Test getting MCP tools when agent has no server restrictions."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config to allow all servers
        agent_manager.config = {
            "allowed_mcp_servers": None,
            "allowed_mcp_tools": None
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Mock the MCP client
            mock_client = AsyncMock()
            mock_tool = MagicMock()
            mock_tool.name = "read_wiki_structure"
            mock_tool.description = "Read wiki structure"
            mock_tool.inputSchema = {"type": "object", "properties": {}}
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.list_tools.return_value = [mock_tool]
                
                tools = await agent_manager.get_mcp_tools()
                
                # Should return all available MCP tools
                assert len(tools) == 1
                assert "deepwiki__read_wiki_structure" in tools[0]["function"]["name"]
    
    @pytest.mark.asyncio
    async def test_get_available_tools_combines_regular_and_mcp(self):
        """Test that get_available_tools combines regular and MCP tools."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock the config
        agent_manager.config = {
            "allowed_regular_tools": ["get_current_datetime"],
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Mock the MCP client
            mock_client = AsyncMock()
            mock_tool = MagicMock()
            mock_tool.name = "read_wiki_structure"
            mock_tool.description = "Read wiki structure"
            mock_tool.inputSchema = {"type": "object", "properties": {}}
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.list_tools.return_value = [mock_tool]
                
                tools = await agent_manager.get_available_tools()
                
                # Should return both regular and MCP tools
                assert len(tools) >= 2
                tool_names = [tool["function"]["name"] for tool in tools]
                assert "get_current_datetime" in tool_names
                assert "deepwiki__read_wiki_structure" in tool_names
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_success(self):
        """Test executing a regular tool successfully."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_regular_tools": ["get_current_datetime"]
        }
        
        result = await agent_manager.execute_regular_tool("get_current_datetime", {"timezone": "UTC"})
        
        # Should return a datetime string
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_unauthorized(self):
        """Test executing a regular tool that agent doesn't have access to."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_regular_tools": ["get_current_datetime"]
        }
        
        with pytest.raises(ValueError, match="Agent research_agent does not have access to tool add_two_numbers"):
            await agent_manager.execute_regular_tool("add_two_numbers", {"a": 1, "b": 2})
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_success(self):
        """Test executing an MCP tool successfully."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Mock the MCP client
            mock_client = AsyncMock()
            mock_client.call_tool.return_value = "Mock wiki structure result"
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                
                result = await agent_manager.execute_mcp_tool("deepwiki__read_wiki_structure", {"query": "test"})
                
                assert result == "Mock wiki structure result"
                mock_client.call_tool.assert_called_once_with("read_wiki_structure", {"query": "test"})
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_unauthorized_server(self):
        """Test executing an MCP tool from unauthorized server."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        with pytest.raises(ValueError, match="Agent research_agent does not have access to MCP server fetch"):
            await agent_manager.execute_mcp_tool("fetch__fetch_url", {"url": "http://example.com"})
    
    @pytest.mark.asyncio
    async def test_execute_mcp_tool_unauthorized_tool(self):
        """Test executing an MCP tool that agent doesn't have access to."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        with pytest.raises(ValueError, match="Agent research_agent does not have access to tool search_wiki from server deepwiki"):
            await agent_manager.execute_mcp_tool("deepwiki__search_wiki", {"query": "test"})
    
    @pytest.mark.asyncio
    async def test_execute_tool_regular_tool(self):
        """Test execute_tool with a regular tool."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_regular_tools": ["get_current_datetime"]
        }
        
        result = await agent_manager.execute_tool("get_current_datetime", {"timezone": "UTC"})
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_mcp_tool(self):
        """Test execute_tool with an MCP tool."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["deepwiki"],
            "allowed_mcp_tools": {
                "deepwiki": ["read_wiki_structure"]
            }
        }
        
        # Mock the MCP servers and client
        mock_server = MagicMock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            mock_client = AsyncMock()
            mock_client.call_tool.return_value = "Mock result"
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                
                result = await agent_manager.execute_tool("deepwiki__read_wiki_structure", {"query": "test"})
                
                assert result == "Mock result"
    
    def test_load_agent_config_existing_agent(self):
        """Test loading config for an existing agent."""
        agent_manager = AgentToolManager("research_agent")
        
        # Should load the config from agent_config.json
        assert agent_manager.config is not None
        assert "allowed_regular_tools" in agent_manager.config
        assert "allowed_mcp_servers" in agent_manager.config
