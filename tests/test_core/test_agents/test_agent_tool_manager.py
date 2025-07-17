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
import asyncio


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

    @pytest.mark.asyncio
    async def test_load_server_tools_command_based_mcp(self):
        """Test loading tools from a command-based MCP server (like searxng)."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock a command-based MCP server
        mock_server = MagicMock()
        mock_server.server_label = "searxng"
        mock_server.server_url = None  # No URL for command-based servers
        mock_server.command = "docker"
        mock_server.args = [
            "run", "-i", "--rm",
            "-e", "SEARXNG_URL=https://searx.be",
            "isokoliuk/mcp-searxng:latest"
        ]
        
        # Mock the subprocess creation
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        
        # Mock the SubprocessMCPClient
        mock_subprocess_client = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "searxng_web_search"
        mock_tool.description = "Search the web using Searxng"
        mock_tool.inputSchema = {"type": "object", "properties": {"query": {"type": "string"}}}
        
        mock_subprocess_client.list_tools.return_value = [mock_tool]
        
        # Fix: Use the correct patch path for asyncio
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_subprocess.return_value = mock_process
            
            # Fix: Use the correct import path for SubprocessMCPClient
            with patch('app.core.tools.subprocess_mcp_client.SubprocessMCPClient') as mock_client_class:
                mock_client_class.return_value = mock_subprocess_client
                
                # Test the method
                tools = await agent_manager.load_server_tools(mock_server)
                
                # Verify the results
                assert len(tools) == 1
                assert tools[0]["function"]["name"] == "searxng__searxng_web_search"
                assert tools[0]["function"]["description"] == "Search the web using Searxng"
                
                # Verify subprocess was created with correct command
                mock_subprocess.assert_called_once_with(
                    "docker", "run", "-i", "--rm",
                    "-e", "SEARXNG_URL=https://searx.be",
                    "isokoliuk/mcp-searxng:latest",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Verify SubprocessMCPClient was created
                mock_client_class.assert_called_once_with(mock_process)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_command_based_server(self):
        """Test executing a tool from a command-based MCP server."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["searxng"],
            "allowed_mcp_tools": {
                "searxng": ["searxng_web_search"]
            }
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "searxng"
        mock_server.server_url = None  # Command-based server
        mock_server.command = "docker"
        mock_server.args = [
            "run", "-i", "--rm",
            "-e", "SEARXNG_URL=https://searx.be",
            "isokoliuk/mcp-searxng:latest"
        ]
        
        # Mock the subprocess creation
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        
        # Mock the SubprocessMCPClient
        mock_subprocess_client = AsyncMock()
        mock_subprocess_client.call_tool.return_value = "Search results for 'AI news'"
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Fix: Use the correct patch path for asyncio
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_subprocess.return_value = mock_process
                
                # Fix: Use the correct import path for SubprocessMCPClient
                with patch('app.core.tools.subprocess_mcp_client.SubprocessMCPClient') as mock_client_class:
                    mock_client_class.return_value = mock_subprocess_client
                    
                    # Test the method
                    result = await agent_manager.execute_mcp_tool("searxng__searxng_web_search", {"query": "AI news"})
                    
                    # Verify the result
                    assert result == "Search results for 'AI news'"
                    
                    # Verify subprocess was created
                    mock_subprocess.assert_called_once_with(
                        "docker", "run", "-i", "--rm",
                        "-e", "SEARXNG_URL=https://searx.be",
                        "isokoliuk/mcp-searxng:latest",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Verify tool was called with correct arguments
                    mock_subprocess_client.call_tool.assert_called_once_with("searxng_web_search", {"query": "AI news"})

    @pytest.mark.asyncio
    async def test_load_server_tools_command_based_mcp_error_handling(self):
        """Test error handling when command-based MCP server fails to start."""
        agent_manager = AgentToolManager("research_agent")
        
        # Mock a command-based MCP server
        mock_server = MagicMock()
        mock_server.server_label = "searxng"
        mock_server.server_url = None
        mock_server.command = "docker"
        mock_server.args = ["run", "-i", "--rm", "isokoliuk/mcp-searxng:latest"]
        
        # Mock subprocess creation to raise an exception
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("docker command not found")
            
            # Test that the method handles the error gracefully
            tools = await agent_manager.load_server_tools(mock_server)
            
            # Should return empty list when server fails to start
            assert tools == []
            
            # Verify subprocess was attempted
            mock_subprocess.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_command_based_server_communication_error(self):
        """Test error handling when command-based MCP server communication fails."""
        agent_manager = AgentToolManager("research_agent")
        agent_manager.config = {
            "allowed_mcp_servers": ["searxng"],
            "allowed_mcp_tools": {
                "searxng": ["searxng_web_search"]
            }
        }
        
        # Mock the MCP servers
        mock_server = MagicMock()
        mock_server.server_label = "searxng"
        mock_server.server_url = None
        mock_server.command = "docker"
        mock_server.args = ["run", "-i", "--rm", "isokoliuk/mcp-searxng:latest"]
        
        # Mock the subprocess creation
        mock_process = AsyncMock()
        
        # Mock the SubprocessMCPClient to raise an exception
        mock_subprocess_client = AsyncMock()
        mock_subprocess_client.call_tool.side_effect = Exception("Communication failed")
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_subprocess.return_value = mock_process
                
                with patch('app.core.tools.subprocess_mcp_client.SubprocessMCPClient') as mock_client_class:
                    mock_client_class.return_value = mock_subprocess_client
                    
                    # Test that the method handles communication errors
                    with pytest.raises(Exception, match="Communication failed"):
                        await agent_manager.execute_mcp_tool("searxng__searxng_web_search", {"query": "test"})

    def test_mcp_model_command_based_server_validation(self):
        """Test that MCP model correctly validates command-based server configurations."""
        from app.models.tools.mcp import MCP
        
        # Test valid command-based server config
        valid_command_server = MCP(
            server_label="searxng",
            command="docker",
            args=["run", "-i", "--rm", "isokoliuk/mcp-searxng:latest"],
            require_approval="never"
        )
        
        assert valid_command_server.server_label == "searxng"
        assert valid_command_server.command == "docker"
        assert valid_command_server.args == ["run", "-i", "--rm", "isokoliuk/mcp-searxng:latest"]
        assert valid_command_server.server_url is None
        assert valid_command_server.header is None
        
        # Test valid HTTP-based server config (existing functionality)
        valid_http_server = MCP(
            server_label="deepwiki",
            server_url="https://mcp.deepwiki.com/mcp",
            require_approval="never",
            header={"Authorization": ""}
        )
        
        assert valid_http_server.server_label == "deepwiki"
        assert valid_http_server.server_url == "https://mcp.deepwiki.com/mcp"
        assert valid_http_server.command is None
        assert valid_http_server.args is None

   