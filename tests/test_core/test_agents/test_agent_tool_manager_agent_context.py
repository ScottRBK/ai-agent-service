"""
Unit tests for AgentToolManager agent_instance integration and agent context passing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.core.agents.agent_tool_manager import AgentToolManager


class TestAgentToolManagerAgentContext:
    """Test cases for AgentToolManager agent context handling"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent instance"""
        agent = Mock()
        agent.agent_id = "test_agent"
        agent.user_id = "test_user"
        agent.session_id = "test_session"
        agent.knowledge_base = Mock()
        return agent
    
    @pytest.fixture
    def mock_config(self):
        """Mock agent configuration"""
        return {
            "agent_id": "test_agent",
            "allowed_regular_tools": ["get_current_datetime"],
            "allowed_mcp_servers": {
                "deepwiki": {
                    "allowed_mcp_tools": ["search_wiki"]
                }
            }
        }
    
    def test_init_without_agent_instance(self):
        """Test AgentToolManager initialization without agent_instance"""
        manager = AgentToolManager("test_agent")
        assert manager.agent_id == "test_agent"
        assert manager.agent_instance is None
    
    def test_init_with_agent_instance(self, mock_agent):
        """Test AgentToolManager initialization with agent_instance"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        assert manager.agent_id == "test_agent"
        assert manager.agent_instance == mock_agent
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_without_agent_context(self, mock_config):
        """Test executing regular tool without agent context"""
        manager = AgentToolManager("test_agent")
        manager.config = mock_config
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="tool result")
            
            result = await manager.execute_regular_tool("get_current_datetime", {"timezone": "UTC"})
            
            # Should pass None as agent_context
            mock_registry.execute_tool_call.assert_called_once_with(
                "get_current_datetime", 
                {"timezone": "UTC"},
                agent_context=None
            )
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_with_agent_context(self, mock_config, mock_agent):
        """Test executing regular tool with agent context"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = mock_config
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="tool result")
            
            result = await manager.execute_regular_tool("get_current_datetime", {"timezone": "UTC"})
            
            # Should pass agent_instance as agent_context
            mock_registry.execute_tool_call.assert_called_once_with(
                "get_current_datetime", 
                {"timezone": "UTC"},
                agent_context=mock_agent
            )
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_unauthorized_with_agent_context(self, mock_config, mock_agent):
        """Test executing unauthorized regular tool with agent context"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = mock_config
        
        with pytest.raises(ValueError, match="Agent test_agent does not have access to tool add_two_numbers"):
            await manager.execute_regular_tool("add_two_numbers", {"a": 1, "b": 2})
    
    @pytest.mark.asyncio
    async def test_execute_regular_tool_when_config_allows_all_tools(self, mock_agent):
        """Test executing regular tool when config allows all tools"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = {
            "agent_id": "test_agent",
            "allowed_regular_tools": None  # Allow all tools
        }
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="tool result")
            
            result = await manager.execute_regular_tool("any_tool", {"param": "value"})
            
            # Should pass agent_context and execute successfully
            mock_registry.execute_tool_call.assert_called_once_with(
                "any_tool", 
                {"param": "value"},
                agent_context=mock_agent
            )
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_regular_tool_with_agent_context(self, mock_config, mock_agent):
        """Test execute_tool with regular tool passing agent context"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = mock_config
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="tool result")
            
            result = await manager.execute_tool("get_current_datetime", {"timezone": "UTC"})
            
            # Should pass agent_context through execute_regular_tool
            mock_registry.execute_tool_call.assert_called_once_with(
                "get_current_datetime", 
                {"timezone": "UTC"},
                agent_context=mock_agent
            )
            assert result == "tool result"
    
    @pytest.mark.asyncio
    async def test_execute_tool_mcp_tool_with_agent_context(self, mock_config, mock_agent):
        """Test execute_tool with MCP tool (agent context not passed to MCP tools)"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = mock_config
        
        # Mock the MCP servers
        mock_server = Mock()
        mock_server.server_label = "deepwiki"
        mock_server.server_url = "http://localhost:8080"
        mock_server.header = Mock()
        mock_server.header.authorization = None
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry.load_mcp_servers') as mock_load:
            mock_load.return_value = [mock_server]
            
            # Mock the MCP client
            mock_client = AsyncMock()
            mock_client.call_tool.return_value = "mcp result"
            
            with patch('app.core.agents.agent_tool_manager.Client') as mock_client_class:
                mock_client_class.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                
                result = await manager.execute_tool("deepwiki__search_wiki", {"query": "test"})
                
                # MCP tools don't use agent context - they use MCP client directly
                mock_client.call_tool.assert_called_once_with("search_wiki", {"query": "test"})
                assert result == "mcp result"
    
    def test_agent_instance_stored_correctly(self, mock_agent):
        """Test that agent_instance is stored correctly and accessible"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        
        # Should be able to access the stored agent_instance
        assert manager.agent_instance is mock_agent
        assert manager.agent_instance.agent_id == "test_agent"
        assert manager.agent_instance.user_id == "test_user"
    
    def test_agent_instance_can_be_modified(self, mock_agent):
        """Test that agent_instance can be modified after initialization"""
        manager = AgentToolManager("test_agent")
        assert manager.agent_instance is None
        
        # Set agent_instance after initialization
        manager.agent_instance = mock_agent
        assert manager.agent_instance == mock_agent
        
        # Modify to a different agent
        new_agent = Mock()
        new_agent.agent_id = "new_agent"
        manager.agent_instance = new_agent
        assert manager.agent_instance == new_agent
    
    @pytest.mark.asyncio
    async def test_agent_context_error_propagation(self, mock_config, mock_agent):
        """Test that errors in tools that use agent context are properly propagated"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        manager.config = mock_config
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(side_effect=AttributeError("Agent context missing attribute"))
            
            with pytest.raises(AttributeError, match="Agent context missing attribute"):
                await manager.execute_regular_tool("get_current_datetime", {"timezone": "UTC"})
    
    @pytest.mark.asyncio
    async def test_agent_context_none_handling(self, mock_config):
        """Test behavior when agent_instance is explicitly None"""
        manager = AgentToolManager("test_agent", agent_instance=None)
        manager.config = mock_config
        
        with patch('app.core.agents.agent_tool_manager.ToolRegistry') as mock_registry:
            mock_registry.execute_tool_call = AsyncMock(return_value="tool result")
            
            result = await manager.execute_regular_tool("get_current_datetime", {"timezone": "UTC"})
            
            # Should pass None as agent_context
            mock_registry.execute_tool_call.assert_called_once_with(
                "get_current_datetime", 
                {"timezone": "UTC"},
                agent_context=None
            )
            assert result == "tool result"
    
    def test_config_loading_with_agent_instance(self, mock_agent):
        """Test that config loading works correctly with agent_instance"""
        manager = AgentToolManager("research_agent", agent_instance=mock_agent)
        
        # Should load config normally regardless of agent_instance
        assert manager.agent_id == "research_agent"
        assert manager.agent_instance == mock_agent
        assert manager.config is not None
        assert "allowed_regular_tools" in manager.config
    
    @pytest.mark.asyncio
    async def test_cache_behavior_with_agent_instance(self, mock_agent):
        """Test that caching behavior is not affected by agent_instance"""
        manager = AgentToolManager("test_agent", agent_instance=mock_agent)
        
        # Clear cache
        manager.clear_cache()
        assert manager.mcp_tools_cache is None
        assert manager.mcp_servers_cache is None
        
        # Cache should work the same way regardless of agent_instance
        manager.mcp_tools_cache = []
        manager.mcp_servers_cache = []
        
        assert manager.mcp_tools_cache == []
        assert manager.mcp_servers_cache == []