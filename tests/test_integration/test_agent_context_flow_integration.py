"""
Integration tests for full agent context flow from agent to tool execution.
Tests the complete path: BaseAgent → Provider → AgentToolManager → ToolRegistry → Tool
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pydantic import BaseModel, Field
from app.core.agents.base_agent import BaseAgent
from app.core.tools.tool_registry import register_tool, TOOL_REGISTRY


class TestAgentContextFlowIntegration:
    """Integration tests for agent context flow"""
    
    @pytest.fixture
    def clean_registry(self):
        """Clean tool registry for each test"""
        original_registry = dict(TOOL_REGISTRY)
        TOOL_REGISTRY.clear()
        yield
        TOOL_REGISTRY.clear()
        TOOL_REGISTRY.update(original_registry)
    
    @pytest.fixture
    def mock_provider_setup(self):
        """Mock provider setup for agent initialization"""
        provider = Mock()
        provider.initialize = AsyncMock()
        provider.send_chat = AsyncMock(return_value="Test response")
        provider.config = Mock(default_model="gpt-3.5-turbo")
        provider.agent_instance = None
        # Use the real execute_tool_call method from BaseProvider
        from app.core.providers.base import BaseProvider
        provider.execute_tool_call = BaseProvider.execute_tool_call.__get__(provider, BaseProvider)
        
        manager = Mock()
        manager.get_provider.return_value = {
            "class": Mock(return_value=provider),
            "config_class": Mock
        }
        
        return provider, manager
    
    @pytest.fixture
    def knowledge_base_tool(self, clean_registry):
        """Register a test knowledge base tool that requires agent context"""
        class SearchParams(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=5, description="Result limit")
        
        @register_tool(
            name="test_search_knowledge_base",
            description="Test search knowledge base",
            tool_type="function",
            examples=["Search for information"],
            params_model=SearchParams
        )
        async def test_search_kb(agent_context, query: str, limit: int = 5) -> str:
            if not hasattr(agent_context, 'knowledge_base') or not agent_context.knowledge_base:
                return "Knowledge base not available"
            
            # Mock search results
            results = [
                {"title": "Result 1", "content": f"Content about {query}"},
                {"title": "Result 2", "content": f"More info on {query}"}
            ]
            
            return f"Found {len(results)} results for '{query}'"
        
        return "test_search_knowledge_base"
    
    @pytest.fixture
    def regular_tool(self, clean_registry):
        """Register a test regular tool that doesn't require agent context"""
        class EchoParams(BaseModel):
            message: str = Field(description="Message to echo")
        
        @register_tool(
            name="test_echo",
            description="Test echo tool",
            tool_type="function",
            examples=["Echo a message"],
            params_model=EchoParams
        )
        def test_echo(message: str) -> str:
            return f"Echo: {message}"
        
        return "test_echo"
    
    @pytest.mark.asyncio
    async def test_full_agent_context_flow_with_knowledge_base(self, mock_provider_setup, knowledge_base_tool):
        """Test complete flow from agent to knowledge base tool execution"""
        provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Mock agent configuration
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resources": ["knowledge_base"],
            "allowed_regular_tools": ["test_search_knowledge_base"]
        }
        
        # Mock knowledge base
        mock_kb = Mock()
        mock_kb.search = AsyncMock(return_value=[])
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch.object(agent, 'create_knowledge_base', AsyncMock(return_value=mock_kb)):
                        # Initialize agent
                        await agent.initialize()
                        
                        # Verify agent instance is set on provider
                        assert provider.agent_instance == agent
                        assert agent.knowledge_base == mock_kb
                        
                        # Execute tool through provider
                        result = await provider.execute_tool_call(
                            "test_search_knowledge_base",
                            {"query": "test query", "limit": 3},
                            agent_id="test_agent"
                        )
                        
                        # Should execute successfully and find results
                        assert "Found 2 results for 'test query'" in result
    
    @pytest.mark.asyncio
    async def test_full_agent_context_flow_without_knowledge_base(self, mock_provider_setup, knowledge_base_tool):
        """Test flow when agent doesn't have knowledge base configured"""
        provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Mock agent configuration without knowledge_base resource
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "resources": [],  # No knowledge_base
            "allowed_regular_tools": ["test_search_knowledge_base"]
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    # Initialize agent (without knowledge base)
                    await agent.initialize()
                    
                    # Verify agent instance is set but no knowledge base
                    assert provider.agent_instance == agent
                    assert agent.knowledge_base is None
                    
                    # Execute tool through provider
                    result = await provider.execute_tool_call(
                        "test_search_knowledge_base",
                        {"query": "test query"},
                        agent_id="test_agent"
                    )
                    
                    # Should return knowledge base not available message
                    assert "Knowledge base not available" in result
    
    @pytest.mark.asyncio
    async def test_full_agent_context_flow_regular_tool(self, mock_provider_setup, regular_tool):
        """Test flow with regular tool that doesn't require agent context"""
        provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Mock agent configuration
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "model": "gpt-4",
            "allowed_regular_tools": ["test_echo"]
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    # Initialize agent
                    await agent.initialize()
                    
                    # Execute regular tool
                    result = await provider.execute_tool_call(
                        "test_echo",
                        {"message": "Hello World"},
                        agent_id="test_agent"
                    )
                    
                    assert result == "Echo: Hello World"
    
    @pytest.mark.asyncio
    async def test_agent_context_fallback_behavior(self, mock_provider_setup, knowledge_base_tool):
        """Test fallback when agent_instance is provided explicitly vs stored"""
        provider, provider_manager = mock_provider_setup
        
        # Create two different agents
        agent1 = BaseAgent("agent1")
        agent2 = BaseAgent("agent2")
        
        # Mock knowledge base for agent2
        mock_kb = Mock()
        agent2.knowledge_base = mock_kb
        
        # Set agent1 as stored instance on provider
        provider.agent_instance = agent1
        
        mock_config = {
            "agent_id": "agent2",
            "provider": "azure_openai_cc",
            "allowed_regular_tools": ["test_search_knowledge_base"]
        }
        
        with patch.object(agent2.tool_manager, 'config', mock_config):
            # Execute tool with explicit agent_instance (should use agent2, not stored agent1)
            result = await provider.execute_tool_call(
                "test_search_knowledge_base",
                {"query": "test"},
                agent_id="agent2",
                agent_instance=agent2  # Explicit agent instance
            )
            
            # Should use agent2's knowledge base and succeed
            assert "Found 2 results" in result
   
    
    @pytest.mark.asyncio
    async def test_agent_context_with_no_agent_id(self, mock_provider_setup, regular_tool):
        """Test tool execution without agent_id falls back to ToolRegistry"""
        provider, provider_manager = mock_provider_setup
        
        # Execute tool without agent_id - should use ToolRegistry directly
        result = await provider.execute_tool_call(
            "test_echo",
            {"message": "Direct execution"}
            # No agent_id provided
        )
        
        assert result == "Echo: Direct execution"
    
    @pytest.mark.asyncio
    async def test_multiple_agent_context_tools_in_sequence(self, mock_provider_setup, clean_registry):
        """Test executing multiple agent context tools in sequence"""
        provider, provider_manager = mock_provider_setup
        
        # Register multiple tools that use agent context
        class InfoParams(BaseModel):
            info_type: str = Field(description="Type of info to get")
        
        @register_tool(
            name="get_user_info",
            description="Get user information",
            tool_type="function",
            examples=["Get user info"],
            params_model=InfoParams
        )
        def get_user_info(agent_context, info_type: str) -> str:
            return f"User {agent_context.user_id} info: {info_type}"
        
        @register_tool(
            name="get_session_info",
            description="Get session information",
            tool_type="function",
            examples=["Get session info"],
            params_model=InfoParams
        )
        def get_session_info(agent_context, info_type: str) -> str:
            return f"Session {agent_context.session_id} info: {info_type}"
        
        agent = BaseAgent("test_agent", user_id="user123", session_id="session456")
        
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "allowed_regular_tools": ["get_user_info", "get_session_info"]
        }
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    await agent.initialize()
                    
                    # Execute multiple tools in sequence
                    result1 = await provider.execute_tool_call(
                        "get_user_info",
                        {"info_type": "preferences"},
                        agent_id="test_agent"
                    )
                    
                    result2 = await provider.execute_tool_call(
                        "get_session_info",
                        {"info_type": "duration"},
                        agent_id="test_agent"
                    )
                    
                    assert result1 == "User user123 info: preferences"
                    assert result2 == "Session session456 info: duration"
    
    @pytest.mark.asyncio
    async def test_agent_context_with_resource_config_integration(self, mock_provider_setup, knowledge_base_tool):
        """Test agent context with resource_config-based configuration"""
        provider, provider_manager = mock_provider_setup
        
        agent = BaseAgent("test_agent")
        
        # Mock configuration with resource_config structure
        mock_config = {
            "agent_id": "test_agent",
            "provider": "azure_openai_cc",
            "resources": ["knowledge_base"],
            "allowed_regular_tools": ["test_search_knowledge_base"],
            "resource_config": {
                "knowledge_base": {
                    "embedding_provider": "azure_openai_cc",
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_size": 800
                }
            }
        }
        
        mock_kb = Mock()
        
        with patch.object(agent, 'provider_manager', provider_manager):
            with patch.object(agent.tool_manager, 'config', mock_config):
                with patch.object(agent.tool_manager, 'get_available_tools', AsyncMock(return_value=[])):
                    with patch.object(agent, 'create_knowledge_base', AsyncMock(return_value=mock_kb)):
                        await agent.initialize()
                        
                        # Verify resource_config was processed correctly
                        kb_config = agent._get_resource_config("knowledge_base")
                        assert kb_config["embedding_provider"] == "azure_openai_cc"
                        assert kb_config["embedding_model"] == "text-embedding-ada-002"
                        assert kb_config["chunk_size"] == 800
                        
                        # Execute tool with agent context
                        result = await provider.execute_tool_call(
                            "test_search_knowledge_base",
                            {"query": "test"},
                            agent_id="test_agent"
                        )
                        
                        assert "Found 2 results" in result