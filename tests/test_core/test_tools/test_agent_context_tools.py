"""
Unit tests for tools that require agent context
"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.core.tools.tool_registry import ToolRegistry, register_tool, TOOL_REGISTRY
from pydantic import BaseModel, Field


class TestAgentContextTools:
    """Tests for tools that require agent context parameter"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with required attributes"""
        agent = Mock()
        agent.user_id = "test_user"
        agent.session_id = "test_session"
        agent.knowledge_base = Mock()
        agent.knowledge_base.search = AsyncMock(return_value=[])
        return agent
    
    @pytest.fixture
    def test_tool_with_context(self):
        """Register a test tool that requires agent context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_context_tool",
            description="Test tool that requires agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def test_context_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        yield
        # Cleanup
        if "test_context_tool" in TOOL_REGISTRY:
            del TOOL_REGISTRY["test_context_tool"]
    
    @pytest.fixture
    def test_tool_without_context(self):
        """Register a test tool that doesn't require agent context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_no_context_tool",
            description="Test tool without agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_no_context_tool(message: str) -> str:
            return f"Message: {message}"
        
        yield
        # Cleanup
        if "test_no_context_tool" in TOOL_REGISTRY:
            del TOOL_REGISTRY["test_no_context_tool"]
    
    @pytest.mark.asyncio
    async def test_tool_with_agent_context(self, mock_agent, test_tool_with_context):
        """Test that tools with agent_context parameter receive it correctly"""
        result = await ToolRegistry.execute_tool_call(
            "test_context_tool",
            {"message": "Hello"},
            agent_context=mock_agent
        )
        assert result == "User test_user says: Hello"
    
    @pytest.mark.asyncio
    async def test_tool_without_agent_context(self, test_tool_without_context):
        """Test that tools without agent_context still work"""
        result = await ToolRegistry.execute_tool_call(
            "test_no_context_tool",
            {"message": "Hello"}
        )
        assert result == "Message: Hello"
    
    @pytest.mark.asyncio
    async def test_tool_with_context_but_no_context_provided(self, test_tool_with_context):
        """Test that tools requiring context fail gracefully when no context provided"""
        # This should work but agent_context will be None
        with pytest.raises(AttributeError):
            await ToolRegistry.execute_tool_call(
                "test_context_tool",
                {"message": "Hello"}
            )
    
    @pytest.mark.asyncio
    async def test_knowledge_base_search_tool(self, mock_agent):
        """Test the actual search_knowledge_base tool"""
        # The tool should be registered already from imports
        assert "search_knowledge_base" in TOOL_REGISTRY
        
        result = await ToolRegistry.execute_tool_call(
            "search_knowledge_base",
            {"query": "test query", "search_type": "all", "limit": 5},
            agent_context=mock_agent
        )
        
        # Should return no results message since mock returns empty list
        assert "No results found" in result
        mock_agent.knowledge_base.search.assert_called_once()