"""
Unit tests for ToolRegistry async execution and signature inspection for agent context.
"""

import pytest
import inspect
from unittest.mock import Mock, AsyncMock
from pydantic import BaseModel, Field, ValidationError
from app.core.tools.tool_registry import register_tool, ToolRegistry, TOOL_REGISTRY


class TestToolRegistryAgentContext:
    """Test cases for ToolRegistry agent context handling"""
    
    @pytest.fixture
    def clean_registry(self):
        """Clean tool registry for each test"""
        original_registry = dict(TOOL_REGISTRY)
        TOOL_REGISTRY.clear()
        yield
        TOOL_REGISTRY.clear()
        TOOL_REGISTRY.update(original_registry)
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent instance"""
        agent = Mock()
        agent.user_id = "test_user"
        agent.session_id = "test_session"
        agent.knowledge_base = Mock()
        return agent
    
    def test_signature_inspection_detects_agent_context_sync(self, clean_registry):
        """Test that signature inspection correctly detects agent_context parameter in sync function"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_sync_with_context",
            description="Test sync tool with agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_sync_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        # Check signature inspection directly
        sig = inspect.signature(test_sync_tool)
        assert 'agent_context' in sig.parameters
        
        # Verify tool was registered
        assert "test_sync_with_context" in TOOL_REGISTRY
        implementation = TOOL_REGISTRY["test_sync_with_context"]["implementation"]
        impl_sig = inspect.signature(implementation)
        assert 'agent_context' in impl_sig.parameters
    
    def test_signature_inspection_detects_agent_context_async(self, clean_registry):
        """Test that signature inspection correctly detects agent_context parameter in async function"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_async_with_context",
            description="Test async tool with agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def test_async_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        # Check that function is async
        assert inspect.iscoroutinefunction(test_async_tool)
        
        # Check signature inspection
        sig = inspect.signature(test_async_tool)
        assert 'agent_context' in sig.parameters
        
        # Verify tool was registered
        assert "test_async_with_context" in TOOL_REGISTRY
        implementation = TOOL_REGISTRY["test_async_with_context"]["implementation"]
        assert inspect.iscoroutinefunction(implementation)
    
    def test_signature_inspection_no_agent_context_sync(self, clean_registry):
        """Test that signature inspection correctly identifies tools without agent_context in sync function"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_sync_no_context",
            description="Test sync tool without agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_sync_tool(message: str) -> str:
            return f"Message: {message}"
        
        # Check signature inspection
        sig = inspect.signature(test_sync_tool)
        assert 'agent_context' not in sig.parameters
        assert 'message' in sig.parameters
    
    def test_signature_inspection_no_agent_context_async(self, clean_registry):
        """Test that signature inspection correctly identifies tools without agent_context in async function"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_async_no_context",
            description="Test async tool without agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def test_async_tool(message: str) -> str:
            return f"Message: {message}"
        
        # Check signature inspection
        sig = inspect.signature(test_async_tool)
        assert 'agent_context' not in sig.parameters
        assert 'message' in sig.parameters
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_sync_with_agent_context(self, clean_registry, mock_agent):
        """Test executing sync tool that requires agent_context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_sync_with_context",
            description="Test sync tool with agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_sync_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        result = await ToolRegistry.execute_tool_call(
            "test_sync_with_context",
            {"message": "Hello"},
            agent_context=mock_agent
        )
        
        assert result == "User test_user says: Hello"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_async_with_agent_context(self, clean_registry, mock_agent):
        """Test executing async tool that requires agent_context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_async_with_context",
            description="Test async tool with agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def test_async_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        result = await ToolRegistry.execute_tool_call(
            "test_async_with_context",
            {"message": "Hello"},
            agent_context=mock_agent
        )
        
        assert result == "User test_user says: Hello"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_sync_without_agent_context(self, clean_registry):
        """Test executing sync tool that doesn't require agent_context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_sync_no_context",
            description="Test sync tool without agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_sync_tool(message: str) -> str:
            return f"Message: {message}"
        
        result = await ToolRegistry.execute_tool_call(
            "test_sync_no_context",
            {"message": "Hello"}
        )
        
        assert result == "Message: Hello"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_async_without_agent_context(self, clean_registry):
        """Test executing async tool that doesn't require agent_context"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_async_no_context",
            description="Test async tool without agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def test_async_tool(message: str) -> str:
            return f"Message: {message}"
        
        result = await ToolRegistry.execute_tool_call(
            "test_async_no_context",
            {"message": "Hello"}
        )
        
        assert result == "Message: Hello"
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_context_but_none_provided(self, clean_registry):
        """Test executing tool that requires agent_context but None is provided"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_context_required",
            description="Test tool that requires agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        # Should raise AttributeError when agent_context is None
        with pytest.raises(AttributeError):
            await ToolRegistry.execute_tool_call(
                "test_context_required",
                {"message": "Hello"},
                agent_context=None
            )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_with_context_but_not_provided(self, clean_registry):
        """Test executing tool that requires agent_context but no parameter is provided"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        @register_tool(
            name="test_context_required",
            description="Test tool that requires agent context",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_tool(agent_context, message: str) -> str:
            return f"User {agent_context.user_id} says: {message}"
        
        # Should raise AttributeError when agent_context is None (default)
        with pytest.raises(AttributeError):
            await ToolRegistry.execute_tool_call(
                "test_context_required",
                {"message": "Hello"}
                # No agent_context parameter
            )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_validation_error(self, clean_registry):
        """Test that validation errors are properly raised"""
        class TestParams(BaseModel):
            number: int = Field(description="A number")
        
        @register_tool(
            name="test_validation",
            description="Test tool for validation",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_tool(number: int) -> str:
            return f"Number: {number}"
        
        # Should raise ValueError with validation error
        with pytest.raises(ValueError, match="Argument validation error"):
            await ToolRegistry.execute_tool_call(
                "test_validation",
                {"number": "not_a_number"}  # Invalid type
            )
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_nonexistent_tool(self, clean_registry):
        """Test that calling non-existent tool raises appropriate error"""
        with pytest.raises(ValueError, match="Tool 'nonexistent_tool' not registered"):
            await ToolRegistry.execute_tool_call(
                "nonexistent_tool",
                {"param": "value"}
            )
    
    @pytest.mark.asyncio
    async def test_signature_inspection_edge_cases(self, clean_registry):
        """Test signature inspection with various parameter configurations"""
        class TestParams(BaseModel):
            message: str = Field(description="Test message")
        
        # Tool with agent_context as second parameter
        @register_tool(
            name="test_context_second",
            description="Test tool with agent_context as second param",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def test_tool_context_second(message: str, agent_context) -> str:
            return f"Message: {message}, User: {agent_context.user_id}"
        
        # Should still detect agent_context regardless of position
        sig = inspect.signature(test_tool_context_second)
        assert 'agent_context' in sig.parameters
        
        # Execution should work with agent_context as keyword argument
        mock_agent = Mock()
        mock_agent.user_id = "test_user"
        
        result = await ToolRegistry.execute_tool_call(
            "test_context_second",
            {"message": "Hello"},
            agent_context=mock_agent
        )
        
        assert "Message: Hello" in result
        assert "User: test_user" in result
    
    @pytest.mark.asyncio
    async def test_async_and_sync_tool_handling(self, clean_registry, mock_agent):
        """Test that both async and sync tools are handled correctly"""
        class TestParams(BaseModel):
            value: str = Field(description="Test value")
        
        # Sync tool
        @register_tool(
            name="test_sync",
            description="Test sync tool",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        def sync_tool(agent_context, value: str) -> str:
            return f"Sync: {value} from {agent_context.user_id}"
        
        # Async tool
        @register_tool(
            name="test_async",
            description="Test async tool",
            tool_type="function",
            examples=["Test example"],
            params_model=TestParams
        )
        async def async_tool(agent_context, value: str) -> str:
            return f"Async: {value} from {agent_context.user_id}"
        
        # Both should work
        sync_result = await ToolRegistry.execute_tool_call(
            "test_sync",
            {"value": "test1"},
            agent_context=mock_agent
        )
        
        async_result = await ToolRegistry.execute_tool_call(
            "test_async",
            {"value": "test2"},
            agent_context=mock_agent
        )
        
        assert sync_result == "Sync: test1 from test_user"
        assert async_result == "Async: test2 from test_user"