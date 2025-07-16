import importlib
import pytest
import json
from unittest.mock import patch, mock_open
from pydantic import BaseModel, Field
from app.core.tools.tool_registry import register_tool, ToolRegistry, TOOL_REGISTRY
from app.models.tools.tools import ToolType


@pytest.fixture()
def clean_tool_registry():
    """Start each test with a fresh registry that only contains the dummy echo tool."""
    import app.core.tools.tool_registry as tr
    # This was way harder than it should have been.

    original_registry = dict(tr.TOOL_REGISTRY)

    # 1) Wipe the registry so we have a clean slate
    tr.TOOL_REGISTRY.clear()

    # 2) Re-register the dummy echo tool so tests can rely on it
    class _EchoParams(BaseModel):
        text: str = Field(description="Some text to echo")

    @tr.register_tool(
        name="echo",
        description="Echo a message",
        tool_type=ToolType.FUNCTION,
        examples=["Echo 'Hello, world!'"] ,
        params_model=_EchoParams,
    )
    def _echo_tool(text: str) -> str:  # noqa: D401
        return f"Echoed: {text}"

    # ---- run the test ----
    yield

    # 3) Clean up again for safety (important if other files run after this one)
    tr.TOOL_REGISTRY.clear()
    tr.TOOL_REGISTRY.update(original_registry)


@pytest.fixture
def mock_mcp_servers_data():
    """Mock MCP servers data for testing"""
    return [
        {
            "type": "mcp",
            "server_label": "example_mcp_server",
            "server_url": "https://example_mcp_server/mcp",
            "require_approval": "never",
            "header": {
                "Authorization": ""
            }
        },
        {
            "type": "mcp",
            "server_label": "another_mcp_server",
            "server_url": "https://another_mcp_server/mcp",
            "require_approval": "always",
            "header": {
                "Authorization": "Bearer token123"
            }
        }
    ]


def test_registration(clean_tool_registry):
    "Test that a tool correctly gets registered"
    assert "echo" in TOOL_REGISTRY
    schema = TOOL_REGISTRY["echo"]["schema"]
    assert schema.name == "echo"
    assert schema.description == "Echo a message"
    assert schema.type == ToolType.FUNCTION
    assert schema.parameters.properties == {"text": {"type": "string", "description": "Some text to echo", "title": "Text"}}
    assert schema.parameters.required == ["text"]
    assert schema.examples == ["Echo 'Hello, world!'"]


def test_convert_tool_registry_to_response_format(clean_tool_registry):
    """Test that the tool registry is converted to the correct format"""
    response_tools = ToolRegistry.convert_tool_registry_to_response_format()
    assert response_tools == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo a message",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Some text to echo", "title": "Text"}},
                "required": ["text"]
            }
        }
    ]


def test_convert_tool_registry_to_chat_completions_format(clean_tool_registry):
    """Test that the tool registry is converted to the correct format"""
    chat_completions_tools = ToolRegistry.convert_tool_registry_to_chat_completions_format()
    assert chat_completions_tools == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a message",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string", "description": "Some text to echo", "title": "Text"}},
                    "required": ["text"]
                }
            }
        }
    ]

def test_execute_tool_call_valid_tool(clean_tool_registry):
    """Test that the tool registry is executed correctly"""
    result = ToolRegistry.execute_tool_call("echo", {"text": "Hello, world!"})
    assert result == "Echoed: Hello, world!"

def test_execute_tool_call_invalid_tool():
    """Test that the tool registry is executed correctly"""
    with pytest.raises(ValueError):
        ToolRegistry.execute_tool_call("invalid_tool", {"text": "Hello, world!"})

@patch('builtins.open', new_callable=mock_open)
def test_load_mcp_servers_with_mock_open(mock_file, mock_mcp_servers_data):
    """Test loading MCP servers using mock_open"""
    mock_file.return_value.read.return_value = json.dumps(mock_mcp_servers_data)
    
    mcp_servers = ToolRegistry.load_mcp_servers()
    
    mock_file.assert_called_once_with("mcp.json", "r")
    
    assert len(mcp_servers) == 2
    assert mcp_servers[0].server_label == "example_mcp_server"
    assert mcp_servers[1].server_label == "another_mcp_server"



