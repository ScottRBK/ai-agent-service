# tests/conftest.py
import pytest
import json
import os
from fastapi.testclient import TestClient
from app.main import app
from app.config.settings import Settings
from unittest.mock import patch, AsyncMock, MagicMock

current_agent_ids = set()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_providers():
    """Automatically mock all providers to prevent real API calls during tests."""
    with patch('app.core.providers.manager.ProviderManager') as mock_provider_manager:
        # Mock provider manager to return fake provider info
        mock_provider_info = {
            "name": "Mock Provider",
            "description": "Mock provider for testing",
            "class": AsyncMock,
            "config_class": AsyncMock
        }
        mock_provider_manager.return_value.get_provider.return_value = mock_provider_info
        
        # Mock the provider instance
        mock_provider_instance = AsyncMock()
        mock_provider_instance.initialize = AsyncMock()
        mock_provider_instance.send_chat = AsyncMock(return_value="Mock response")
        mock_provider_instance.config.default_model = "mock-model"
        
        # Mock the provider class to return our mock instance
        mock_provider_class = AsyncMock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_info["class"] = mock_provider_class
        
        yield mock_provider_manager

def load_example_agent_configs():
    """Load agent configurations from agent_config.example.json for testing."""
    try:
        config_path = "agent_config.example.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                configs = json.load(f)
            
            if isinstance(configs, dict):
                return [configs]
            elif isinstance(configs, list):
                return configs
            else:
                return []
        else:
            # Fallback to default configs if example file doesn't exist
            return [
                {"agent_id": "test_research_agent", "provider": "azure_openai_cc", "model": "gpt-4o-mini", "resources": ["memory"]},
                {"agent_id": "test_cli_agent", "provider": "ollama", "model": "qwen3:4b", "resources": ["memory"]},
                {"agent_id": "test_agent", "provider": "azure_openai_cc", "model": "gpt-4o-mini", "resources": ["memory"]}
            ]
    except Exception as e:
        # Fallback to default configs if there's an error
        return [
            {"agent_id": "test_research_agent", "provider": "azure_openai_cc", "model": "gpt-4o-mini", "resources": ["memory"]},
            {"agent_id": "test_cli_agent", "provider": "ollama", "model": "qwen3:4b", "resources": ["memory"]},
            {"agent_id": "test_agent", "provider": "azure_openai_cc", "model": "gpt-4o-mini", "resources": ["memory"]}
        ]

@pytest.fixture(autouse=True)
def mock_load_agent_configs(monkeypatch):
    """Patch load_agent_configs to load from agent_config.example.json for testing."""
    from app.api.routes import agents as agents_module
    from app.api.routes import openai_compatible as openai_compatible_module
    
    # Create a mock function that loads from the example config
    def mock_load_agent_configs(*args, **kwargs):
        global current_agent_ids
        configs = load_example_agent_configs()
        current_agent_ids = set()
        if isinstance(configs, list):
            for agent in configs:
                if isinstance(agent, dict) and "agent_id" in agent:
                    current_agent_ids.add(agent["agent_id"])
        elif isinstance(configs, dict) and "agent_id" in configs:
            current_agent_ids.add(configs["agent_id"])
        return configs
    
    monkeypatch.setattr(agents_module, "load_agent_configs", mock_load_agent_configs)
    monkeypatch.setattr(openai_compatible_module, "load_agent_configs", mock_load_agent_configs)
    yield

@pytest.fixture(autouse=True)
def mock_api_agent():
    """Automatically mock APIAgent to prevent real initialization during tests, and raise 404 for non-existent agents."""
    with patch('app.api.routes.agents.APIAgent') as mock_agent_class, \
         patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class2:
        def agent_side_effect(*args, **kwargs):
            agent_id = kwargs.get("agent_id") or (args[0] if args else None)
            global current_agent_ids
            if agent_id not in current_agent_ids:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            mock_agent = MagicMock()
            mock_agent.available_tools = [
                {"type": "function", "function": {"name": "get_current_datetime"}}
            ]
            mock_agent.model = "mock-model"
            # Use AsyncMock with side_effect that can be overridden
            mock_agent.chat = AsyncMock(return_value="Mock response")
            mock_agent.get_conversation_history = AsyncMock(return_value=[])
            mock_agent.clear_conversation = AsyncMock()
            def mock_initialize():
                mock_agent.available_tools = [
                    {"type": "function", "function": {"name": "get_current_datetime"}}
                ]
                mock_agent.model = "mock-model"
                return None
            mock_agent.initialize = AsyncMock(side_effect=mock_initialize)
            return mock_agent
        mock_agent_class.side_effect = agent_side_effect
        mock_agent_class2.side_effect = agent_side_effect
        yield mock_agent_class

@pytest.fixture(autouse=True)
def mock_agent_initialize():
    """Mock the initialize method specifically to prevent real provider calls."""
    with patch('app.core.agents.api_agent.APIAgent.initialize') as mock_initialize:
        mock_initialize.return_value = None
        yield mock_initialize

@pytest.fixture(autouse=True)
def mock_tool_manager():
    """Automatically mock tool manager to prevent real tool calls during tests."""
    with patch('app.core.agents.agent_tool_manager.AgentToolManager') as mock_tool_manager:
        mock_instance = AsyncMock()
        mock_instance.config = {
            "agent_id": "test_agent",
            "provider": "mock_provider",
            "model": "mock-model",
            "allowed_regular_tools": ["get_current_datetime"],
            "resources": ["memory"]
        }
        mock_instance.get_available_tools = AsyncMock(return_value=[])
        mock_tool_manager.return_value = mock_instance
        yield mock_tool_manager

@pytest.fixture(autouse=True)
def mock_resource_manager():
    """Automatically mock resource manager to prevent real resource calls during tests."""
    with patch('app.core.agents.agent_resource_manager.AgentResourceManager') as mock_resource_manager:
        mock_instance = AsyncMock()
        mock_instance.get_model_config.return_value = ("mock-model", {})
        mock_instance.get_memory_resource = AsyncMock(return_value=None)
        mock_resource_manager.return_value = mock_instance
        yield mock_resource_manager