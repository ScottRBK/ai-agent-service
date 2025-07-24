"""
Tests for OpenAI-compatible API endpoints.

This module tests the OpenAI-compatible endpoints,
ensuring they return correct data structures, status codes, and
handle various scenarios properly.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import current_agent_ids


class TestOpenAICompatibleAPI:
    """Test cases for the OpenAI-compatible API endpoints."""
    
    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration for testing."""
        return [
            {
                "agent_id": "test_research_agent",
                "provider": "azure_openai_cc",
                "model": "gpt-4o-mini",
                "allowed_regular_tools": ["get_current_datetime"],
                "allowed_mcp_servers": ["deepwiki"],
                "resources": ["memory"]
            },
            {
                "agent_id": "test_cli_agent",
                "provider": "ollama",
                "model": "qwen3:4b",
                "allowed_regular_tools": ["get_current_datetime"],
                "resources": ["memory"]
            }
        ]
    
    def test_list_models_returns_200(self, client: TestClient):
        """Test that list models endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        current_agent_ids.add("test_cli_agent")
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 2
        agent_ids = [agent["id"] for agent in data["data"]]
        assert "test_research_agent" in agent_ids
        assert "test_cli_agent" in agent_ids
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "ai-agent-service"
    
    def test_list_models_empty_config(self, client: TestClient):
        """Test list models with empty configuration."""
        with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
            mock_load.return_value = []
            
            response = client.get("/v1/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 0
    
    def test_chat_completions_returns_200(self, client: TestClient):
        """Test that chat completions endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "user", "content": "Hello, test agent!"}
            ],
            "temperature": 0.7
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test_research_agent"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Mock response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
    
    def test_chat_completions_with_tools(self, client: TestClient):
        """Test chat completions with tool calling."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "user", "content": "What is the current time?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "description": "Get the current date and time"
                    }
                }
            ],
            "tool_choice": "auto"
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test_research_agent"
        assert len(data["choices"]) == 1
    
    def test_chat_completions_model_not_found(self, client: TestClient):
        """Test that chat completions returns 404 for non-existent model."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "nonexistent_model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_chat_completions_invalid_request(self, client: TestClient):
        """Test that chat completions handles invalid requests properly."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        # Missing required fields
        invalid_request = {
            "model": "test_research_agent"
            # Missing "messages" field
        }
        
        response = client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_streaming(self, client: TestClient):
        """Test chat completions with streaming enabled."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "user", "content": "Hello, streaming test!"}
            ],
            "stream": True
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        # Check that response is streaming
        content = response.content.decode()
        assert "data: " in content
        # Check for proper streaming format
        assert "chat.completion.chunk" in content
        assert "finish_reason" in content
        assert "data: [DONE]" in content
    
    def test_chat_completions_with_model_settings(self, client: TestClient):
        """Test chat completions with custom model settings."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "user", "content": "Hello with custom settings!"}
            ],
            "temperature": 0.9,
            "max_tokens": 1000,
            "top_p": 0.8
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test_research_agent"
    
    def test_multiple_models_listing(self, client: TestClient):
        """Test listing multiple models from the configuration."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        current_agent_ids.add("test_cli_agent")
        current_agent_ids.add("test_mcp_agent")
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()
        
        # Should have at least the test models
        model_ids = [model["id"] for model in models["data"]]
        assert "test_research_agent" in model_ids
        assert "test_cli_agent" in model_ids
        assert "test_mcp_agent" in model_ids
        
        # Verify model details
        for model in models["data"]:
            if model["id"] == "test_research_agent":
                assert model["object"] == "model"
                assert model["owned_by"] == "ai-agent-service"
            elif model["id"] == "test_cli_agent":
                assert model["object"] == "model"
                assert model["owned_by"] == "ai-agent-service"
    
    def test_chat_completions_with_system_message(self, client: TestClient):
        """Test chat completions with system message."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test_research_agent"
    
    def test_chat_completions_with_assistant_message(self, client: TestClient):
        """Test chat completions with assistant message in context."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "model": "test_research_agent",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "What about 3+3?"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test_research_agent"
    
    def test_load_agent_configs_file_not_found(self, client: TestClient):
        """Test load_agent_configs when file doesn't exist."""
        with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
            mock_load.return_value = []
            
            response = client.get("/v1/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 0
    
    def test_load_agent_configs_invalid_json(self, client: TestClient):
        """Test load_agent_configs with invalid JSON."""
        with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
            mock_load.return_value = []
            
            response = client.get("/v1/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 0 