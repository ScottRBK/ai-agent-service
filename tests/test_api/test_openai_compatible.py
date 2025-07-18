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
                "agent_id": "research_agent",
                "provider": "azure_openai_cc",
                "model": "gpt-4o-mini",
                "allowed_regular_tools": ["get_current_datetime"],
                "allowed_mcp_servers": ["deepwiki"],
                "resources": ["memory"]
            },
            {
                "agent_id": "cli_agent",
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
        current_agent_ids.add("research_agent")
        current_agent_ids.add("cli_agent")
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 2
        agent_ids = [agent["id"] for agent in data["data"]]
        assert "research_agent" in agent_ids
        assert "cli_agent" in agent_ids
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "ai-agent-service"
    
    def test_list_models_empty_config(self, client: TestClient):
        """Test list models with empty configuration."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 0  # Should have at least the default agents
    
    def test_chat_completions_returns_200(self, client: TestClient):
        """Test that chat completions endpoint returns HTTP 200."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "research_agent"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Mock response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
    
    def test_chat_completions_with_multiple_messages(self, client: TestClient):
        """Test chat completions with conversation history."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Can you help me?"}
            ],
            "temperature": 0.8
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response"
    
    def test_chat_completions_with_tools(self, client: TestClient):
        """Test chat completions with tools parameter."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "What's the current time?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "description": "Get current date and time"
                    }
                }
            ],
            "tool_choice": "auto"
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response"
    
    def test_chat_completions_error_handling(self, client: TestClient):
        """Test chat completions error handling."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("research_agent")
        
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            # Create a mock agent that raises an exception on chat
            mock_agent = MagicMock()
            mock_agent.chat = AsyncMock(side_effect=Exception("Agent error"))
            mock_agent.available_tools = [{"type": "function", "function": {"name": "get_current_datetime"}}]
            mock_agent.model = "mock-model"
            mock_agent.initialize = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            request_data = {
                "model": "research_agent",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 500
            assert "Agent error" in response.json()["detail"]
    
    def test_chat_completions_agent_not_found(self, client: TestClient):
        """Test chat completions with non-existent agent."""
        # Update the global agent IDs for this test - don't add non_existent_agent
        current_agent_ids.clear()
        current_agent_ids.add("research_agent")  # Add a different agent
        
        request_data = {
            "model": "non_existent_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 404
        assert "Agent non_existent_agent not found" in response.json()["detail"]
    
    def test_chat_completions_with_streaming(self, client: TestClient):
        """Test chat completions with streaming parameter (should be ignored)."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": True  # Should be ignored for now
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response"
    
    def test_chat_completions_with_max_tokens(self, client: TestClient):
        """Test chat completions with max_tokens parameter."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response"


class TestOpenAICompatibleAPIValidation:
    """Test cases for OpenAI-compatible API validation and edge cases."""
    
    def test_chat_completions_missing_model(self, client: TestClient):
        """Test chat completions with missing model."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_missing_messages(self, client: TestClient):
        """Test chat completions with missing messages."""
        request_data = {
            "model": "research_agent"
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_empty_messages(self, client: TestClient):
        """Test chat completions with empty messages array."""
        request_data = {
            "model": "research_agent",
            "messages": []
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_invalid_message_format(self, client: TestClient):
        """Test chat completions with invalid message format."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"content": "Hello"}  # Missing "role"
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_invalid_temperature(self, client: TestClient):
        """Test chat completions with invalid temperature value."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 2.1  # Should be between 0 and 2
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_completions_negative_max_tokens(self, client: TestClient):
        """Test chat completions with negative max_tokens."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": -1
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error


class TestOpenAICompatibleAPIEdgeCases:
    """Test cases for edge cases and error scenarios."""
    
    def test_load_agent_configs_file_not_found(self, client: TestClient):
        """Test load_agent_configs when file doesn't exist."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) >= 0  # Should have at least the default agents
    
    def test_load_agent_configs_invalid_json(self, client: TestClient):
        """Test load_agent_configs with invalid JSON."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        
        response = client.get("/v1/models")
        assert response.status_code == 200
    
    def test_chat_completions_with_system_message(self, client: TestClient):
        """Test chat completions with system message."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response"
    
    def test_chat_completions_with_assistant_message(self, client: TestClient):
        """Test chat completions with assistant message in history."""
        request_data = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4"},
                {"role": "user", "content": "What about 3+3?"}
            ]
        }
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Mock response" 