"""
Tests for agent management API endpoints.

This module tests the agent management endpoints,
ensuring they return correct data structures, status codes, and
handle various scenarios properly.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.models.agents import ChatRequest, ChatResponse, AgentInfo, ConversationHistory
from tests.conftest import current_agent_ids


class TestAgentManagementAPI:
    """Test cases for the agent management API endpoints."""
    
    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration for testing."""
        return [
            {
                "agent_id": "test_agent",
                "provider": "azure_openai_cc",
                "model": "gpt-4o-mini",
                "allowed_regular_tools": ["get_current_datetime"],
                "allowed_mcp_servers": ["deepwiki"],
                "resources": ["memory"]
            },
            {
                "agent_id": "research_agent",
                "provider": "ollama",
                "model": "qwen3:4b",
                "allowed_regular_tools": ["get_current_datetime"],
                "resources": ["memory"]
            }
        ]
    
    @pytest.fixture
    def mock_agent_config_file(self, sample_agent_config):
        """Create a temporary agent config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_agent_config, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_list_agents_returns_200(self, client: TestClient, mock_agent_config_file):
        """Test that list agents endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_agent")
        
        response = client.get("/agents/")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # Should have at least the test_agent
        assert any(agent["agent_id"] == "test_agent" for agent in data)
    
    def test_list_agents_empty_config(self, client: TestClient):
        """Test list agents with empty configuration."""
        with patch('app.api.routes.agents.load_agent_configs') as mock_load:
            mock_load.return_value = []
            
            response = client.get("/agents/")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 0
    
    def test_get_agent_info_returns_200(self, client: TestClient):
        """Test that get agent info endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_agent")
        
        response = client.get("/agents/test_agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "test_agent"
        assert data["provider"] == "azure_openai_cc"
        assert data["model"] == "gpt-4o-mini"
        assert "memory" in data["resources"]
        assert data["has_memory"] is True
    
    def test_get_agent_info_not_found(self, client: TestClient):
        """Test get agent info with non-existent agent."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_agent")
        
        response = client.get("/agents/non_existent_agent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_chat_with_agent_returns_200(self, client: TestClient):
        """Test that chat with agent endpoint returns HTTP 200."""
        request_data = {
            "message": "Hello",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        response = client.post("/agents/test_agent/chat", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Mock response"
        assert data["agent_id"] == "test_agent"
        assert data["user_id"] == "test_user"
        assert data["session_id"] == "test_session"
        assert data["model_used"] == "mock-model"
        assert data["tools_available"] == 1
    
    def test_chat_with_agent_with_model_override(self, client: TestClient):
        """Test chat with agent with model override."""
        request_data = {
            "message": "Hello",
            "user_id": "test_user",
            "session_id": "test_session",
            "model": "custom-model",
            "model_settings": {"temperature": 0.8}
        }
        response = client.post("/agents/test_agent/chat", json=request_data)
        assert response.status_code == 200
    
    def test_chat_with_agent_error_handling(self, client: TestClient):
        """Test chat with agent error handling."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_agent")
        
        # Simulate error by patching the mock agent's chat method to raise
        with patch('app.api.routes.agents.APIAgent') as mock_agent_class:
            # Create a mock agent that raises an exception on chat
            mock_agent = MagicMock()
            mock_agent.chat = AsyncMock(side_effect=Exception("Test error"))
            mock_agent.available_tools = [{"type": "function", "function": {"name": "get_current_datetime"}}]
            mock_agent.model = "mock-model"
            mock_agent.initialize = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            request_data = {
                "message": "Hello",
                "user_id": "test_user",
                "session_id": "test_session"
            }
            response = client.post("/agents/test_agent/chat", json=request_data)
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]
    
    def test_get_conversation_history_returns_200(self, client: TestClient):
        """Test that get conversation history endpoint returns HTTP 200."""
        response = client.get("/agents/test_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["agent_id"] == "test_agent"
        assert data["user_id"] == "test_user"
        assert isinstance(data["messages"], list)
    
    def test_clear_conversation_history_returns_200(self, client: TestClient):
        """Test that clear conversation history endpoint returns HTTP 200."""
        response = client.delete("/agents/test_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 200
        data = response.json()
        assert "cleared" in data["message"]
        assert "test_session" in data["message"]
    
    def test_clear_conversation_history_error_handling(self, client: TestClient):
        """Test clear conversation history error handling."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_agent")
        
        with patch('app.api.routes.agents.APIAgent') as mock_agent_class:
            # Create a mock agent that raises an exception on clear_conversation
            mock_agent = MagicMock()
            mock_agent.clear_conversation = AsyncMock(side_effect=Exception("Clear error"))
            mock_agent.available_tools = [{"type": "function", "function": {"name": "get_current_datetime"}}]
            mock_agent.model = "mock-model"
            mock_agent.initialize = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            response = client.delete("/agents/test_agent/conversation/test_session?user_id=test_user")
            assert response.status_code == 500
            assert "Clear error" in response.json()["detail"]


class TestAgentManagementAPIValidation:
    """Test cases for API validation and edge cases."""
    
    def test_chat_request_validation(self, client: TestClient):
        """Test chat request validation."""
        # Test missing required field
        request_data = {
            "user_id": "test_user",
            "session_id": "test_session"
            # Missing "message" field
        }
        
        response = client.post("/agents/test_agent/chat", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_chat_request_with_optional_fields(self, client: TestClient):
        """Test chat request with all optional fields."""
        request_data = {
            "message": "Hello",
            "user_id": "custom_user",
            "session_id": "custom_session",
            "model": "custom_model",
            "model_settings": {"temperature": 0.9, "max_tokens": 1000}
        }
        response = client.post("/agents/test_agent/chat", json=request_data)
        assert response.status_code == 200 