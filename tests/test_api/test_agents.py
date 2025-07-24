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
                "agent_id": "test_research_agent",
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
        current_agent_ids.add("test_research_agent")
        
        response = client.get("/agents/")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # Should have at least the test_research_agent
        assert any(agent["agent_id"] == "test_research_agent" for agent in data)
    
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
        current_agent_ids.add("test_cli_agent")
        
        response = client.get("/agents/test_cli_agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "test_cli_agent"
        assert data["provider"] == "ollama"
        assert data["model"] == "qwen3:1.7b"
        assert "memory" in data["resources"]
        assert data["has_memory"] is True
    
    def test_get_agent_info_not_found(self, client: TestClient):
        """Test that get agent info returns 404 for non-existent agent."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        response = client.get("/agents/nonexistent_agent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_chat_with_agent_returns_200(self, client: TestClient):
        """Test that chat with agent endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "message": "Hello, test agent!",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = client.post("/agents/test_research_agent/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "test_research_agent"
        assert data["user_id"] == "test_user"
        assert data["session_id"] == "test_session"
        assert data["response"] == "Mock response"
    
    def test_chat_with_agent_model_override(self, client: TestClient):
        """Test chat with agent using model override."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "message": "Hello with custom model!",
            "user_id": "test_user",
            "session_id": "test_session",
            "model": "custom-model",
            "model_settings": {"temperature": 0.8, "max_tokens": 1000}
        }
        
        response = client.post("/agents/test_research_agent/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "test_research_agent"
        assert data["response"] == "Mock response"
    
    def test_chat_with_agent_not_found(self, client: TestClient):
        """Test that chat with agent returns 404 for non-existent agent."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "message": "Hello!",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        
        response = client.post("/agents/nonexistent_agent/chat", json=chat_request)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_chat_with_agent_invalid_request(self, client: TestClient):
        """Test that chat with agent handles invalid requests properly."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        # Missing required fields
        invalid_request = {
            "user_id": "test_user",
            "session_id": "test_session"
            # Missing "message" field
        }
        
        response = client.post("/agents/test_research_agent/chat", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_get_conversation_history_returns_200(self, client: TestClient):
        """Test that get conversation history endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        response = client.get("/agents/test_research_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "test_research_agent"
        assert data["user_id"] == "test_user"
        assert data["session_id"] == "test_session"
        assert isinstance(data["messages"], list)
    
    def test_get_conversation_history_not_found(self, client: TestClient):
        """Test that get conversation history returns 404 for non-existent agent."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        response = client.get("/agents/nonexistent_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_clear_conversation_history_returns_200(self, client: TestClient):
        """Test that clear conversation history endpoint returns HTTP 200."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        response = client.delete("/agents/test_research_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert "cleared" in data["message"].lower()
    
    def test_clear_conversation_history_not_found(self, client: TestClient):
        """Test that clear conversation history returns 404 for non-existent agent."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        response = client.delete("/agents/nonexistent_agent/conversation/test_session?user_id=test_user")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_multiple_agents_listing(self, client: TestClient):
        """Test listing multiple agents from the configuration."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        current_agent_ids.add("test_cli_agent")
        current_agent_ids.add("test_mcp_agent")
        
        response = client.get("/agents/")
        assert response.status_code == 200
        agents = response.json()
        
        # Should have at least the test agents
        agent_ids = [agent["agent_id"] for agent in agents]
        assert "test_research_agent" in agent_ids
        assert "test_cli_agent" in agent_ids
        assert "test_mcp_agent" in agent_ids
        
        # Verify agent details
        for agent in agents:
            if agent["agent_id"] == "test_research_agent":
                assert agent["provider"] == "azure_openai_cc"
                assert agent["model"] == "gpt-4.1-nano"
            elif agent["agent_id"] == "test_cli_agent":
                assert agent["provider"] == "ollama"
                assert agent["model"] == "qwen3:1.7b"
                assert agent["has_memory"] is True 