"""
Integration tests for the complete API flow.

This module tests the complete API integration,
ensuring all components work together properly.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import current_agent_ids


class TestAPIIntegration:
    """Integration tests for the complete API flow."""
    
    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration for integration testing."""
        return [
            {
                "agent_id": "test_integration_agent",
                "provider": "azure_openai_cc",
                "model": "gpt-4o-mini",
                "allowed_regular_tools": ["get_current_datetime"],
                "allowed_mcp_servers": ["deepwiki"],
                "resources": ["memory"]
            }
        ]
    
    def test_complete_agent_management_flow(self, client: TestClient, sample_agent_config):
        """Test complete agent management API flow."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")  # Use an agent that exists in the config
        
        response = client.get("/agents/")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) >= 1
        assert any(agent["agent_id"] == "test_research_agent" for agent in agents)
        
        response = client.get("/agents/test_research_agent")
        assert response.status_code == 200
        agent_info = response.json()
        assert agent_info["agent_id"] == "test_research_agent"
        assert agent_info["provider"] == "azure_openai_cc"
        assert "model" in agent_info
        assert "has_memory" in agent_info
        
        chat_request = {
            "message": "Hello from integration test",
            "user_id": "integration_user",
            "session_id": "integration_session"
        }
        response = client.post("/agents/test_research_agent/chat", json=chat_request)
        assert response.status_code == 200
        chat_response = response.json()
        assert chat_response["response"] == "Mock response"
        assert chat_response["agent_id"] == "test_research_agent"
        assert chat_response["user_id"] == "integration_user"
        assert chat_response["session_id"] == "integration_session"
        
        response = client.get("/agents/test_research_agent/conversation/integration_session?user_id=integration_user")
        assert response.status_code == 200
        history = response.json()
        assert history["session_id"] == "integration_session"
        assert history["agent_id"] == "test_research_agent"
        assert history["user_id"] == "integration_user"
        assert isinstance(history["messages"], list)
        
        # Test clearing conversation
        response = client.delete("/agents/test_research_agent/conversation/integration_session?user_id=integration_user")
        assert response.status_code == 200
        
        # Verify conversation is cleared
        response = client.get("/agents/test_research_agent/conversation/integration_session?user_id=integration_user")
        assert response.status_code == 200
        history = response.json()
        assert len(history["messages"]) == 0
    
    def test_agent_not_found_handling(self, client: TestClient):
        """Test handling of non-existent agents."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        # Test getting info for non-existent agent
        response = client.get("/agents/nonexistent_agent")
        assert response.status_code == 404
        
        # Test chatting with non-existent agent
        chat_request = {
            "message": "Hello",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        response = client.post("/agents/nonexistent_agent/chat", json=chat_request)
        assert response.status_code == 404
    
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
    
    def test_agent_chat_with_model_override(self, client: TestClient):
        """Test agent chat with model override functionality."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")
        
        chat_request = {
            "message": "Test with custom model",
            "user_id": "override_user",
            "session_id": "override_session",
            "model": "custom-model",
            "model_settings": {"temperature": 0.9, "max_tokens": 1000}
        }
        response = client.post("/agents/test_research_agent/chat", json=chat_request)
        assert response.status_code == 200
    
    def test_api_model_override_integration(self, client: TestClient, sample_agent_config):
        """Test API model override functionality."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("test_research_agent")  # Use an agent that exists in the config
        
        chat_request = {
            "message": "Test with custom model",
            "user_id": "override_user",
            "session_id": "override_session",
            "model": "custom-model",
            "model_settings": {"temperature": 0.9, "max_tokens": 1000}
        }
        response = client.post("/agents/test_research_agent/chat", json=chat_request)
        assert response.status_code == 200
    
    def test_api_root_endpoint_integration(self, client: TestClient):
        """Test API root endpoint integration."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
        assert "agents" in data["endpoints"]
        assert "openai_compatible" in data["endpoints"]
        assert "health" in data["endpoints"]
    
    def test_api_health_endpoint_integration(self, client: TestClient):
        """Test API health endpoint integration."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data
        assert "version" in data 