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
                "agent_id": "integration_test_agent",
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
        current_agent_ids.add("research_agent")  # Use an agent that exists in the config
        
        response = client.get("/agents/")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) >= 1
        assert any(agent["agent_id"] == "research_agent" for agent in agents)
        
        response = client.get("/agents/research_agent")
        assert response.status_code == 200
        agent_info = response.json()
        assert agent_info["agent_id"] == "research_agent"
        assert agent_info["provider"] == "azure_openai_cc"
        assert "model" in agent_info
        assert "has_memory" in agent_info
        
        chat_request = {
            "message": "Hello from integration test",
            "user_id": "integration_user",
            "session_id": "integration_session"
        }
        response = client.post("/agents/research_agent/chat", json=chat_request)
        assert response.status_code == 200
        chat_response = response.json()
        assert chat_response["response"] == "Mock response"
        assert chat_response["agent_id"] == "research_agent"
        assert chat_response["user_id"] == "integration_user"
        assert chat_response["session_id"] == "integration_session"
        
        response = client.get("/agents/research_agent/conversation/integration_session?user_id=integration_user")
        assert response.status_code == 200
        history = response.json()
        assert history["session_id"] == "integration_session"
        assert history["agent_id"] == "research_agent"
        assert history["user_id"] == "integration_user"
        assert isinstance(history["messages"], list)
        
        response = client.delete("/agents/research_agent/conversation/integration_session?user_id=integration_user")
        assert response.status_code == 200
        clear_response = response.json()
        assert "cleared" in clear_response["message"]
    
    def test_complete_openai_compatible_flow(self, client: TestClient, sample_agent_config):
        """Test complete OpenAI-compatible API flow."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("research_agent")  # Use an agent that exists in the config
        
        response = client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()
        assert models["object"] == "list"
        assert len(models["data"]) >= 1
        agent_ids = [model["id"] for model in models["data"]]
        assert "research_agent" in agent_ids
        assert models["data"][0]["object"] == "model"
        assert models["data"][0]["owned_by"] == "ai-agent-service"
        
        chat_request = {
            "model": "research_agent",
            "messages": [
                {"role": "user", "content": "Hello from OpenAI compatible test"}
            ],
            "temperature": 0.7
        }
        response = client.post("/v1/chat/completions", json=chat_request)
        assert response.status_code == 200
        completion = response.json()
        assert completion["object"] == "chat.completion"
        assert completion["model"] == "research_agent"
        assert len(completion["choices"]) == 1
        assert completion["choices"][0]["message"]["role"] == "assistant"
        assert completion["choices"][0]["message"]["content"] == "Mock response"
        assert completion["choices"][0]["finish_reason"] == "stop"
        assert "usage" in completion
    
    def test_api_error_handling_integration(self, client: TestClient):
        """Test API error handling in integration scenarios."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        
        response = client.get("/agents/non_existent_agent")
        assert response.status_code == 404
        
        chat_request = {
            "message": "Hello",
            "user_id": "test_user",
            "session_id": "test_session"
        }
        response = client.post("/agents/non_existent_agent/chat", json=chat_request)
        assert response.status_code == 404  # Should handle gracefully
    
    def test_api_validation_integration(self, client: TestClient):
        """Test API validation in integration scenarios."""
        invalid_request = {
            "user_id": "test_user",
            "session_id": "test_session"
        }
        response = client.post("/agents/test_agent/chat", json=invalid_request)
        assert response.status_code == 422
        
        invalid_openai_request = {
            "model": "test_agent"
        }
        response = client.post("/v1/chat/completions", json=invalid_openai_request)
        assert response.status_code == 422
    
    def test_api_concurrent_requests(self, client: TestClient, sample_agent_config):
        """Test API handling of concurrent requests."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("research_agent")  # Use an agent that exists in the config
        
        import concurrent.futures
        
        def make_request():
            chat_request = {
                "message": "Concurrent test",
                "user_id": "concurrent_user",
                "session_id": "concurrent_session"
            }
            response = client.post("/agents/research_agent/chat", json=chat_request)
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [future.result() for future in futures]
            assert all(status == 200 for status in results)
    
    def test_api_memory_integration(self, client: TestClient, sample_agent_config):
        """Test API memory integration."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("cli_agent")  # Use an agent that has memory
        
        user_id = "memory_user"
        session_id = "memory_session"
        
        chat_request = {
            "message": "First message",
            "user_id": user_id,
            "session_id": session_id
        }
        response = client.post("/agents/cli_agent/chat", json=chat_request)
        assert response.status_code == 200
        
        chat_request = {
            "message": "Second message",
            "user_id": user_id,
            "session_id": session_id
        }
        response = client.post("/agents/cli_agent/chat", json=chat_request)
        assert response.status_code == 200
        
        response = client.get(f"/agents/cli_agent/conversation/{session_id}?user_id={user_id}")
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history["messages"], list)
        
        response = client.delete(f"/agents/cli_agent/conversation/{session_id}?user_id={user_id}")
        assert response.status_code == 200
    
    def test_api_model_override_integration(self, client: TestClient, sample_agent_config):
        """Test API model override functionality."""
        # Update the global agent IDs for this test
        current_agent_ids.clear()
        current_agent_ids.add("research_agent")  # Use an agent that exists in the config
        
        chat_request = {
            "message": "Test with custom model",
            "user_id": "override_user",
            "session_id": "override_session",
            "model": "custom-model",
            "model_settings": {"temperature": 0.9, "max_tokens": 1000}
        }
        response = client.post("/agents/research_agent/chat", json=chat_request)
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