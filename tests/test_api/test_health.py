"""
Tests for health check endpoints.

This module tests the basic health check endpoint,
ensuring it returns correct data structures, status codes, and
handles various scenarios properly.
"""

import pytest
from fastapi.testclient import TestClient

class TestBasicHealthEndpoint:
    """Test cases for the basic health check endpoint."""
    
    def test_health_endpoint_returns_200(self, client: TestClient):
        """Test that health endpoint returns HTTP 200."""
        response = client.get("/health/")
        assert response.status_code == 200