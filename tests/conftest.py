# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config.settings import Settings

@pytest.fixture
def client():
    return TestClient(app)