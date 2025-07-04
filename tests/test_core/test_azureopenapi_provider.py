"""
Tests for Azure OpenAI provider.

This module tests the Azure OpenAI provider, ensuring it
interacts with the API correctly and handles various scenarios.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import importlib
import sys
from app.core.providers.azureopenapi import AzureOpenAIProvider
from app.models.providers import AzureOpenAIConfig

@pytest.fixture
def mock_config():
    return AzureOpenAIConfig(
        name="test-provider",
        api_version="2023-05-15",
        base_url="https://example.openai.azure.com/",
        api_key="test-key",
        model_list=["gpt-35-turbo", "gpt-4"]
    )

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_initialization(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    assert provider.client is mock_client
    assert provider.config.model_list == ["gpt-35-turbo", "gpt-4"]

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_health_check_healthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(return_value=MagicMock())
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "healthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is None

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_health_check_unhealthy(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.responses.create = AsyncMock(side_effect=Exception("Test error"))
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    health = await provider.health_check()
    assert health.status == "unhealthy"
    assert health.timestamp is not None
    assert health.service == "test-provider"
    assert health.version is not None
    assert health.error_details is "Test error"

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_get_model_list(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    models = await provider.get_model_list()
    assert models == ["gpt-35-turbo", "gpt-4"]

@patch.dict("os.environ", {
    "AZURE_OPENAI_MODEL_LIST": "gpt-4o,gpt-4o-mini,claude-3",
    "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/",
    "AZURE_OPENAI_DEFAULT_MODEL": "gpt-4o",
    "AZURE_OPENAI_API_VERSION": "2023-05-15",
    "AZURE_OPENAI_API_KEY": "test-key"
}, clear=False)
@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_get_config_from_env_var(mock_azure_openai):
    """Test that config is correctly read from environment variable"""

    # Force reload the providers module to pick up our environment changes
    import app.models.providers
    importlib.reload(app.models.providers)
    
    # Import after reload to get the updated config
    from app.models.providers import AzureOpenAIConfig
    
    # Create config without explicitly setting model_list, so it reads from env var
    config = AzureOpenAIConfig(
        name="test-provider"
    )
    
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-4o", "gpt-4o-mini", "claude-3"]
    mock_azure_openai.return_value = mock_client
    
    provider = AzureOpenAIProvider(config)
    models = await provider.get_model_list()
    
    # Verify the model list matches what was set in the environment variable
    assert models == ["gpt-4o", "gpt-4o-mini", "claude-3"]
    
    # Verify all environment variables were read correctly
    assert config.model_list == ["gpt-4o", "gpt-4o-mini", "claude-3"]
    assert config.base_url == "https://example.openai.azure.com/"
    assert config.default_model == "gpt-4o"
    assert config.api_version == "2023-05-15"
    assert config.api_key == "test-key"

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_cleanup_noop(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_client.models.list.return_value = ["gpt-35-turbo", "gpt-4"]
    mock_azure_openai.return_value = mock_client
    provider = AzureOpenAIProvider(mock_config)
    # Should not raise
    await provider.cleanup()

@patch("app.core.providers.azureopenapi.AsyncAzureOpenAI")
@pytest.mark.asyncio
async def test_send_chat_returns_response(mock_azure_openai, mock_config):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Hello, world!"
    mock_client.responses.create = AsyncMock(return_value=mock_response)
    mock_azure_openai.return_value = mock_client

    provider = AzureOpenAIProvider(mock_config)
    await provider.initialize()
    result = await provider.send_chat(model="gpt-35-turbo", 
                                      instructions="instructions", 
                                      context="hi", tools=[])
    assert result == "Hello, world!"
    mock_client.responses.create.assert_called_once_with(
        model="gpt-35-turbo",
        instructions="instructions",
        input="hi",
        tools=[]
    )

