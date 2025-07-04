# tests/test_integration.py
import pytest
from app.core.providers.manager import ProviderManager
from app.models.health import HealthStatus

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["azure_openai", "azure_openai_cc", "ollama"])
async def test_provider_if_healthy(provider_id):
    """Test any provider if healthy."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()

    provider = provider_info["class"](config)
        
    await provider.initialize()
    # Health check
    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")
    
    # Basic test
    try:
        await provider.initialize()
        
        # Test model list
        models = await provider.get_model_list()
        assert len(models) > 0
        
        # Test basic chat
        response = await provider.send_chat(
            context=[{"role": "user", "content": "Hi"}],
            model=config.default_model,
            instructions="Be brief",
            tools=[]
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()