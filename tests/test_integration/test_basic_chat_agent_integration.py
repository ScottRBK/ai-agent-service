# tests/test_integration/test_basic_chat_agent_integration.py
import pytest
from app.core.providers.manager import ProviderManager
from app.core.providers.base import ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.core.tools.function_calls.arithmetic_tool import ArithmeticTool
from app.core.tools.function_calls.date_tool import DateTool

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_provider_agent_integration(provider_id):
    """Test agent functionality with any provider if healthy."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()

    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")
    

    try:
        await provider.initialize()

        models = await provider.get_model_list()
        assert len(models) > 0

        instructions = """
                        You are a helpful assistant that can use tools to answer questions.
                        You can use the following tools:
                        - add_two_numbers: Add two numbers
                        - get_current_datetime: Get the current date and time

                        When you are asked to use a tool, you must use it, providing it is available to you.
                        You must not use a tool if you are not asked to.
                        """
        
        # Test with agent_id for agent-specific tool filtering
        agent_id = "test_agent"
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is 6000+12312 please use the add_two_numbers tool and also what is the current date and time in Tokyo?"}],
            model=config.default_model,
            instructions=instructions,
            tools=None,  # Let agent manager handle tool selection
            agent_id=agent_id
        )
        print(response)
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_provider_agent_integration_backward_compatibility(provider_id):
    """Test that agent functionality maintains backward compatibility with existing tool usage."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()

    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        await provider.initialize()

        models = await provider.get_model_list()
        assert len(models) > 0

        instructions = """
                        You are a helpful assistant that can use tools to answer questions.
                        You can use the following tools:
                        - add_two_numbers: Add two numbers
                        - get_current_datetime: Get the current date and time

                        When you are asked to use a tool, you must use it, providing it is available to you.
                        You must not use a tool if you are not asked to.
                        """
        tools = ["get_current_datetime", "add_two_numbers"]
        
        # Test without agent_id (backward compatibility)
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current date and time?"}],
            model=config.default_model,
            instructions=instructions,
            tools=tools
            # No agent_id parameter - should work as before
        )
        print(response)
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_provider_agent_integration_restricted_tools(provider_id):
    """Test agent functionality with restricted tool access."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()

    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        await provider.initialize()

        models = await provider.get_model_list()
        assert len(models) > 0

        instructions = """
                        You are a helpful assistant that can use tools to answer questions.
                        You can use the following tools:
                        - get_current_datetime: Get the current date and time

                        When you are asked to use a tool, you must use it, providing it is available to you.
                        You must not use a tool if you are not asked to.
                        """
        
        # Test with agent_id that has restricted tool access
        agent_id = "restricted_agent"
        
        try:
            response = await provider.send_chat(
                context=[{"role": "user", "content": "What is 6000+12312 please use the add_two_numbers tool and also what is the current date and time in Tokyo?"}],
                model=config.default_model,
                instructions=instructions,
                tools=None,  # Let agent manager handle tool selection
                agent_id=agent_id
            )
        except ProviderMaxToolIterationsError:
            pytest.skip(f"Max tool iterations reached")
        else:
            print(response)
            assert isinstance(response, str)
            assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_provider_agent_integration_mcp_tools(provider_id):
    """Test agent functionality with MCP tools (when available)."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()

    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        await provider.initialize()

        models = await provider.get_model_list()
        assert len(models) > 0

        instructions = """
                        You are a helpful assistant that can use tools to answer questions.
                        You can use various tools including MCP tools for web search and content fetching.

                        When you are asked to use a tool, you must use it, providing it is available to you.
                        You must not use a tool if you are not asked to.
                        """
        
        # Test with agent_id that has access to MCP tools
        agent_id = "mcp_agent"
        
        try:
            response = await provider.send_chat(
                context=[{"role": "user", "content": "Can you tell me about the current date and time, and also fetch some information about Python programming?"}],
                model=config.default_model,
                instructions=instructions,
                tools=None,  # Let agent manager handle tool selection
                agent_id=agent_id
            )
        except ProviderMaxToolIterationsError:
            pytest.skip(f"Max tool iterations reached")
        else:
            print(response)
            assert isinstance(response, str)
            assert len(response) > 0
        
    finally:
        await provider.cleanup() 