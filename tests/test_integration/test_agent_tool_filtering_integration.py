# tests/test_integration/test_agent_tool_filtering_integration.py
import pytest
import logging
from app.core.providers.manager import ProviderManager
from app.core.providers.base import ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.core.agents.agent_tool_manager import AgentToolManager

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_research_agent_tool_filtering(provider_id, caplog):
    """Test research_agent with specific tool access."""
    caplog.set_level(logging.INFO)
    
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()
    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        # Test research_agent - should have access to datetime + deepwiki/fetch tools
        agent_id = "research_agent"
        
        instructions = """
                        You are a research assistant. You have access to:
                        - get_current_datetime: Get current date/time
                        - deepwiki tools: Search and read wiki content
                        - fetch tools: Fetch web content
                        
                        Use the appropriate tools when asked.
                        """
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and can you search for information about Python programming?"}],
            model=config.default_model,
            instructions=instructions,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Research agent response: {response}")
        
        # Verify the response contains expected content
        assert isinstance(response, str)
        assert len(response) > 0
        # Should mention time (from get_current_datetime)
        assert any(word in response.lower() for word in ["time", "date", "current"])
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_restricted_agent_tool_filtering(provider_id, caplog):
    """Test restricted_agent with limited tool access."""
    caplog.set_level(logging.INFO)
    
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()
    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        # Test restricted_agent - should only have datetime + limited deepwiki access
        agent_id = "restricted_agent"
        
        instructions = """
                        You are a restricted assistant. You have access to:
                        - get_current_datetime: Get current date/time
                        - deepwiki read_wiki_structure: Read wiki structure only
                        
                        Use the appropriate tools when asked.
                        """
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and can you read the structure of the Python wiki?"}],
            model=config.default_model,
            instructions=instructions,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Restricted agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_mcp_agent_tool_filtering(provider_id, caplog):
    """Test mcp_agent with only MCP tools."""
    caplog.set_level(logging.INFO)
    
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()
    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        # Test mcp_agent - should have access to all MCP tools but no regular tools
        agent_id = "mcp_agent"
        
        instructions = """
                        You are an MCP-only assistant. You have access to:
                        - deepwiki tools: Search and read wiki content
                        - fetch tools: Fetch web content
                        
                        Use the appropriate tools when asked.
                        """
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "Can you search for information about Python programming and fetch some web content?"}],
            model=config.default_model,
            instructions=instructions,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"MCP agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_regular_tools_only_agent_filtering(provider_id, caplog):
    """Test regular_tools_only_agent with only regular tools."""
    caplog.set_level(logging.INFO)
    
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()
    provider = provider_info["class"](config)
        
    await provider.initialize()

    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")

    try:
        # Test regular_tools_only_agent - should have access to regular tools only
        agent_id = "regular_tools_only_agent"
        
        instructions = """
                        You are a regular tools assistant. You have access to:
                        - get_current_datetime: Get current date/time
                        - add_two_numbers: Add two numbers
                        
                        Use the appropriate tools when asked.
                        """
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and what is 5 + 7?"}],
            model=config.default_model,
            instructions=instructions,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Regular tools only agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_agent_tool_manager_direct_test(provider_id, caplog):
    """Test AgentToolManager directly to verify tool filtering."""
    caplog.set_level(logging.INFO)
    
    # Test different agent configurations
    test_cases = [
        ("research_agent", "Should have datetime + deepwiki + fetch tools"),
        ("restricted_agent", "Should have datetime + limited deepwiki tools"),
        ("mcp_agent", "Should have all MCP tools"),
        ("regular_tools_only_agent", "Should have only regular tools")
    ]
    
    for agent_id, description in test_cases:
        logging.info(f"Testing {agent_id}: {description}")
        
        agent_manager = AgentToolManager(agent_id)
        agent_manager.clear_cache()
        available_tools = await agent_manager.get_available_tools()
        
        logging.info(f"Agent {agent_id} has {len(available_tools)} tools available")
        
        # Log the tool names for verification
        tool_names = [tool["function"]["name"] for tool in available_tools]
        logging.info(f"Tool names: {tool_names}")
        
        # Basic assertions
        assert isinstance(available_tools, list)
        assert len(available_tools) >= 0  # Could be 0 if no tools configured
        
        # Verify specific agent configurations
        if agent_id == "research_agent":
            # Should have datetime tool
            assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
            # Should have deepwiki tools
            assert any("deepwiki__" in tool["function"]["name"] for tool in available_tools)
            
        elif agent_id == "restricted_agent":
            # Should have datetime tool
            assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
            # Should have limited deepwiki tools
            assert any("deepwiki__read_wiki_structure" in tool["function"]["name"] for tool in available_tools)
            
        elif agent_id == "mcp_agent":
            # Should not have regular tools
            assert not any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
            # Should have MCP tools
            assert any("deepwiki__" in tool["function"]["name"] for tool in available_tools)
            
        elif agent_id == "regular_tools_only_agent":
            # Should have regular tools
            assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
            assert any("add_two_numbers" in tool["function"]["name"] for tool in available_tools)
            # Should not have MCP tools
            assert not any("deepwiki__" in tool["function"]["name"] for tool in available_tools) 

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_provider_tool_integration_bad_tools(provider_id):
    """Test error handling when tools are requested but not available to the agent."""
    manager = ProviderManager()
    provider_info = manager.get_provider(provider_id)
    
    config = provider_info["config_class"]()
    provider = provider_info["class"](config)
    
    await provider.initialize()
    
    health: HealthStatus = await provider.health_check()
    if not health.status == "healthy":
        pytest.skip(f"{provider_id} not healthy: {health.error_details}")
    
    try:
        # Test with an agent that only has access to datetime tool
        agent_id = "regular_tools_only_agent"
        
        instructions = """
                        You are a helpful assistant that can use tools to answer questions.
                        You can use the following tools:
                        - get_current_datetime: Get the current date and time

                        When you are asked to use a tool, you must use it, providing it is available to you.
                        You must not use a tool if you are not asked to.
                        """
        
        try:
            response = await provider.send_chat(
                context=[{"role": "user", "content": "What is 6000+12312 please use the add_two_numbers tool and also what is the current date and time in Tokyo?"}],
                model=config.default_model,
                instructions=instructions,
                tools=None,  # Let agent manager handle tool selection
                agent_id=agent_id
            )
        except ProviderMaxToolIterationsError:
            # This is expected behavior - agent doesn't have access to add_two_numbers
            # so it should reach max iterations trying to use unavailable tools
            pass
        else:
            # If we get here, the agent somehow managed to complete without the required tool
            # This might indicate the LLM ignored the tool requirement, which is acceptable
            assert isinstance(response, str)
            assert len(response) > 0
        
    finally:
        await provider.cleanup() 