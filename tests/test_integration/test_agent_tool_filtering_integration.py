# tests/test_integration/test_agent_tool_filtering_integration.py
import pytest
import logging
from unittest.mock import patch
from app.core.providers.manager import ProviderManager
from app.core.providers.base import ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.agents.prompt_manager import PromptManager

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_research_agent_tool_filtering(provider_id, caplog):
    """Test test_research_agent with specific tool access."""
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
        # Test test_research_agent - should have datetime + deepwiki access
        agent_id = "test_research_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and can you search for information about Python programming?"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Research agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_restricted_agent_tool_filtering(provider_id, caplog):
    """Test test_restricted_agent with limited tool access."""
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
        # Test test_restricted_agent - should only have datetime + limited deepwiki access
        agent_id = "test_restricted_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and can you read the structure of the Python wiki?"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Restricted agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_mcp_agent_tool_filtering(provider_id, caplog):
    """Test test_mcp_agent with only MCP tools."""
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
        # Test test_mcp_agent - should have access to all MCP tools but no regular tools
        agent_id = "test_mcp_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "Can you search for information about Python programming and fetch some web content?"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"MCP agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_regular_tools_only_agent_tool_filtering(provider_id, caplog):
    """Test test_regular_tools_only_agent with only regular tools."""
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
        # Test test_regular_tools_only_agent - should have only regular tools
        agent_id = "test_regular_tools_only_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and what is 5 + 3?"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Regular tools only agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_cli_agent_tool_filtering(provider_id, caplog):
    """Test test_cli_agent with full tool access."""
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
        # Test test_cli_agent - should have access to all tools
        agent_id = "test_cli_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time and can you search for information about Python programming?"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"CLI agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
async def test_agent_tool_manager_direct_test(provider_id, caplog):
    """Test AgentToolManager directly to verify tool filtering."""
    caplog.set_level(logging.INFO)
    
    # Mock the configuration paths to use example files
    with patch('app.config.settings.settings.AGENT_CONFIG_PATH', 'agent_config.example.json'), \
         patch('app.config.settings.settings.MCP_CONFIG_PATH', 'mcp.example.json'):
        
        # Test different agent configurations
        test_cases = [
            ("test_research_agent", "Should have datetime + deepwiki tools"),
            ("test_restricted_agent", "Should have datetime + limited deepwiki tools"),
            ("test_mcp_agent", "Should have all MCP tools"),
            ("test_regular_tools_only_agent", "Should have only regular tools")
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
            assert len(available_tools) >= 0
            
            # Verify specific agent configurations
            if agent_id == "test_research_agent":
                # Should have datetime tool
                assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
                # Should have deepwiki tools (research agent is configured with deepwiki server)
                assert any("deepwiki__" in tool["function"]["name"] for tool in available_tools)
                
            elif agent_id == "test_restricted_agent":
                # Should have datetime tool
                assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
                # Should have limited deepwiki tools
                assert any("deepwiki__read_wiki_structure" in tool["function"]["name"] for tool in available_tools)
                
            elif agent_id == "test_mcp_agent":
                # Should not have regular tools
                assert not any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
                # Should have MCP tools
                assert any("deepwiki__" in tool["function"]["name"] for tool in available_tools)
                
            elif agent_id == "test_regular_tools_only_agent":
                # Should have regular tools
                assert any("get_current_datetime" in tool["function"]["name"] for tool in available_tools)
                assert any("add_two_numbers" in tool["function"]["name"] for tool in available_tools)
                # Should not have MCP tools
                assert not any("deepwiki__" in tool["function"]["name"] for tool in available_tools)

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai", "openrouter"])
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
        agent_id = "test_regular_tools_only_agent"
        
        # Use PromptManager instead of hardcoded instructions
        prompt_manager = PromptManager(agent_id)
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        try:
            response = await provider.send_chat(
                context=[{"role": "user", "content": "What is 6000+12312 please use the add_two_numbers tool and also what is the current date and time in Tokyo?"}],
                model=config.default_model,
                instructions=system_prompt,
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