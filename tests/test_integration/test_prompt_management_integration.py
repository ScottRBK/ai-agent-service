"""
Integration tests for prompt management with agents and providers.
"""

import pytest
import logging
from app.core.providers.manager import ProviderManager
from app.core.providers.base import ProviderMaxToolIterationsError
from app.models.health import HealthStatus
from app.core.agents.prompt_manager import PromptManager
from app.core.agents.agent_tool_manager import AgentToolManager

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_prompt_manager_integration(provider_id, caplog):
    """Test that PromptManager integrates correctly with providers."""
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
        # Test research_agent with PromptManager
        agent_id = "research_agent"
        
        # Create PromptManager and get system prompt
        prompt_manager = PromptManager(agent_id)
        system_prompt = prompt_manager.get_system_prompt()
        
        # Verify prompt is loaded from file
        assert "research assistant" in system_prompt.lower()
        assert "web search" in system_prompt.lower() or "document retrieval" in system_prompt.lower()
        
        # Get available tools
        agent_tool_manager = AgentToolManager(agent_id)
        available_tools = await agent_tool_manager.get_available_tools()
        
        # Generate system prompt with tools
        full_system_prompt = prompt_manager.get_system_prompt_with_tools(available_tools)
        
        # Verify tools are included in prompt
        tool_names = [tool["function"]["name"] for tool in available_tools]
        for tool_name in tool_names:
            assert tool_name in full_system_prompt
        
        # Test with provider
        response = await provider.send_chat(
            context=[{"role": "user", "content": "What is the current time?"}],
            model=config.default_model,
            instructions=full_system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Research agent response with prompt manager: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_prompt_manager_fallback_integration(provider_id, caplog):
    """Test PromptManager fallback behavior with non-existent agent."""
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
        # Test with non-existent agent (should use default prompt)
        agent_id = "nonexistent_agent"
        
        prompt_manager = PromptManager(agent_id)
        system_prompt = prompt_manager.get_system_prompt()
        
        # Should use default prompt
        assert "helpful AI assistant" in system_prompt
        
        # Test with provider
        response = await provider.send_chat(
            context=[{"role": "user", "content": "Hello"}],
            model=config.default_model,
            instructions=system_prompt,
            tools=None,
            agent_id=agent_id
        )
        
        logging.info(f"Non-existent agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        await provider.cleanup()

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["ollama", "azure_openai_cc", "azure_openai"])
async def test_cli_agent_prompt_integration(provider_id, caplog, monkeypatch):
    """Test CLI agent with PromptManager integration."""
    caplog.set_level(logging.INFO)
    
    from app.core.agents.cli_agent import CLIAgent
    
    # Mock the get_model_config method to return None for model_settings
    def mock_get_model_config(self):
        return None, None
    
    # Apply the mock
    monkeypatch.setattr("app.core.agents.agent_resource_manager.AgentResourceManager.get_model_config", mock_get_model_config)
    
    try:
        # Test CLI agent initialization
        agent = CLIAgent("cli_agent", provider_id)
        await agent.initialize()
        
        # Verify system prompt is generated
        assert hasattr(agent, 'system_prompt')
        assert len(agent.system_prompt) > 0
        
        # Verify prompt contains expected content
        assert "CLI-enabled assistant" in agent.system_prompt or "command-line" in agent.system_prompt
        
        # Test a simple chat
        response = await agent.chat("What is the current time?")
        
        logging.info(f"CLI agent response: {response}")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    except Exception as e:
        if "not healthy" in str(e):
            pytest.skip(f"{provider_id} not healthy")
        else:
            raise 

@pytest.mark.asyncio
async def test_prompt_manager_tool_name_extraction():
    """Test that PromptManager correctly extracts tool names from dictionaries."""
    
    # Test with tool dictionaries (from get_available_tools())
    tool_dicts = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get current date and time"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "deepwiki__search_wiki",
                "description": "Search wiki content"
            }
        }
    ]
    
    prompt_manager = PromptManager("research_agent")
    system_prompt = prompt_manager.get_system_prompt_with_tools(tool_dicts)
    
    # Verify tool names are extracted and included
    assert "get_current_datetime" in system_prompt
    assert "deepwiki__search_wiki" in system_prompt
    assert "Available tools:" in system_prompt 