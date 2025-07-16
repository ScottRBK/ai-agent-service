#!/usr/bin/env python3
"""
Example script to run different agents with the AI Agent Service framework.

Usage:
    python examples/run_agent.py [agent_id] [provider_id]

Examples:
    python examples/run_agent.py research_agent azure_openai_cc
    python examples/run_agent.py data_agent ollama
    python examples/run_agent.py mcp_agent azure_openai_cc
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.agents.cli_agent import CLIAgent
from app.core.providers.manager import ProviderManager
from app.utils.logging import logger


def print_usage():
    """Print usage information."""
    print("�� AI Agent Service - Agent Runner")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python examples/run_agent.py [agent_id] [provider_id]")
    print()
    print("Available Agents:")
    print("  research_agent  - Has access to datetime, deepwiki, fetch tools")
    print("  data_agent     - Has access to datetime, arithmetic tools")
    print("  mcp_agent      - Has access to MCP tools only (deepwiki, fetch)")
    print("  restricted_agent - Limited access to specific tools")
    print("  cli_agent      - Has access to datetime tool and all MCP tools")
    print()
    print("Available Providers:")
    print("  azure_openai_cc - Azure OpenAI (Chat Completions)")
    print("  azure_openai    - Azure OpenAI (Responses API)")
    print("  ollama          - Local Ollama provider")
    print()
    print("Examples:")
    print("  python examples/run_agent.py research_agent azure_openai_cc")
    print("  python examples/run_agent.py data_agent ollama")
    print("  python examples/run_agent.py mcp_agent azure_openai_cc")
    print()


def validate_agent_config(agent_id: str) -> bool:
    """Check if the agent configuration exists."""
    try:
        from app.core.agents.agent_tool_manager import AgentToolManager
        agent_manager = AgentToolManager(agent_id)
        return agent_manager.config is not None
    except Exception as e:
        logger.error(f"Error validating agent {agent_id}: {e}")
        return False


def validate_provider(provider_id: str) -> bool:
    """Check if the provider is available."""
    try:
        provider_manager = ProviderManager()
        provider_info = provider_manager.get_provider(provider_id)
        return provider_info is not None
    except Exception as e:
        logger.error(f"Error validating provider {provider_id}: {e}")
        return False


async def main():
    """Main function to run the agent."""
    # Parse command line arguments
    agent_id = sys.argv[1] if len(sys.argv) > 1 else "research_agent"
    provider_id = sys.argv[2] if len(sys.argv) > 2 else "azure_openai_cc"
    
    # Show usage if help is requested
    if agent_id in ["-h", "--help", "help"]:
        print_usage()
        return
    
    print(f"�� Starting {agent_id} agent with {provider_id} provider...")
    print()
    
    # Validate agent and provider
    if not validate_agent_config(agent_id):
        print(f"❌ Agent '{agent_id}' not found or invalid configuration")
        print("   Check agent_config.json for available agents")
        print()
        print_usage()
        return
    
    if not validate_provider(provider_id):
        print(f"❌ Provider '{provider_id}' not available")
        print("   Check your provider configuration")
        print()
        print_usage()
        return
    
    print(f"✅ Agent '{agent_id}' validated")
    print(f"✅ Provider '{provider_id}' validated")
    print()
    
    try:
        # Create and run the agent
        agent = CLIAgent(agent_id, provider_id)
        await agent.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        logger.error(f"Error running agent {agent_id} with provider {provider_id}: {e}")


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists("agent_config.json"):
        print("❌ agent_config.json not found in current directory")
        print("   Please run this script from the project root directory")
        print("   Example: python examples/run_agent.py")
        sys.exit(1)
    
    asyncio.run(main()) 