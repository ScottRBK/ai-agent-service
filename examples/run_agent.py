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
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.agents.cli_agent import CLIAgent
from app.core.providers.manager import ProviderManager
from app.utils.logging import logger


def print_usage():
    """Print usage information."""
    print("🤖 AI Agent Service - Agent Runner")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python examples/run_agent.py [agent_id] [provider_id]")
    print("  python examples/run_agent.py [agent_id]  # Uses provider from agent config")
    print()
    print("Available Agents:")
    print("  research_agent  - Has access to datetime, deepwiki, fetch tools")
    print("  data_agent     - Has access to datetime, arithmetic tools")
    print("  mcp_agent      - Has access to MCP tools only (deepwiki, fetch)")
    print("  restricted_agent - Limited access to specific tools")
    print("  cli_agent      - Has access to datetime tool and all MCP tools")
    print("  azure_agent    - Azure OpenAI agent with memory")
    print()
    print("Available Providers:")
    print("  azure_openai_cc - Azure OpenAI (Chat Completions)")
    print("  azure_openai    - Azure OpenAI (Responses API)")
    print("  ollama          - Local Ollama provider")
    print()
    print("Examples:")
    print("  python examples/run_agent.py research_agent azure_openai_cc")
    print("  python examples/run_agent.py cli_agent  # Uses provider from agent config")
    print("  python examples/run_agent.py azure_agent  # Uses Azure OpenAI from config")
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI agents with custom settings")
    parser.add_argument("agent_id", help="Agent ID to run")
    parser.add_argument("provider_id", nargs="?", help="Provider ID to use (optional, will use agent config if not specified)")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--stream", action="store_true", help="Stream the response")

    # Generic settings - any key-value pair
    parser.add_argument("--setting", action="append", nargs=2, 
                       metavar=("KEY", "VALUE"), 
                       help="Model setting KEY VALUE (can be used multiple times)")
    
    return parser.parse_args()

def get_provider_from_agent_config(agent_id: str) -> str:
    """Get the provider from agent configuration."""
    try:
        from app.core.agents.agent_tool_manager import AgentToolManager
        agent_manager = AgentToolManager(agent_id)
        provider = agent_manager.config.get("provider")
        if provider:
            return provider
        else:
            logger.warning(f"No provider specified in agent config for {agent_id}, using default")
            return "azure_openai_cc"  # Default fallback
    except Exception as e:
        logger.error(f"Error getting provider from agent config for {agent_id}: {e}")
        return "azure_openai_cc"  # Default fallback

def parse_setting_value(value):
    """Parse a setting value, trying to convert to appropriate type."""
    # Boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # String (default)
    return value

async def main():
    args = parse_arguments()
    
    # Get provider - use CLI argument if provided, otherwise get from agent config
    provider_id = args.provider_id
    if not provider_id:
        provider_id = get_provider_from_agent_config(args.agent_id)
        print(f"📋 Using provider '{provider_id}' from agent configuration")
    
    # Build model settings dictionary
    model_settings = {}
    
    # Parse all --setting arguments
    if args.setting:
        for key, value in args.setting:
            model_settings[key] = parse_setting_value(value)
    
    print(f"🤖 Starting {args.agent_id} agent with {provider_id} provider...")
    if model_settings:
        print(f"⚙️  Model settings: {model_settings}")
    print()

    if args.stream:
        print("🔄 Streaming mode enabled")
    
    # Validate agent and provider
    if not validate_agent_config(args.agent_id):
        print(f"❌ Agent '{args.agent_id}' not found or invalid configuration")
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
    
    print(f"✅ Agent '{args.agent_id}' validated")
    print(f"✅ Provider '{provider_id}' validated")
    print()
    
    try:
        # Create agent with settings
        agent = CLIAgent(args.agent_id, provider_id, model=args.model, model_settings=model_settings)
        await agent.interactive_mode(stream=args.stream)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        logger.error(f"Error running agent {args.agent_id} with provider {provider_id}: {e}")


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists("agent_config.json"):
        print("❌ agent_config.json not found in current directory")
        print("   Please run this script from the project root directory")
        print("   Example: python examples/run_agent.py")
        sys.exit(1)
    
    asyncio.run(main()) 