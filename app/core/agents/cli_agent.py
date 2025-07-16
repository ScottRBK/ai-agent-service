"""
CLI Agent implementation using the AgentToolManager framework.
"""

import asyncio
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.utils.logging import logger


class CLIAgent:
    """
    Interactive command-line agent with tool capabilities.
    """
    
    def __init__(self, agent_id: str = "cli_agent", provider_id: str = "azure_openai_cc"):
        self.agent_id = agent_id
        self.provider_id = provider_id
        self.tool_manager = AgentToolManager(agent_id)
        self.provider_manager = ProviderManager()
        self.provider = None
        self.conversation_history = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent and provider."""
        if self.initialized:
            return
            
        # Get provider
        provider_info = self.provider_manager.get_provider(self.provider_id)
        config = provider_info["config_class"]()
        self.provider = provider_info["class"](config)
        await self.provider.initialize()
        
        # Verify agent has access to tools
        available_tools = await self.tool_manager.get_available_tools()
        logger.info(f"Agent {self.agent_id} initialized with {len(available_tools)} tools")
        
        self.initialized = True
    
    async def chat(self, user_input: str) -> str:
        """Send a message to the agent and get response."""
        if not self.initialized:
            await self.initialize()
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get response from provider
        response = await self.provider.send_chat(
            context=self.conversation_history,
            model=self.provider.config.default_model,
            instructions="You are a helpful AI assistant. Use available tools when needed to provide accurate and helpful responses.",
            agent_id=self.agent_id
        )
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def interactive_mode(self):
        """Run interactive chat mode."""
        await self.initialize()
        
        print(f"🤖 {self.agent_id} Agent Ready!")
        print(f"�� Available tools: {len(await self.tool_manager.get_available_tools())}")
        print("💬 Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...")
                response = await self.chat(user_input)
                print(f"�� {self.agent_id}: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main function to run the CLI agent."""
    agent = CLIAgent("cli_agent", "ollama")
    await agent.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())