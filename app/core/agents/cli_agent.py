"""
CLI Agent implementation using the AgentToolManager framework.
"""

import asyncio
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.core.agents.prompt_manager import PromptManager
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.utils.logging import logger
from app.models.resources.memory import MemoryEntry
from app.core.resources.memory import PostgreSQLMemoryResource


class CLIAgent:
    """
    Interactive command-line agent with tool capabilities.
    """
    
    def __init__(self, agent_id: str = "cli_agent", 
             provider_id: str = "azure_openai_cc", 
             user_id: str = "default_user", 
             session_id: str = "default_session",
             model: Optional[str] = None,
             model_settings: Optional[dict] = None):
        self.agent_id = agent_id
        self.provider_id = provider_id
        self.user_id = user_id
        self.session_id = session_id
        self.tool_manager = AgentToolManager(agent_id)
        self.resource_manager = AgentResourceManager(agent_id)
        self.provider_manager = ProviderManager()
        self.prompt_manager = PromptManager(agent_id)
        self.provider = None
        self.conversation_history = []
        self.initialized = False
        
        # Store model configuration
        self.requested_model = model
        self.requested_model_settings = model_settings
    
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
        self.available_tools = await self.tool_manager.get_available_tools()
        logger.info(f"Agent {self.agent_id} initialized with {len(self.available_tools)} tools")

        self.system_prompt = self.prompt_manager.get_system_prompt_with_tools(self.available_tools)
        logger.debug(f"System prompt: {self.system_prompt}")

        # Get model and settings with priority: CLI args > agent config > provider default
        agent_model, agent_model_settings = self.resource_manager.get_model_config()
        
        # Use requested model/settings if provided, otherwise fall back to agent config
        self.model = self.requested_model or agent_model or self.provider.config.default_model
        self.model_settings = self.requested_model_settings or agent_model_settings
        
        logger.debug(f"Model: {self.model}, Model settings: {self.model_settings}")

        self.memory_resource = await self.resource_manager.get_memory_resource()

        if self.memory_resource:
            memories = await self.memory_resource.get_memories(self.user_id, session_id=self.session_id, agent_id=self.agent_id)
            self.conversation_history = [{"role": memory.content["role"], "content": memory.content["content"]} for memory in memories]
            for memory in self.conversation_history:
                logger.debug(f"Memory: {memory}")
        else:
            logger.warning(f"No memory resource found for agent {self.agent_id}")
        
        self.initialized = True

    def _clean_response_for_memory(self, response: str) -> str:
        """Clean response before storing in memory."""
        # Remove content between <think> tags
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove \n characters and replace with actual newlines
        response = response.replace('\\n', '\n')
        
        # Clean up extra whitespace
        response = response.strip()
        
        return response

    async def save_memory(self, role: str, content: str):
        """Save a memory entry to the memory resource."""
        if self.memory_resource:
            memory_entry = MemoryEntry(
                user_id=self.user_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                content={"role": role, "content": content}
            )
            await self.memory_resource.store_memory(memory_entry)
        else:
            logger.warning(f"No memory resource found for agent {self.agent_id}")
    
    async def chat(self, user_input: str) -> str:
        """Send a message to the agent and get response."""
        if not self.initialized:
            await self.initialize()
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        await self.save_memory("user", user_input)
        
        # Get response from provider
        response = await self.provider.send_chat(
            context=self.conversation_history,
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            model_settings=self.model_settings
        )
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        await self.save_memory("assistant", self._clean_response_for_memory(response))
        return response
    
    async def interactive_mode(self):
        """Run interactive chat mode."""
        await self.initialize()
        
        print(f"🤖 {self.agent_id} Agent Ready!")
        print(f"🛠️ Available tools: {len(self.available_tools)}")
        print(f"🧠 Memory: {'Enabled' if self.memory_resource else 'Disabled'}")
        print(f"📝 System prompt: {self.prompt_manager.get_system_prompt()[:50]}...")
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