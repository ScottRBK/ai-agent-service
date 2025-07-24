"""
CLI Agent implementation using the AgentToolManager framework.
"""

import asyncio
from typing import Optional
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.core.agents.prompt_manager import PromptManager
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.core.agents.memory_compression_agent import MemoryCompressionAgent
from app.utils.logging import logger
from app.models.resources.memory import MemoryEntry, MemorySessionSummary
from typing import List, AsyncGenerator

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
        self.summary = None
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

    async def load_memory(self):
        """Load memory from the memory resource."""
        try:
            memories: List[MemoryEntry] = await self.memory_resource.get_memories(self.user_id, self.session_id, self.agent_id, order_direction="asc")
            summary: MemorySessionSummary = await self.memory_resource.get_session_summary(self.user_id, self.session_id, self.agent_id)
            if summary:
                conversation_history = [{"role": "system", "content": summary.summary}]
            else:
                conversation_history = []
            conversation_history.extend([{"role": memory.content["role"], "content": memory.content["content"]} for memory in memories])
            return conversation_history
        
        except Exception as e:
            logger.error(f"client agent - load_memory - Error loading memory for agent {self.agent_id}: {e}")
            return []
        
    async def chat_stream_with_memory(self, user_input: str) -> AsyncGenerator[str, None]:
        """Send a message to the agent and stream the response"""
        if not self.initialized:
            await self.initialize()
        
        conversation_history = await self.load_memory()
        conversation_history.append({"role": "user", "content": user_input})

        full_response = ""
        async for chunk in self.provider.send_chat_with_streaming(
            context=conversation_history,
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            model_settings=self.model_settings
        ):
            full_response += chunk
            yield chunk

        clean_response = self._clean_response_for_memory(full_response)
        await self.save_memory("assistant", clean_response)
        memory_compression_agent = MemoryCompressionAgent()
        compression_config = {
                    "threshold_tokens": 500,
                    "recent_messages_to_keep": 4,
                    "enabled": True
                    }
        await memory_compression_agent.compress_conversation(self.agent_id, 
                                                            compression_config,
                                                            self.user_id,
                                                            self.session_id)

    async def chat_with_memory(self, user_input: str) -> str:
        """Send a message to the agent and get response."""
        if not self.initialized:
            await self.initialize()
        
        conversation_history = await self.load_memory()
        # Add user message to history
        await self.save_memory("user", user_input)     

        # Get response from provider
        response = await self.provider.send_chat(
            context=[*conversation_history, {"role": "user", "content": user_input}],
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            model_settings=self.model_settings
        )
        
        # Add assistant response to history
        clean_response = self._clean_response_for_memory(response)
        await self.save_memory("assistant", clean_response)

        compression_config = {
                    "threshold_tokens": 500,
                    "recent_messages_to_keep": 4,
                    "enabled": True
                    }
        memory_compression_agent = MemoryCompressionAgent()
        await memory_compression_agent.compress_conversation(self.agent_id, 
                                                            compression_config,
                                                            self.user_id,
                                                            self.session_id)
       
        return response
    
    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Send a message to the agent and stream the response"""
        if not self.initialized:
            await self.initialize()
            
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})

        full_response = ""
        async for chunk in self.provider.send_chat_with_streaming(
            context=self.conversation_history,
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            model_settings=self.model_settings
        ):
            full_response += chunk
            yield chunk

        self.conversation_history.append({"role": "assistant", "content": full_response})
    
    async def chat(self, user_input: str,) -> str:
        """Send a message to the agent and get response."""
        if not self.initialized:
            await self.initialize()

        if self.memory_resource:
            return await self.chat_with_memory(user_input)
     
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})  

        # Get response from provider
        response = await self.provider.send_chat(
            context=[*self.conversation_history, {"role": "user", "content": user_input}],
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            model_settings=self.model_settings
        )
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response})
       
        return response
    
    async def interactive_mode(self, stream: bool = False):
        """Run interactive chat mode."""
        await self.initialize()
        
        print(f"🤖 {self.agent_id} Agent Ready!")
        print(f"🛠️ Available tools: {len(self.available_tools)}")
        print(f"🧠 Memory: {'Enabled' if self.memory_resource else 'Disabled'}")
        print(f"📝 System prompt: {self.prompt_manager.get_system_prompt()}...")
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
                if stream:
                    if self.memory_resource:
                        async for chunk in self.chat_stream_with_memory(user_input):
                            print(chunk, end="", flush=True)
                    else:
                        async for chunk in self.chat_stream(user_input):
                            print(chunk, end="", flush=True)
                    print("\n")  # Add newline after streaming
                else:
                    response = await self.chat(user_input)
                    print(f"🤖 {self.agent_id}: {response}\n")
                  
                        
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