"""
CLI Agent implementation using the AgentToolManager framework.
"""

import asyncio
from typing import Optional, AsyncGenerator
from app.core.agents.base_agent import BaseAgent
from app.utils.logging import logger

class CLIAgent(BaseAgent):
    """
    Interactive command-line agent with tool capabilities.
    """
    
    def __init__(self, agent_id: str = "cli_agent", 
             provider_id: Optional[str] = None, 
             user_id: str = "default_user", 
             session_id: str = "default_session",
             model: Optional[str] = None,
             model_settings: Optional[dict] = None):
        self.provider_id = provider_id
        super().__init__(agent_id, user_id, session_id, model, model_settings)
    
    def _get_provider_from_config(self) -> str:
        """Override to use CLI-specific provider_id if provided"""
        if self.provider_id is not None:
            return self.provider_id
        return super()._get_provider_from_config()
        
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
        
        compression_config = {
            "threshold_tokens": 500,
            "recent_messages_to_keep": 4,
            "enabled": True
        }
        await self._trigger_memory_compression(compression_config)

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
            "threshold_tokens": 4000,
            "recent_messages_to_keep": 4,
            "enabled": True
        }
        await self._trigger_memory_compression(compression_config)
       
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