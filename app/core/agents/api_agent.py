"""
API Agent implementation for FastAPI integration.
Based on BaseAgent but optimized for API usage.
"""

from typing import Optional, Dict, Any, AsyncGenerator
from app.core.agents.base_agent import BaseAgent
from app.utils.logging import logger

class APIAgent(BaseAgent):
    """
    API-optimized agent with tool capabilities.
    Similar to BaseAgent but designed for stateless API usage.
    """
    
    def __init__(self, 
                 agent_id: str,
                 user_id: str = "default_user",
                 session_id: str = "default_session",
                 model: Optional[str] = None,
                 model_settings: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, user_id, session_id, model, model_settings)
    
    def _get_provider_from_config(self) -> str:
        """Get provider from agent configuration"""
        config = self.tool_manager.config
        return config.get("provider", "azure_openai_cc")
    
    
    async def chat(self, user_input: str) -> str:
        """Send a message to the agent and get response"""
        if not self.initialized:
            await self.initialize()
        
        # Get conversation history
        conversation_history = await self.get_conversation_history()
        
        # Add user message
        conversation_history.append({"role": "user", "content": user_input})
        await self.save_memory("user", user_input)
        
        # Get response from provider
        response = await self.provider.send_chat(
            context=conversation_history,
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            tools=None,
            model_settings=self.model_settings
        )
        
        # Save assistant response
        await self.save_memory("assistant", self._clean_response_for_memory(response))

        # Trigger memory compression
        await self._trigger_memory_compression()
        
        return response
    
    async def clear_conversation(self):
        """Clear conversation history for current session"""
        if self.memory:
            await self.memory.clear_session(
                self.user_id, 
                self.session_id
            )

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Send a message to the agent and stream the response"""
        if not self.initialized:
            await self.initialize()
        
        # Get conversation history
        conversation_history = await self.get_conversation_history()
        
        # Add user message
        conversation_history.append({"role": "user", "content": user_input})
        if self.memory:
            await self.save_memory("user", user_input)
        
        # Stream response from provider
        full_response = ""
        async for chunk in self.provider.send_chat_with_streaming(
            context=conversation_history,
            model=self.model,
            instructions=self.system_prompt,
            agent_id=self.agent_id,
            tools=None,
            model_settings=self.model_settings
        ):
            full_response += chunk
            yield chunk
        
        # Save assistant response after streaming is complete
        await self.save_memory("assistant", self._clean_response_for_memory(full_response))

        # Trigger memory compression
        await self._trigger_memory_compression()
