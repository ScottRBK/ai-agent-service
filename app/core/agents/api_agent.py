"""
API Agent implementation for FastAPI integration.
Based on CLIAgent but optimized for API usage.
"""

from typing import Optional, Dict, Any
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.core.agents.prompt_manager import PromptManager
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.utils.logging import logger
from app.models.resources.memory import MemoryEntry

class APIAgent:
    """
    API-optimized agent with tool capabilities.
    Similar to CLIAgent but designed for stateless API usage.
    """
    
    def __init__(self, 
                 agent_id: str,
                 user_id: str = "default_user",
                 session_id: str = "default_session",
                 model: Optional[str] = None,
                 model_settings: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id
        self.tool_manager = AgentToolManager(agent_id)
        self.resource_manager = AgentResourceManager(agent_id)
        self.provider_manager = ProviderManager()
        self.prompt_manager = PromptManager(agent_id)
        self.provider = None
        self.initialized = False
        
        # Initialize memory_resource to None
        self.memory_resource = None
        
        # Store model configuration
        self.requested_model = model
        self.requested_model_settings = model_settings
        
        # Get provider from agent config
        self.provider_id = self._get_provider_from_config()
    
    def _get_provider_from_config(self) -> str:
        """Get provider from agent configuration"""
        config = self.tool_manager.config
        return config.get("provider", "azure_openai_cc")
    
    async def initialize(self):
        """Initialize the agent and provider"""
        if self.initialized:
            return
            
        try:
            # Get provider
            provider_info = self.provider_manager.get_provider(self.provider_id)
            config = provider_info["config_class"]()
            self.provider = provider_info["class"](config)
            await self.provider.initialize()
            
            # Get available tools
            self.available_tools = await self.tool_manager.get_available_tools()
            logger.info(f"Agent {self.agent_id} initialized with {len(self.available_tools)} tools")

            # Get system prompt
            self.system_prompt = self.prompt_manager.get_system_prompt_with_tools(self.available_tools)

            # Get model and settings
            agent_model, agent_model_settings = self.resource_manager.get_model_config()
            self.model = self.requested_model or agent_model or self.provider.config.default_model
            self.model_settings = self.requested_model_settings or agent_model_settings
            
            # Get memory resource
            self.memory_resource = await self.resource_manager.get_memory_resource()
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Error during agent initialization: {e}")
            raise
    
    async def get_conversation_history(self) -> list:
        """Get conversation history for current session"""
        if not self.memory_resource:
            return []
            
        memories = await self.memory_resource.get_memories(
            self.user_id, 
            session_id=self.session_id, 
            agent_id=self.agent_id
        )
        return [{"role": memory.content["role"], "content": memory.content["content"]} 
                for memory in memories]
    
    def _clean_response_for_memory(self, response: str) -> str:
        """Clean response before storing in memory"""
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.replace('\\n', '\n')
        return response.strip()
    
    async def save_memory(self, role: str, content: str):
        """Save a memory entry"""
        if self.memory_resource:
            memory_entry = MemoryEntry(
                user_id=self.user_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                content={"role": role, "content": content}
            )
            await self.memory_resource.store_memory(memory_entry)
    
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
            model_settings=self.model_settings
        )
        
        # Save assistant response
        await self.save_memory("assistant", self._clean_response_for_memory(response))
        
        return response
    
    async def clear_conversation(self):
        """Clear conversation history for current session"""
        if self.memory_resource:
            await self.memory_resource.clear_session_memories(
                self.user_id, 
                self.session_id, 
                self.agent_id
            )
