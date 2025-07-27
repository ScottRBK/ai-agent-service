"""
Base Agent implementation providing common functionality for all agent types.
Memory functionality is optional and configuration-driven.
"""

from typing import Optional, Dict, Any, List, AsyncGenerator
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.core.agents.prompt_manager import PromptManager
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.utils.logging import logger
from app.utils.chat_utils import clean_response_for_memory
from app.models.resources.memory import MemoryEntry, MemorySessionSummary


class BaseAgent:
    """
    Base class for all agents providing common functionality.
    Memory features are optional and activate only when configured.
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
        self.conversation_history = []
        self.summary = None
        self.initialized = False
        
        # Memory resource - None if not configured
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
            logger.debug(f"System prompt: {self.system_prompt[:10]}")

            # Get model and settings with priority: CLI args > agent config > provider default
            agent_model, agent_model_settings = self.resource_manager.get_model_config()
            self.model = self.requested_model or agent_model or self.provider.config.default_model
            self.model_settings = self.requested_model_settings or agent_model_settings
            
            logger.debug(f"Model: {self.model}, Model settings: {self.model_settings}")

            # Get memory resource if configured
            self.memory_resource = await self.resource_manager.get_memory_resource()
            
            self.initialized = True
        except Exception as e:
            logger.error(f"Error during agent initialization: {e}")
            raise
    
    
    def _clean_response_for_memory(self, response: str) -> str:
        """Clean response before storing in memory."""
        return clean_response_for_memory(response)
    
    async def save_memory(self, role: str, content: str):
        """Save a memory entry to the memory resource. Only executes if memory is configured."""
        if self.memory_resource:
            memory_entry = MemoryEntry(
                user_id=self.user_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                content={"role": role, "content": content}
            )
            await self.memory_resource.store_memory(memory_entry)
        else:
            logger.debug(f"No memory resource found for agent {self.agent_id}")
    
    async def load_memory(self) -> List[Dict[str, str]]:
        """Load memory from the memory resource. Returns empty list if no memory configured."""
        if not self.memory_resource:
            return []
            
        try:
            memories: List[MemoryEntry] = await self.memory_resource.get_memories(
                self.user_id, 
                self.session_id, 
                self.agent_id, 
                order_direction="asc"
            )
            summary: MemorySessionSummary = await self.memory_resource.get_session_summary(
                self.user_id, 
                self.session_id, 
                self.agent_id
            )
            
            conversation_history = []
            if summary:
                conversation_history.append({"role": "system", "content": summary.summary})
            
            conversation_history.extend([
                {"role": memory.content["role"], "content": memory.content["content"]} 
                for memory in memories
            ])
            
            return conversation_history
        
        except Exception as e:
            logger.error(f"Error loading memory for agent {self.agent_id}: {e}")
            return []
    
    async def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history. Alias for load_memory for compatibility."""
        return await self.load_memory()
    
    async def _trigger_memory_compression(self, compression_config: Optional[Dict[str, Any]] = None):
        """Trigger memory compression if memory is configured."""
        if not self.memory_resource:
            return
            
        try:
            # Import here to avoid circular dependency
            from app.core.agents.memory_compression_agent import MemoryCompressionAgent
            
            if not compression_config:
                compression_config = {
                    "threshold_tokens": 10000,
                    "recent_messages_to_keep": 10,
                    "enabled": True
                }
            
            memory_compression_agent = MemoryCompressionAgent()
            await memory_compression_agent.compress_conversation(
                self.agent_id, 
                compression_config,
                self.user_id,
                self.session_id
            )
        except Exception as e:
            logger.error(f"Error during memory compression: {e}")