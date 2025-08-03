"""
Base Agent implementation providing common functionality for all agent types.
Memory functionality is optional and configuration-driven.
Uses direct composition instead of complex manager layers.
"""

from typing import Optional, Dict, Any, List, AsyncGenerator
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager
from app.core.agents.prompt_manager import PromptManager
from app.utils.logging import logger
from app.utils.chat_utils import clean_response_for_memory
from app.models.resources.memory import MemoryEntry, MemorySessionSummary


class BaseAgent:
    """
    Base class for all agents providing common functionality.
    Memory features are optional and activate only when configured.
    Uses direct composition instead of complex manager layers.
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
        
        # Direct resource composition (no managers)
        self.memory = None
        self.knowledge_base = None
        
        # Keep only essential managers
        self.tool_manager = AgentToolManager(agent_id)
        self.provider_manager = ProviderManager()
        self.prompt_manager = PromptManager(agent_id)
        
        # Provider attributes
        self.provider = None
        self.embedding_provider = None
        self.rerank_provider = None
        
        self.conversation_history = []
        self.summary = None
        self.initialized = False
        
        # Store model configuration
        self.requested_model = model
        self.requested_model_settings = model_settings
        
        # Get provider from agent config
        self.provider_id = self._get_provider_from_config()
    
    def _get_provider_from_config(self) -> str:
        """Get provider from agent configuration"""
        config = self.tool_manager.config
        return config.get("provider", "azure_openai_cc")

    def get_resource(self, resource_type: str):
        """Get resource by type name."""
        if resource_type == "memory":
            return self.memory
        elif resource_type == "knowledge_base":
            return self.knowledge_base
        return None
    
    def wants_memory(self) -> bool:
        """Check if agent is configured for memory."""
        resources = self.tool_manager.config.get("resources", [])
        return "memory" in resources

    def wants_knowledge_base(self) -> bool:
        """Check if agent is configured for knowledge base."""
        resources = self.tool_manager.config.get("resources", [])
        return "knowledge_base" in resources
    
    async def setup_providers(self):
        """Initialize all providers needed by this agent."""
        # Main provider (chat)
        provider_id = self.tool_manager.config.get("provider", "azure_openai_cc")
        provider_info = self.provider_manager.get_provider(provider_id)
        config = provider_info["config_class"]()
        self.provider = provider_info["class"](config)
        await self.provider.initialize()
        
        # Embedding provider (may be same as main)
        embedding_provider_id = self.tool_manager.config.get("embedding_provider", provider_id)
        if embedding_provider_id != provider_id:
            embedding_info = self.provider_manager.get_provider(embedding_provider_id)
            embedding_config = embedding_info["config_class"]()
            self.embedding_provider = embedding_info["class"](embedding_config)
            await self.embedding_provider.initialize()
        else:
            self.embedding_provider = self.provider
        
        # Optional rerank provider
        rerank_provider_id = self.tool_manager.config.get("rerank_provider")
        if rerank_provider_id:
            rerank_info = self.provider_manager.get_provider(rerank_provider_id)
            rerank_config = rerank_info["config_class"]()
            self.rerank_provider = rerank_info["class"](rerank_config)
            await self.rerank_provider.initialize()
    
    async def create_memory(self):
        """Create memory resource with explicit dependencies."""
        from app.core.resources.memory import PostgreSQLMemoryResource
        from app.config.settings import settings
        
        connection_string = (
            f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        
        config = {
            "connection_string": connection_string,
            "default_ttl_hours": 24 * 7
        }
        
        memory = PostgreSQLMemoryResource(f"{self.agent_id}_memory", config)
        await memory.initialize()
        return memory

    async def create_knowledge_base(self):
        """Create knowledge base with all dependencies injected."""
        from app.core.resources.knowledge_base import KnowledgeBaseResource
        from app.config.settings import settings
        
        # Get config
        kb_config = self.tool_manager.config.get("knowledge_base", {})
        
        # Add database connection
        connection_string = (
            f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        kb_config["connection_string"] = connection_string
        
        # Create with all dependencies
        kb = KnowledgeBaseResource(f"{self.agent_id}_kb", kb_config)
        
        # Inject providers
        kb.set_chat_provider(self.provider)
        
        # Get embedding model (use agent model if not specified)
        agent_model = self.tool_manager.config.get("model")
        default_model = self.requested_model or agent_model or self.provider.config.default_model
        embedding_model = self.tool_manager.config.get("embedding_model", default_model)
        
        kb.set_embedding_provider(self.embedding_provider, embedding_model)
        
        if self.rerank_provider:
            kb.set_rerank_provider(
                self.rerank_provider,
                self.tool_manager.config.get("rerank_model")
            )
        
        # Initialize with all dependencies ready
        await kb.initialize()
        return kb
    
    async def initialize(self):
        """Initialize agent with explicit dependency flow."""
        if self.initialized:
            return
        
        try:
            # 1. Setup all providers first
            await self.setup_providers()
            
            # 2. Get tools and model config
            self.available_tools = await self.tool_manager.get_available_tools()
            self.system_prompt = self.prompt_manager.get_system_prompt_with_tools(self.available_tools)
            
            agent_model = self.tool_manager.config.get("model")
            agent_model_settings = self.tool_manager.config.get("model_settings")
            self.model = self.requested_model or agent_model or self.provider.config.default_model
            self.model_settings = self.requested_model_settings or agent_model_settings
            
            logger.info(f"Agent {self.agent_id} initialized with {len(self.available_tools)} tools")
            logger.debug(f"Model: {self.model}, Model settings: {self.model_settings}")
            
            # 3. Create resources if configured
            if self.wants_memory():
                self.memory = await self.create_memory()
            
            if self.wants_knowledge_base():
                self.knowledge_base = await self.create_knowledge_base()
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error during agent initialization: {e}")
            raise
    
    def _clean_response_for_memory(self, response: str) -> str:
        """Clean response before storing in memory."""
        return clean_response_for_memory(response)
    
    async def save_memory(self, role: str, content: str):
        """Save memory if available."""
        if self.memory:
            memory_entry = MemoryEntry(
                user_id=self.user_id,
                session_id=self.session_id,
                agent_id=self.agent_id,
                content={"role": role, "content": content}
            )
            await self.memory.store_memory(memory_entry)
    
    async def load_memory(self) -> List[Dict[str, str]]:
        """Load memory if available."""
        if not self.memory:
            return []
        
        try:
            memories: List[MemoryEntry] = await self.memory.get_memories(
                self.user_id, 
                self.session_id, 
                self.agent_id, 
                order_direction="asc"
            )
            summary: MemorySessionSummary = await self.memory.get_session_summary(
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
        if not self.memory:
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
                self.session_id,
                self.memory
            )
        except Exception as e:
            logger.error(f"Error during memory compression: {e}")