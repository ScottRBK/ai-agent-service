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
    
    def _get_resource_config(self, resource_name: str) -> dict:
        """Get resource configuration directly from resource_config"""
        resource_config = self.tool_manager.config.get("resource_config", {})
        return resource_config.get(resource_name, {})
    
    async def setup_providers(self):
        """Initialize all providers needed by this agent."""
        # Main provider (chat)
        provider_id = self.tool_manager.config.get("provider", "azure_openai_cc")
        provider_info = self.provider_manager.get_provider(provider_id)
        config = provider_info["config_class"]()
        self.provider = provider_info["class"](config)
        # Set agent instance on provider for tool execution
        self.provider.agent_instance = self
        await self.provider.initialize()
        
        # Embedding provider (may be same as main)
        # Check in knowledge_base resource config first, then top-level config
        kb_config = self._get_resource_config("knowledge_base")
        embedding_provider_id = kb_config.get("embedding_provider") or self.tool_manager.config.get("embedding_provider", provider_id)
        if embedding_provider_id != provider_id:
            embedding_info = self.provider_manager.get_provider(embedding_provider_id)
            embedding_config = embedding_info["config_class"]()
            self.embedding_provider = embedding_info["class"](embedding_config)
            await self.embedding_provider.initialize()
        else:
            self.embedding_provider = self.provider
        
        # Optional rerank provider
        # Check in knowledge_base resource config first, then top-level
        kb_config = self._get_resource_config("knowledge_base")
        rerank_provider_id = kb_config.get("rerank_provider") or self.tool_manager.config.get("rerank_provider")
        
        if rerank_provider_id:
            # Check if it's the same as main or embedding provider to reuse instances
            if rerank_provider_id == provider_id:
                self.rerank_provider = self.provider
            elif hasattr(self, 'embedding_provider') and rerank_provider_id == embedding_provider_id:
                self.rerank_provider = self.embedding_provider
            else:
                # Create new provider instance
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
        
        # Get config from resource_config
        kb_config = self._get_resource_config("knowledge_base")
        
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
        
        # Get embedding model from knowledge base config
        agent_model = self.tool_manager.config.get("model")
        default_model = self.requested_model or agent_model or self.provider.config.default_model
        
        # Check in resource_config first, then fall back to top-level config
        embedding_model = kb_config.get("embedding_model") or self.tool_manager.config.get("embedding_model", default_model)
        
        kb.set_embedding_provider(self.embedding_provider, embedding_model)
        
        if self.rerank_provider:
            # Check resource_config first, then top-level config
            rerank_model = kb_config.get("rerank_model") or self.tool_manager.config.get("rerank_model")
            if rerank_model:
                kb.set_rerank_provider(self.rerank_provider, rerank_model)
        
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
        """Load memory with automatic cross-session context if enabled."""
        if not self.memory:
            return []
        
        try:
            # Load current session memory
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
            
            # Add cross-session context if enabled
            kb_config = self._get_resource_config("knowledge_base")
            if (kb_config.get("enable_cross_session_context", False) and 
                self.knowledge_base and conversation_history):
                
                # Check if latest message warrants cross-session search
                last_user_msg = next((m for m in reversed(conversation_history) 
                                    if m["role"] == "user"), None)
                if last_user_msg and self._should_search_cross_session(last_user_msg["content"]):
                    cross_session_context = await self._get_cross_session_context(
                        last_user_msg["content"], self.session_id
                    )
                    if cross_session_context:
                        # Insert cross-session context before recent messages
                        insert_position = max(len(conversation_history) - 5, 1)  # Keep last 5 messages
                        conversation_history.insert(insert_position, {
                            "role": "system",
                            "content": cross_session_context
                        })
            
            return conversation_history
        
        except Exception as e:
            logger.error(f"Error loading memory for agent {self.agent_id}: {e}")
            return []
    
    async def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history. Alias for load_memory for compatibility."""
        return await self.load_memory()
    
    async def _trigger_memory_compression(self):
        """Trigger memory compression when threshold is reached"""
        if not self.memory:
            return
        
        memory_config = self._get_resource_config("memory")
        compression_config = memory_config.get("compression", {
            "threshold_tokens": 10000,
            "recent_messages_to_keep": 10,
            "enabled": True
        })
        
        if not compression_config.get("enabled", True):
            return
            
        try:
            # Import here to avoid circular dependency
            from app.core.agents.memory_compression_agent import MemoryCompressionAgent
            
            # Pass knowledge base resource if archival is enabled
            knowledge_base_resource = None
            if compression_config.get("archive_conversations", False) and self.knowledge_base:
                knowledge_base_resource = self.knowledge_base
            
            memory_compression_agent = MemoryCompressionAgent()
            await memory_compression_agent.compress_conversation(
                parent_agent_id=self.agent_id,
                compression_config=compression_config,
                user_id=self.user_id,
                session_id=self.session_id,
                parent_memory_resource=self.memory,
                knowledge_base_resource=knowledge_base_resource  # New parameter
            )
        except Exception as e:
            logger.error(f"Error during memory compression: {e}")
    
    def _should_search_cross_session(self, user_message: str) -> bool:
        """Determine if cross-session search is needed"""
        kb_config = self._get_resource_config("knowledge_base")
        
        # Check configuration for custom triggers
        triggers = kb_config.get("search_triggers", [
            "what did we discuss", "previous conversation", "last time",
            "remind me", "we mentioned", "earlier you said"
        ])
        
        return any(trigger in user_message.lower() for trigger in triggers)
    
    async def _get_cross_session_context(self, query: str, current_session_id: str) -> Optional[str]:
        """Retrieve relevant context from other sessions"""
        if not self.knowledge_base:
            return None
        
        try:
            kb_config = self._get_resource_config("knowledge_base")
            relevance_threshold = kb_config.get("relevance_threshold", 0.7)
            
            # Search for relevant past conversations
            results = await self.knowledge_base.search(
                query=query,
                namespaces=[f"conversations:{self.user_id}"],
                limit=10,
                use_reranking=True
            )
            
            # Filter by relevance threshold and exclude current session
            relevant_results = []
            for r in results:
                if r.score > relevance_threshold:
                    # Check if this is from a different session
                    metadata = r.document.metadata if r.document else {}
                    if metadata.get('session_id') != current_session_id:
                        relevant_results.append(r)
                        if len(relevant_results) >= 3:
                            break
            
            if not relevant_results:
                return None
            
            # Format context
            context_parts = ["## Relevant Past Conversations\n"]
            for result in relevant_results:
                metadata = result.document.metadata if result.document else {}
                date_range = metadata.get('date_range', {})
                context_parts.append(
                    f"### Session from {date_range.get('start', 'Unknown date')}\n"
                    f"{result.chunk.content}\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Cross-session context retrieval failed: {e}")
            return None  # Graceful degradation