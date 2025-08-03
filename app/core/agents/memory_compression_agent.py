"""
Memory Compression Agent is responsible for compressing conversation history.
It acts as a specialized agent for creating summaries and managing compression logic.
"""

from typing import List, Dict, Any, Optional
from app.core.agents.base_agent import BaseAgent
from app.core.resources.memory_compression_manager import MemoryCompressionManager
from app.models.resources.memory import MemorySessionSummary
from app.utils.logging import logger

class MemoryCompressionAgent(BaseAgent):
    """
    Dedicated agent for memory compression tasks.
    Handles conversation summarization and compression logic.
    """
    
    def __init__(self, 
                 agent_id: str = "memory_compression_agent",
                 model: Optional[str] = None,
                 model_settings: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "default_user", "default_session", model, model_settings)
    
    
    async def compress_conversation(self, parent_agent_id: str, 
                                  compression_config: Dict[str, Any],
                                  user_id: str,
                                  session_id: str,
                                  parent_memory_resource):
        """
        Compress conversation history using the compression agent.
        Returns compressed history but does not update database.
        
        Args:
            parent_agent_id: ID of the parent agent
            compression_config: Compression configuration
            user_id: User ID for memory session
            session_id: Session ID for memory session
            parent_memory_resource: Memory resource from the parent agent
            
        Returns:
            Compressed conversation history
        """
        if not self.initialized:
            await self.initialize()
        
        # Initialize compression manager with config
        compression_manager = MemoryCompressionManager(parent_agent_id, compression_config)
        
        conversation_history = await parent_memory_resource.get_memories(user_id, session_id, parent_agent_id, order_by="created_at", order_direction="asc")        
        # Check if compression is needed
        if not compression_manager.should_compress(conversation_history):
            logger.info("MemoryCompressionAgent- compress_conversation - No compression needed")
            return ""

        # Split conversation for compression
        older_messages, recent_messages = compression_manager.split_conversation_for_compression(conversation_history)
        logger.info(f"MemoryCompressionAgent- compress_conversation - Older messages: {len(older_messages)}, Recent messages: {len(recent_messages)}")
        
        # Get existing summary
        existing_summary: MemorySessionSummary = await parent_memory_resource.get_session_summary(user_id, session_id, parent_agent_id)
        if existing_summary:
            summary = existing_summary.summary
        else:
            summary = ""

        # Create/Update summary using the compression agent        
        summary_header = f"# Summary of older messages:"
        summary_body = await self._create_summary(older_messages, summary, compression_manager)
        summary = f"{summary_header}\n\n{summary_body}"

        logger.info(f"MemoryCompressionAgent- compress_conversation - Summary Created")

        # Store summary
        await parent_memory_resource.store_session_summary(MemorySessionSummary(
            user_id=user_id,
            session_id=session_id,
            agent_id=parent_agent_id,
            summary=summary
        ))
        logger.info(f"MemoryCompressionAgent- compress_conversation - Summary Stored")
        for message in older_messages:
            await parent_memory_resource.delete_memory(message.id)
        logger.info(f"MemoryCompressionAgent- compress_conversation - Older messages deleted")
    
    async def _create_summary(self, messages: List[Dict[str, str]], existing_summary: str, compression_manager: MemoryCompressionManager) -> str:
        """
        Create a summary of conversation messages using the compression agent.
        
        Args:
            messages: Messages to summarize
            existing_summary: Existing summary to update
            compression_manager: Compression manager instance
            
        Returns:
            Summary text
        """
        try:
            # Create a fresh provider instance to avoid inheriting context from other agents
            provider_info = self.provider_manager.get_provider(self.provider_id)
            config = provider_info["config_class"]()
            fresh_provider = provider_info["class"](config)
            await fresh_provider.initialize()

            formatted_messages = compression_manager.format_messages_for_summary(messages)
            summary_instructions = "Please summarize the following conversation:"
            if existing_summary:
                summary_instructions += f"\n\nThe previous conversation summary was: {existing_summary}"
                summary_instructions += "\n\nPlease update the summary to reflect the new conversation."
                summary_instructions += "\n\nNew messages to factor into the summary:"
            summary_instructions += f"\n\n{formatted_messages}"

            message = [
                {"role": "user", "content": 
                 summary_instructions}
            ]
            
            
            # Use the fresh provider to create summary
            response = await fresh_provider.send_chat(
                context=message,
                model=self.model,
                instructions=self.system_prompt,
                agent_id=self.agent_id,
                model_settings=self.model_settings
            )
            
            # Clean up the fresh provider
            await fresh_provider.cleanup()
            
            return self._clean_response_for_memory(response.strip())
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Previous conversation context (summary unavailable)"
    
    def get_compression_stats(self, conversation_history: List[Dict[str, str]], 
                            compression_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get compression statistics for monitoring.
        
        Args:
            conversation_history: Conversation history to analyze
            compression_config: Optional compression configuration
            
        Returns:
            Compression statistics
        """
        compression_manager = MemoryCompressionManager(self.agent_id, compression_config)
        return compression_manager.get_compression_stats(conversation_history)
