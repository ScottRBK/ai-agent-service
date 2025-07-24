"""
Agent Memory Compression Manager is responsible for managing conversation history compression.
It provides compression logic and token counting without direct agent dependencies.
"""
from typing import List, Dict, Any, Optional
from app.utils.token_counter import TokenCounter
from app.utils.logging import logger
from app.core.resources.memory import MemoryEntry
class MemoryCompressionManager:
    """
    Manages memory compression logic for specific agents.
    Provides compression utilities without direct agent dependencies.
    """

    def __init__(self, agent_id: str, compression_config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = compression_config or {}
        self.threshold_tokens = self.config.get("threshold_tokens", 8000)
        self.recent_messages_to_keep = self.config.get("recent_messages_to_keep", 4)
        self.enabled = self.config.get("enabled", True)
        self.token_counter = TokenCounter()

    def should_compress(self, conversation_history: List[MemoryEntry]) -> bool:
        """
        Check if conversation history exceeds token threshold.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            True if compression is needed, False otherwise
        """
        if not self.enabled:
            return False
            
        if len(conversation_history) <= self.recent_messages_to_keep:
            return False
        
            
        total_tokens = self.token_counter.count_conversation_tokens([{"role": memory.content["role"], "content": memory.content["content"]} for memory in conversation_history])
        logger.info(f"MemoryCompressionManager- should_compress - Total tokens: {total_tokens}, Threshold tokens: {self.threshold_tokens}")
        should_compress = total_tokens > self.threshold_tokens
        
        if should_compress:
            logger.info(f"Conversation history for agent {self.agent_id} exceeds threshold: {total_tokens} > {self.threshold_tokens} tokens")
        
        return should_compress
    
    def split_conversation_for_compression(self, conversation_history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split conversation into older messages (to summarize) and recent messages (to keep).
        
        Args:
            conversation_history: Full conversation history
            
        Returns:
            Tuple of (older_messages, recent_messages)
        """
        if len(conversation_history) <= self.recent_messages_to_keep:
            return [], conversation_history
        
        older_messages = conversation_history[:-self.recent_messages_to_keep]
        recent_messages = conversation_history[-self.recent_messages_to_keep:]
        
        return older_messages, recent_messages
    
    def format_messages_for_summary(self, messages: List[MemoryEntry]) -> str:
        """
        Format messages for summary generation.
        
        Args:
            messages: Messages to format
            
        Returns:
            Formatted message string
        """
        formatted = []
        for i, message in enumerate(messages, 1):
            role = message.content["role"]
            content = message.content["content"]
            formatted.append(f"{i}. {role.upper()}: {content}")
        
        return "\n\n".join(formatted)
    
    def get_compression_stats(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Get compression statistics for monitoring."""
        total_tokens = self.token_counter.count_conversation_tokens(conversation_history)
        
        return {
            "agent_id": self.agent_id,
            "enabled": self.enabled,
            "threshold_tokens": self.threshold_tokens,
            "current_tokens": total_tokens,
            "should_compress": total_tokens > self.threshold_tokens,
            "message_count": len(conversation_history),
            "recent_messages_to_keep": self.recent_messages_to_keep
        }