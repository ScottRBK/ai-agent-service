"""
Memory Compression Agent is responsible for compressing conversation history.
It acts as a specialized agent for creating summaries and managing compression logic.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from app.core.agents.base_agent import BaseAgent
from app.core.resources.memory_compression_manager import MemoryCompressionManager
from app.models.resources.memory import MemorySessionSummary
from app.models.resources.knowledge_base import DocumentType
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
                                  parent_memory_resource,
                                  knowledge_base_resource=None):
        """
        Compress conversation history using the compression agent.
        Returns compressed history but does not update database.
        
        Args:
            parent_agent_id: ID of the parent agent
            compression_config: Compression configuration
            user_id: User ID for memory session
            session_id: Session ID for memory session
            parent_memory_resource: Memory resource from the parent agent
            knowledge_base_resource: Optional knowledge base resource for archival
            
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

        # Determine if we should archive to KB
        archive_to_kb = (knowledge_base_resource is not None and 
                        compression_config.get("archive_conversations", False))
        
        # Create/Update summary using the compression agent        
        summary_result = await self._create_summary(older_messages, summary, 
                                                  compression_manager, archive_to_kb)
        
        if archive_to_kb:
            summary = summary_result['summary']
            metadata = summary_result['metadata']
            
            # Archive to knowledge base
            await self._archive_compressed_session(
                summary=summary,
                metadata=metadata,
                parent_agent_id=parent_agent_id,
                user_id=user_id,
                session_id=session_id,
                knowledge_base_resource=knowledge_base_resource
            )
        else:
            summary = summary_result
        
        summary_header = f"# Summary of older messages:"
        summary_body = summary
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
    
    async def _create_summary(self, messages: List[Dict[str, str]], existing_summary: str, 
                         compression_manager: MemoryCompressionManager,
                         archive_to_kb: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Create a summary of conversation messages using the compression agent.
        
        Args:
            messages: Messages to summarize
            existing_summary: Existing summary to update
            compression_manager: Compression manager instance
            archive_to_kb: Whether to extract metadata for KB archival
            
        Returns:
            str: Summary text (if not archiving to KB)
            Dict: {"summary": str, "metadata": dict} (if archiving to KB)
        """
        try:
            # Create a fresh provider instance to avoid inheriting context from other agents
            provider_info = self.provider_manager.get_provider(self.provider_id)
            config = provider_info["config_class"]()
            fresh_provider = provider_info["class"](config)
            await fresh_provider.initialize()

            formatted_messages = compression_manager.format_messages_for_summary(messages)
            
            # Enhanced prompt for KB archival
            if archive_to_kb:
                summary_instructions = """Please create a comprehensive summary of this conversation and extract key metadata.

Format your response EXACTLY as follows:

## SUMMARY
[Detailed narrative summary of the conversation]

## TOPICS
[Comma-separated list of main topics discussed]

## ENTITIES
[Comma-separated list of key people, projects, technologies mentioned]

## DECISIONS
[List each decision on a new line, or "None" if no decisions were made]

## QUESTIONS
[List each unresolved question on a new line, or "None" if no questions remain]"""
                if existing_summary:
                    summary_instructions += f"\n\nThe previous conversation summary was: {existing_summary}"
                    summary_instructions += "\n\nPlease update the summary to include the new messages."
                summary_instructions += f"\n\nNew messages to summarize:\n{formatted_messages}"
            else:
                # Original simple summary
                summary_instructions = "Please summarize the following conversation:"
                if existing_summary:
                    summary_instructions += f"\n\nPrevious summary: {existing_summary}"
                    summary_instructions += "\n\nUpdate the summary with new messages:"
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
            
            clean_response = self._clean_response_for_memory(response.strip())
            
            if archive_to_kb:
                return self._parse_summary_with_metadata(clean_response, messages)
            else:
                return clean_response
            
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
    
    def _parse_summary_with_metadata(self, summary_response: str, messages: List[Dict]) -> Dict[str, Any]:
        """Parse structured summary response into content and metadata."""
        sections = {}
        current_section = None
        content_lines = []
        
        for line in summary_response.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(content_lines).strip()
                current_section = line[3:].strip().lower()
                content_lines = []
            else:
                content_lines.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(content_lines).strip()
        
        # Extract metadata from parsed sections
        metadata = {
            'topics': [t.strip() for t in sections.get('topics', '').split(',') if t.strip()],
            'entities': [e.strip() for e in sections.get('entities', '').split(',') if e.strip()],
            'decisions': [d.strip() for d in sections.get('decisions', '').split('\n') if d.strip() and d.strip() != 'None'],
            'questions': [q.strip() for q in sections.get('questions', '').split('\n') if q.strip() and q.strip() != 'None'],
            'message_count': len(messages)
        }
        
        # Extract dates from messages
        if messages:
            first_msg = messages[0]
            last_msg = messages[-1]
            # Handle both MemoryEntry objects and dicts
            metadata['start_date'] = self._extract_timestamp(first_msg)
            metadata['end_date'] = self._extract_timestamp(last_msg)
        
        return {
            'summary': sections.get('summary', summary_response),
            'metadata': metadata
        }
    
    def _extract_timestamp(self, msg) -> str:
        """Extract timestamp from message (handles both dict and MemoryEntry)."""
        if hasattr(msg, 'created_at'):
            return msg.created_at.isoformat()
        elif isinstance(msg, dict) and 'created_at' in msg:
            return msg['created_at'].isoformat()
        else:
            return datetime.now().isoformat()
    
    async def _archive_compressed_session(self, summary: str, metadata: Dict[str, Any],
                                        parent_agent_id: str, user_id: str, 
                                        session_id: str, knowledge_base_resource):
        """Archive compressed session to knowledge base with extracted metadata."""
        try:
            # Format conversation document
            document_content = f"""# Conversation Summary

## Date Range: {metadata.get('start_date', 'Unknown')} to {metadata.get('end_date', 'Unknown')}

## Summary:
{summary}"""
            
            if metadata.get('topics'):
                document_content += f"\n\n## Topics Discussed:\n- " + "\n- ".join(metadata['topics'])
            
            if metadata.get('decisions'):
                document_content += f"\n\n## Key Decisions:\n- " + "\n- ".join(metadata['decisions'])
            
            if metadata.get('questions'):
                document_content += f"\n\n## Open Questions:\n- " + "\n- ".join(metadata['questions'])
            
            # Store in knowledge base
            await knowledge_base_resource.ingest_document(
                content=document_content,
                namespace=f"conversations:{user_id}",  # Simplified namespace
                doc_type=DocumentType.CONVERSATION,
                source=f"session:{session_id}",
                title=f"Conversation - {session_id} - {metadata.get('end_date', 'Unknown')}",
                metadata={
                    "session_id": session_id,
                    "agent_id": parent_agent_id,
                    "conversation_topics": metadata.get('topics', []),
                    "entities_mentioned": metadata.get('entities', []),
                    "decisions_made": metadata.get('decisions', []),
                    "open_questions": metadata.get('questions', []),
                    "message_count": metadata.get('message_count', 0),
                    "compression_timestamp": datetime.now(timezone.utc).isoformat(),
                    "date_range": {
                        "start": metadata.get('start_date'),
                        "end": metadata.get('end_date')
                    }
                }
            )
            logger.info(f"Archived conversation {session_id} to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to archive session to knowledge base: {e}")
            # Graceful degradation - compression continues even if archival fails
