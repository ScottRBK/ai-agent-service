"""
PostgreSQL memory resource for conversation history persistence.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Boolean, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from app.core.resources.base import BaseResource, ResourceType, ResourceError, ResourceConnectionError
from app.models.resources.memory import MemoryEntry, MemorySessionSummary 
from app.utils.logging import logger

Base = declarative_base()  # Add this line back

class MemoryEntryTable(Base):  # Renamed to avoid confusion
    """Database table model for memory entries."""
    __tablename__ = 'memory_entries'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    entry_metadata = Column(Text, nullable=True) 
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class MemorySessionSummaryTable(Base):
    """Database table model for memory session summaries."""
    __tablename__ = 'memory_session_summaries'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class PostgreSQLMemoryResource(BaseResource):
    """PostgreSQL memory resource for conversation history."""
    
    def __init__(self, resource_id: str, config: dict):
        super().__init__(resource_id, config)
        self.engine = None
        self.SessionLocal = None
        self.connection_string = config.get("connection_string")
        self.default_ttl_hours = config.get("default_ttl_hours", 24 * 7)  # 1 week default
        
        if not self.connection_string:
            raise ResourceError("PostgreSQL connection string is required", resource_id)
    
    def _get_resource_type(self) -> ResourceType:
        return ResourceType.MEMORY
    
    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables."""
        try:
            self.engine = create_engine(self.connection_string)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            self.initialized = True
            logger.info(f"PostgreSQL Memory Resource {self.resource_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Memory Resource {self.resource_id}: {e}")
            raise ResourceConnectionError(f"Failed to connect to PostgreSQL: {e}", self.resource_id)
    
    async def cleanup(self) -> None:
        """Cleanup PostgreSQL connection."""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info(f"PostgreSQL Memory Resource {self.resource_id} cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup of PostgreSQL Memory Resource {self.resource_id}: {e}")
    
    async def health_check(self) -> bool:
        """Check if PostgreSQL connection is healthy."""
        try:
            if not self.engine:
                logger.error("Health check failed: engine is None")
                return False
            
            logger.info("Attempting health check...")
            with self.engine.connect() as conn:
                logger.info("Connected to database, executing SELECT 1")
                result = conn.execute(text("SELECT 1"))
                logger.info(f"Health check query result: {result}")
            logger.info("Health check successful")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for PostgreSQL Memory Resource {self.resource_id}: {e}")
            return False
    
    def _get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise ResourceError("Resource not initialized", self.resource_id)
        return self.SessionLocal()
    
    
    async def store_memory(self, memory_entry: MemoryEntry) -> str:
        """Store a memory entry."""
        try:
            db_session = self._get_session()
            
            # Calculate expiration if not provided
            expires_at = memory_entry.expires_at
            if not expires_at and self.default_ttl_hours:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=self.default_ttl_hours)
            
            # Create database entry
            db_entry = MemoryEntryTable(  # Using the renamed table model
                id=memory_entry.id or str(uuid.uuid4()),
                user_id=memory_entry.user_id,
                session_id=memory_entry.session_id,
                agent_id=memory_entry.agent_id,
                content=json.dumps(memory_entry.content),
                entry_metadata=json.dumps(memory_entry.entry_metadata) if memory_entry.entry_metadata else None,  
                expires_at=expires_at,
                is_active=memory_entry.is_active
            )
            
            db_session.add(db_entry)
            db_session.commit()
            
            await self.record_successful_call()
            logger.debug(f"Stored memory entry {db_entry.id} for user {memory_entry.user_id}")
            
            return db_entry.id
            
        except SQLAlchemyError as e:
            db_session.rollback()
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to store memory: {e}", self.resource_id)
        finally:
            db_session.close()
    
    async def store_session_summary(self, session_summary: MemorySessionSummary) -> str:
        """Store a session summary."""
        try:
            db_session = self._get_session()
            
            db_summary = MemorySessionSummaryTable(
                id=session_summary.id or str(uuid.uuid4()),
                user_id=session_summary.user_id,
                session_id=session_summary.session_id,
                agent_id=session_summary.agent_id,
                summary=session_summary.summary,
            )

            db_session.add(db_summary)
            db_session.commit()
            
            await self.record_successful_call()
            logger.debug(f"Stored session summary {db_summary.id} for user {session_summary.user_id}")

            return db_summary.id
            
        except SQLAlchemyError as e:
            db_session.rollback()
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to store session summary: {e}", self.resource_id)
        finally:
            db_session.close()

    async def get_session_summary(self, user_id: str, session_id: str, agent_id: str) -> MemorySessionSummary:
        """Get a session summary."""
        try:
            db_session = self._get_session()
            
            summary = db_session.query(MemorySessionSummaryTable).filter(
                MemorySessionSummaryTable.user_id == user_id,
                MemorySessionSummaryTable.session_id == session_id,
                MemorySessionSummaryTable.agent_id == agent_id
            ).order_by(MemorySessionSummaryTable.created_at.desc()).first()
            
            if summary:
                return MemorySessionSummary(
                    id=summary.id,
                    user_id=summary.user_id,
                    session_id=summary.session_id,
                    agent_id=summary.agent_id,
                    summary=summary.summary,
                    created_at=summary.created_at,
                    updated_at=summary.updated_at
                )
            else:
                return None
            
        except SQLAlchemyError as e:
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to get session summary: {e}", self.resource_id)
        finally:
            db_session.close()

    async def get_memories(
        self, 
        user_id: str, 
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
        order_by: str = "created_at",
        order_direction: str = "desc"
    ) -> List[MemoryEntry]:
        """Get memory entries for a user."""
        try:
            db_session = self._get_session()
            
            query = db_session.query(MemoryEntryTable).filter(
                MemoryEntryTable.user_id == user_id,
                MemoryEntryTable.is_active == True
            )
            
            if session_id:
                query = query.filter(MemoryEntryTable.session_id == session_id)
            
            if agent_id:
                query = query.filter(MemoryEntryTable.agent_id == agent_id)
            
            # Filter out expired entries
            query = query.filter(
                (MemoryEntryTable.expires_at.is_(None)) | 
                (MemoryEntryTable.expires_at > datetime.now(timezone.utc))
            )
            
            # Simple dynamic sorting
            column = getattr(MemoryEntryTable, order_by)
            if order_direction.lower() == "desc":
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())
            
            query = query.limit(limit)
            entries = query.all()
            
            # Convert to Pydantic models
            memories = []
            for entry in entries:
                try:
                    content = json.loads(entry.content)
                except json.JSONDecodeError:
                    content = entry.content
                
                try:
                    metadata = json.loads(entry.entry_metadata) if entry.entry_metadata else None  # Updated column name
                except json.JSONDecodeError:
                    metadata = None
                
                memory_entry = MemoryEntry(  # Using the imported Pydantic model
                    id=entry.id,
                    user_id=entry.user_id,
                    session_id=entry.session_id,
                    agent_id=entry.agent_id,
                    content=content,
                    entry_metadata=metadata,  # Fixed field name
                    created_at=entry.created_at,
                    updated_at=entry.updated_at,
                    expires_at=entry.expires_at,
                    is_active=entry.is_active
                )
                memories.append(memory_entry)
            
            await self.record_successful_call()
            return memories
            
        except SQLAlchemyError as e:
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to retrieve memories: {e}", self.resource_id)
        finally:
            db_session.close()
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        try:
            db_session = self._get_session()
            
            entry = db_session.query(MemoryEntryTable).filter(MemoryEntryTable.id == memory_id).first()
            if entry:
                entry.is_active = False
                db_session.commit()
                await self.record_successful_call()
                logger.debug(f"Deleted memory entry {memory_id}")
                return True
            else:
                logger.warning(f"Memory entry {memory_id} not found")
                return False
                
        except SQLAlchemyError as e:
            db_session.rollback()
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to delete memory: {e}", self.resource_id)
        finally:
            db_session.close()
    
    async def clear_session(self, user_id: str, session_id: str) -> int:
        """Clear all memories for a specific session."""
        try:
            db_session = self._get_session()
            
            count = db_session.query(MemoryEntryTable).filter(
                MemoryEntryTable.user_id == user_id,
                MemoryEntryTable.session_id == session_id,
                MemoryEntryTable.is_active == True
            ).update({"is_active": False})
            
            db_session.commit()
            await self.record_successful_call()
            logger.info(f"Cleared {count} memory entries for user {user_id}, session {session_id}")
            return count
            
        except SQLAlchemyError as e:
            db_session.rollback()
            await self.record_failed_call(ResourceError(f"Database error: {e}", self.resource_id))
            raise ResourceError(f"Failed to clear session: {e}", self.resource_id)
        finally:
            db_session.close()
    
    async def replace_memories_with_compressed(self, 
                                              user_id: str, 
                                              session_id: str, 
                                              agent_id: str,
                                              messages_to_delete: List[MemoryEntry],
                                              compressed_history: List[Dict[str, str]]):
        """
        Replace existing memories with compressed version.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            compressed_history: Compressed conversation history
        """
        try:
            db_session = self._get_session()
            
            for message in messages_to_delete:
                await self.delete_memory(message.id)
            
            # Store compressed history
            for message in compressed_history:
                memory_entry = MemoryEntry(
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    content={"role": message["role"], "content": message["content"]}

                )
                await self.store_memory(memory_entry)
            
            logger.info(f"Replaced memories with compressed version for {user_id}/{session_id}")
            
        except Exception as e:
            logger.error(f"Error replacing memories with compressed version: {e}")
            raise