"""
Integration tests for PostgreSQL Memory Resource.
Tests actual database operations with automatic test database creation/teardown.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
import os
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

from app.core.resources.memory import PostgreSQLMemoryResource
from app.models.resources.memory import MemoryEntry
from app.config.settings import settings

load_dotenv()


class TestMemoryResourceIntegration:
    """Integration tests for memory resource with real database."""
    
    @pytest.fixture(scope="class")
    def test_db_config(self):
        """Generate unique test database configuration from settings."""
        test_db_name = f"test_memory_{uuid.uuid4().hex[:8]}"
        return {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "user": settings.POSTGRES_USER,
            "password": settings.POSTGRES_PASSWORD,
            "database": test_db_name
        }
    
    @pytest.fixture(scope="class")
    def test_db_connection(self, test_db_config):
        """Create test database and provide connection."""
        # Connect to default database to create test database
        conn = psycopg2.connect(
            host=test_db_config["host"],
            port=test_db_config["port"],
            user=test_db_config["user"],
            password=test_db_config["password"],
            database=settings.POSTGRES_DB  # Connect to our configured database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create test database
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {test_db_config['database']}")
        
        conn.close()
        
        # Connect to test database
        test_conn = psycopg2.connect(**test_db_config)
        yield test_conn
        
        # Cleanup: drop test database
        test_conn.close()
        
        # Reconnect to default database to drop test database
        cleanup_conn = psycopg2.connect(
            host=test_db_config["host"],
            port=test_db_config["port"],
            user=test_db_config["user"],
            password=test_db_config["password"],
            database=settings.POSTGRES_DB
        )
        cleanup_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with cleanup_conn.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {test_db_config['database']}")
        
        cleanup_conn.close()
    
    @pytest_asyncio.fixture
    async def memory_resource(self, test_db_connection, test_db_config) -> AsyncGenerator[PostgreSQLMemoryResource, None]:
        """Create memory resource with test database."""
        # Build connection string for test database
        connection_string = (
            f"postgresql://{test_db_config['user']}:{test_db_config['password']}"
            f"@{test_db_config['host']}:{test_db_config['port']}/{test_db_config['database']}"
        )
        
        config = {
            "connection_string": connection_string,
            "default_ttl_hours": 1  # Short TTL for testing
        }
        
        resource = PostgreSQLMemoryResource("test_memory", config)
        
        try:
            await resource.initialize()
            yield resource
        finally:
            await resource.cleanup()
    
    @pytest.fixture
    def sample_memory_entry(self) -> MemoryEntry:
        """Sample memory entry for testing."""
        return MemoryEntry(
            user_id="test_user_123",
            session_id="test_session_456",
            agent_id="test_agent_789",
            content={"role": "user", "message": "Hello, this is a test message"},
            entry_metadata={"test_type": "integration", "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    @pytest.fixture(autouse=True)
    async def cleanup_database(self, memory_resource):
        """Clean up database before each test."""
        # Clear all memory entries
        db_session = memory_resource._get_session()
        try:
            db_session.query(memory_resource.MemoryEntryTable).delete()
            db_session.commit()
        finally:
            db_session.close()
        yield

    @pytest.mark.asyncio
    async def test_store_and_retrieve_memory(self, memory_resource, sample_memory_entry):
        """Test storing and retrieving a memory entry."""
        # Store memory
        memory_id = await memory_resource.store_memory(sample_memory_entry)
        assert memory_id is not None
        assert len(memory_id) > 0
        
        # Retrieve memories for user
        memories = await memory_resource.get_memories("test_user_123")
        assert len(memories) == 1
        
        retrieved_memory = memories[0]
        assert retrieved_memory.id == memory_id
        assert retrieved_memory.user_id == sample_memory_entry.user_id
        assert retrieved_memory.session_id == sample_memory_entry.session_id
        assert retrieved_memory.agent_id == sample_memory_entry.agent_id
        assert retrieved_memory.content == sample_memory_entry.content
        assert retrieved_memory.entry_metadata == sample_memory_entry.entry_metadata
        assert retrieved_memory.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_memories_with_filters(self, memory_resource):
        """Test retrieving memories with different filters."""
        # Create multiple memory entries
        entries = [
            MemoryEntry(
                user_id="user1",
                session_id="session1",
                agent_id="agent1",
                content={"message": "First message"}
            ),
            MemoryEntry(
                user_id="user1",
                session_id="session1",
                agent_id="agent2",
                content={"message": "Second message"}
            ),
            MemoryEntry(
                user_id="user1",
                session_id="session2",
                agent_id="agent1",
                content={"message": "Third message"}
            )
        ]
        
        # Store all entries
        for entry in entries:
            await memory_resource.store_memory(entry)
        
        # Test filtering by user only
        user_memories = await memory_resource.get_memories("user1")
        assert len(user_memories) == 3
        
        # Test filtering by user and session
        session_memories = await memory_resource.get_memories("user1", session_id="session1")
        assert len(session_memories) == 2
        
        # Test filtering by user, session, and agent
        agent_memories = await memory_resource.get_memories("user1", session_id="session1", agent_id="agent1")
        assert len(agent_memories) == 1
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_resource, sample_memory_entry):
        """Test deleting a memory entry."""
        # Use unique user ID for this test
        sample_memory_entry.user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Store memory
        memory_id = await memory_resource.store_memory(sample_memory_entry)
        
        # Verify it exists
        memories = await memory_resource.get_memories(sample_memory_entry.user_id)
        assert len(memories) == 1
        
        # Delete memory
        success = await memory_resource.delete_memory(memory_id)
        assert success is True
        
        # Verify it's no longer active
        memories = await memory_resource.get_memories(sample_memory_entry.user_id)
        assert len(memories) == 0
    
    @pytest.mark.asyncio
    async def test_clear_session(self, memory_resource):
        """Test clearing all memories for a session."""
        # Create multiple entries for same user/session
        entries = [
            MemoryEntry(
                user_id="user2",
                session_id="session3",
                agent_id="agent1",
                content={"message": f"Message {i}"}
            ) for i in range(3)
        ]
        
        # Store all entries
        for entry in entries:
            await memory_resource.store_memory(entry)
        
        # Verify they exist
        memories = await memory_resource.get_memories("user2", session_id="session3")
        assert len(memories) == 3
        
        # Clear session
        cleared_count = await memory_resource.clear_session("user2", "session3")
        assert cleared_count == 3
        
        # Verify they're gone
        memories = await memory_resource.get_memories("user2", session_id="session3")
        assert len(memories) == 0
    
    @pytest.mark.asyncio
    async def test_memory_expiration(self, memory_resource):
        """Test memory expiration functionality."""
        # Create memory with short expiration
        entry = MemoryEntry(
            user_id="user3",
            session_id="session4",
            agent_id="agent1",
            content={"message": "Expiring message"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=1)  # Expire in 1 second
        )
        
        # Store memory
        await memory_resource.store_memory(entry)
        
        # Verify it exists immediately
        memories = await memory_resource.get_memories("user3")
        assert len(memories) == 1
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Verify it's expired and not returned
        memories = await memory_resource.get_memories("user3")
        assert len(memories) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, memory_resource):
        """Test health check functionality."""
        # Health check should pass when initialized
        is_healthy = await memory_resource.health_check()
        assert is_healthy is True

