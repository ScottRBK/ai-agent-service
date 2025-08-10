# tests/test_core/test_resources/test_memory.py
"""
Unit tests for PostgreSQL Memory Resource.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.resources.memory import PostgreSQLMemoryResource, MemoryEntryTable
from app.models.resources.memory import MemoryEntry
from app.core.resources.base import ResourceError, ResourceConnectionError


class TestPostgreSQLMemoryResource:
    """Test cases for PostgreSQL Memory Resource."""
    
    @pytest.fixture
    def memory_config(self):
        """Test configuration for memory resource."""
        return {
            "connection_string": "postgresql://test:test@localhost:5432/test_db",
            "default_ttl_hours": 24
        }
    
    @pytest.fixture
    def memory_resource(self, memory_config):
        """Create memory resource instance."""
        return PostgreSQLMemoryResource("test_memory", memory_config)
    
    @pytest.fixture
    def sample_memory_entry(self):
        """Sample memory entry for testing."""
        return MemoryEntry(
            user_id="user123",
            session_id="session456",
            agent_id="test_agent",
            content={"role": "user", "message": "Hello world"},
            entry_metadata={"type": "conversation"}
        )
    
    def test_init_with_valid_config(self, memory_config):
        """Test initialization with valid configuration."""
        resource = PostgreSQLMemoryResource("test_memory", memory_config)
        assert resource.resource_id == "test_memory"
        assert resource.connection_string == memory_config["connection_string"]
        assert resource.default_ttl_hours == 24
        assert resource.resource_type.value == "memory"
    
    def test_init_without_connection_string(self):
        """Test initialization fails without connection string."""
        with pytest.raises(ResourceError, match="PostgreSQL connection string is required"):
            PostgreSQLMemoryResource("test_memory", {})
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, memory_resource):
        """Test successful initialization."""
        with patch('app.core.resources.memory.create_engine') as mock_create_engine:  # Patch at module level
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            
            with patch('app.core.resources.memory.sessionmaker') as mock_sessionmaker:  # Patch at module level
                mock_session = MagicMock()
                mock_sessionmaker.return_value = mock_session
                
                await memory_resource.initialize()
                
                assert memory_resource.initialized is True
                # Expect the connection string to be modified to use psycopg dialect
                expected_connection_string = memory_resource.connection_string.replace('postgresql://', 'postgresql+psycopg://', 1)
                mock_create_engine.assert_called_once_with(expected_connection_string)
    
    @pytest.mark.asyncio
    async def test_initialize_failure(self, memory_resource):
        """Test initialization failure."""
        with patch('sqlalchemy.create_engine', side_effect=Exception("Connection failed")):
            with pytest.raises(ResourceConnectionError, match="Failed to connect to PostgreSQL"):
                await memory_resource.initialize()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, memory_resource):
        """Test successful health check."""
        memory_resource.engine = MagicMock()
        mock_conn = MagicMock()
        memory_resource.engine.connect.return_value.__enter__.return_value = mock_conn
        
        result = await memory_resource.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, memory_resource):
        """Test health check failure."""
        memory_resource.engine = None
        result = await memory_resource.health_check()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_store_memory_success(self, memory_resource, sample_memory_entry):
        """Test successful memory storage."""
        # Mock database session
        mock_session = MagicMock()
        mock_db_entry = MagicMock()
        mock_db_entry.id = "test_id"
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            # Mock the MemoryEntryTable constructor
            with patch('app.core.resources.memory.MemoryEntryTable') as mock_table_class:
                mock_table_class.return_value = mock_db_entry
                
                # Mock the record_successful_call method
                with patch.object(memory_resource, 'record_successful_call') as mock_record:
                    # Mock the session context
                    mock_session.add = MagicMock()
                    mock_session.commit = MagicMock()
                    mock_session.rollback = MagicMock()
                    mock_session.close = MagicMock()
                    
                    result = await memory_resource.store_memory(sample_memory_entry)
                    
                    assert result == "test_id"
                    mock_session.add.assert_called_once_with(mock_db_entry)
                    mock_session.commit.assert_called_once()
                    mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_memory_database_error(self, memory_resource, sample_memory_entry):
        """Test memory storage with database error."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session = MagicMock()
        mock_session.add.side_effect = SQLAlchemyError("Database error")
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            with pytest.raises(ResourceError, match="Failed to store memory"):
                await memory_resource.store_memory(sample_memory_entry)
            
            mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_memories_success(self, memory_resource):
        """Test successful memory retrieval."""
        # Mock database entries
        mock_entry = MagicMock()
        mock_entry.id = "test_id"
        mock_entry.user_id = "user123"
        mock_entry.session_id = "session456"
        mock_entry.agent_id = "test_agent"
        mock_entry.content = '{"role": "user", "message": "Hello"}'
        mock_entry.entry_metadata = '{"type": "conversation"}'
        mock_entry.created_at = datetime.now(timezone.utc)
        mock_entry.updated_at = datetime.now(timezone.utc)
        mock_entry.expires_at = None
        mock_entry.is_active = True
        
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [mock_entry]
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            result = await memory_resource.get_memories("user123")
            
            assert len(result) == 1
            assert result[0].user_id == "user123"
            assert result[0].session_id == "session456"
            assert result[0].agent_id == "test_agent"
    
    @pytest.mark.asyncio
    async def test_get_memories_with_filters(self, memory_resource):
        """Test memory retrieval with filters."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            result = await memory_resource.get_memories(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                limit=50
            )
            
            assert result == []
            # Verify filters were applied
            assert mock_query.filter.call_count >= 3  # user_id, session_id, agent_id
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, memory_resource):
        """Test successful memory deletion."""
        mock_entry = MagicMock()
        mock_entry.is_active = True
        
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_entry
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            result = await memory_resource.delete_memory("test_id")
            
            assert result is True
            assert mock_entry.is_active is False
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, memory_resource):
        """Test memory deletion when entry not found."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            result = await memory_resource.delete_memory("nonexistent_id")
            
            assert result is False
            mock_session.commit.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_clear_session_success(self, memory_resource):
        """Test successful session clearing."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 5  # 5 entries cleared
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            result = await memory_resource.clear_session("user123", "session456")
            
            assert result == 5
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_all_sessions_for_user_success(self, memory_resource):
        """Test successful clearing of all sessions for a user."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 20  # 20 entries cleared across multiple sessions
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            with patch.object(memory_resource, 'record_successful_call') as mock_record:
                result = await memory_resource.clear_all_sessions_for_user("user123")
                
                assert result == 20
                mock_session.commit.assert_called_once()
                mock_record.assert_called_once()
                # Verify the update was called with is_active = False
                mock_query.update.assert_called_once_with({"is_active": False})
    
    @pytest.mark.asyncio
    async def test_clear_all_sessions_for_user_no_entries(self, memory_resource):
        """Test clearing all sessions when no entries exist."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 0  # No entries to clear
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            with patch.object(memory_resource, 'record_successful_call') as mock_record:
                result = await memory_resource.clear_all_sessions_for_user("nonexistent_user")
                
                assert result == 0
                mock_session.commit.assert_called_once()
                mock_record.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_all_sessions_for_user_database_error(self, memory_resource):
        """Test clearing all sessions with database error."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.update.side_effect = SQLAlchemyError("Database error")
        
        mock_session.query.return_value = mock_query
        
        with patch.object(memory_resource, '_get_session') as mock_get_session:
            mock_get_session.return_value = mock_session
            
            with patch.object(memory_resource, 'record_failed_call') as mock_record_failed:
                with pytest.raises(ResourceError, match="Failed to clear user sessions"):
                    await memory_resource.clear_all_sessions_for_user("user123")
                
                mock_session.rollback.assert_called_once()
                mock_session.close.assert_called_once()
                mock_record_failed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, memory_resource):
        """Test resource cleanup."""
        memory_resource.engine = MagicMock()
        
        await memory_resource.cleanup()
        
        memory_resource.engine.dispose.assert_called_once()


class TestMemoryEntryTable:
    """Test cases for database table model."""
    
    def test_table_creation(self):
        """Test that table can be created."""
        # This tests that the SQLAlchemy model is properly defined
        assert MemoryEntryTable.__tablename__ == 'memory_entries'
        assert hasattr(MemoryEntryTable, 'id')
        assert hasattr(MemoryEntryTable, 'user_id')
        assert hasattr(MemoryEntryTable, 'session_id')
        assert hasattr(MemoryEntryTable, 'agent_id')
        assert hasattr(MemoryEntryTable, 'content')
        assert hasattr(MemoryEntryTable, 'entry_metadata')
        assert hasattr(MemoryEntryTable, 'created_at')
        assert hasattr(MemoryEntryTable, 'updated_at')
        assert hasattr(MemoryEntryTable, 'expires_at')
        assert hasattr(MemoryEntryTable, 'is_active')


class TestMemoryEntryIntegration:
    """Integration tests for memory entry model."""
    
    def test_memory_entry_validation(self):
        """Test Pydantic model validation."""
        # Valid entry
        entry = MemoryEntry(
            user_id="user123",
            session_id="session456",
            agent_id="test_agent",
            content={"message": "Hello"}
        )
        assert entry.user_id == "user123"
        assert entry.is_active is True
        
        # Invalid entry (empty user_id)
        with pytest.raises(ValueError, match="ID cannot be empty"):
            MemoryEntry(
                user_id="",
                session_id="session456",
                agent_id="test_agent",
                content={"message": "Hello"}
            )
        
        # Invalid entry (None content) - Pydantic catches this before our validator
        with pytest.raises(ValueError, match="Input should be a valid string"):
            MemoryEntry(
                user_id="user123",
                session_id="session456",
                agent_id="test_agent",
                content=None
            )
        
        # Test that validators are called
        # Empty string for user_id should be caught by validator
        with pytest.raises(ValueError, match="ID cannot be empty"):
            MemoryEntry(
                user_id="   ",  # Whitespace only
                session_id="session456",
                agent_id="test_agent",
                content={"message": "Hello"}
            )