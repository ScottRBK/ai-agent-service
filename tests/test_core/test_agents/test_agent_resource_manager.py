"""
Unit tests for AgentResourceManager.
Tests agent-specific resource filtering and memory resource management.
"""

import pytest
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from app.core.agents.agent_resource_manager import AgentResourceManager
from app.core.resources.base import BaseResource, ResourceType, ResourceError
from app.core.resources.memory import PostgreSQLMemoryResource


class MockResource(BaseResource):
    """Mock resource for testing."""
    
    def _get_resource_type(self) -> ResourceType:
        return ResourceType.MEMORY
    
    async def initialize(self) -> None:
        self.initialized = True
    
    async def cleanup(self) -> None:
        pass
    
    async def health_check(self) -> bool:
        return True
    
    def get_stats(self) -> dict:
        return {"type": "mock", "initialized": self.initialized}


class TestAgentResourceManager:
    """Test cases for AgentResourceManager."""
    
    def test_init_with_valid_agent_id(self):
        """Test initialization with a valid agent ID."""
        agent_manager = AgentResourceManager("cli_agent")
        assert agent_manager.agent_id == "cli_agent"
        assert agent_manager.config is not None
        assert agent_manager.resource_manager is not None
    
    def test_init_with_invalid_agent_id(self):
        """Test initialization with an invalid agent ID (should use default config)."""
        agent_manager = AgentResourceManager("nonexistent_agent")
        assert agent_manager.agent_id == "nonexistent_agent"
        # Should return default config, not None
        assert agent_manager.config is not None
        assert agent_manager.config["agent_id"] == "nonexistent_agent"
        assert agent_manager.config["resources"] == []
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_agent_config_single_agent(self, mock_exists, mock_file):
        """Test loading config for a single agent configuration."""
        mock_exists.return_value = True
        
        # Mock single agent config
        mock_config = {
            "agent_id": "cli_agent",
            "resources": ["memory"],
            "allowed_regular_tools": ["get_current_datetime"]
        }
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
        
        agent_manager = AgentResourceManager("cli_agent")
        
        assert agent_manager.config["agent_id"] == "cli_agent"
        assert agent_manager.config["resources"] == ["memory"]
        assert agent_manager.config["allowed_regular_tools"] == ["get_current_datetime"]
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_agent_config_multiple_agents(self, mock_exists, mock_file):
        """Test loading config for an agent from multiple agent configuration."""
        mock_exists.return_value = True
        
        # Mock multiple agent configs
        mock_configs = [
            {
                "agent_id": "research_agent",
                "resources": ["knowledge_base"],
                "allowed_regular_tools": ["get_current_datetime"]
            },
            {
                "agent_id": "cli_agent",
                "resources": ["memory"],
                "allowed_regular_tools": ["get_current_datetime"]
            }
        ]
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_configs)
        
        agent_manager = AgentResourceManager("cli_agent")
        
        assert agent_manager.config["agent_id"] == "cli_agent"
        assert agent_manager.config["resources"] == ["memory"]
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_agent_config_agent_not_found(self, mock_exists, mock_file):
        """Test loading config when agent is not found in configuration."""
        mock_exists.return_value = True
        
        # Mock config without the requested agent
        mock_configs = [
            {
                "agent_id": "research_agent",
                "resources": ["knowledge_base"]
            }
        ]
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_configs)
        
        agent_manager = AgentResourceManager("cli_agent")
        
        # Should return default config
        assert agent_manager.config["agent_id"] == "cli_agent"
        assert agent_manager.config["resources"] == []
    
    @patch('os.path.exists')
    def test_load_agent_config_file_not_found(self, mock_exists):
        """Test loading config when agent_config.json doesn't exist."""
        mock_exists.return_value = False
        
        agent_manager = AgentResourceManager("cli_agent")
        
        # Should return default config
        assert agent_manager.config["agent_id"] == "cli_agent"
        assert agent_manager.config["resources"] == []
    
    @patch('builtins.open', side_effect=Exception("File read error"))
    @patch('os.path.exists')
    def test_load_agent_config_file_error(self, mock_exists, mock_file):
        """Test loading config when file read fails."""
        mock_exists.return_value = True
        
        agent_manager = AgentResourceManager("cli_agent")
        
        # Should return default config on error
        assert agent_manager.config["agent_id"] == "cli_agent"
        assert agent_manager.config["resources"] == []
    
    @pytest.mark.asyncio
    async def test_get_agent_resources(self):
        """Test getting resources for an agent."""
        agent_manager = AgentResourceManager("cli_agent")
        
        # Mock the resource manager's get_agent_resources method
        mock_resource = MockResource("test_memory", {"test": "config"})
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[mock_resource])
        
        resources = await agent_manager.get_agent_resources()
        
        assert len(resources) == 1
        assert resources[0].resource_id == "test_memory"
        agent_manager.resource_manager.get_agent_resources.assert_called_once_with("cli_agent")
    
    def test_has_resource_with_memory(self):
        """Test has_resource when agent has memory configured."""
        agent_manager = AgentResourceManager("cli_agent")
        agent_manager.config = {
            "agent_id": "cli_agent",
            "resources": ["memory", "knowledge_base"]
        }
        
        assert agent_manager.has_resource("memory") is True
        assert agent_manager.has_resource("knowledge_base") is True
        assert agent_manager.has_resource("cache") is False
    
    def test_has_resource_without_memory(self):
        """Test has_resource when agent doesn't have memory configured."""
        agent_manager = AgentResourceManager("research_agent")
        agent_manager.config = {
            "agent_id": "research_agent",
            "resources": ["knowledge_base"]
        }
        
        assert agent_manager.has_resource("memory") is False
        assert agent_manager.has_resource("knowledge_base") is True
    
    def test_has_resource_empty_resources(self):
        """Test has_resource when agent has no resources configured."""
        agent_manager = AgentResourceManager("restricted_agent")
        agent_manager.config = {
            "agent_id": "restricted_agent",
            "resources": []
        }
        
        assert agent_manager.has_resource("memory") is False
        assert agent_manager.has_resource("knowledge_base") is False
    
    @pytest.mark.asyncio
    async def test_get_memory_resource_existing(self):
        """Test getting memory resource when it already exists."""
        agent_manager = AgentResourceManager("cli_agent")
        agent_manager.config = {
            "agent_id": "cli_agent",
            "resources": ["memory"]
        }
        
        # Mock existing memory resource
        mock_memory_resource = MockResource("global_memory", {"test": "config"})
        mock_memory_resource.resource_type = ResourceType.MEMORY
        
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[mock_memory_resource])
        
        memory_resource = await agent_manager.get_memory_resource()
        
        assert memory_resource is not None
        assert memory_resource.resource_id == "global_memory"
        assert memory_resource.resource_type == ResourceType.MEMORY
    
    @pytest.mark.asyncio
    async def test_get_memory_resource_not_configured(self):
        """Test getting memory resource when agent doesn't have memory configured."""
        agent_manager = AgentResourceManager("research_agent")
        agent_manager.config = {
            "agent_id": "research_agent",
            "resources": ["knowledge_base"]
        }
        
        memory_resource = await agent_manager.get_memory_resource()
        
        assert memory_resource is None
    
    @pytest.mark.asyncio
    async def test_get_memory_resource_create_new(self):
        """Test getting memory resource when it needs to be created."""
        agent_manager = AgentResourceManager("cli_agent")
        agent_manager.config = {
            "agent_id": "cli_agent",
            "resources": ["memory"]
        }
        
        # Mock no existing resources
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[])
        
        # Mock the create_memory_resource method
        agent_manager.create_memory_resource = AsyncMock()
        
        # Mock the resource manager's create_resource and assign_resource_to_agent methods
        agent_manager.resource_manager.create_resource = AsyncMock()
        agent_manager.resource_manager.assign_resource_to_agent = AsyncMock()
        
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.POSTGRES_USER = "test_user"
        mock_settings.POSTGRES_PASSWORD = "test_password"
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "test_db"
        
        with patch('app.config.settings.settings', mock_settings):
            with patch('app.core.resources.memory.PostgreSQLMemoryResource') as mock_memory_class:
                mock_memory_resource = MockResource("global_memory", {"test": "config"})
                mock_memory_resource.resource_type = ResourceType.MEMORY
                mock_memory_class.return_value = mock_memory_resource
                
                # After creation, return the new resource
                agent_manager.resource_manager.get_agent_resources = AsyncMock(
                    side_effect=[[], [mock_memory_resource]]
                )
                
                memory_resource = await agent_manager.get_memory_resource()
                
                assert memory_resource is not None
                assert memory_resource.resource_id == "global_memory"
                agent_manager.create_memory_resource.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_memory_resource_success(self):
        """Test successful memory resource creation."""
        agent_manager = AgentResourceManager("cli_agent")
        
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.POSTGRES_USER = "test_user"
        mock_settings.POSTGRES_PASSWORD = "test_password"
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "test_db"
        
        # Mock the resource manager methods
        agent_manager.resource_manager.create_resource = AsyncMock()
        agent_manager.resource_manager.assign_resource_to_agent = AsyncMock()
        
        with patch('app.config.settings.settings', mock_settings):
            with patch('app.core.resources.memory.PostgreSQLMemoryResource') as mock_memory_class:
                mock_memory_resource = MockResource("global_memory", {"test": "config"})
                mock_memory_class.return_value = mock_memory_resource
                
                await agent_manager.create_memory_resource()
                
                # Verify PostgreSQLMemoryResource was created with correct config
                mock_memory_class.assert_called_once_with("global_memory", {
                    "connection_string": "postgresql://test_user:test_password@localhost:5432/test_db",
                    "default_ttl_hours": 24 * 7
                })
                
                # Verify resource was created and assigned
                agent_manager.resource_manager.create_resource.assert_called_once_with(mock_memory_resource)
                agent_manager.resource_manager.assign_resource_to_agent.assert_called_once_with("cli_agent", "global_memory")
    
    @pytest.mark.asyncio
    async def test_create_memory_resource_failure(self):
        """Test memory resource creation failure."""
        agent_manager = AgentResourceManager("cli_agent")
        
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.POSTGRES_USER = "test_user"
        mock_settings.POSTGRES_PASSWORD = "test_password"
        mock_settings.POSTGRES_HOST = "localhost"
        mock_settings.POSTGRES_PORT = "5432"
        mock_settings.POSTGRES_DB = "test_db"
        
        # Mock the resource manager to raise an exception
        agent_manager.resource_manager.create_resource = AsyncMock(side_effect=ResourceError("Test error", "global_memory"))
        
        with patch('app.config.settings.settings', mock_settings):
            with patch('app.core.resources.memory.PostgreSQLMemoryResource') as mock_memory_class:
                mock_memory_resource = MockResource("global_memory", {"test": "config"})
                mock_memory_class.return_value = mock_memory_resource
                
                with pytest.raises(ResourceError, match="Test error"):
                    await agent_manager.create_memory_resource()
    
    @pytest.mark.asyncio
    async def test_get_resources_by_type(self):
        """Test getting resources by type."""
        agent_manager = AgentResourceManager("cli_agent")
        
        # Mock resources of different types
        mock_memory_resource = MockResource("global_memory", {"test": "config"})
        mock_memory_resource.resource_type = ResourceType.MEMORY
        
        mock_cache_resource = MockResource("global_cache", {"test": "config"})
        mock_cache_resource.resource_type = ResourceType.CACHE
        
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[mock_memory_resource, mock_cache_resource])
        
        memory_resources = await agent_manager.get_resources_by_type(ResourceType.MEMORY)
        cache_resources = await agent_manager.get_resources_by_type(ResourceType.CACHE)
        
        assert len(memory_resources) == 1
        assert memory_resources[0].resource_type == ResourceType.MEMORY
        
        assert len(cache_resources) == 1
        assert cache_resources[0].resource_type == ResourceType.CACHE
    
    @pytest.mark.asyncio
    async def test_get_resources_by_type_empty(self):
        """Test getting resources by type when no resources exist."""
        agent_manager = AgentResourceManager("cli_agent")
        
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[])
        
        resources = await agent_manager.get_resources_by_type(ResourceType.MEMORY)
        
        assert len(resources) == 0
    
    def test_has_resource_with_none_resources(self):
        """Test has_resource when resources config is None."""
        agent_manager = AgentResourceManager("test_agent")
        agent_manager.config = {
            "agent_id": "test_agent",
            "resources": None
        }
        
        assert agent_manager.has_resource("memory") is False
        assert agent_manager.has_resource("knowledge_base") is False
    
    @pytest.mark.asyncio
    async def test_get_memory_resource_after_creation_failure(self):
        """Test getting memory resource when creation fails."""
        agent_manager = AgentResourceManager("cli_agent")
        agent_manager.config = {
            "agent_id": "cli_agent",
            "resources": ["memory"]
        }
        
        # Mock no existing resources
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[])
        
        # Mock create_memory_resource to raise an exception
        agent_manager.create_memory_resource = AsyncMock(side_effect=ResourceError("Creation failed", "global_memory"))
        
        with pytest.raises(ResourceError, match="Creation failed"):
            await agent_manager.get_memory_resource()
    
    @pytest.mark.asyncio
    async def test_get_memory_resource_mixed_resource_types(self):
        """Test getting memory resource when agent has multiple resource types."""
        agent_manager = AgentResourceManager("cli_agent")
        agent_manager.config = {
            "agent_id": "cli_agent",
            "resources": ["memory", "knowledge_base"]
        }
        
        # Mock resources of different types
        mock_memory_resource = MockResource("global_memory", {"test": "config"})
        mock_memory_resource.resource_type = ResourceType.MEMORY
        
        mock_kb_resource = MockResource("global_kb", {"test": "config"})
        mock_kb_resource.resource_type = ResourceType.KNOWLEDGE_BASE
        
        agent_manager.resource_manager.get_agent_resources = AsyncMock(return_value=[mock_memory_resource, mock_kb_resource])
        
        memory_resource = await agent_manager.get_memory_resource()
        
        assert memory_resource is not None
        assert memory_resource.resource_type == ResourceType.MEMORY
        assert memory_resource.resource_id == "global_memory" 