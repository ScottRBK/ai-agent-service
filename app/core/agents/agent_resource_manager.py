"""
Agent Resource Manager is responsible for managing which resources are available to specific agents.
It provides agent-specific resource filtering to ensure agents only have access to appropriate resources.
"""

import json
import os
from typing import List, Dict, Any, Optional
from app.core.resources.manager import ResourceManager
from app.core.resources.base import BaseResource, ResourceType
from app.utils.logging import logger


class AgentResourceManager:
    """
    Manages resource availability for specific agents.
    Provides filtering capabilities to ensure agents only have access to appropriate resources.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.resource_manager = ResourceManager()
        self.config = self.load_agent_config()
    
    def load_agent_config(self) -> Dict[str, Any]:
        """
        Load agent configuration from agent_config.json or return default config.
        """
        try:
            from app.config.settings import settings
            config_path = settings.AGENT_CONFIG_PATH
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    all_configs = json.load(f)
                
                # Find config for this agent
                if isinstance(all_configs, dict):
                    # Single agent config
                    if all_configs.get("agent_id") == self.agent_id:
                        return all_configs
                elif isinstance(all_configs, list):
                    # Multiple agent configs
                    for config in all_configs:
                        if config.get("agent_id") == self.agent_id:
                            return config
                
                logger.warning(f"No config found for agent {self.agent_id}, using default")
            
        except Exception as e:
            logger.error(f"Error loading agent config for {self.agent_id}: {e}")
        
        # Return default config (no resources)
        return {
            "agent_id": self.agent_id,
            "resources": []
        }
    
    async def get_agent_resources(self) -> List[BaseResource]:
        """
        Get all resources available to this specific agent.
        
        Returns:
            List of resources available to this agent
        """
        return await self.resource_manager.get_agent_resources(self.agent_id)
    
    async def get_memory_resource(self) -> Optional[BaseResource]:
        """
        Get memory resource for this agent.
        
        Returns:
            Memory resource if available, None otherwise
        """
        # First check if agent is configured for memory
        if not self.has_resource("memory"):
            return None
        
        # Check if memory resource already exists
        resources = await self.get_agent_resources()
        for resource in resources:
            if resource.resource_type == ResourceType.MEMORY:
                return resource
        
        # If agent is configured for memory but resource doesn't exist, create it
        logger.info(f"Creating memory resource for agent {self.agent_id}")
        await self.create_memory_resource()
        
        # Now get the newly created resource
        resources = await self.get_agent_resources()
        for resource in resources:
            if resource.resource_type == ResourceType.MEMORY:
                return resource
        
        return None
    
    async def create_memory_resource(self):
        """Create memory resource if not exists."""
        try:
            from app.core.resources.memory import PostgreSQLMemoryResource
            from app.config.settings import settings
            
            # Build connection string
            connection_string = (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
            
            config = {
                "connection_string": connection_string,
                "default_ttl_hours": 24 * 7
            }
            
            # Create memory resource
            memory_resource = PostgreSQLMemoryResource("global_memory", config)
            
            # Register with resource manager
            await self.resource_manager.create_resource(memory_resource)
            
            # Assign to this agent
            await self.resource_manager.assign_resource_to_agent(self.agent_id, "global_memory")
            
            logger.info(f"Memory resource created and assigned to agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to create memory resource for agent {self.agent_id}: {e}")
            raise
    
    async def get_resources_by_type(self, resource_type: ResourceType) -> List[BaseResource]:
        """
        Get all resources of a specific type for this agent.
        
        Args:
            resource_type: Type of resources to retrieve
            
        Returns:
            List of resources of the specified type
        """
        resources = await self.get_agent_resources()
        return [r for r in resources if r.resource_type == resource_type]
    
    def has_resource(self, resource_name: str) -> bool:
        """
        Check if agent has access to a specific resource.
        
        Args:
            resource_name: Name of the resource to check
            
        Returns:
            True if agent has access to the resource
        """
        allowed_resources = self.config.get("resources", [])
        # Handle None case
        if allowed_resources is None:
            return False
        return resource_name in allowed_resources
    
    def get_model_config(self) -> tuple[str, Optional[dict]]:
        """Get the model and settings for this agent."""
        model = self.config.get("model")
        model_settings = self.config.get("model_settings")
        
        return model, model_settings