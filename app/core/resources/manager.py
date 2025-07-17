"""
Manager class that handles the management of resources and their interactions.
The class is responsible for generating and managing resource instances, as well as handling their lifecycle and interactions.
methods
- list_resources: Returns a list of all registered resource instances.
- get_resource: Retrieves a specific resource instance.
- create_resource: Adds a new resource instance.
- remove_resource: Removes a resource instance.
"""

from typing import Dict, List, Optional, Any
from app.core.resources.base import BaseResource, ResourceType, ResourceError
from app.utils.logging import logger

class ResourceManager:
    """Manages all resources for the application."""
    
    def __init__(self):
        self.resources: Dict[str, BaseResource] = {}
        self.agent_resources: Dict[str, List[str]] = {}  # agent_id -> resource_ids
    
    def list_resources(self) -> Dict[str, BaseResource]:
        """Returns a list of all registered resource instances."""
        return self.resources.copy()
    
    def get_resource(self, resource_id: str) -> Optional[BaseResource]:
        """Retrieves a specific resource instance."""
        return self.resources.get(resource_id)
    
    async def create_resource(self, resource: BaseResource) -> None:
        """Adds a new resource instance."""
        try:
            await resource.initialize()
            self.resources[resource.resource_id] = resource
            logger.info(f"Resource {resource.resource_id} created successfully")
        except Exception as e:
            logger.error(f"Failed to create resource {resource.resource_id}: {e}")
            raise ResourceError(f"Failed to create resource: {e}", resource.resource_id)
    
    async def remove_resource(self, resource_id: str) -> None:
        """Removes a resource instance."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            await resource.cleanup()
            del self.resources[resource_id]
            logger.info(f"Resource {resource_id} removed successfully")
        else:
            logger.warning(f"Resource {resource_id} not found for removal")
    
    async def get_agent_resources(self, agent_id: str) -> List[BaseResource]:
        """Get all resources available to an agent."""
        resource_ids = self.agent_resources.get(agent_id, [])
        agent_resources = []
        
        for resource_id in resource_ids:
            resource = self.resources.get(resource_id)
            if resource:
                agent_resources.append(resource)
            else:
                logger.warning(f"Resource {resource_id} not found for agent {agent_id}")
        
        return agent_resources
    
    async def assign_resource_to_agent(self, agent_id: str, resource_id: str) -> None:
        """Assign a resource to an agent."""
        if resource_id not in self.resources:
            raise ResourceError(f"Resource {resource_id} not found", resource_id)
        
        if agent_id not in self.agent_resources:
            self.agent_resources[agent_id] = []
        
        if resource_id not in self.agent_resources[agent_id]:
            self.agent_resources[agent_id].append(resource_id)
            logger.info(f"Resource {resource_id} assigned to agent {agent_id}")
    
    async def remove_resource_from_agent(self, agent_id: str, resource_id: str) -> None:
        """Remove a resource from an agent."""
        if agent_id in self.agent_resources:
            if resource_id in self.agent_resources[agent_id]:
                self.agent_resources[agent_id].remove(resource_id)
                logger.info(f"Resource {resource_id} removed from agent {agent_id}")
    
    async def get_resources_by_type(self, resource_type: ResourceType) -> List[BaseResource]:
        """Get all resources of a specific type."""
        return [
            resource for resource in self.resources.values()
            if resource.resource_type == resource_type
        ]
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all resources."""
        health_status = {}
        
        for resource_id, resource in self.resources.items():
            try:
                health_status[resource_id] = await resource.health_check()
            except Exception as e:
                logger.error(f"Health check failed for resource {resource_id}: {e}")
                health_status[resource_id] = False
        
        return health_status
    
    async def cleanup_all(self) -> None:
        """Cleanup all resources."""
        for resource in self.resources.values():
            try:
                await resource.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed for resource {resource.resource_id}: {e}")
        
        self.resources.clear()
        self.agent_resources.clear()
        logger.info("All resources cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all resources."""
        stats = {
            "total_resources": len(self.resources),
            "total_agents": len(self.agent_resources),
            "resources": {}
        }
        
        for resource_id, resource in self.resources.items():
            stats["resources"][resource_id] = resource.get_stats()
        
        return stats