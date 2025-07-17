"""
Base resource interface for all resource types.
Abstract base class defining the standard interface for all resources.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ResourceType(Enum):
    """Types of resources available."""
    KNOWLEDGE_BASE = "knowledge_base"
    MEMORY = "memory"
    CACHE = "cache"
    CONFIG = "config"

class ResourceError(Exception):
    """Base exception for resource errors."""
    def __init__(self, message: str, resource_name: str, error_code: Optional[str] = None):
        self.message = message
        self.resource_name = resource_name
        self.error_code = error_code
        super().__init__(f"[{resource_name}] {message}")

class ResourceConnectionError(ResourceError):
    """Raised when resource connection fails."""
    pass

class ResourceNotFoundError(ResourceError):
    """Raised when resource is not found."""
    pass

class BaseResource(ABC):
    """Abstract base class for all resources."""
    
    def __init__(self, resource_id: str, config: Dict[str, Any]):
        self.resource_id = resource_id
        self.config = config
        self.resource_type = self._get_resource_type()
        
        # Health tracking
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.last_successful_call: Optional[datetime] = None
        self.last_error: Optional[ResourceError] = None
        self.error_count = 0
        
        self.client = None
        self.initialized = False

    @abstractmethod
    def _get_resource_type(self) -> ResourceType:
        """Get the resource type."""
        pass

    @abstractmethod 
    async def initialize(self) -> None:
        """Initialize the resource."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check the health of the resource."""
        pass

    async def record_successful_call(self) -> None:
        """Record a successful call."""
        self.total_requests += 1
        self.success_requests += 1
        self.last_successful_call = datetime.now()

    async def record_failed_call(self, error: ResourceError) -> None:
        """Record a failed call."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error = error
        self.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "initialized": self.initialized,
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "last_successful_call": self.last_successful_call.isoformat() if self.last_successful_call else None,
            "last_error": str(self.last_error) if self.last_error else None,
            "error_count": self.error_count
        }