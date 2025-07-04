"""
This module defines the HealthStatus model for the health check endpoints and provider models.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class HealthStatus(BaseModel):
    """
    Basic health check response model.
    
    This model defines what a simple health check should return.
    """
    status: str = Field(
        ..., 
        description="Service health status",
        json_schema_extra={"example": "healthy"}
    )
    timestamp: datetime = Field(
        ..., 
        description="When the health check was performed",
        json_schema_extra={"example": "2024-12-30T20:12:25.673396"}
    )
    service: str = Field(
        ..., 
        description="Name of the service",
        json_schema_extra={"example": "API Microservice"}
    )
    version: str = Field(
        ..., 
        description="Service version",
        json_schema_extra={"example": "1.0.0"}
    )
    error_details: Optional[str] = Field(
        None,
        description="Error details if the service is not healthy",
        json_schema_extra={"example": "Error details if the service is not healthy"}
    )