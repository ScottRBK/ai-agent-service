"""
This module defines the HealthStatus model for the health check endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime

class HealthStatus(BaseModel):
    """
    Basic health check response model.
    
    This model defines what a simple health check should return.
    """
    status: str = Field(
        ..., 
        description="Service health status",
        example="healthy"
    )
    timestamp: datetime = Field(
        ..., 
        description="When the health check was performed",
        example="2024-12-30T20:12:25.673396"
    )
    service: str = Field(
        ..., 
        description="Name of the service",
        example="API Microservice"
    )
    version: str = Field(
        ..., 
        description="Service version",
        example="1.0.0"
    )