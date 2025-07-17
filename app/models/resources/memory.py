"""
Pydantic models for memory management.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class MemoryEntry(BaseModel):
    """Model for a memory entry."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    id: Optional[str] = Field(None, description="Unique identifier for the memory entry")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    agent_id: str = Field(..., description="Agent identifier")
    content: Union[str, Dict[str, Any]] = Field(..., description="Memory content")
    entry_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")  # Renamed from 'metadata'
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(True, description="Whether the entry is active")
    
    @field_validator('user_id', 'session_id', 'agent_id')
    @classmethod
    def validate_ids(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("ID cannot be empty")
        return v.strip()
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("Content cannot be None")
        return v

