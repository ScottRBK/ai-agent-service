from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from enum import Enum

class ToolType(str, Enum):
    """Enum for tool types."""
    FUNCTION = "function"
    MCP = "mcp"

class ToolParameters(BaseModel):
    """Base class for all tool parameters."""
    type: str = "object"
    properties: Dict[str, Any]
    required: List[str]

class Tool(BaseModel):
    """Base class for all tools."""
    name: str
    description: str
    type: ToolType
    parameters: ToolParameters
    examples: Optional[list[str]] = None
