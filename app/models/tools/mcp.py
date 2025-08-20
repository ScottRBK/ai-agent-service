from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict

class MCPHeader(BaseModel):
    """MCP header model - supports any headers via extra='allow'"""
    model_config = ConfigDict(extra='allow')
    
    authorization: Optional[str] = Field(None, description="The authorization token for the MCP server")

class MCP(BaseModel):
    """Response MCP model"""
    server_label: str
    server_url: Optional[str] = None  # Make optional for command-based servers
    command: Optional[str] = None     # Add command support
    args: Optional[List[str]] = None  # Add args support
    require_approval: str
    header: Optional[MCPHeader] = None  # Make optional for command-based servers
    env: Optional[Dict[str, str]] = None  # Add environment variables support

