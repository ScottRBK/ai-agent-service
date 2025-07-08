from pydantic import BaseModel, Field

class MCPHeader(BaseModel):
    """Response MCP header model"""
    Authorization: str = Field(description="The authorization token for the MCP server")

class MCP(BaseModel):
    """Response MCP model"""
    server_label: str
    server_url: str
    require_approval: str
    header: MCPHeader

