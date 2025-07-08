from pydantic import BaseModel, Field

class ResponseMCPHeader(BaseModel):
    """Response MCP header model"""
    Authorization: str = Field(description="The authorization token for the MCP server")

class ResponseMCP(BaseModel):
    """Response MCP model"""
    type: str = "mcp"
    server_label: str
    server_url: str
    require_approval: str
    header: ResponseMCPHeader

