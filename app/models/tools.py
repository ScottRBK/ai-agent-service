from pydantic import BaseModel

class Tool(BaseModel):
    """Base class for all tools."""
    name: str
    description: str
    type: str
    parameters: dict
    examples: list[str]

class ToolResponse(BaseModel):
    """Response from a tool."""
    result: str