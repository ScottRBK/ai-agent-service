from app.models.tools.tools import Tool, ToolType
from pydantic import BaseModel, Field


class CurrentDateParameters(BaseModel):
    timezone: str = Field(description="The timezone to get the date and time for")

class CurrentDateTool(Tool):
    type: ToolType = ToolType.FUNCTION
    name: str = "date"
    description: str = "Get the current date and time for a given timezone (e.g., 'Asia/Tokyo', 'UTC', 'America/New_York')"
    parameters: CurrentDateParameters
    examples: list[str] = ["What is the date and time in Tokyo?"]