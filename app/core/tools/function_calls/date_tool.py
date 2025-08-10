"""
Date Tool for working with dates and times
"""
from datetime import datetime
import zoneinfo
from app.core.tools.tool_registry import register_tool
from app.models.tools.date import CurrentDateTool as CurrentDateToolModel
from pydantic import BaseModel, Field
from app.models.tools.tools import Tool, ToolType


class CurrentDateParameters(BaseModel):
    timezone: str = Field(description="The timezone to get the date and time for (e.g., 'Asia/Tokyo', 'UTC', 'America/New_York')")


class DateTool:
    @register_tool(
        name="get_current_datetime",
        description="Get the current date and time in the given timezone",
        tool_type=ToolType.FUNCTION,
        examples=["What is the date and time in Tokyo?"],
        params_model=CurrentDateParameters
    )
    def get_current_datetime(timezone: str) -> str:
        """Get the current date and time in the given timezone
        Args:
            timezone: str The timezone name (e.g., 'Asia/Tokyo', 'UTC', 'America/New_York')
        Returns:
            The current date and time in the given timezone in the format YYYY-MM-DD HH:MM:SS
        """

        # Convert string to timezone object
        tz = zoneinfo.ZoneInfo(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    


