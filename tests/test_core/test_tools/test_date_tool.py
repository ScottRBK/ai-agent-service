"""
Tests for the Date Tool
"""

from unittest.mock import patch
from datetime import datetime
from app.core.tools.function_calls.date_tool import DateTool

@patch("app.core.tools.function_calls.date_tool.datetime")
def test_get_current_time(mock_datetime):
    mock_datetime.now.return_value = datetime(2021, 1, 1, 10, 0, 0)
    assert DateTool.get_current_datetime("Asia/Tokyo") == "2021-01-01 10:00:00"

def test_get_current_time_invalid_timezone():
    assert DateTool.get_current_datetime("Invalid/Timezone") == "Error: Invalid timezone 'Invalid/Timezone'. Please use a valid timezone name like 'Asia/Tokyo' or 'UTC'."