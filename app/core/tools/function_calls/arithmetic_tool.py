"""
Arithmetic Tool for performing arithmetic operations
"""
from typing import Union
from app.core.tools.tool_registry import register_tool
from app.models.tools.tools import Tool, ToolType
from pydantic import BaseModel, Field


class ArithmeticToolParameters(BaseModel):
    a: Union[int, float] = Field(description="The first number")
    b: Union[int, float] = Field(description="The second number")


class ArithmeticTool:
  @register_tool(
        name="add_two_numbers",
        description="Add two numbers",
        tool_type=ToolType.FUNCTION,
        examples=["What is 1 + 1?"],
        params_model=ArithmeticToolParameters
    )
  def add_two_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Add two numbers

    Args:
      a (float): The first number
      b (float): The second number

    Returns:
      Union[int, float]: The sum of the two numbers (a plus b)
    """

    return float(a) + float(b)
  