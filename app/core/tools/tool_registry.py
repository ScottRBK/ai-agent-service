"""
Tool Registry is a registry of tools that can be used by the agent.
It is used to convert the tool registry to the different formats that are supported by the different providers.
It is also used to execute the tool calls.
"""

import json
from typing import Callable, Dict, Any, Type
from pydantic import BaseModel
from app.models.tools.tools import Tool, ToolParameters
from pydantic import ValidationError
from app.utils.logging import logger

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_tool(
    *,
    name: str,
    description: str,
    tool_type: str,
    examples: list[str],
    params_model: Type[BaseModel]
):
    """Decorator to register a tool dynamically from a params_model."""
    def decorator(func: Callable[[BaseModel], Any]):
        TOOL_REGISTRY[name] = {
            "schema": Tool(
                name=name,
                description=description,
                type=tool_type,
                parameters=ToolParameters(
                    properties=params_model.model_json_schema()["properties"],
                    required=params_model.model_json_schema()["required"]
                ),
                examples=examples
            ),
            "implementation": func,
            "params_model": params_model
        }
        return func
    return decorator

class ToolRegistry:
    @staticmethod
    def convert_tool_registry_to_response_format() -> list[dict]:
        """
        Convert TOOL_REGISTRY to OpenAI Chat Completions tools format.
        
        Returns:
            List of tools in OpenAI format
        """
        response_tools = []
        
        for tool_name,tool_entry in TOOL_REGISTRY.items():
            tool_schema = tool_entry["schema"]
            
            # Convert to Azure OpenAI format
            response_tool = {
                "type": "function",
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": {
                    "type": "object",
                    "properties": tool_schema.parameters.properties,
                    "required": tool_schema.parameters.required
                }
            
            }
            
            response_tools.append(response_tool)
        
        return response_tools

    @staticmethod
    def convert_tool_registry_to_chat_completions_format() -> list[dict]:
        """
        Convert TOOL_REGISTRY to OpenAI Chat Completions tools format.
        
        Returns:
            List of tools in OpenAI format
        """
        cc_tools = []
        
        for tool_name, tool_entry in TOOL_REGISTRY.items():
            tool_schema = tool_entry["schema"]
            
            # Convert to Azure OpenAI format
            cc_tool = {
                "type": "function",
                "function": {
                    "name": tool_schema.name,
                    "description": tool_schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool_schema.parameters.properties,
                        "required": tool_schema.parameters.required
                    }
                }
            }
            
            cc_tools.append(cc_tool)
        
        return cc_tools
    
    @staticmethod
    def execute_tool_call(tool_name: str, arguments: dict) -> str:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool_entry = TOOL_REGISTRY[tool_name]
        params_model = tool_entry["params_model"]
        implementation = tool_entry["implementation"]

        try:
            validated_args = params_model(**arguments)
        except ValidationError as e:
            raise ValueError(f"Argument validation error: {e}")

        return implementation(**validated_args.model_dump())
        

