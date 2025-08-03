"""
Tool Registry is a registry of tools that can be used by the agent.
It is used to convert the tool registry to the different formats that are supported by the different providers.
It is also used to execute the tool calls.
"""

import os
import re
import json
from typing import Callable, Dict, Any, Type
from pydantic import BaseModel
from pydantic import ValidationError
from mcp.types import Tool as MCPTool
from app.models.tools.tools import Tool, ToolParameters
from app.models.tools.mcp import MCP
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
                    required=params_model.model_json_schema().get("required", [])
                ),
                examples=examples
            ),
            "implementation": func,
            "params_model": params_model
        }
        return func
    return decorator

class ToolRegistry:
    """
    Tool Registry is a registry of tools that can be used by the agent.
    It is used to convert the tool registry to the different formats that are supported by the different providers.
    It is also used to execute the tool calls.
    """

    @staticmethod
    def _substitute_environment_variables(obj):
        """Recursively substitute environment variables in strings using ${VAR_NAME} syntax."""
        if isinstance(obj, dict):
            return {key: ToolRegistry._substitute_environment_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ToolRegistry._substitute_environment_variables(item) for item in obj]
        elif isinstance(obj, str):
            def replace_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found, using empty string")
                    return ""
                return env_value
            return re.sub(r'\$\{([^}]+)\}', replace_var, obj)
        else:
            return obj

    @staticmethod
    def load_mcp_servers() -> list[MCP]:
        """Loads the MCP servers from the mcp.json file with environment variable substitution."""
        try:
            from app.config.settings import settings
            config_path = settings.MCP_CONFIG_PATH
            with open(config_path, "r") as f:
                mcp_data = json.load(f)
            
            # Substitute environment variables in the loaded data
            mcp_data = ToolRegistry._substitute_environment_variables(mcp_data)
            
            if isinstance(mcp_data, dict):
                mcp_servers = [mcp_data]
            elif isinstance(mcp_data, list):
                mcp_servers = mcp_data
            else:
                logger.error(f"Invalid mcp.json format: expected dict or list, got {type(mcp_data)}")
                return []
            
            return [MCP(**server) for server in mcp_servers]
        
        except FileNotFoundError:
            logger.error(f"mcp.json file not found at {config_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in mcp.json: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading MCP servers: {e}")
            return []

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
    async def execute_tool_call(tool_name: str, arguments: dict, agent_context=None) -> str:
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool_entry = TOOL_REGISTRY[tool_name]
        params_model = tool_entry["params_model"]
        implementation = tool_entry["implementation"]

        try:
            validated_args = params_model(**arguments)
        except ValidationError as e:
            raise ValueError(f"Argument validation error: {e}")

        # Check if the tool expects agent_context using inspect
        import inspect
        
        sig = inspect.signature(implementation)
        
        if 'agent_context' in sig.parameters:
            # Tool expects agent context - pass it as keyword argument
            validated_args_dict = validated_args.model_dump()
            validated_args_dict['agent_context'] = agent_context
            
            if inspect.iscoroutinefunction(implementation):
                # For async functions, await them
                return await implementation(**validated_args_dict)
            else:
                return implementation(**validated_args_dict)
        else:
            # Regular tool without agent context
            if inspect.iscoroutinefunction(implementation):
                return await implementation(**validated_args.model_dump())
            else:
                return implementation(**validated_args.model_dump())
    
    @staticmethod
    def convert_mcp_tools_to_chatcompletions(mcp_server_label: str, mcp_tools: list[MCPTool]) -> list[dict[str, Any]]:
        """
        Converts a list of MCPTool objects to a list of dictionaries
        in the format expected by the Chat Completions Model.
        """
        # Use a more unique separator to avoid conflicts with tool names
        separator = "__"
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{mcp_server_label}{separator}{tool.name}",
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_tools
        ]