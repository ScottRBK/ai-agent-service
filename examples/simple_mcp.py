"""
This example shows how to use the Azure OpenAI API to send a chat completion request to the MCP server.
"""

import asyncio
import logging
import json

from app.core.providers.azureopenapi_cc import AzureOpenAIProviderCC, AzureOpenAIConfig
from openai import AzureOpenAI
from fastmcp import FastMCP, Client
from mcp.types import Tool as MCPTool
from typing import Any
from app.core.tools.tool_registry import ToolRegistry


async def convert_mcp_tools_to_chatcompletions(mcp_server_label: str, mcp_tools: list[MCPTool]) -> list[dict[str, Any]]:
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
async def send_chat_completion(client: AzureOpenAI, provider: AzureOpenAIProviderCC, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
    """
    Sends a chat completion request to the Azure OpenAI API.
    """
    response = await client.chat.completions.create(
        model=provider.config.default_model,
        messages=messages,
        tools=tools,
    )
    return response

async def call_tool(mcp_client: Client, mcp_server_label: str, tool_name: str, tool_arguments: dict[str, Any]) -> Any:
    """
    Calls a tool on the MCP server.
    """
    # Tool name is already properly extracted, no need to split again
    return await mcp_client.call_tool(tool_name, tool_arguments)


async def main():

    provider = AzureOpenAIProviderCC(AzureOpenAIConfig())
    await provider.initialize()
    client = provider.client
    mcp_tools: list[MCPTool] = []
    mcp_servers = ToolRegistry.load_mcp_servers()
    print(f"Found {len(mcp_servers)} MCP servers: {[s.server_label for s in mcp_servers]}")
    
    # Create a mapping for easy lookup
    servers_by_label = {server.server_label: server for server in mcp_servers}

    chatcompletions_formatted_tools = []

    for mcp_server in mcp_servers:
        print(f"Loading tools from {mcp_server.server_label} at {mcp_server.server_url}")
        mcp_server_label = mcp_server.server_label
        mcp_client = Client(mcp_server.server_url)

        async with mcp_client:
            print(f"Pinging {mcp_server.server_label}")
            print(await mcp_client.ping())
            print(f"Listing tools from {mcp_server.server_label}")
            server_tools = await mcp_client.list_tools()
            print(server_tools)
            mcp_tools.extend(server_tools)

            chatcompletions_formatted_tools.extend(ToolRegistry.convert_mcp_tools_to_chatcompletions(mcp_server_label, server_tools))
    
    #add the non-mcp tools to the chatcompletions_formatted_tools
    non_mcp_tools = ["get_current_datetime"]
    tools_list = ToolRegistry.convert_tool_registry_to_chat_completions_format()
    registered_tools = [tool for tool in tools_list if tool["function"]["name"] in non_mcp_tools]
    chatcompletions_formatted_tools.extend(registered_tools)

    messages = [{"role": "user", "content": """Hello can you tell me about the bytedance/trae-agent repo.
    Also, can you tell me the current date and time and and also fetch some cool mcp servers from https://mcpservers.org/remote-mcp-servers"""}]
    print(f"Sending chat completion request to {provider.config.default_model}")
    response = await send_chat_completion(client, provider, messages, chatcompletions_formatted_tools)
    
    for _ in range(provider.max_tool_iterations):
        if response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"Calling tool: {tool_call.function.name}")

                # Use the same separator as in tool name creation
                separator = "__"
                if separator not in tool_call.function.name:
                    #check to see if it is a non-mcp tool:
                    if tool_call.function.name in non_mcp_tools:
                        tool_result = ToolRegistry.execute_tool_call(tool_call.function.name, json.loads(tool_call.function.arguments))
                        messages.append({
                            "role": "tool",
                            "content": str(tool_result),
                            "tool_call_id": tool_call.id
                        })
                    else:
                        raise ValueError(f"Invalid tool name format: {tool_call.function.name}")
                else:
                    
                    mcp_server_label, tool_name = tool_call.function.name.split(separator, 1)
                    mcp_server_url = servers_by_label[mcp_server_label].server_url

                    async with Client(mcp_server_url) as mcp_client:
                        tool_result = await call_tool(mcp_client, mcp_server_label, tool_name, 
                                                    json.loads(tool_call.function.arguments))
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.id
                    })
            
            response = await send_chat_completion(client, provider, 
                                                    messages, chatcompletions_formatted_tools)
        else:
            break
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())