import asyncio
import json
from typing import Any, Dict, List
from mcp.types import Tool as MCPTool
from app.utils.logging import logger

class SubprocessMCPClient:
    """Simple MCP client for subprocess-based servers."""
    
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self.request_id = 0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.warning(f"Error cleaning up subprocess: {e}")
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/list"
        }
        self.request_id += 1
        
        response = await self._send_request(request)
        tools_data = response.get("result", {}).get("tools", [])
        return [MCPTool(**tool) for tool in tools_data]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        self.request_id += 1
        
        response = await self._send_request(request)
        content = response.get("result", {}).get("content", [{}])
        return content[0].get("text", "") if content else ""
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the subprocess."""
        try:
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
            
            response_str = await self.process.stdout.readline()
            if not response_str:
                raise Exception("No response from subprocess")
            
            return json.loads(response_str.decode())
        except Exception as e:
            logger.error(f"Error communicating with subprocess: {e}")
            raise 