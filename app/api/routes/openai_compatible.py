from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.core.agents.api_agent import APIAgent
import time
import json
import os
from app.utils.logging import logger

router = APIRouter(prefix="/v1", tags=["openai-compatible"])

def load_agent_configs() -> List[Dict[str, Any]]:
    """
    Load all agent configurations from agent_config.json.
    
    Returns:
        List of agent configurations
    """
    try:
        from app.config.settings import settings
        config_path = settings.AGENT_CONFIG_PATH
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                all_configs = json.load(f)
            
            # Handle both single agent config and multiple agent configs
            if isinstance(all_configs, dict):
                # Single agent config
                return [all_configs]
            elif isinstance(all_configs, list):
                # Multiple agent configs
                return all_configs
            else:
                logger.warning("agent_config.json contains invalid format")
                return []
        else:
            logger.warning(f"agent_config.json not found at {config_path}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading agent configs: {e}")
        return []

class ChatCompletionRequest(BaseModel):
    model: str  # This will be interpreted as agent_id
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    The 'model' parameter is interpreted as the agent_id.
    """
    logger.info(f"Chat completions request: {request}")
    try:
        # Input validation
        if not isinstance(request.messages, list) or len(request.messages) == 0:
            raise HTTPException(status_code=422, detail="'messages' must be a non-empty list")
        for msg in request.messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise HTTPException(status_code=422, detail="Each message must have 'role' and 'content'")
        if request.temperature is not None and not (0 <= request.temperature <= 2):
            raise HTTPException(status_code=422, detail="'temperature' must be between 0 and 2")
        if request.max_tokens is not None and request.max_tokens < 0:
            raise HTTPException(status_code=422, detail="'max_tokens' must be non-negative")

        # Extract agent_id from model parameter
        agent_id = request.model

        # Check if agent exists
        agent_configs = load_agent_configs()
        if not any(agent.get("agent_id") == agent_id for agent in agent_configs):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Create API agent
        agent = APIAgent(
            agent_id=agent_id,
            user_id="default_user",  # Could be extracted from auth
            session_id="default_session"
        )

        # Initialize the agent
        await agent.initialize()

        # Use the agent's chat method instead of calling provider directly
        # This ensures proper memory handling and conversation flow
        try:
            user_message = request.messages[-1]["content"]
        except IndexError:
            raise HTTPException(status_code=422, detail="'messages' must be a non-empty list")
        response = await agent.chat(user_message)

        # Format response in OpenAI format
        return ChatCompletionResponse(
            id="chatcmpl-123",
            created=int(time.time()),
            model=agent_id,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,  # Could be calculated
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """
    OpenAI-compatible models endpoint.
    Returns available agents as "models".
    """
    # Read agent_config.json and return agents as models
    agents = load_agent_configs()
    
    return {
        "object": "list",
        "data": [
            {
                "id": agent["agent_id"],
                "object": "model",
                "created": 0,
                "owned_by": "ai-agent-service"
            }
            for agent in agents
        ]
    }
