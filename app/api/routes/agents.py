from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from app.models.agents import ChatRequest, ChatResponse, AgentInfo, ConversationHistory, ChatMessage
from app.core.agents.api_agent import APIAgent
from app.core.agents.agent_tool_manager import AgentToolManager
from app.utils.logging import logger
from app.api.dependencies import get_auth_context_dep
from app.models.auth import AuthContext
from app.config.settings import settings
from datetime import datetime
import json
import os

router = APIRouter(prefix="/agents", tags=["agents"])

# Dependency to get agent manager
def get_agent_tool_manager(agent_id: str):
    return AgentToolManager(agent_id)


def load_agent_configs() -> List[Dict[str, Any]]:
    """Load all agent configurations from agent_config.json."""
    try:
        from app.config.settings import settings
        config_path = settings.AGENT_CONFIG_PATH
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                all_configs = json.load(f)
            
            if isinstance(all_configs, dict):
                return [all_configs]
            elif isinstance(all_configs, list):
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

@router.get("/", response_model=List[AgentInfo])
async def list_agents():
    """List all available agents with their configurations"""
    try:
        agents = load_agent_configs()
        agent_info_list = []
        
        for agent_config in agents:
            agent_id = agent_config.get("agent_id")
            if not agent_id:
                continue
                
            # Create temporary agent to get tool info
            agent = APIAgent(agent_id=agent_id)
            await agent.initialize()
            
            # Get tool names
            tool_names = []
            for tool in agent.available_tools:
                if tool.get("type") == "function" and "function" in tool:
                    tool_names.append(tool["function"]["name"])
            
            agent_info = AgentInfo(
                agent_id=agent_id,
                provider=agent_config.get("provider", "unknown"),
                model=agent_config.get("model", "unknown"),
                tools_available=tool_names,
                resources=agent_config.get("resources", []),
                has_memory="memory" in agent_config.get("resources", [])
            )
            agent_info_list.append(agent_info)
        
        return agent_info_list
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent_info(agent_id: str):
    """Get information about a specific agent"""
    try:
        agents = load_agent_configs()
        agent_config = None
        
        for config in agents:
            if config.get("agent_id") == agent_id:
                agent_config = config
                break
        
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Create agent to get tool info
        agent = APIAgent(agent_id=agent_id)
        await agent.initialize()
        
        # Get tool names
        tool_names = []
        for tool in agent.available_tools:
            if tool.get("type") == "function" and "function" in tool:
                tool_names.append(tool["function"]["name"])
        
        return AgentInfo(
            agent_id=agent_id,
            provider=agent_config.get("provider", "unknown"),
            model=agent_config.get("model", "unknown"),
            tools_available=tool_names,
            resources=agent_config.get("resources", []),
            has_memory="memory" in agent_config.get("resources", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    auth_context: AuthContext = Depends(get_auth_context_dep),
    agent_tool_manager: AgentToolManager = Depends(get_agent_tool_manager)
):
    """Send a message to an agent and get response"""
    try:
        # Check if agent exists
        agents = load_agent_configs()
        if not any(agent.get("agent_id") == agent_id for agent in agents):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Priority: Headers > Request Body > Defaults
        user_id = auth_context.user.user_id
        session_id = auth_context.session.session_id
        
        # Allow request body to override if headers not present
        if auth_context.user.user_id == settings.AUTH_FALLBACK_USER_ID:
            user_id = request.user_id or user_id
            session_id = request.session_id or session_id
        

        # Create API agent instance
        agent = APIAgent(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            model=request.model,
            model_settings=request.model_settings
        )
        
        # Get response
        response = await agent.chat(request.message)
        
        return ChatResponse(
            response=response,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            model_used=agent.model,
            tools_available=len(agent.available_tools)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(
    agent_id: str,
    session_id: str,
    user_id: str = "default_user"
):
    """Get conversation history for a specific session"""
    try:
        # Check if agent exists
        agents = load_agent_configs()
        if not any(agent.get("agent_id") == agent_id for agent in agents):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent = APIAgent(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id
        )
        await agent.initialize()
        
        messages = await agent.get_conversation_history()
        
        return ConversationHistory(
            messages=[ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages],
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{agent_id}/conversation/{session_id}")
async def clear_conversation_history(
    agent_id: str,
    session_id: str,
    user_id: str = "default_user"
):
    """Clear conversation history for a specific session"""
    try:
        # Check if agent exists
        agents = load_agent_configs()
        if not any(agent.get("agent_id") == agent_id for agent in agents):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent = APIAgent(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id
        )
        await agent.initialize()
        
        await agent.clear_conversation()
        
        return {"message": f"Conversation history cleared for session {session_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
