from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ChatMessage(BaseModel):
    """Individual chat message model"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Request model for agent chat"""
    message: str = Field(..., description="User message to send to agent")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    session_id: Optional[str] = Field("default_session", description="Session identifier")
    model: Optional[str] = Field(None, description="Override model from agent config")
    model_settings: Optional[Dict[str, Any]] = Field(None, description="Override model settings")

class ChatResponse(BaseModel):
    """Response model for agent chat"""
    response: str = Field(..., description="Agent response")
    agent_id: str = Field(..., description="Agent identifier")
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    model_used: str = Field(..., description="Model that was used")
    tools_available: int = Field(..., description="Number of tools available to agent")

class AgentInfo(BaseModel):
    """Agent information model"""
    agent_id: str = Field(..., description="Agent identifier")
    provider: str = Field(..., description="AI provider")
    model: str = Field(..., description="Default model")
    tools_available: List[str] = Field(..., description="Available tools")
    resources: List[str] = Field(..., description="Available resources")
    has_memory: bool = Field(..., description="Whether agent has memory enabled")

class ConversationHistory(BaseModel):
    """Conversation history model"""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    session_id: str = Field(..., description="Session identifier")
    agent_id: str = Field(..., description="Agent identifier")
    user_id: str = Field(..., description="User identifier")
