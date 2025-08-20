from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserInfo(BaseModel):
    """User information from upstream service"""
    user_id: str = Field(..., description="Unique user identifier")
    email: Optional[EmailStr] = Field(None, description="User email")
    name: Optional[str] = Field(None, description="User display name")
    role: Optional[str] = Field("user", description="User role (user/admin)")
    groups: Optional[List[str]] = Field(default_factory=list, description="User groups")
    
class SessionInfo(BaseModel):
    """Session context information"""
    session_id: str = Field(..., description="Session identifier")
    chat_id: Optional[str] = Field(None, description="Chat/conversation ID")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuthContext(BaseModel):
    """Complete authentication context"""
    user: UserInfo
    session: SessionInfo