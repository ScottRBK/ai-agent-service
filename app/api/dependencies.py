from typing import Optional
from fastapi import Request, Depends
from app.models.auth import UserInfo, SessionInfo, AuthContext
from app.utils.auth_headers import get_auth_context
from app.config.settings import settings
from app.utils.logging import logger

def get_current_user(request: Request) -> UserInfo:
    """Get current user from request or use defaults"""
    auth_context = get_auth_context(request)
    
    if auth_context:
        return auth_context.user
    
    # Fallback to defaults
    return UserInfo(
        user_id=settings.AUTH_FALLBACK_USER_ID,
        name="Default User",
        role="user"
    )

def get_current_session(request: Request) -> SessionInfo:
    """Get current session from request or use defaults"""
    auth_context = get_auth_context(request)
    
    if auth_context:
        return auth_context.session
    
    # Fallback to defaults
    return SessionInfo(
        session_id=settings.AUTH_FALLBACK_SESSION_ID
    )

def get_auth_context_dep(request: Request) -> AuthContext:
    """Dependency to get auth context"""
    user = get_current_user(request)
    session = get_current_session(request)

    return AuthContext(user=user, session=session)