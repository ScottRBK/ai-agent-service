from typing import Optional, Dict, Any, List
from fastapi import Request
from app.models.auth import UserInfo, SessionInfo, AuthContext
from app.config.settings import settings
from app.utils.logging import logger

def extract_user_from_headers(request: Request) -> Optional[UserInfo]:
    """Extract user information from request headers"""
    if not settings.ENABLE_USER_SESSION_MANAGEMENT:
        return None

    logger.info(f"auth headers - extract_user_from_headers - request - {request}")

    headers = dict(request.headers)
    
    logger.info(f"auth_headers - extract_user_from_headers - headers - {headers}")

    # Extract user ID (required)
    user_id = headers.get(settings.AUTH_TRUSTED_ID_HEADER.lower())
    if not user_id:
        return None


    # Extract optional fields
    email = headers.get(settings.AUTH_TRUSTED_EMAIL_HEADER.lower())
    name = headers.get(settings.AUTH_TRUSTED_NAME_HEADER.lower())
    role = headers.get(settings.AUTH_TRUSTED_ROLE_HEADER.lower(), "user")
    
    # Parse groups (comma-separated)
    groups_header = headers.get(settings.AUTH_TRUSTED_GROUPS_HEADER.lower(), "")
    groups = [g.strip() for g in groups_header.split(",") if g.strip()]
    
    user = UserInfo(
        user_id=user_id,
        email=email,
        name=name or email or user_id,  # Fallback chain for name
        role=role,
        groups=groups
    )
    
    logger.debug(f"Extracted user from headers: {user.user_id}")
    return user

def extract_session_from_headers(request: Request) -> Optional[SessionInfo]:
    """Extract session information from request headers"""
    if not settings.ENABLE_USER_SESSION_MANAGEMENT:
        return None
        
    headers = dict(request.headers)
    
    session_id = headers.get(settings.AUTH_SESSION_HEADER.lower()) if settings.AUTH_SESSION_HEADER else None
    chat_id = headers.get(settings.AUTH_CHAT_ID_HEADER.lower()) if settings.AUTH_CHAT_ID_HEADER else None
    
    if not session_id and chat_id:
        session_id = chat_id
    
    if not session_id:
        return None 
    
    if not chat_id:
        chat_id = session_id

    session = SessionInfo(
        session_id=session_id,
        chat_id=chat_id
    )
    
    logger.debug(f"Extracted session from headers: {session.session_id}")
    return session

def get_auth_context(request: Request) -> Optional[AuthContext]:
    """Get complete auth context from request"""
    user = extract_user_from_headers(request)
    session = extract_session_from_headers(request)
    
    if user and session:
        return AuthContext(user=user, session=session)
    elif user:
        session = SessionInfo(session_id=user.user_id,
                              chat_id=user.user_id)
        return AuthContext(user=user, session=session)
    return None