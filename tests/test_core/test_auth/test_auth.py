"""
Unit tests for auth models
"""
import pytest
from datetime import datetime
from typing import Dict, Any

from app.models.auth import UserInfo, SessionInfo, AuthContext


class TestUserInfo:
    """Test UserInfo model"""
    
    def test_user_info_minimal(self):
        """Test creating UserInfo with minimal required fields"""
        user = UserInfo(user_id="test123")
        
        assert user.user_id == "test123"
        assert user.email is None
        assert user.name is None
        assert user.role == "user"  # Default value
        assert user.groups == []  # Default empty list
    
    def test_user_info_full(self):
        """Test creating UserInfo with all fields"""
        user = UserInfo(
            user_id="user456",
            email="test@example.com",
            name="Test User",
            role="admin",
            groups=["group1", "group2"]
        )
        
        assert user.user_id == "user456"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.role == "admin"
        assert user.groups == ["group1", "group2"]
    
    def test_user_info_serialization(self):
        """Test UserInfo serialization to dict"""
        user = UserInfo(
            user_id="user789",
            email="user@test.com",
            name="John Doe"
        )
        
        data = user.model_dump()
        assert data["user_id"] == "user789"
        assert data["email"] == "user@test.com"
        assert data["name"] == "John Doe"
        assert data["role"] == "user"
        assert data["groups"] == []
    
    def test_user_info_from_dict(self):
        """Test creating UserInfo from dictionary"""
        data = {
            "user_id": "dict_user",
            "email": "dict@test.com",
            "name": "Dict User",
            "role": "moderator",
            "groups": ["editors", "reviewers"]
        }
        
        user = UserInfo(**data)
        assert user.user_id == "dict_user"
        assert user.email == "dict@test.com"
        assert user.role == "moderator"
        assert len(user.groups) == 2


class TestSessionInfo:
    """Test SessionInfo model"""
    
    def test_session_info_minimal(self):
        """Test creating SessionInfo with minimal required fields"""
        session = SessionInfo(session_id="sess_abc123")
        
        assert session.session_id == "sess_abc123"
        assert session.chat_id is None
        assert isinstance(session.created_at, datetime)
        assert session.metadata == {}
    
    def test_session_info_with_chat_id(self):
        """Test creating SessionInfo with chat_id"""
        session = SessionInfo(
            session_id="sess_xyz789",
            chat_id="chat_456"
        )
        
        assert session.session_id == "sess_xyz789"
        assert session.chat_id == "chat_456"
    
    def test_session_info_with_metadata(self):
        """Test creating SessionInfo with metadata"""
        metadata = {"source": "web", "ip": "192.168.1.1"}
        session = SessionInfo(
            session_id="sess_meta",
            metadata=metadata
        )
        
        assert session.session_id == "sess_meta"
        assert session.metadata == metadata
        assert session.metadata["source"] == "web"
    
    def test_session_info_created_at(self):
        """Test that created_at is automatically set"""
        before = datetime.now()
        session = SessionInfo(session_id="sess_time")
        after = datetime.now()
        
        assert before <= session.created_at <= after
    
    def test_session_info_serialization(self):
        """Test SessionInfo serialization"""
        session = SessionInfo(
            session_id="sess_serial",
            chat_id="chat_serial",
            metadata={"key": "value"}
        )
        
        data = session.model_dump()
        assert data["session_id"] == "sess_serial"
        assert data["chat_id"] == "chat_serial"
        assert "created_at" in data
        assert data["metadata"] == {"key": "value"}


class TestAuthContext:
    """Test AuthContext model"""
    
    def test_auth_context_creation(self):
        """Test creating AuthContext with user and session"""
        user = UserInfo(user_id="ctx_user")
        session = SessionInfo(session_id="ctx_session")
        
        auth_context = AuthContext(user=user, session=session)
        
        assert auth_context.user.user_id == "ctx_user"
        assert auth_context.session.session_id == "ctx_session"
    
    def test_auth_context_full(self):
        """Test AuthContext with fully populated user and session"""
        user = UserInfo(
            user_id="full_user",
            email="full@test.com",
            name="Full User",
            role="admin",
            groups=["admins"]
        )
        
        session = SessionInfo(
            session_id="full_session",
            chat_id="full_chat",
            metadata={"client": "mobile"}
        )
        
        auth_context = AuthContext(user=user, session=session)
        
        assert auth_context.user.email == "full@test.com"
        assert auth_context.user.role == "admin"
        assert auth_context.session.chat_id == "full_chat"
        assert auth_context.session.metadata["client"] == "mobile"
    
    def test_auth_context_serialization(self):
        """Test AuthContext serialization to dict"""
        user = UserInfo(user_id="serial_user")
        session = SessionInfo(session_id="serial_session")
        auth_context = AuthContext(user=user, session=session)
        
        data = auth_context.model_dump()
        
        assert "user" in data
        assert "session" in data
        assert data["user"]["user_id"] == "serial_user"
        assert data["session"]["session_id"] == "serial_session"
    
    def test_auth_context_from_dict(self):
        """Test creating AuthContext from nested dictionary"""
        data = {
            "user": {
                "user_id": "nested_user",
                "email": "nested@test.com"
            },
            "session": {
                "session_id": "nested_session",
                "chat_id": "nested_chat"
            }
        }
        
        auth_context = AuthContext(**data)
        
        assert auth_context.user.user_id == "nested_user"
        assert auth_context.user.email == "nested@test.com"
        assert auth_context.session.session_id == "nested_session"
        assert auth_context.session.chat_id == "nested_chat"


class TestModelValidation:
    """Test model validation and error handling"""
    
    def test_user_info_missing_required_field(self):
        """Test that UserInfo requires user_id"""
        with pytest.raises(ValueError):
            UserInfo()
    
    def test_session_info_missing_required_field(self):
        """Test that SessionInfo requires session_id"""
        with pytest.raises(ValueError):
            SessionInfo()
    
    def test_auth_context_missing_user(self):
        """Test that AuthContext requires user"""
        session = SessionInfo(session_id="test_session")
        with pytest.raises(ValueError):
            AuthContext(session=session)
    
    def test_auth_context_missing_session(self):
        """Test that AuthContext requires session"""
        user = UserInfo(user_id="test_user")
        with pytest.raises(ValueError):
            AuthContext(user=user)
    
    def test_invalid_email_format(self):
        """Test that invalid email raises validation error"""
        with pytest.raises(ValueError):
            UserInfo(user_id="test", email="not_an_email")