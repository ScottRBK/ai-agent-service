"""
Unit tests for auth_headers utility functions
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import Request

from app.models.auth import UserInfo, SessionInfo, AuthContext
from app.utils.auth_headers import (
    extract_user_from_headers,
    extract_session_from_headers,
    get_auth_context
)


class TestExtractUserFromHeaders:
    """Test extract_user_from_headers function"""
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_with_all_headers(self, mock_settings):
        """Test extracting user with all headers present"""
        # Configure mock settings
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = "X-User-Email"
        mock_settings.AUTH_TRUSTED_NAME_HEADER = "X-User-Name"
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = "X-User-Role"
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = "X-User-Groups"
        
        # Create mock request with headers
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "user123",
            "x-user-email": "test@example.com",
            "x-user-name": "Test User",
            "x-user-role": "admin",
            "x-user-groups": "group1,group2,group3"
        }
        
        user = extract_user_from_headers(mock_request)
        
        assert user is not None
        assert user.user_id == "user123"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.role == "admin"
        assert user.groups == ["group1", "group2", "group3"]
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_minimal_headers(self, mock_settings):
        """Test extracting user with only required headers"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "minimal_user"
        }
        
        user = extract_user_from_headers(mock_request)
        
        assert user is not None
        assert user.user_id == "minimal_user"
        assert user.email is None
        assert user.name == "minimal_user"  # Falls back to user_id
        assert user.role == "user"  # Default value
        assert user.groups == []
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_disabled(self, mock_settings):
        """Test that extraction returns None when disabled"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = False
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {"x-user-id": "test"}
        
        user = extract_user_from_headers(mock_request)
        assert user is None
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_missing_user_id(self, mock_settings):
        """Test that None is returned when user_id is missing"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {}  # No user ID header
        
        user = extract_user_from_headers(mock_request)
        assert user is None
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_name_fallback(self, mock_settings):
        """Test name fallback chain: name -> email -> user_id"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = "X-User-Email"
        mock_settings.AUTH_TRUSTED_NAME_HEADER = "X-User-Name"
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = ""
        
        # Test with email but no name
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "user456",
            "x-user-email": "fallback@test.com"
        }
        
        user = extract_user_from_headers(mock_request)
        assert user.name == "fallback@test.com"  # Falls back to email
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_user_empty_groups(self, mock_settings):
        """Test handling of empty groups header"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = "X-User-Groups"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "user789",
            "x-user-groups": ""  # Empty groups string
        }
        
        user = extract_user_from_headers(mock_request)
        assert user.groups == []


class TestExtractSessionFromHeaders:
    """Test extract_session_from_headers function"""
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_session_with_all_headers(self, mock_settings):
        """Test extracting session with all headers"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        mock_settings.AUTH_CHAT_ID_HEADER = "X-Chat-Id"
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-session-id": "sess_123",
            "x-chat-id": "chat_456"
        }
        
        session = extract_session_from_headers(mock_request)
        
        assert session is not None
        assert session.session_id == "sess_123"
        assert session.chat_id == "chat_456"
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_session_minimal(self, mock_settings):
        """Test extracting session with only session_id - chat_id defaults to session_id"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        mock_settings.AUTH_CHAT_ID_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-session-id": "sess_minimal"
        }
        
        session = extract_session_from_headers(mock_request)
        
        assert session is not None
        assert session.session_id == "sess_minimal"
        # When chat_id is not provided, it defaults to session_id
        assert session.chat_id == "sess_minimal"
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_session_disabled(self, mock_settings):
        """Test that extraction returns None when disabled"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = False
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {"x-session-id": "test"}
        
        session = extract_session_from_headers(mock_request)
        assert session is None
    
    @patch('app.utils.auth_headers.settings')
    def test_extract_session_missing_session_id(self, mock_settings):
        """Test that None is returned when session_id is missing"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {}  # No session ID header
        
        session = extract_session_from_headers(mock_request)
        assert session is None


class TestGetAuthContext:
    """Test get_auth_context function"""
    
    @patch('app.utils.auth_headers.settings')
    def test_get_auth_context_complete(self, mock_settings):
        """Test getting complete auth context"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = ""
        mock_settings.AUTH_CHAT_ID_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "context_user",
            "x-session-id": "context_session"
        }
        
        auth_context = get_auth_context(mock_request)
        
        assert auth_context is not None
        assert auth_context.user.user_id == "context_user"
        assert auth_context.session.session_id == "context_session"
    
    @patch('app.utils.auth_headers.settings')
    def test_get_auth_context_missing_user(self, mock_settings):
        """Test that None is returned when user is missing"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-session-id": "sess_only"  # No user ID
        }
        
        auth_context = get_auth_context(mock_request)
        assert auth_context is None
    
    @patch('app.utils.auth_headers.settings')
    def test_get_auth_context_missing_session(self, mock_settings):
        """Test that fallback session is created when session is missing but user exists"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_SESSION_HEADER = "X-Session-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = ""
        mock_settings.AUTH_CHAT_ID_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "user_only"  # No session ID
        }
        
        auth_context = get_auth_context(mock_request)
        # When user exists but no session, fallback session uses user_id
        assert auth_context is not None
        assert auth_context.user.user_id == "user_only"
        assert auth_context.session.session_id == "user_only"
        assert auth_context.session.chat_id == "user_only"
    
    @patch('app.utils.auth_headers.settings')
    def test_get_auth_context_disabled(self, mock_settings):
        """Test that None is returned when management is disabled"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = False
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "test",
            "x-session-id": "test"
        }
        
        auth_context = get_auth_context(mock_request)
        assert auth_context is None


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    @patch('app.utils.auth_headers.settings')
    def test_groups_with_spaces(self, mock_settings):
        """Test parsing groups with spaces"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = "X-User-Groups"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "user_spaces",
            "x-user-groups": " group1 , group2 , group3 "  # Spaces around groups
        }
        
        user = extract_user_from_headers(mock_request)
        assert user.groups == ["group1", "group2", "group3"]  # Spaces stripped
    
    @patch('app.utils.auth_headers.settings')
    def test_empty_header_settings(self, mock_settings):
        """Test behavior with empty header settings"""
        mock_settings.ENABLE_USER_SESSION_MANAGEMENT = True
        mock_settings.AUTH_TRUSTED_ID_HEADER = "X-User-Id"
        mock_settings.AUTH_TRUSTED_EMAIL_HEADER = ""  # Empty setting
        mock_settings.AUTH_TRUSTED_NAME_HEADER = ""   # Empty setting
        mock_settings.AUTH_TRUSTED_ROLE_HEADER = ""   # Empty setting
        mock_settings.AUTH_TRUSTED_GROUPS_HEADER = "" # Empty setting
        
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "test_empty",
            "x-user-email": "should_not_be_extracted@test.com"  # Should be ignored
        }
        
        user = extract_user_from_headers(mock_request)
        assert user.user_id == "test_empty"
        assert user.email is None  # Not extracted due to empty setting