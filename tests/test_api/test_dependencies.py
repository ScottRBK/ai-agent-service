"""
Unit tests for API dependencies module.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import Request
from app.api.dependencies import get_current_user, get_current_session, get_auth_context_dep
from app.models.auth import UserInfo, SessionInfo, AuthContext


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""
    
    @patch('app.api.dependencies.get_auth_context')
    def test_returns_user_from_auth_context(self, mock_get_auth_context):
        """Should return user from auth context when available."""
        # Arrange
        request = Mock(spec=Request)
        expected_user = UserInfo(
            user_id="test_user",
            name="Test User",
            email="test@example.com",
            role="admin"
        )
        mock_auth_context = Mock(user=expected_user)
        mock_get_auth_context.return_value = mock_auth_context
        
        # Act
        result = get_current_user(request)
        
        # Assert
        assert result == expected_user
        mock_get_auth_context.assert_called_once_with(request)
    
    @patch('app.api.dependencies.get_auth_context')
    @patch('app.api.dependencies.settings')
    def test_returns_default_user_when_no_auth_context(self, mock_settings, mock_get_auth_context):
        """Should return default user when auth context is None."""
        # Arrange
        request = Mock(spec=Request)
        mock_get_auth_context.return_value = None
        mock_settings.AUTH_FALLBACK_USER_ID = "default_user"
        
        # Act
        result = get_current_user(request)
        
        # Assert
        assert result.user_id == "default_user"
        assert result.name == "Default User"
        assert result.role == "user"


class TestGetCurrentSession:
    """Tests for get_current_session dependency."""
    
    @patch('app.api.dependencies.get_auth_context')
    def test_returns_session_from_auth_context(self, mock_get_auth_context):
        """Should return session from auth context when available."""
        # Arrange
        request = Mock(spec=Request)
        expected_session = SessionInfo(
            session_id="test_session",
            chat_id="test_chat"
        )
        mock_auth_context = Mock(session=expected_session)
        mock_get_auth_context.return_value = mock_auth_context
        
        # Act
        result = get_current_session(request)
        
        # Assert
        assert result == expected_session
        mock_get_auth_context.assert_called_once_with(request)
    
    @patch('app.api.dependencies.get_auth_context')
    @patch('app.api.dependencies.settings')
    def test_returns_default_session_when_no_auth_context(self, mock_settings, mock_get_auth_context):
        """Should return default session when auth context is None."""
        # Arrange
        request = Mock(spec=Request)
        mock_get_auth_context.return_value = None
        mock_settings.AUTH_FALLBACK_SESSION_ID = "default_session"
        
        # Act
        result = get_current_session(request)
        
        # Assert
        assert result.session_id == "default_session"


class TestGetAuthContextDep:
    """Tests for get_auth_context_dep dependency."""
    
    @patch('app.api.dependencies.get_current_session')
    @patch('app.api.dependencies.get_current_user')
    def test_combines_user_and_session(self, mock_get_user, mock_get_session):
        """Should combine user and session into AuthContext."""
        # Arrange
        request = Mock(spec=Request)
        expected_user = UserInfo(
            user_id="test_user",
            name="Test User",
            role="user"
        )
        expected_session = SessionInfo(
            session_id="test_session"
        )
        mock_get_user.return_value = expected_user
        mock_get_session.return_value = expected_session
        
        # Act
        result = get_auth_context_dep(request)
        
        # Assert
        assert isinstance(result, AuthContext)
        assert result.user == expected_user
        assert result.session == expected_session
        mock_get_user.assert_called_once_with(request)
        mock_get_session.assert_called_once_with(request)