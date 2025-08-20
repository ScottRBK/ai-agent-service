import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi import Request
from fastapi.testclient import TestClient
from app.api.routes.openai_compatible import chat_completions, stream_chat_response, ChatCompletionRequest
from app.core.agents.api_agent import APIAgent
from app.models.auth import AuthContext, UserInfo, SessionInfo

@pytest.fixture
def mock_agent():
    return AsyncMock(spec=APIAgent)

@pytest.fixture
def chat_request():
    return ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True
    )

@pytest.fixture
def mock_fastapi_request():
    """Create a mock FastAPI Request object."""
    request = Mock(spec=Request)
    request.headers = {}
    return request

@pytest.fixture
def mock_auth_context():
    """Create a mock AuthContext with default user and session."""
    return AuthContext(
        user=UserInfo(
            user_id="test_user",
            name="Test User",
            role="user"
        ),
        session=SessionInfo(
            session_id="test_session"
        )
    )

### chat_completions streaming
@pytest.mark.asyncio
async def test_chat_completions_streaming_response(mock_agent, chat_request, mock_fastapi_request, mock_auth_context):
    # Arrange
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent_class.return_value = mock_agent
            mock_agent.initialize = AsyncMock()
            
            # Act
            response = await chat_completions(
                chat_request, 
                mock_fastapi_request,
                mock_auth_context
            )
            
            # Assert
            assert hasattr(response, 'body_iterator')
            assert response.media_type == "text/event-stream"

@pytest.mark.asyncio
async def test_chat_completions_non_streaming_response(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False
    )
    
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent = AsyncMock()
            mock_agent.chat.return_value = "Hello there!"
            mock_agent_class.return_value = mock_agent
            
            # Act
            response = await chat_completions(
                chat_request,
                mock_fastapi_request,
                mock_auth_context
            )
            
            # Assert
            assert response.choices[0]["message"]["content"] == "Hello there!"
            assert response.model == "test_agent"

@pytest.mark.asyncio
async def test_chat_completions_agent_not_found(chat_request, mock_fastapi_request, mock_auth_context):
    # Arrange
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        mock_load.return_value = [{"agent_id": "other_agent"}]
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await chat_completions(chat_request, mock_fastapi_request, mock_auth_context)
        assert "Agent test_agent not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_messages(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[],  # Empty messages
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request, mock_fastapi_request, mock_auth_context)
    assert "'messages' must be a non-empty list" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_message_format(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user"}],  # Missing content
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request, mock_fastapi_request, mock_auth_context)
    assert "Each message must have 'role' and 'content'" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_temperature(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=3.0,  # Invalid temperature
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request, mock_fastapi_request, mock_auth_context)
    assert "'temperature' must be between 0 and 2" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_max_tokens(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=-1,  # Invalid max_tokens
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request, mock_fastapi_request, mock_auth_context)
    assert "'max_tokens' must be non-negative" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_summary_agent_trigger(mock_fastapi_request, mock_auth_context):
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "### Task: Summarize this"}],
        stream=False
    )
    
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            # Mock both test_agent and summary_agent to exist
            mock_load.return_value = [{"agent_id": "test_agent"}, {"agent_id": "summary_agent"}]
            mock_agent = AsyncMock()
            mock_agent.chat.return_value = "Summary response"
            mock_agent_class.return_value = mock_agent
            
            # Act
            response = await chat_completions(
                chat_request,
                mock_fastapi_request,
                mock_auth_context
            )
            
            # Assert
            assert response.choices[0]["message"]["content"] == "Summary response"

### stream_chat_response
@pytest.mark.asyncio
async def test_stream_chat_response_format(mock_agent):
    # Arrange
    # Create a proper async generator function
    async def mock_streaming_generator(user_message):
        yield "Hello"
        yield " world"
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    assert len(chunks) == 5  # Initial + 2 content chunks + final + [DONE]
    assert chunks[0].startswith("data: ")  # Initial chunk
    assert chunks[1].startswith("data: ")  # First content chunk
    assert chunks[2].startswith("data: ")  # Second content chunk
    assert chunks[3].startswith("data: ")  # Final chunk with finish_reason
    assert chunks[4].startswith("data: [DONE]")  # [DONE] chunk

@pytest.mark.asyncio
async def test_stream_chat_response_initial_chunk(mock_agent):
    # Arrange
    mock_agent.chat_stream.return_value.__aiter__ = AsyncMock(
        return_value=iter(["Hello", " world"])
    )
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
        break  # Just get the first chunk
    
    # Assert
    data = json.loads(chunks[0].replace("data: ", ""))
    assert data["object"] == "chat.completion.chunk"
    assert data["choices"][0]["delta"]["role"] == "assistant"

@pytest.mark.asyncio
async def test_stream_chat_response_content_chunks(mock_agent):
    # Arrange
    # Create a proper async generator function
    async def mock_streaming_generator(user_message):
        yield "Hello"
        yield " world"
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    # Check that content chunks contain the expected data
    content_chunks = [chunk for chunk in chunks if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]")]
    assert len(content_chunks) == 4  # Initial + 2 content chunks + final
    
    # Parse the content chunks to verify structure
    for chunk in content_chunks[1:-1]:  # Skip initial and final chunks
        data = json.loads(chunk[6:])  # Remove "data: " prefix
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "delta" in data["choices"][0]
        assert "content" in data["choices"][0]["delta"]

@pytest.mark.asyncio
async def test_stream_chat_response_final_chunk(mock_agent):
    # Arrange
    # Create a proper async generator function
    async def mock_streaming_generator(user_message):
        yield "Response"
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    assert chunks[-1] == "data: [DONE]\n\n"  # Final chunk should be [DONE]

@pytest.mark.asyncio
async def test_stream_chat_response_completion_id_format(mock_agent):
    # Arrange
    mock_agent.chat_stream.return_value.__aiter__ = AsyncMock(return_value=iter([]))
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
        break
    
    # Assert
    data = json.loads(chunks[0].replace("data: ", ""))
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "test_agent"

@pytest.mark.asyncio
async def test_stream_chat_response_empty_content(mock_agent):
    # Arrange
    # Create a proper async generator function for empty response
    async def mock_streaming_generator(user_message):
        if False:  # This ensures it's an async generator even with no yields
            yield ""
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    assert len(chunks) == 3  # Initial + final + [DONE] chunks
    assert chunks[0].startswith("data: ")  # Initial chunk
    assert chunks[1].startswith("data: ")  # Final chunk with finish_reason
    assert chunks[2] == "data: [DONE]\n\n"  # [DONE] chunk

@pytest.mark.asyncio
async def test_stream_chat_response_single_content_chunk(mock_agent):
    # Arrange
    # Create a proper async generator function
    async def mock_streaming_generator(user_message):
        yield "Single response"
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    assert len(chunks) == 4  # Initial + content + final + [DONE]
    assert chunks[1].startswith("data: ")  # Content chunk
    data = json.loads(chunks[1][6:])  # Remove "data: " prefix
    assert data["choices"][0]["delta"]["content"] == "Single response"

@pytest.mark.asyncio
async def test_stream_chat_response_json_structure(mock_agent):
    # Arrange
    # Create a proper async generator function
    async def mock_streaming_generator(user_message):
        yield "Test"
    
    # Mock the async method to return the async generator
    mock_agent.chat_stream = mock_streaming_generator
    
    # Act
    chunks = []
    async for chunk in stream_chat_response(mock_agent, "Hi", "test_agent", None):
        chunks.append(chunk)
    
    # Assert
    # Verify the JSON structure of content chunks
    content_chunks = [chunk for chunk in chunks if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]")]
    
    for chunk in content_chunks:
        data = json.loads(chunk[6:])  # Remove "data: " prefix
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "test_agent"


### User Session Management Tests
@pytest.mark.asyncio
async def test_chat_completions_uses_auth_context_user_session(mock_fastapi_request):
    """Test that chat_completions uses user_id and session_id from auth context."""
    # Arrange
    custom_auth_context = AuthContext(
        user=UserInfo(
            user_id="custom_user",
            name="Custom User",
            email="custom@example.com",
            role="admin"
        ),
        session=SessionInfo(
            session_id="custom_session",
            chat_id="custom_chat"
        )
    )
    
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False
    )
    
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent = AsyncMock()
            mock_agent.chat.return_value = "Response"
            mock_agent_class.return_value = mock_agent
            
            # Act
            await chat_completions(
                chat_request,
                mock_fastapi_request,
                custom_auth_context
            )
            
            # Assert - Verify APIAgent was created with correct user/session
            mock_agent_class.assert_called_once_with(
                agent_id="test_agent",
                user_id="custom_user",
                session_id="custom_session"
            )


@pytest.mark.asyncio
async def test_chat_completions_with_headers_from_openwebui(mock_fastapi_request):
    """Test handling of Open WebUI style headers."""
    # Arrange
    mock_fastapi_request.headers = {
        "x-openwebui-user-id": "webui_user",
        "x-openwebui-user-email": "user@webui.com",
        "x-openwebui-user-name": "WebUI User",
        "x-openwebui-chat-id": "chat_123"
    }
    
    # Create auth context that would be extracted from these headers
    auth_context_from_headers = AuthContext(
        user=UserInfo(
            user_id="webui_user",
            name="WebUI User",
            email="user@webui.com",
            role="user"
        ),
        session=SessionInfo(
            session_id="chat_123",  # Using chat_id as session_id
            chat_id="chat_123"
        )
    )
    
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello from Open WebUI"}],
        stream=False
    )
    
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent = AsyncMock()
            mock_agent.chat.return_value = "Response to WebUI"
            mock_agent_class.return_value = mock_agent
            
            # Act
            response = await chat_completions(
                chat_request,
                mock_fastapi_request,
                auth_context_from_headers
            )
            
            # Assert
            mock_agent_class.assert_called_once_with(
                agent_id="test_agent",
                user_id="webui_user",
                session_id="chat_123"
            )
            assert response.choices[0]["message"]["content"] == "Response to WebUI"


@pytest.mark.asyncio
async def test_chat_completions_different_sessions_isolated():
    """Test that different sessions for the same user are isolated."""
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False
    )
    
    # First session context
    auth_context_session1 = AuthContext(
        user=UserInfo(user_id="same_user", name="User", role="user"),
        session=SessionInfo(session_id="session1")
    )
    
    # Second session context
    auth_context_session2 = AuthContext(
        user=UserInfo(user_id="same_user", name="User", role="user"),
        session=SessionInfo(session_id="session2")
    )
    
    mock_request = Mock(spec=Request)
    mock_request.headers = {}
    
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent = AsyncMock()
            mock_agent.chat.return_value = "Response"
            mock_agent_class.return_value = mock_agent
            
            # Act - Call with first session
            await chat_completions(chat_request, mock_request, auth_context_session1)
            
            # Assert first call
            first_call_args = mock_agent_class.call_args_list[0]
            assert first_call_args[1]["user_id"] == "same_user"
            assert first_call_args[1]["session_id"] == "session1"
            
            # Act - Call with second session
            await chat_completions(chat_request, mock_request, auth_context_session2)
            
            # Assert second call
            second_call_args = mock_agent_class.call_args_list[1]
            assert second_call_args[1]["user_id"] == "same_user"
            assert second_call_args[1]["session_id"] == "session2" 