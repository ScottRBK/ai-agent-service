import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.api.routes.openai_compatible import chat_completions, stream_chat_response, ChatCompletionRequest
from app.core.agents.api_agent import APIAgent

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

### chat_completions streaming
@pytest.mark.asyncio
async def test_chat_completions_streaming_response(mock_agent, chat_request):
    # Arrange
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        with patch('app.api.routes.openai_compatible.APIAgent') as mock_agent_class:
            mock_load.return_value = [{"agent_id": "test_agent"}]
            mock_agent_class.return_value = mock_agent
            mock_agent.initialize = AsyncMock()
            
            # Act
            response = await chat_completions(chat_request)
            
            # Assert
            assert hasattr(response, 'body_iterator')
            assert response.media_type == "text/event-stream"

@pytest.mark.asyncio
async def test_chat_completions_non_streaming_response():
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
            response = await chat_completions(chat_request)
            
            # Assert
            assert response.choices[0]["message"]["content"] == "Hello there!"
            assert response.model == "test_agent"

@pytest.mark.asyncio
async def test_chat_completions_agent_not_found(chat_request):
    # Arrange
    with patch('app.api.routes.openai_compatible.load_agent_configs') as mock_load:
        mock_load.return_value = [{"agent_id": "other_agent"}]
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await chat_completions(chat_request)
        assert "Agent test_agent not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_messages():
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[],  # Empty messages
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request)
    assert "'messages' must be a non-empty list" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_message_format():
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user"}],  # Missing content
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request)
    assert "Each message must have 'role' and 'content'" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_temperature():
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=3.0,  # Invalid temperature
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request)
    assert "'temperature' must be between 0 and 2" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_invalid_max_tokens():
    # Arrange
    chat_request = ChatCompletionRequest(
        model="test_agent",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=-1,  # Invalid max_tokens
        stream=False
    )
    
    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        await chat_completions(chat_request)
    assert "'max_tokens' must be non-negative" in str(exc_info.value)

@pytest.mark.asyncio
async def test_chat_completions_summary_agent_trigger():
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
            response = await chat_completions(chat_request)
            
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