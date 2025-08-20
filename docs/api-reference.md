# API Reference

This document provides detailed API reference information for the AI Agent Service.

## Health & Documentation Endpoints

### Health Status
```bash
GET /health
```
Returns detailed health status with provider and resource monitoring.

### API Documentation
```bash
GET /docs        # Interactive Swagger UI documentation
GET /redoc       # Alternative API documentation interface
GET /            # Basic service information and status
```

## Agent Management API

### List All Agents
```bash
GET /agents/
```

**Example Response:**
```json
[
  {
    "agent_id": "research_agent",
    "allowed_regular_tools": ["get_current_datetime"],
    "allowed_mcp_servers": {
      "deepwiki": {"allowed_mcp_tools": ["search_wiki"]},
      "searxng": {"allowed_mcp_tools": null}
    },
    "resources": ["memory"],
    "provider": "azure_openai_cc",
    "model": "gpt-4o-mini"
  }
]
```

### Get Specific Agent
```bash
GET /agents/{agent_id}
```

**Example:**
```bash
curl -X GET "http://localhost:8001/agents/research_agent"
```

### Send Message to Agent
```bash
POST /agents/{agent_id}/chat
```

**Request Body:**
```json
{
  "message": "What is the current time in Tokyo?",
  "user_id": "user123",        // Optional if provided via headers
  "session_id": "session456",   // Optional if provided via headers
  "stream": false
}
```

**Example with Request Body:**
```bash
curl -X POST "http://localhost:8001/agents/research_agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the current time in Tokyo?",
    "user_id": "user123",
    "session_id": "session456"
  }'
```

**Example with Headers (preferred for Open WebUI integration):**
```bash
curl -X POST "http://localhost:8001/agents/research_agent/chat" \
  -H "Content-Type: application/json" \
  -H "X-OpenWebUI-User-Id: user123" \
  -H "X-OpenWebUI-Chat-Id: chat789" \
  -d '{
    "message": "What is the current time in Tokyo?"
  }'
```

**Response:**
```json
{
  "response": "The current time in Tokyo is...",
  "tool_calls": [],
  "memory_saved": true
}
```

### Get Conversation History
```bash
GET /agents/{agent_id}/conversation/{session_id}?user_id={user_id}
```

**Example:**
```bash
curl -X GET "http://localhost:8001/agents/research_agent/conversation/session456?user_id=user123"
```

**Response:**
```json
{
  "conversation": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2023-01-01T12:00:00"
    },
    {
      "role": "assistant", 
      "content": "Hello! How can I help you?",
      "timestamp": "2023-01-01T12:00:01"
    }
  ],
  "session_summary": "Initial greeting exchange"
}
```

### Clear Conversation History
```bash
DELETE /agents/{agent_id}/conversation/{session_id}?user_id={user_id}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8001/agents/research_agent/conversation/session456?user_id=user123"
```

## OpenAI-Compatible API

> **Note**: The OpenAI-compatible endpoints fully support header-based authentication. When integrated with Open WebUI, user and session context is automatically extracted from headers, making these endpoints ideal for seamless integration.

### List Models
```bash
GET /v1/models
```

**Example:**
```bash
curl -X GET "http://localhost:8001/v1/models"
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "research_agent",
      "object": "model",
      "created": 1640995200,
      "owned_by": "ai-agent-service"
    },
    {
      "id": "cli_agent",
      "object": "model", 
      "created": 1640995200,
      "owned_by": "ai-agent-service"
    }
  ]
}
```

### Chat Completions (Non-Streaming)
```bash
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "research_agent",
  "messages": [
    {"role": "user", "content": "What is the current time in Tokyo?"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": false
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "research_agent",
    "messages": [
      {"role": "user", "content": "What is the current time in Tokyo?"}
    ],
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1640995200,
  "model": "research_agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The current time in Tokyo is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Chat Completions (Streaming)
```bash
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "research_agent",
  "messages": [
    {"role": "user", "content": "What is the current time in Tokyo?"}
  ],
  "stream": true,
  "temperature": 0.7
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "research_agent",
    "messages": [
      {"role": "user", "content": "What is the current time in Tokyo?"}
    ],
    "stream": true,
    "temperature": 0.7
  }'
```

**Streaming Response (Server-Sent Events):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1640995200,"model":"research_agent","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1640995200,"model":"research_agent","choices":[{"index":0,"delta":{"content":" current"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1640995200,"model":"research_agent","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Programmatic Usage Examples

### Python with requests
```python
import requests

# List all agents
response = requests.get("http://localhost:8001/agents/")
agents = response.json()
print(f"Available agents: {[agent['agent_id'] for agent in agents]}")

# Chat with an agent
response = requests.post(
    "http://localhost:8001/agents/research_agent/chat",
    json={
        "message": "Research Python async programming",
        "user_id": "user123",
        "session_id": "session456"
    }
)
result = response.json()
print(f"Agent response: {result['response']}")

# Use OpenAI-compatible endpoint
response = requests.post(
    "http://localhost:8001/v1/chat/completions",
    json={
        "model": "research_agent",
        "messages": [
            {"role": "user", "content": "What is the current time in Tokyo?"}
        ]
    }
)
result = response.json()
print(f"OpenAI response: {result['choices'][0]['message']['content']}")
```

### Python with OpenAI SDK
```python
from openai import OpenAI

# Initialize client with custom base URL
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # API key not required
)

# List available models (agents)
models = client.models.list()
print(f"Available models: {[model.id for model in models.data]}")

# Chat completion
response = client.chat.completions.create(
    model="research_agent",
    messages=[
        {"role": "user", "content": "What is the current time in Tokyo?"}
    ],
    temperature=0.7
)
print(f"Response: {response.choices[0].message.content}")

# Streaming chat completion
stream = client.chat.completions.create(
    model="research_agent",
    messages=[
        {"role": "user", "content": "What is the current time in Tokyo?"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Error Responses

### Standard Error Format
```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "status_code": 400
}
```

### Common Error Codes
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Agent or resource not found
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server-side error

### Error Examples

**Agent Not Found:**
```json
{
  "detail": "Agent 'unknown_agent' not found",
  "status_code": 404
}
```

**Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "status_code": 422
}
```

## Rate Limiting and Performance

- Rate limiting is out of scope for this service
- Async/await throughout for optimal performance
- Connection pooling for database operations
- Tool caching per agent for improved response times
- Streaming support reduces perceived latency

## Authentication and Security

> **Important**: The service currently implements user and session management through trusted headers, but does **NOT** perform authentication. It trusts that authentication has been handled by an upstream service (like Open WebUI or a reverse proxy). Actual authentication/authorization will be added in a future release.

### User and Session Management

The service supports header-based user identification and session management. This allows the service to maintain separate contexts for different users and sessions when deployed behind an authenticating proxy or service like Open WebUI.

#### Trusted Headers for User Context

When `ENABLE_USER_SESSION_MANAGEMENT=true` is configured, the service trusts and extracts user and session information from the following headers:

| Header | Environment Variable | Description |
|--------|---------------------|-------------|
| `X-OpenWebUI-User-Id` | `AUTH_TRUSTED_ID_HEADER` | Unique user identifier (required) |
| `X-OpenWebUI-User-Email` | `AUTH_TRUSTED_EMAIL_HEADER` | User's email address |
| `X-OpenWebUI-User-Name` | `AUTH_TRUSTED_NAME_HEADER` | User's display name |
| `X-OpenWebUI-User-Role` | `AUTH_TRUSTED_ROLE_HEADER` | User role (e.g., "user", "admin") |
| `X-OpenWebUI-User-Groups` | `AUTH_TRUSTED_GROUPS_HEADER` | Comma-separated list of user groups |
| `X-OpenWebUI-Session-Id` | `AUTH_SESSION_HEADER` | Session identifier |
| `X-OpenWebUI-Chat-Id` | `AUTH_CHAT_ID_HEADER` | Chat/conversation identifier |

#### User Context Priority

The service uses the following priority for determining user and session context:

1. **Trusted Headers (highest priority)**: If user headers are present from a trusted upstream service, they take precedence
2. **Request body**: If no headers are present, `user_id` and `session_id` from the request body are used
3. **Defaults (lowest priority)**: Falls back to `default_user` and `default_session`

> **Security Note**: Since there is no authentication, anyone can pass these headers or body parameters. Only use this in trusted environments or behind an authenticating proxy.

#### Session Management Behavior

- **Chat ID as Session**: When Open WebUI provides a `chat_id` header but no `session_id`, the chat ID is automatically used as the session ID
- **Memory Isolation**: Each session maintains its own conversation memory
- **User Isolation**: Memory and resources are isolated per user

#### Example: Request with User Context Headers

```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-OpenWebUI-User-Id: user123" \
  -H "X-OpenWebUI-User-Email: user@example.com" \
  -H "X-OpenWebUI-User-Name: John Doe" \
  -H "X-OpenWebUI-Chat-Id: chat789" \
  -d '{
    "model": "research_agent",
    "messages": [
      {"role": "user", "content": "What is the current time?"}
    ]
  }'
```

In this case:
- User context is extracted from trusted headers
- `chat789` is used as both chat_id and session_id
- Memory is maintained specific to this user and chat session
- **Note**: The service trusts these headers without verification

#### Open WebUI Integration

When integrated with Open WebUI:
1. Open WebUI handles user authentication
2. Enable header forwarding in Open WebUI: `ENABLE_FORWARD_USER_INFO_HEADERS=true`
3. Configure the agent service to trust headers: `ENABLE_USER_SESSION_MANAGEMENT=true`
4. Open WebUI automatically sends user context headers with each request
5. Each Open WebUI chat maintains its own session context using the chat ID

> **Trust Model**: The agent service trusts that Open WebUI has properly authenticated users. It does not verify the headers themselves.

#### Configuration Example

```env
# Enable user session management
ENABLE_USER_SESSION_MANAGEMENT=true

# Configure header names (case-insensitive)
AUTH_TRUSTED_ID_HEADER=x-openwebui-user-id
AUTH_TRUSTED_EMAIL_HEADER=x-openwebui-user-email
AUTH_TRUSTED_NAME_HEADER=x-openwebui-user-name
AUTH_TRUSTED_ROLE_HEADER=x-openwebui-user-role
AUTH_TRUSTED_GROUPS_HEADER=x-openwebui-user-groups
AUTH_SESSION_HEADER=x-openwebui-session-id
AUTH_CHAT_ID_HEADER=x-openwebui-chat-id

# Fallback values when headers are not present
AUTH_FALLBACK_USER_ID=default_user
AUTH_FALLBACK_SESSION_ID=default_session
```

### Security Considerations

**Current Security Features:**
- User and session context management via trusted headers
- Session isolation for memory (each session has separate memory)
- User isolation for resources
- Agent-specific tool access control
- Environment variable substitution for secure token management
- MCP server authorization support

**Security Limitations:**
- **No Authentication**: The service does not authenticate users
- **No Authorization**: The service does not verify user permissions
- **Trust-based**: Headers and parameters are trusted without verification
- **Deployment Requirement**: Must be deployed behind an authenticating proxy or service

**Recommended Deployment:**
- Deploy behind Open WebUI or another authenticating service
- Use in trusted internal networks only
- Do not expose directly to the internet without authentication layer