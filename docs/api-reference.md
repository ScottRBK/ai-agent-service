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
  "user_id": "user123",
  "session_id": "session456",
  "stream": false
}
```

**Example:**
```bash
curl -X POST "http://localhost:8001/agents/research_agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the current time in Tokyo?",
    "user_id": "user123",
    "session_id": "session456"
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

- Authentication will be added at a later date
- Agent-specific tool access control
- Session isolation for memory
- Environment variable substitution for secure token management
- MCP server authorization support