# AI Agent Service

A modern, intelligent AI Agent Service framework built with FastAPI & FastMCP that demonstrates how to implement agent tool management, prompt handling, and multi-provider AI integration from scratch. This project showcases a production-ready implementation of agent-specific tool filtering, dynamic system prompts, and unified provider interfaces without relying on abstraction frameworks. Built with Docker, comprehensive logging, and enterprise-grade features. The service now includes comprehensive streaming support across all providers and API endpoints, enabling real-time response delivery and enhanced user experience.

## ✨ Features

- **Framework Design** - Complete implementation showing how to build AI agents from the ground up
- **FastAPI Framework** - Modern, fast web framework with automatic API documentation
- **AI Agent Capabilities** - Intelligent automation and decision-making
- **Health Check Endpoints** - Built-in monitoring and status endpoints
- **Multi-Provider AI Support** - Azure OpenAI, Ollama with unified interface
- **MCP Integration** - Model Context Protocol for external tools using fastmcp library
- **Tool Filtering** - Agent-specific permissions and authorization
- **Prompt Management** - Dynamic system prompts with tool integration
- **Model Configuration** - Flexible model selection and parameter management
- **CLI Parameter Overrides** - Runtime model and setting customization
- **Docker Support** - Multi-stage builds with development and production targets 
- **Environment Configuration** - Flexible settings with environment variable support
- **Structured Logging** - Comprehensive logging setup for debugging and monitoring
- **Type Safety** - Full type hints throughout the codebase
- **Auto-Generated Docs** - Interactive API documentation with Swagger UI and ReDoc
- **Hot Reload** - Development mode with automatic code reloading
- **Resource Management** - Global resource lifecycle management with agent-specific filtering
- **Memory Persistence** - PostgreSQL-based conversation history with automatic cleanup
- **Memory Compression** - Intelligent conversation history management with AI-powered summarization
- **Agent Resource Manager** - Per-agent resource access control and automatic resource creation
- **Streaming Support** - Real-time response streaming across all providers and API endpoints
- **Response Processing** - Automatic response cleaning and formatting for memory storage
- **OpenAI-Compatible API** - Full OpenAI protocol compliance with streaming support

## 🏗️ Agent Architecture

The service uses a unified agent architecture built around inheritance:

### BaseAgent Foundation
- **BaseAgent** - Common functionality shared by all agents
- **Optional Memory** - Memory features activate only when configured in `agent_config.json`
- **Consistent Interface** - All agents inherit the same core methods and behaviors
- **Error Handling** - Centralized error handling and logging across all agent types

### Agent Inheritance Hierarchy
```
BaseAgent (base class)
├── CLIAgent (interactive command-line interface)
├── APIAgent (web API optimized)
└── MemoryCompressionAgent (conversation summarization)
```

### Key Benefits
- **Code Reuse** - ~135 lines of duplicate code eliminated
- **Consistency** - Uniform memory handling and error management
- **Configuration-Driven** - Memory behavior controlled via agent configuration
- **Maintainability** - Single source of truth for common functionality

### Memory Behavior
- Memory functionality is **optional** and **configuration-driven**
- Agents without `"memory"` in their `resources` array operate without persistence
- Agents with memory support automatic conversation history and compression
- Error handling ensures graceful fallback to empty history when memory issues occur

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/ScottRBK/ai-agent-service
cd ai-agent-service

# Run in development mode
cd docker
docker-compose --profile dev up --build
```

The service will be available at:
- **API**: http://localhost:8001
- **Health Check**: http://localhost:8001/health
- **API Docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.main
```

## Software Architecture
![High Level Architecture](architecture.png "High Level Architecture")

## 📁 Project Structure 

```
ai-agent-service/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py              # Health check endpoints
│   │       ├── agents.py              # Agent management API
│   │       └── openai_compatible.py   # OpenAI-compatible API with streaming
│   ├── core/
│   │   ├── agents/
│   │   │   ├── base_agent.py            # Base agent class with common functionality
│   │   │   ├── agent_tool_manager.py    # Agent tool filtering with fastmcp
│   │   │   ├── agent_resource_manager.py # Agent resource management
│   │   │   ├── prompt_manager.py        # System prompt management
│   │   │   ├── cli_agent.py             # CLI agent implementation (inherits from BaseAgent)
│   │   │   ├── api_agent.py             # API agent implementation with streaming (inherits from BaseAgent)
│   │   │   └── memory_compression_agent.py # Memory compression agent (inherits from BaseAgent)
│   │   ├── providers/
│   │   │   ├── base.py                  # Base provider interface
│   │   │   ├── azureopenapi.py          # Azure OpenAI (Responses API) with streaming
│   │   │   ├── azureopenapi_cc.py       # Azure OpenAI (Chat Completions) with streaming
│   │   │   └── ollama.py                # Ollama provider with streaming
│   │   ├── resources/
│   │   │   ├── base.py                  # Base resource interface
│   │   │   ├── manager.py               # Global resource management
│   │   │   ├── memory.py                # PostgreSQL memory resource
│   │   │   └── memory_compression_manager.py # Memory compression logic
│   │   └── tools/
│   │       ├── tool_registry.py         # Tool management
│   │       └── function_calls/          # Built-in tools
│   ├── config/
│   │   └── settings.py                  # Application configuration
│   ├── models/
│   │   ├── agents.py                    # Agent API models
│   │   └── resources/
│   │       └── memory.py                # Memory data models
│   └── utils/
│       ├── logging.py                   # Logging configuration
│       └── chat_utils.py                # Response cleaning utilities
├── tests/
│   ├── test_core/
│   │   ├── test_agents/                 # Agent unit tests
│   │   ├── test_providers/              # Provider tests with streaming
│   │   ├── test_resources/              # Resource tests
│   │   └── test_tools/                  # Tool tests
│   ├── test_api/
│   │   ├── test_agents.py               # Agent API tests
│   │   └── test_openai_compatible_integration.py # OpenAI API tests with streaming
│   └── test_integration/                # End-to-end tests
├── examples/
│   └── run_agent.py                     # CLI agent runner
├── docker/
│   ├── Dockerfile                       # Multi-stage Docker build
│   └── docker-compose.yml               # Development environment
├── agent_config.json                    # Agent configurations
├── prompts/                             # System Prompt Files per agent
├── mcp.json                             # MCP server config
└── requirements.txt
```

## 🔧 Configuration

The application uses environment-based configuration with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | "AI Agent Service" | Name of the service |
| `SERVICE_VERSION` | "1.0.0" | Version of the service |
| `HOST` | "0.0.0.0" | Server host address |
| `PORT` | 8000 | Server port |
| `LOG_LEVEL` | "DEBUG" | Logging level |
| `AGENT_CONFIG_PATH` | "agent_config.json" | Path to agent configuration file |
| `MCP_CONFIG_PATH` | "mcp.json" | Path to MCP server configuration file |
| `PROMPTS_DIR_PATH` | "prompts" | Path to prompts directory |
| `GITHUB_TOKEN` | (dynamic) | GitHub token for MCP server authentication (resolved via environment variable substitution) |

### MCP Server Authorization

The service supports secure authorization for MCP servers through centralized environment variable substitution:

#### Environment Variable Substitution
MCP server configurations can reference environment variables using `${VARIABLE_NAME}` syntax. The substitution is handled automatically when MCP servers are loaded:

```json
{
  "server_label": "github",
  "server_url": "https://api.githubcopilot.com/mcp/",
  "require_approval": "never",
  "header": {
    "authorization": "${GITHUB_TOKEN}"
  }
}
```

The `${GITHUB_TOKEN}` reference is automatically resolved to the value of the `GITHUB_TOKEN` environment variable when the MCP servers are loaded by the `ToolRegistry`.

#### Security Best Practices
1. **Set environment variables**: Configure token values as environment variables (no code changes required)
2. **Reference in mcp.json**: Use `${VARIABLE_NAME}` syntax in MCP server configurations
3. **Centralized processing**: Environment variable substitution is handled automatically during MCP server loading
4. **Docker environment variables**: Pass tokens through Docker environment variables
5. **Add to .gitignore**: Ensure `.env` files containing tokens are not committed to version control
6. **No code changes**: Add new MCP servers with authorization by editing only `mcp.json` and setting environment variables
7. **Pydantic configuration**: The `Settings` class is configured with `extra = "ignore"` to allow dynamic environment variables

#### Adding New Authorization Tokens
To add support for new MCP server tokens (no code changes required):

1. **Reference in mcp.json**:
```json
{
  "server_label": "new_service",
  "server_url": "https://api.newservice.com/mcp/",
  "header": {
    "authorization": "${NEW_SERVICE_TOKEN}"
  }
}
```

2. **Set environment variable**:
```bash
export NEW_SERVICE_TOKEN="your_actual_token_here"
```

The environment variable substitution is handled automatically by the `ToolRegistry.load_mcp_servers()` method. No code changes or application restarts are required.

### Configuration File Management

The service supports flexible configuration file management through environment variables and Docker volume mounts:

#### Environment Variables
- `AGENT_CONFIG_PATH`: Path to the agent configuration file (inside container)
- `MCP_CONFIG_PATH`: Path to the MCP server configuration file (inside container)  
- `PROMPTS_DIR_PATH`: Path to the prompts directory (inside container)

#### Docker Volume Mounts
- `AGENT_CONFIG_FILE`: Host path to agent_config.json
- `MCP_CONFIG_FILE`: Host path to mcp.json
- `PROMPTS_DIR`: Host path to prompts directory

### Environment Files

Create a `.env` file in the `docker/` directory to override defaults:

```env
# docker/.env
PORT=8001
LOG_LEVEL=INFO
DEV_CONTAINER_NAME=my-agent-service

# Configuration file paths (inside container)
AGENT_CONFIG_PATH=/app/config/agent_config.json
MCP_CONFIG_PATH=/app/config/mcp.json
PROMPTS_DIR_PATH=/app/config/prompts

# Configuration file mounts (host paths)
AGENT_CONFIG_FILE=../agent_config.json
MCP_CONFIG_FILE=../mcp.json
PROMPTS_DIR=../prompts
```

### Docker Configuration Examples

#### Basic Usage (Default Paths)
```bash
cd docker
docker-compose --profile dev up --build
```

#### Custom Configuration Paths
```bash
# Using custom configuration files
AGENT_CONFIG_FILE=/path/to/custom/agent_config.json \
MCP_CONFIG_FILE=/path/to/custom/mcp.json \
PROMPTS_DIR=/path/to/custom/prompts \
docker-compose --profile dev up --build
```

#### Environment Variable Override
```bash
# Override configuration paths via environment variables
AGENT_CONFIG_PATH=/custom/config/agents.json \
MCP_CONFIG_PATH=/custom/config/mcp_servers.json \
PROMPTS_DIR_PATH=/custom/prompts \
docker-compose --profile dev up --build
```

## 📡 API Endpoints

### Root Endpoint
- **GET** `/` - Service information and status

### Health Check
- **GET** `/health` - Detailed health status with timestamp

### Documentation
- **GET** `/docs` - Interactive Swagger UI documentation
- **GET** `/redoc` - Alternative ReDoc documentation

### Example Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000000",
  "service": "AI Agent Service",
  "version": "1.0.0"
}
```

## 🐳 Docker Usage

### Development Mode
```bash
cd docker
docker-compose --profile dev up --build
```

### Production Build
```bash
docker build -f docker/Dockerfile --target production -t ai-agent-service:latest .
docker run -p 8000:8000 ai-agent-service:latest
```

### Docker Environment Variables
```bash
# Override default port
docker run -e PORT=8001 -p 8001:8001 ai-agent-service:latest

# PostgreSQL Configuration
POSTGRES_DB=ai_agent_db
POSTGRES_USER=ai_agent_user
POSTGRES_PASSWORD=ai_agent_password
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# MCP Server Authorization
GITHUB_TOKEN=your_github_token_here
```

### Docker Compose Environment Variables
```yaml
environment:
  # ... existing environment variables ...
  GITHUB_TOKEN: ${GITHUB_TOKEN}
  # Add other MCP server tokens as needed
  NEW_SERVICE_TOKEN: ${NEW_SERVICE_TOKEN}
```

**Note**: Environment variable substitution for MCP server tokens is handled automatically by the `ToolRegistry` when loading the `mcp.json` configuration file.

### Database Setup
The service includes PostgreSQL for memory persistence:

```bash
# Start with database
cd docker
docker-compose --profile dev up --build

# Database will be automatically initialized
# Access via DBeaver or other PostgreSQL client
```

## 🤖 AI Agent & Tool Management

### Agent Configuration
Configure agent-specific tool access, resources, and model settings via `agent_config.json` (or `agent_config.example.json` for testing):

```json
[
  {
    "agent_id": "research_agent",
    "system_prompt_file": "prompts/research_agent.txt",
    "allowed_regular_tools": ["get_current_datetime"],
    "allowed_mcp_servers": {
      "deepwiki": {
        "allowed_mcp_tools": ["read_wiki_structure", "search_wiki"]
      },
      "fetch": {
        "allowed_mcp_tools": ["fetch_url"]
      },
      "searxng": {
        "allowed_mcp_tools": null  # all tools from searxng
      }
    },
    "resources": ["memory"],
    "provider": "azure_openai_cc",
    "model": "qwen3:4b",
    "model_settings": {
      "temperature": 0.7,
      "max_tokens": 2000,
      "num_ctx": 8192,
      "num_predict": 2048
    }
  }
]
```

**MCP Server Configuration:**
- **Server-level access**: Each server in `allowed_mcp_servers` defines which MCP servers the agent can access
- **Tool-level filtering**: Each server can specify `allowed_mcp_tools` to control which tools are available:
  - `null`: All tools from this server are available
  - `[]`: No tools from this server are available
  - `["tool1", "tool2"]`: Only specified tools from this server are available
**Resource Configuration:**
- `resources` - Array of resource types available to the agent
- Automatic resource creation when agents request access
- Memory resources provide conversation persistence across sessions
- Memory compression automatically manages long conversation histories

**Model Configuration:**
- `model` - AI model identifier (e.g., "gpt-4o-mini", "qwen3:4b")
- `model_settings` - Provider-specific parameters (temperature, max_tokens, num_ctx, etc.)
- Settings are passed directly to the provider without validation for maximum flexibility

**Memory Compression Configuration:**
- **threshold_tokens**: Token limit before compression is triggered (default: 8000)
- **recent_messages_to_keep**: Number of recent messages to preserve (default: 4)
- **enabled**: Enable/disable compression for specific agents (default: true)
- **Agent-specific settings**: Different compression configurations per agent

### MCP (Model Context Protocol) Integration
The service supports both HTTP-based and command-based MCP servers using the fastmcp library:

- **HTTP-based Servers**: DeepWiki, Fetch, GitHub Copilot - Connect via URLs with standard HTTP/WebSocket
- **Command-based Servers**: Searxng and other local/containerized servers - Use fastmcp StdioTransport for subprocess execution
- **DeepWiki Server** - Interrogate information on Deepwiki page, https://docs.devin.ai/work-with-devin/deepwiki-mcp
- **Fetch Server** - An MCP server that provides web content fetching capabilities. This server enables LLMs to retrieve and process content from web pages, converting HTML to markdown for easier consumption
- **GitHub Copilot Server** - GitHub Copilot integration with secure token authentication via environment variables
- **Searxng Server** - Web search capabilities via Docker container using fastmcp StdioTransport
- **Extensible** - Add custom MCP servers via `mcp.json` with support for both server types
- **Secure Authorization** - Dynamic environment variable substitution for token management using `${VARIABLE_NAME}` syntax (no code changes required)
- **Automatic Processing** - Environment variables are resolved once during MCP server loading in `ToolRegistry`

### Tool Filtering
- **Agent-specific permissions** - Control which tools each agent can access
- **Regular tools** - Built-in functions like date/time, arithmetic, these are both just very basic examples 
- **MCP tools** - External server capabilities with proper authorization
- **Dual protocol support** - HTTP-based and command-based MCP servers with standardized fastmcp implementation

### System Prompt Management
The service includes a flexible prompt management system:

- **External prompt files** - Store prompts in `prompts/` directory
- **Inline prompts** - Define prompts directly in agent config
- **Tool integration** - Automatically include available tools in system prompts
- **Fallback prompts** - Default prompts when no configuration is found

#### Prompt File Example (`prompts/research_agent.txt`):

## Provider Support

### Supported AI Providers
- **Azure OpenAI (Chat Completions)** - Full MCP integration
- **Azure OpenAI (Responses API)** - Full MCP integration  
- **Ollama** - Full MCP integration

### Provider Features
- **Tool calling** - Execute functions and MCP tools (HTTP and command-based)
- **Agent filtering** - Provider-agnostic tool management
- **Health monitoring** - Provider status and metrics
- **Error handling** - Robust error management with fastmcp integration
- **Model settings** - Flexible parameter handling for provider-specific options
  - Azure OpenAI: temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, seed, response_format
  - Ollama: num_ctx, num_predict, top_k, repeat_penalty, repeat_last_n, temperature, top_p
  - Settings are passed directly to providers without validation for maximum flexibility
  - Support for both agent config and CLI parameter overrides

## 🤖 Running Agents

### Quick Start with CLI Agent

Use the example script to run different agents with various providers and memory:

```bash
# Run the research agent with Azure OpenAI and memory (includes Searxng web search)
python examples/run_agent.py research_agent azure_openai_cc

# Run the CLI agent with conversation memory (full MCP access including command-based servers)
python examples/run_agent.py cli_agent azure_openai_cc

# Run the MCP-only agent (HTTP and command-based MCP servers)
python examples/run_agent.py mcp_agent azure_openai_cc

# Override model and parameters via CLI
python examples/run_agent.py cli_agent ollama --model qwen3:4b --setting num_ctx 200 --setting num_predict 200 --setting temperature 0.7

# Use Azure OpenAI with custom settings
python examples/run_agent.py research_agent azure_openai_cc --model gpt-4o-mini --setting temperature 0.8 --setting max_tokens 3000
```

**MCP Server Access:**
- **HTTP-based**: DeepWiki, Fetch - Available immediately
- **Command-based**: Searxng - Requires Docker container running, uses fastmcp StdioTransport
- **Mixed usage**: Agents can use both types simultaneously

**CLI Parameter Override:**
- `--model` - Override the model specified in agent config
- `--setting key value` - Override individual model parameters
- CLI settings take precedence over agent config settings
- Supports any provider-specific parameters

### Memory Features
- **Conversation Persistence** - Agents remember previous interactions
- **Session Isolation** - Separate memory per user and session
- **Automatic Cleanup** - Expired memories automatically removed
- **Content Filtering** - Internal tags and formatting automatically cleaned
- **Memory Compression** - Intelligent conversation history management with AI-powered summarization
- **Token-based Compression** - Automatically compresses conversations when token limits are exceeded
- **Configurable Thresholds** - Set token limits and recent message retention per agent
- **Session Summaries** - Creates concise summaries of older conversation parts

### Available Agents

| Agent | Description | Tools Available |
|-------|-------------|-----------------|
| `research_agent` | Research assistant with web access | datetime, deepwiki, fetch, searxng |
| `cli_agent` | Command-line interface assistant | datetime, all MCP tools (HTTP and command-based) |
| `mcp_agent` | MCP tools only | deepwiki, fetch, searxng (no regular tools) |
| `restricted_agent` | Limited access example | specific tools only |
| `regular_tools_only_agent` | Basic functionality without MCP | datetime, arithmetic |

### Available Providers

| Provider | Description | Use Case |
|----------|-------------|----------|
| `azure_openai_cc` | Azure OpenAI (Chat Completions) | Production, full features |
| `azure_openai` | Azure OpenAI (Responses API) | Legacy compatibility |
| `ollama` | Local Ollama provider | Development, offline |

### Example Session

```bash
$ python examples/run_agent.py research_agent azure_openai_cc

🤖 research_agent Agent Ready!
🛠️ Available tools: 6
🧠 Memory: Enabled
💬 Type 'quit' to exit

You: What's the current time in Tokyo?
🤔 Thinking...
🤖 research_agent: The current time in Tokyo is 2025-01-16 15:30:45.

You: Search for the latest AI news
🤔 Thinking...
🤖 research_agent: I'll search for the latest AI news using web search...

You: quit
👋 Goodbye!
```

### Memory Compression Workflow

The system automatically manages conversation history through intelligent compression:

1. **Token Monitoring**: After each exchange, the system calculates total conversation tokens
2. **Compression Trigger**: When token threshold is exceeded (configurable per agent)
3. **Intelligent Splitting**: Older messages are separated from recent context
4. **AI Summarization**: A specialized compression agent creates concise summaries
5. **Context Preservation**: Recent messages are kept for immediate context
6. **Summary Integration**: Future conversations include the summary as system context

This ensures agents maintain conversation awareness while staying within token limits.

Invoking a specific model and model parameters: 

```bash
$ python examples/run_agent.py cli_agent azure_openai_cc --model gpt-4o-mini --setting temperature 0.7 --setting max_tokens 2000
```

## 🌐 Agent API Endpoints

The service provides comprehensive REST API endpoints for managing and interacting with AI agents:

### Agent Management API

#### List All Agents
```bash
GET /agents/
```
Returns all configured agents with their capabilities, tools, and resources.

#### Get Agent Information
```bash
GET /agents/{agent_id}
```
Returns detailed information about a specific agent including available tools and model configuration.

#### Chat with Agent
```bash
POST /agents/{agent_id}/chat
```
Send a message to an agent and get a response with conversation context and memory support.

**Request Body:**
```json
{
  "message": "What's the current time in Tokyo?",
  "user_id": "user123",
  "session_id": "session456",
  "model": "gpt-4o-mini",
  "model_settings": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Response:**
```json
{
  "response": "The current time in Tokyo is...",
  "agent_id": "research_agent",
  "user_id": "user123",
  "session_id": "session456",
  "timestamp": "2024-01-16T15:30:45.123456",
  "model_used": "gpt-4o-mini",
  "tools_available": 6
}
```

#### Conversation Management
```bash
GET /agents/{agent_id}/conversation/{session_id}?user_id=user123
```
Get complete conversation history for any session.

```bash
DELETE /agents/{agent_id}/conversation/{session_id}?user_id=user123
```
Clear conversation history for privacy and storage management.

### OpenAI-Compatible API

The service also provides OpenAI-compatible endpoints for seamless integration:

#### Chat Completions
```bash
POST /v1/chat/completions
```
Standard OpenAI-compatible chat completions where the `model` parameter is interpreted as the `agent_id`. Supports both streaming and non-streaming responses.

**Non-streaming Request:**
```json
{
  "model": "research_agent",
  "messages": [
    {"role": "user", "content": "What's the current time in Tokyo?"}
  ],
  "temperature": 0.7
}
```

**Streaming Request:**
```json
{
  "model": "research_agent",
  "messages": [
    {"role": "user", "content": "What's the current time in Tokyo?"}
  ],
  "stream": true,
  "temperature": 0.7
}
```

#### List Models
```bash
GET /v1/models
```
Returns all configured agents as available "models" for OpenAI-compatible clients.

### API Features

- **Agent Discovery**: Automatic discovery of agents from `agent_config.json`
- **Memory Integration**: Automatic conversation persistence and retrieval
- **Model Override**: Override agent's default model and settings via API
- **Session Management**: Maintain conversation history per user and session
- **Tool Integration**: Full tool calling support with agent-specific permissions
- **Multi-session Support**: Support for multiple concurrent sessions per user
- **Streaming Support**: Real-time streaming responses with Server-Sent Events (SSE)
- **Response Processing**: Automatic response cleaning and formatting for memory storage

### Example API Usage

```bash
# List all available agents
curl -X GET "http://localhost:8001/agents/"

# Get specific agent information
curl -X GET "http://localhost:8001/agents/research_agent"

# Send a message to an agent
curl -X POST "http://localhost:8001/agents/research_agent/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the current time in Tokyo?",
    "user_id": "user123",
    "session_id": "session456"
  }'

# Use OpenAI-compatible endpoint (non-streaming)
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "research_agent",
    "messages": [
      {"role": "user", "content": "What is the current time in Tokyo?"}
    ]
  }'

# Use OpenAI-compatible endpoint (streaming)
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "research_agent",
    "messages": [
      {"role": "user", "content": "What is the current time in Tokyo?"}
    ],
    "stream": true
  }'
```

### Agent Configuration

Agents are configured in `agent_config.json` (or `agent_config.example.json` for testing):

```json
[
  {
    "agent_id": "research_agent",
    "allowed_regular_tools": ["get_current_datetime"],
    "allowed_mcp_servers": {
      "deepwiki": {
        "allowed_mcp_tools": ["search_wiki"]
      },
      "fetch": {
        "allowed_mcp_tools": ["fetch_url"]
      },
      "searxng": {
        "allowed_mcp_tools": null  # all tools from searxng
      }
    }
  }
]
```

**Configuration Examples:**

**Full access to all tools from a server:**
```json
{
  "deepwiki": {
    "allowed_mcp_tools": null  # All tools from deepwiki
  }
}
```

**Limited access to specific tools:**
```json
{
  "github": {
    "allowed_mcp_tools": ["search_code", "get_repository"]  # Only these tools
  }
}
```

**No access to tools from a server:**
```json
{
  "searxng": {
    "allowed_mcp_tools": []  # No tools from searxng
  }
}
```

**All MCP servers with all tools (legacy behavior):**
```json
{
  "allowed_mcp_servers": null  # All servers with all tools
}
```

### Troubleshooting

**Agent not found:**
- Check `agent_config.json` (or `agent_config.example.json` for testing) exists and is valid JSON
- Verify agent_id matches configuration

**Provider not available:**
- Check provider credentials and configuration
- Ensure required environment variables are set

**Tools not working:**
- Verify MCP servers are running (for MCP tools)
- Check tool permissions in agent configuration
- For command-based servers: Ensure Docker containers are running and accessible if a docker based command
- **Authorization issues**: Verify environment variables are set for MCP server tokens (e.g., `GITHUB_TOKEN`)
- **Token substitution**: Ensure environment variables referenced in `mcp.json` (e.g., `${GITHUB_TOKEN}`) are properly set

**Prompt issues:**
- Verify prompt files exist in `prompts/` directory
- Check file permissions and encoding (UTF-8)
- Ensure prompt files are not empty

**Model configuration issues:**
- Verify model name is supported by the selected provider
- Check that model settings are valid for the provider (e.g., num_ctx for Ollama, max_tokens for Azure OpenAI)
- Ensure CLI parameter format is correct: `--setting key value`
- Model settings in agent config take precedence over provider defaults
- CLI settings take precedence over agent config settings

**Security and Authorization:**
- Verify environment variable substitution syntax is correct (e.g., `${GITHUB_TOKEN}`)
- Check that required environment variables are set in Docker containers
- Ensure `.env` files containing tokens are added to `.gitignore`
- **Environment variable substitution**: Check that `ToolRegistry.load_mcp_servers()` is properly processing environment variables
- **Token resolution**: Verify that environment variables are being resolved during MCP server loading
- **No code changes required**: MCP server tokens are handled dynamically via environment variable substitution

## 🧪 Testing

### Test Coverage
- **285+ tests** including unit and integration tests with BaseAgent architecture validation
- **BaseAgent architecture** - Memory behavior validation across agent inheritance hierarchy
- **Agent tool filtering** - Comprehensive permission testing
- **Resource management** - Memory resource CRUD operations and lifecycle
- **Memory compression management** - Comprehensive testing of compression logic, token counting, and conversation splitting
- **Memory compression agent** - Testing of AI-powered summarization and compression workflows
- **Agent resource filtering** - Per-agent resource access control
- **Prompt management** - System prompt loading and integration
- **MCP integration** - End-to-end tool execution
- **MCP server types** - HTTP-based and command-based server testing
- **fastmcp integration** - StdioTransport testing, error handling, and mixed server environments
- **Provider compatibility** - All providers tested
- **Model configuration** - Agent config model settings and CLI parameter overrides
- **Model settings flow** - Testing of settings passing from agent config to providers
- **Streaming functionality** - Comprehensive testing of streaming capabilities across all components
  - **Ollama Provider Streaming** - Testing of `send_chat_with_streaming` async generator
  - **Azure OpenAI Provider Streaming** - Testing of streaming capabilities for both Azure providers
  - **API Agent Streaming** - Testing of `chat_stream` method with memory integration
  - **OpenAI-Compatible Streaming** - Testing of SSE streaming endpoint with proper format validation
  - **Tool Call Streaming** - Testing of tool calls during streaming with max iteration limits
  - **Streaming Error Handling** - Testing of error scenarios in streaming contexts
  - **Streaming Memory Integration** - Testing of memory persistence during streaming
  - **Response Cleaning** - Testing of `chat_utils.py` response cleaning functionality

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_core/test_agents/     # Agent unit tests
pytest tests/test_core/test_resources/  # Resource tests
pytest tests/test_integration/          # Integration tests

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Examples
```bash
# Test resource management
pytest tests/test_core/test_resources/test_memory_resource.py

# Test memory compression management
pytest tests/test_core/test_resources/test_memory_compression_manager.py

# Test agent tool filtering
pytest tests/test_integration/test_agent_tool_filtering_integration.py

# Test prompt management
pytest tests/test_integration/test_prompt_management_integration.py

# Test MCP integration
pytest tests/test_integration/test_basic_chat_agent_integration.py

# Test fastmcp integration
pytest tests/test_core/test_agents/test_agent_tool_manager.py

# Test streaming functionality
pytest tests/test_core/test_providers/test_ollama_provider.py -k "streaming"
pytest tests/test_core/test_providers/test_azureopenapi_provider.py -k "streaming"
pytest tests/test_api/test_openai_compatible_integration.py -k "streaming"
```

## 🛠️ Development

### Adding New Endpoints

1. Create a new route file in `app/api/routes/`
2. Define your Pydantic models in `app/models/`
3. Include the router in `app/main.py`

### Using Agents with BaseAgent Architecture

```python
from app.core.agents.api_agent import APIAgent
from app.core.agents.cli_agent import CLIAgent

# Create an API agent (inherits from BaseAgent)
api_agent = APIAgent("research_agent", user_id="user123", session_id="session456")
await api_agent.initialize()

# Send message with automatic memory handling
response = await api_agent.chat("What's the current time in Tokyo?")

# Create a CLI agent (inherits from BaseAgent) 
cli_agent = CLIAgent("cli_agent", provider_id="azure_openai_cc")
await cli_agent.initialize()

# Interactive mode with memory support
await cli_agent.interactive_mode()
```

### BaseAgent Common Interface

All agents inherit these core methods from `BaseAgent`:

```python
# Memory operations (work only if memory is configured)
await agent.save_memory("user", "Hello")
history = await agent.load_memory()  # Returns [] if no memory configured

# Memory compression (automatic when thresholds are exceeded)
await agent._trigger_memory_compression(config)

# Response cleaning for memory storage
clean_text = agent._clean_response_for_memory(response)

# Initialization with provider and tool setup
await agent.initialize()
```

### Using Agents with Prompt Management

```python
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.agents.prompt_manager import PromptManager
from app.core.providers.manager import ProviderManager

# Create agent with prompt management
agent_manager = AgentToolManager("research_agent")
prompt_manager = PromptManager("research_agent")

# Get available tools and system prompt
tools = await agent_manager.get_available_tools()
system_prompt = prompt_manager.get_system_prompt_with_tools(tools)

# Use with any provider
provider = ProviderManager().get_provider("azure_openai_cc")
response = await provider.send_chat(
    context=[{"role": "user", "content": "What's the current time?"}],
    model="gpt-4",
    instructions=system_prompt,
    agent_id="research_agent"
```

### Creating Custom Agents

1. **Add agent configuration** to `agent_config.json` (or `agent_config.example.json` for testing)
2. **Create prompt file** in `prompts/` directory (optional)
3. **Define tool permissions** for the agent
4. **Test the agent** using the run script

Example:
```json
{
  "agent_id": "custom_agent",
  "system_prompt_file": "prompts/custom_agent.txt",
  "allowed_regular_tools": ["get_current_datetime", "add_two_numbers"],
  "allowed_mcp_servers": {
    "deepwiki": {
      "allowed_mcp_tools": ["search_wiki"]
    },
    "searxng": {
      "allowed_mcp_tools": null  # all tools from searxng
    }
  }
}
```

### Hot Reload
Development mode automatically reloads on code changes:
```bash
docker-compose --profile dev up
```

## 📦 Dependencies

Key dependencies include:
- **FastAPI** - Modern web framework
- **FastMCP** - Modern MCP Integration framework with StdioTransport support
- **Uvicorn** - ASGI server with uvloop for performance
- **Pydantic** - Data validation and settings management
- **Pydantic-settings** - Environment-based configuration
- **Pytest** - Testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the [API documentation](http://localhost:8001/docs) when running locally
2. Review the logs: `docker logs <container-name>`
3. Open an issue in this repository

---

**Happy coding!** 🚀