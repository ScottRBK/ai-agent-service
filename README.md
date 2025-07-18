# AI Agent Service

A modern, intelligent AI Agent Service framework built with FastAPI & FastMCP that demonstrates how to implement agent tool management, prompt handling, and multi-provider AI integration from scratch. This project showcases a production-ready implementation of agent-specific tool filtering, dynamic system prompts, and unified provider interfaces without relying on abstraction frameworks. Built with Docker, comprehensive logging, and enterprise-grade features.

## ✨ Features

- **Framework Design** - Complete implementation showing how to build AI agents from the ground up
- **FastAPI Framework** - Modern, fast web framework with automatic API documentation
- **AI Agent Capabilities** - Intelligent automation and decision-making
- **Health Check Endpoints** - Built-in monitoring and status endpoints
- **Multi-Provider AI Support** - Azure OpenAI, Ollama with unified interface
- **MCP Integration** - Model Context Protocol for external tools
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
- **Agent Resource Manager** - Per-agent resource access control and automatic resource creation

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
│   │       └── health.py
│   ├── core/
│   │   ├── agents/
│   │   │   ├── agent_tool_manager.py    # Agent tool filtering
│   │   │   ├── agent_resource_manager.py # Agent resource management
│   │   │   └── prompt_manager.py        # System prompt management
│   │   ├── providers/
│   │   │   ├── base.py                  # Base provider interface
│   │   │   ├── azureopenapi.py          # Azure OpenAI (Responses API)
│   │   │   ├── azureopenapi_cc.py       # Azure OpenAI (Chat Completions)
│   │   │   └── ollama.py                # Ollama provider
│   │   ├── resources/
│   │   │   ├── base.py                  # Base resource interface
│   │   │   ├── manager.py               # Global resource management
│   │   │   └── memory.py                # PostgreSQL memory resource
│   │   └── tools/
│   │       ├── tool_registry.py         # Tool management
│   │       └── function_calls/          # Built-in tools
│   ├── config/
│   ├── models/
│   │   └── resources/
│   │       └── memory.py                # Memory data models
│   └── utils/
├── tests/
│   ├── test_core/
│   │   ├── test_agents/                 # Agent unit tests
│   │   ├── test_providers/              # Provider tests
│   │   ├── test_resources/              # Resource tests
│   │   └── test_tools/                  # Tool tests
│   └── test_integration/                # End-to-end tests
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
```

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
Configure agent-specific tool access, resources, and model settings via `agent_config.json`:

```json
[
  {
    "agent_id": "research_agent",
    "system_prompt_file": "prompts/research_agent.txt",
    "allowed_regular_tools": ["get_current_datetime"],
    "allowed_mcp_servers": ["deepwiki", "fetch"],
    "allowed_mcp_tools": {
      "deepwiki": ["read_wiki_structure", "search_wiki"],
      "fetch": ["fetch_url"]
    },
    "resources": ["memory"],
    "provider": "azure_openai_cc",
    "model": "gpt-4o-mini",
    "model_settings": {
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }
]
```
**Resource Configuration:**
- `resources` - Array of resource types available to the agent
- Automatic resource creation when agents request access
- Memory resources provide conversation persistence across sessions

**Model Configuration:**
- `model` - AI model identifier (e.g., "gpt-4o-mini", "qwen3:4b")
- `model_settings` - Provider-specific parameters (temperature, max_tokens, num_ctx, etc.)
- Settings are passed directly to the provider without validation for maximum flexibility

### MCP (Model Context Protocol) Integration
I have included two example MCP servers for examples
- **DeepWiki Server** - Interrogate information on Deepwiki page, https://docs.devin.ai/work-with-devin/deepwiki-mcp
- **Fetch Server** - An MCP server that provides web content fetching capabilities. This server enables LLMs to retrieve and process content from web pages, converting HTML to markdown for easier consumption 
- **Extensible** - Add custom MCP servers via `mcp.json`

### Tool Filtering
- **Agent-specific permissions** - Control which tools each agent can access
- **Regular tools** - Built-in functions like date/time, arithmetic, these are both just very basic examples 
- **MCP tools** - External server capabilities with proper authorization

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
- **Tool calling** - Execute functions and MCP tools
- **Agent filtering** - Provider-agnostic tool management
- **Health monitoring** - Provider status and metrics
- **Error handling** - Robust error management
- **Model settings** - Flexible parameter handling for provider-specific options
  - Azure OpenAI: temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, seed, response_format
  - Ollama: num_ctx, num_predict, top_k, repeat_penalty, repeat_last_n, temperature, top_p
  - Settings are passed directly to providers without validation for maximum flexibility
  - Support for both agent config and CLI parameter overrides

## 🤖 Running Agents

### Quick Start with CLI Agent

Use the example script to run different agents with various providers and memory:

```bash
# Run the research agent with Azure OpenAI and memory
python examples/run_agent.py research_agent azure_openai_cc

# Run the CLI agent with conversation memory
python examples/run_agent.py cli_agent azure_openai_cc

# Run the MCP-only agent
python examples/run_agent.py mcp_agent azure_openai_cc

# Override model and parameters via CLI
python examples/run_agent.py cli_agent ollama --model qwen3:4b --setting num_ctx 200 --setting num_predict 200 --setting temperature 0.7

# Use Azure OpenAI with custom settings
python examples/run_agent.py research_agent azure_openai_cc --model gpt-4o-mini --setting temperature 0.8 --setting max_tokens 3000
```

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

### Available Agents

| Agent | Description | Tools Available |
|-------|-------------|-----------------|
| `research_agent` | Research assistant with web access | datetime, deepwiki, fetch |
| `data_agent` | Data analysis specialist | datetime, arithmetic |
| `mcp_agent` | MCP tools only | deepwiki, fetch |
| `restricted_agent` | Limited access example | specific tools only |

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
💬 Type 'quit' to exit

You: What's the current time in Tokyo?
🤔 Thinking...
🤖 research_agent: The current time in Tokyo is 2025-01-16 15:30:45.

You: Research Python async programming
🤔 Thinking...
🤖 research_agent: I'll search for information about Python async programming...

You: quit
👋 Goodbye!
```

Invoking a specific model and model parameters: 

```bash
$ python examples/run_agent.py cli_agent azure_openai_cc --model gpt-4o-mini --setting temperature 0.7 --setting max_tokens 2000
```
### Agent Configuration

Agents are configured in `agent_config.json`:

```json
[
  {
    "agent_id": "research_agent",
    "allowed_regular_tools": ["get_current_datetime"],
    "allowed_mcp_servers": ["deepwiki", "fetch"],
    "allowed_mcp_tools": {
      "deepwiki": ["read_wiki_structure", "search_wiki"],
      "fetch": ["fetch_url"]
    }
  }
]
```
### Troubleshooting

**Agent not found:**
- Check `agent_config.json` exists and is valid JSON
- Verify agent_id matches configuration

**Provider not available:**
- Check provider credentials and configuration
- Ensure required environment variables are set

**Tools not working:**
- Verify MCP servers are running (for MCP tools)
- Check tool permissions in agent configuration

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

## 🧪 Testing

### Test Coverage
- **103+ tests** including unit and integration tests
- **Agent tool filtering** - Comprehensive permission testing
- **Resource management** - Memory resource CRUD operations and lifecycle
- **Agent resource filtering** - Per-agent resource access control
- **Prompt management** - System prompt loading and integration
- **MCP integration** - End-to-end tool execution
- **Provider compatibility** - All providers tested
- **Model configuration** - Agent config model settings and CLI parameter overrides
- **Model settings flow** - Testing of settings passing from agent config to providers

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

# Test agent tool filtering
pytest tests/test_integration/test_agent_tool_filtering_integration.py

# Test prompt management
pytest tests/test_integration/test_prompt_management_integration.py

# Test MCP integration
pytest tests/test_integration/test_basic_chat_agent_integration.py
```

## 🛠️ Development

### Adding New Endpoints

1. Create a new route file in `app/api/routes/`
2. Define your Pydantic models in `app/models/`
3. Include the router in `app/main.py`

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
)
```

### Creating Custom Agents

1. **Add agent configuration** to `agent_config.json`
2. **Create prompt file** in `prompts/` directory (optional)
3. **Define tool permissions** for the agent
4. **Test the agent** using the run script

Example:
```json
{
  "agent_id": "custom_agent",
  "system_prompt_file": "prompts/custom_agent.txt",
  "allowed_regular_tools": ["get_current_datetime", "add_two_numbers"],
  "allowed_mcp_servers": ["deepwiki"],
  "allowed_mcp_tools": {
    "deepwiki": ["search_wiki"]
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
- **FastMCP** - Modern MCP Integration framework
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
