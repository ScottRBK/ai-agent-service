# AI Agent Service

A modern, intelligent AI Agent Service built with FastAPI that provides automated decision-making and AI-powered capabilities. Built with Docker, comprehensive logging, and production-ready features.

## ✨ Features

- **FastAPI Framework** - Modern, fast web framework with automatic API documentation
- **AI Agent Capabilities** - Intelligent automation and decision-making
- **Health Check Endpoints** - Built-in monitoring and status endpoints
- **Multi-Provider AI Support** - Azure OpenAI, Ollama with unified interface
- **MCP Integration** - Model Context Protocol for external tools
- **Tool Filtering** - Agent-specific permissions and authorization
- **Docker Support** - Multi-stage builds with development and production targets 
- **Environment Configuration** - Flexible settings with environment variable support
- **Structured Logging** - Comprehensive logging setup for debugging and monitoring
- **Type Safety** - Full type hints throughout the codebase
- **Auto-Generated Docs** - Interactive API documentation with Swagger UI and ReDoc
- **Hot Reload** - Development mode with automatic code reloading

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
│   │   │   └── agent_tool_manager.py    # Agent tool filtering
│   │   ├── providers/
│   │   │   ├── base.py                  # Base provider interface
│   │   │   ├── azureopenapi.py          # Azure OpenAI (Responses API)
│   │   │   ├── azureopenapi_cc.py       # Azure OpenAI (Chat Completions)
│   │   │   └── ollama.py                # Ollama provider
│   │   └── tools/
│   │       ├── tool_registry.py         # Tool management
│   │       └── function_calls/          # Built-in tools
│   ├── config/
│   ├── models/
│   └── utils/
├── tests/
│   ├── test_core/
│   │   ├── test_agents/                 # Agent unit tests
│   │   ├── test_providers/              # Provider tests
│   │   └── test_tools/                  # Tool tests
│   └── test_integration/                # End-to-end tests
├── agent_config.json                    # Agent configurations
├── mcp.json                            # MCP server config
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

### Environment Files

Create a `.env` file in the `docker/` directory to override defaults:

```env
# docker/.env
PORT=8001
LOG_LEVEL=INFO
DEV_CONTAINER_NAME=my-agent-service
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
```

## 🤖 AI Agent & Tool Management

### Agent Configuration
Configure agent-specific tool access via `agent_config.json`:

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

### MCP (Model Context Protocol) Integration
I have included two example MCP servers for examples
- **DeepWiki Server** - Interrogate information on Deepwiki page, https://docs.devin.ai/work-with-devin/deepwiki-mcp
- **Fetch Server** - An MCP server that provides web content fetching capabilities. This server enables LLMs to retrieve and process content from web pages, converting HTML to markdown for easier consumption 
- **Extensible** - Add custom MCP servers via `mcp.json`

### Tool Filtering
- **Agent-specific permissions** - Control which tools each agent can access
- **Regular tools** - Built-in functions like date/time, arithmetic, these are both just very basic examples 
- **MCP tools** - External server capabilities with proper authorization

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

## 🧪 Testing

```bash
# Install test dependencies (included in requirements.txt)
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app
```

### Example Test
```bash
# Test health endpoint
curl http://localhost:8001/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000000",
  "service": "AI Agent Service", 
  "version": "1.0.0"
}
```

## 🛠️ Development

### Adding New Endpoints

1. Create a new route file in `app/api/routes/`
2. Define your Pydantic models in `app/models/`
3. Include the router in `app/main.py`

Example:
### Using Agents with MCP Tools

```python
from app.core.agents.agent_tool_manager import AgentToolManager
from app.core.providers.manager import ProviderManager

# Create agent with specific tool access
agent_manager = AgentToolManager("research_agent")

# Get available tools (regular + MCP)
tools = await agent_manager.get_available_tools()

# Use with any provider
provider = ProviderManager().get_provider("azure_openai_cc")
response = await provider.send_chat(
    context=[{"role": "user", "content": "What's the current time?"}],
    model="gpt-4",
    instructions="You are a helpful assistant.",
    agent_id="research_agent"
)
```

### Hot Reload
Development mode automatically reloads on code changes:
```bash
docker-compose --profile dev up
```

## 📦 Dependencies

Key dependencies include:
- **FastAPI** - Modern web framework
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