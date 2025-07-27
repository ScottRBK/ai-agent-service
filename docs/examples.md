# Usage Examples

This document provides comprehensive examples for using the AI Agent Service in various ways.

## CLI Usage Examples

### Basic Agent Execution
```bash
# Run research agent with Azure OpenAI (includes Searxng web search)
python examples/run_agent.py research_agent azure_openai_cc

# Run CLI agent with memory (full MCP access including command-based servers)
python examples/run_agent.py cli_agent azure_openai_cc

# Run MCP-only agent (HTTP and command-based MCP servers)
python examples/run_agent.py mcp_agent azure_openai_cc

# Run agent with Ollama provider
python examples/run_agent.py cli_agent ollama --model qwen3:4b --setting temperature 0.7
```

### CLI Parameter Overrides
```bash
# Override model and settings
python examples/run_agent.py research_agent azure_openai_cc \
  --model gpt-4o-mini \
  --setting temperature 0.8 \
  --setting max_tokens 3000

# Multiple setting overrides
python examples/run_agent.py cli_agent ollama \
  --model llama3:8b \
  --setting temperature 0.9 \
  --setting num_ctx 4096 \
  --setting top_p 0.95

# Override with Ollama-specific parameters
python examples/run_agent.py research_agent ollama \
  --model codellama:13b \
  --setting num_predict 1000 \
  --setting repeat_penalty 1.1 \
  --setting top_k 40
```

**Parameter Override Priority:**
1. CLI arguments (highest priority)
2. Agent configuration file
3. Provider defaults (lowest priority)

## Programmatic Usage Examples

### Using BaseAgent Directly
```python
from app.core.agents.cli_agent import CLIAgent
from app.core.agents.api_agent import APIAgent
from app.core.agents.base_agent import BaseAgent

# Create and run agent with default config
agent = CLIAgent("research_agent", "azure_openai_cc")
await agent.interactive_mode()

# Create agent with custom model and settings
agent = CLIAgent(
    agent_id="research_agent", 
    provider_id="azure_openai_cc",
    model="gpt-4o-mini",
    model_settings={"temperature": 0.8, "max_tokens": 2000}
)
await agent.interactive_mode()

# Create API agent with memory (when configured in agent_config.json)
api_agent = APIAgent("cli_agent", user_id="user123", session_id="session456")
response = await api_agent.chat("Hello, how are you?")
print(response)
```

### Custom Agent Implementation
```python
from app.core.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, provider_id: str = None, **kwargs):
        super().__init__(agent_id, provider_id, **kwargs)
    
    async def custom_method(self):
        # Inherits all BaseAgent functionality
        await self.initialize()
        return await self.chat("Custom message")
    
    async def process_batch(self, messages: list):
        """Process multiple messages in sequence"""
        results = []
        for msg in messages:
            response = await self.chat(msg)
            results.append(response)
        return results

# Usage
custom_agent = CustomAgent("research_agent", "azure_openai_cc")
await custom_agent.initialize()
result = await custom_agent.custom_method()
```

### Memory Management Examples
```python
from app.core.agents.api_agent import APIAgent

# Create agent with memory support
agent = APIAgent("cli_agent", user_id="user123", session_id="session456")

# Chat with automatic memory persistence
response1 = await agent.chat("My name is John")
response2 = await agent.chat("What is my name?")  # Will remember from previous message

# Clear conversation history
await agent.clear_conversation()

# Get conversation history
history = await agent.get_conversation_history()
for entry in history:
    print(f"{entry['role']}: {entry['content']}")
```

## Agent Configuration Examples

### Research Agent Configuration
```json
{
  "agent_id": "research_agent",
  "allowed_regular_tools": ["get_current_datetime"],
  "allowed_mcp_servers": {
    "deepwiki": {
      "allowed_mcp_tools": ["search_wiki"]
    },
    "searxng": {
      "allowed_mcp_tools": null
    },
    "fetch": {
      "allowed_mcp_tools": ["fetch_url", "fetch_rss"]
    }
  },
  "resources": ["memory"],
  "provider": "azure_openai_cc",
  "model": "gpt-4o-mini",
  "model_settings": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

### CLI Agent Configuration
```json
{
  "agent_id": "cli_agent",
  "allowed_regular_tools": ["get_current_datetime"],
  "allowed_mcp_servers": {
    "deepwiki": {"allowed_mcp_tools": null},
    "fetch": {"allowed_mcp_tools": null},
    "searxng": {"allowed_mcp_tools": null}
  },
  "resources": ["memory"],
  "provider": "azure_openai_cc",
  "model": "gpt-4o",
  "model_settings": {
    "temperature": 0.5,
    "max_tokens": 4000
  }
}
```

### Restricted Agent Configuration
```json
{
  "agent_id": "restricted_agent",
  "allowed_regular_tools": ["get_current_datetime"],
  "allowed_mcp_servers": {
    "deepwiki": {
      "allowed_mcp_tools": ["search_wiki"]
    }
  },
  "resources": [],
  "provider": "ollama",
  "model": "llama3:8b",
  "model_settings": {
    "temperature": 0.1,
    "num_ctx": 2048
  }
}
```

## MCP Server Configuration Examples

### HTTP-Based MCP Servers
```json
[
  {
    "server_label": "github",
    "server_url": "https://api.githubcopilot.com/mcp/",
    "header": {
      "authorization": "${GITHUB_TOKEN}"
    }
  },
  {
    "server_label": "deepwiki",
    "server_url": "https://deepwiki-mcp.example.com/",
    "header": {
      "api-key": "${DEEPWIKI_API_KEY}"
    }
  },
  {
    "server_label": "fetch",
    "server_url": "https://fetch-mcp.example.com/"
  }
]
```

### Command-Based MCP Servers
```json
[
  {
    "server_label": "searxng",
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-e", "SEARXNG_URL=http://searxng:8089",
      "isokoliuk/mcp-searxng:latest"
    ],
    "require_approval": "never"
  },
  {
    "server_label": "local_tools",
    "command": "python",
    "args": ["-m", "local_mcp_server"],
    "require_approval": "once"
  }
]
```

### Mixed Environment Configuration
```json
[
  {
    "server_label": "github",
    "server_url": "https://api.githubcopilot.com/mcp/",
    "header": {
      "authorization": "${GITHUB_TOKEN}"
    }
  },
  {
    "server_label": "searxng",
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-e", "SEARXNG_URL=http://searxng:8089",
      "isokoliuk/mcp-searxng:latest"
    ],
    "require_approval": "never"
  },
  {
    "server_label": "fetch",
    "server_url": "https://fetch-mcp.example.com/"
  }
]
```

## Docker Development Examples

### Basic Development Setup
```bash
# Start development environment
cd docker
docker-compose --profile dev up --build

# Access services
# API: http://localhost:8001
# Health: http://localhost:8001/health
# Docs: http://localhost:8001/docs
# Open WebUI: http://localhost:3000
```

### Custom Docker Compose
```yaml
version: '3.8'
services:
  ai-agent-service:
    build: .
    ports:
      - "8001:8000"
    environment:
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - DATABASE_URL=postgresql://agent_user:secure_password@postgres:5432/agent_service
    volumes:
      - ./agent_config.json:/app/agent_config.json
      - ./mcp.json:/app/mcp.json
      - ./prompts:/app/prompts
    depends_on:
      - postgres
    networks:
      - ai-network

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=secure_password
      - POSTGRES_DB=agent_service
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ai-network

volumes:
  postgres_data:

networks:
  ai-network:
    driver: bridge
```

## Testing Examples

### Unit Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_core/test_agents/     # Agent unit tests
pytest tests/test_core/test_resources/  # Resource tests
pytest tests/test_integration/          # Integration tests

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Test specific components
pytest tests/test_core/test_agents/test_base_agent.py -v
pytest tests/test_core/test_providers/test_azureopenapi_cc_provider.py -v
```

### Integration Testing
```bash
# Test agent tool filtering
pytest tests/test_integration/test_agent_tool_filtering_integration.py

# Test memory integration
pytest tests/test_integration/test_memory_integration.py

# Test API integration
pytest tests/test_integration/test_api_integration.py
```

### Evaluation Testing
```bash
# Test evaluation system
pytest tests/test_evaluation/

# Test specific evaluation components
pytest tests/test_evaluation/test_scenario_models.py      # Scenario Pydantic models
pytest tests/test_evaluation/test_yaml_parser_integration.py  # YAML parsing
pytest tests/test_evaluation/test_template_generator.py   # Template generation
pytest tests/test_evaluation/test_e2e_scenario_workflow.py # End-to-end workflow
```

## Evaluation Examples

The service includes comprehensive AI agent evaluation capabilities using DeepEval framework:

### Running Evaluation Examples
```bash
# Tool correctness evaluation
python evaluations/examples/tool_correctness.py

# Hallucination detection evaluation
python evaluations/examples/hallucination.py

# Summarization quality evaluation
python evaluations/examples/summarization.py

# Custom GEval metrics with observability
python evaluations/examples/geval_observe.py

# Generate synthetic test data with tools
python evaluations/examples/synthesizer_with_tools.py --generate

# Run evaluation with existing golden dataset
python evaluations/examples/synthesizer_with_tools.py

# Generate synthetic data from scratch
python evaluations/examples/synthesizer_from_scratch.py

# Task completion evaluation with observability
python evaluations/examples/task_completion_observe.py
```

### Evaluation Categories

**Tool Correctness** - Validates that agents select and use appropriate tools for given tasks
**Hallucination Detection** - Measures factual accuracy and identifies false information generation
**Summarization Quality** - Assesses summary accuracy, conciseness, and completeness
**GEval Metrics** - Custom evaluation criteria with real-time observability and tracing
**Synthetic Data Generation** - Creates test datasets from contexts and generates data from scratch
**Task Completion** - Evaluates agent performance on specific tasks with detailed observability

## Environment Variable Examples

### Development Environment
```bash
# .env.development
SERVICE_NAME=ai-agent-service-dev
SERVICE_VERSION=1.0.0-dev
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://agent_user:dev_password@localhost:5432/agent_service_dev

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_dev_key
AZURE_OPENAI_ENDPOINT=https://your-dev-resource.openai.azure.com/

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# MCP Tokens
GITHUB_TOKEN=your_github_token
DEEPWIKI_API_KEY=your_deepwiki_key
```

### Production Environment
```bash
# .env.production
SERVICE_NAME=ai-agent-service
SERVICE_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://agent_user:secure_prod_password@prod-db:5432/agent_service

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_prod_key
AZURE_OPENAI_ENDPOINT=https://your-prod-resource.openai.azure.com/

# MCP Tokens
GITHUB_TOKEN=your_prod_github_token
DEEPWIKI_API_KEY=your_prod_deepwiki_key

# Memory Configuration
MEMORY_COMPRESSION_TOKEN_THRESHOLD=8000
MEMORY_CLEANUP_INTERVAL_HOURS=24
```

## Advanced Usage Patterns

### Batch Processing
```python
from app.core.agents.api_agent import APIAgent
import asyncio

async def batch_process_questions(questions: list, agent_id: str):
    """Process multiple questions with the same agent"""
    agent = APIAgent(agent_id, user_id="batch_user", session_id="batch_session")
    
    results = []
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}")
        response = await agent.chat(question)
        results.append({
            "question": question,
            "response": response,
            "index": i
        })
    
    return results

# Usage
questions = [
    "What is the capital of France?",
    "Explain quantum computing",
    "What are the benefits of renewable energy?"
]

results = asyncio.run(batch_process_questions(questions, "research_agent"))
```

### Multi-Agent Workflow
```python
from app.core.agents.api_agent import APIAgent
import asyncio

async def multi_agent_research(topic: str):
    """Use multiple agents for comprehensive research"""
    
    # Research agent gathers information
    research_agent = APIAgent("research_agent", user_id="multi", session_id="research")
    research_result = await research_agent.chat(f"Research the topic: {topic}")
    
    # API agent structures the information
    api_agent = APIAgent("api_agent", user_id="multi", session_id="structure")
    structured_result = await api_agent.chat(
        f"Structure this research information: {research_result}"
    )
    
    return {
        "topic": topic,
        "raw_research": research_result,
        "structured_output": structured_result
    }

# Usage
result = asyncio.run(multi_agent_research("Artificial Intelligence in Healthcare"))
```

### Custom Tool Integration
```python
from app.core.agents.base_agent import BaseAgent
from app.core.tools.tool_registry import ToolRegistry

class CustomToolAgent(BaseAgent):
    async def initialize_custom_tools(self):
        """Add custom tools to the agent"""
        await self.initialize()
        
        # Custom tool function
        def custom_calculator(expression: str) -> str:
            """Safely evaluate mathematical expressions"""
            try:
                # Simple expression evaluation (in real use, use ast.literal_eval)
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Add custom tool to agent
        self.tool_manager.add_custom_tool("custom_calculator", custom_calculator)
    
    async def calculate(self, expression: str):
        """Use custom calculator tool"""
        return await self.chat(f"Calculate: {expression}")

# Usage
agent = CustomToolAgent("cli_agent", "azure_openai_cc")
await agent.initialize_custom_tools()
result = await agent.calculate("2 + 2 * 3")
```