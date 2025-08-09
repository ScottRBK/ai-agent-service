# Deployment and Operations

This document covers deployment strategies, operational considerations, and system behavior characteristics.

## Docker Deployment

### Development Environment
```bash
# Start development environment with all services
cd docker
docker-compose --profile dev up --build

# Access services
# API: http://localhost:8001
# Health: http://localhost:8001/health
# Docs: http://localhost:8001/docs
# Open WebUI: http://localhost:3000
```

### Production Build
The service uses a multi-stage Dockerfile optimized for production:

```dockerfile
# Multi-stage build with production target
FROM python:3.11-slim as production

# Optimized layers for minimal image size
# Security hardening included
# Health checks configured
```

### Environment Configuration

#### Required Environment Variables
```bash
# Service Configuration
SERVICE_NAME=ai-agent-service
SERVICE_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/agent_service
POSTGRES_USER=agent_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=agent_service

# Provider Credentials
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OLLAMA_BASE_URL=http://localhost:11434

# OpenRouter Configuration
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_DEFAULT_MODEL=openrouter/auto
OPENROUTER_MODEL_LIST=openrouter/auto,meta-llama/llama-3.1-8b-instruct,openai/gpt-4

# MCP Server Tokens (Dynamic Substitution)
GITHUB_TOKEN=your_github_token
DEEPWIKI_API_KEY=your_deepwiki_key
```

#### Optional Environment Variables
```bash
# Memory Configuration
MEMORY_COMPRESSION_TOKEN_THRESHOLD=8000
MEMORY_CLEANUP_INTERVAL_HOURS=24

# Tool Configuration
MCP_TOOL_CACHE_TTL=3600
TOOL_EXECUTION_TIMEOUT=30

# Evaluation Configuration
EVALUATION_OUTPUT_DIR=evaluations/output
DEEPEVAL_TELEMETRY_OPT_OUT=NO  # Set to YES to disable telemetry to Confident AI
CONFIDENT_TRACING_ENABLED=true  # Set to false to disable tracing for Confident AI

# Performance Tuning
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
```

## Open WebUI Integration

### Compatibility Overview
The AI Agent Service is fully compatible with Open WebUI through OpenAI protocol compliance:
- **Standard Endpoints**: Implements `/v1/chat/completions` and `/v1/models`
- **Message Format**: Uses standard OpenAI message format
- **Response Format**: Returns responses in OpenAI-compatible format
- **Tool Calling**: Full support for function calling with agent-specific tools
- **Streaming Support**: Real-time streaming responses for enhanced user experience
- **Agent-as-Model Concept**: Each agent appears as a "model" in Open WebUI

### Configuration Methods

#### Web Interface Configuration
1. **Access Open WebUI**: Navigate to the Open WebUI web interface
2. **Add Provider**: Go to Settings → Providers → Add Provider
3. **Configure Provider**: 
   - Provider Type: OpenAI
   - Base URL: `http://your-service:8001/v1`
   - API Key: (leave empty or use any value)
4. **Import Models**: Models will be automatically discovered and available

#### Docker Environment Variables
```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -e CUSTOM_PROVIDERS='[{"id":"ai-agent-service","name":"AI Agent Service","base_url":"http://ai-agent-service:8001/v1","api_key":"not-needed","models":[{"id":"research_agent","name":"Research Agent"},{"id":"cli_agent","name":"CLI Agent"}]}]' \
  ghcr.io/open-webui/open-webui:main
```

#### Docker Compose Configuration
```yaml
version: '3.8'
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - CUSTOM_PROVIDERS='[{"id":"ai-agent-service","name":"AI Agent Service","base_url":"http://ai-agent-service:8001/v1","api_key":"not-needed"}]'
    networks:
      - ai-network

  ai-agent-service:
    build: .
    ports:
      - "8001:8000"
    volumes:
      - ./agent_config.json:/app/agent_config.json
      - ./prompts:/app/prompts
    networks:
      - ai-network

networks:
  ai-network:
    driver: bridge
```

## Infrastructure Requirements

### Minimum System Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB
- **Network**: Outbound HTTPS access for MCP servers

### Recommended Production Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ (for logs and database)
- **Network**: High-bandwidth for streaming responses
- **Database**: PostgreSQL 12+ (separate instance recommended)

### Scaling Considerations
- **Horizontal Scaling**: Stateless design supports load balancing
- **Database Scaling**: PostgreSQL read replicas for memory queries
- **Resource Isolation**: Each agent maintains separate tool and resource contexts
- **Connection Pooling**: Built-in database connection management

## System Behavior Characteristics

### Security and Permissions
- **Agent-specific tool access control**: Each agent has granular control over which tools it can access
- **MCP server authorization**: Both HTTP and command-based servers support secure authentication
- **Environment Variable Substitution**: Secure token management through `${VARIABLE_NAME}` syntax
- **Authorization Headers**: HTTP-based MCP servers support secure authentication
- **Token Security**: Sensitive credentials stored as environment variables, not in configuration files
- **Session isolation**: Memory and conversation history isolated per user/session
- **Configuration-Driven Security**: Memory and resource access controlled via agent configuration

### Performance Optimization
- **Tool caching per agent**: Improves response times for repeated operations
- **Connection pooling**: Efficient database connection management
- **Async/await throughout**: Non-blocking operations across the entire codebase
- **Efficient memory management**: Automatic cleanup and compression
- **fastmcp optimization**: Efficient StdioTransport handling for command-based MCP servers
- **Streaming optimization**: Real-time response delivery with minimal latency

### Reliability Features
- **Comprehensive error handling**: Graceful degradation on failures
- **Health monitoring**: Detailed health checks for all system components
- **Automatic resource cleanup**: Prevents resource leaks
- **Graceful degradation**: System continues operating with reduced functionality
- **fastmcp lifecycle management**: Proper StdioTransport startup and cleanup
- **Streaming reliability**: Robust error handling and recovery for streaming scenarios

### Extensibility
- **Plugin-based tool system**: Easy addition of new tools and capabilities
- **Configurable MCP servers**: Support for both HTTP and command-based servers
- **Custom agent definitions**: Create new agent types through configuration
- **Provider abstraction layer**: Support for multiple AI providers
- **Dynamic MCP Server Addition**: Add new MCP servers with authorization by editing only `mcp.json` and setting environment variables
- **Streaming extensibility**: Easy addition of streaming support to new providers
- **Unified Agent Architecture**: Easy creation of new agent types by inheriting from BaseAgent
- **Configuration-Driven Extensibility**: Add new agent capabilities through configuration without code changes

## Monitoring and Observability

### Health Checks
```bash
# Basic health check
curl http://localhost:8001/health

# Detailed health with provider status
curl http://localhost:8001/health?detailed=true
```

### Logging
- **Structured logging**: JSON format for log aggregation
- **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Request tracing**: Unique request IDs for tracking
- **Performance metrics**: Response times and resource usage

### Metrics Collection
- **Request metrics**: Count, duration, status codes
- **Memory usage**: Conversation storage and compression statistics
- **Tool performance**: Execution times and success rates
- **Provider metrics**: Model usage and response quality

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump -U agent_user -h localhost agent_service > backup.sql

# Restore
psql -U agent_user -h localhost agent_service < backup.sql
```

### Configuration Backup
```bash
# Backup configuration files
tar -czf config-backup.tar.gz agent_config.json mcp.json prompts/
```

### Disaster Recovery
- **Database replication**: PostgreSQL streaming replication
- **Configuration management**: Version control for all configuration
- **Stateless design**: Quick recovery with container restart
- **Memory reconstruction**: Conversation history preserved in database

## Security Considerations

### Network Security
- **TLS/HTTPS**: Required for production deployments
- **Firewall rules**: Restrict access to necessary ports only
- **Network segmentation**: Isolate service components

### Data Protection
- **Encryption at rest**: Database encryption for conversation history
- **Encryption in transit**: TLS for all external communications
- **Data retention**: Configurable conversation history retention policies

### Access Control
- **API authentication**: Will be added at a later date
- **Authorization headers**: MCP server authentication support
- **Environment isolation**: Separate environments for dev/staging/prod

## Troubleshooting

### Common Issues

#### Service Won't Start
1. Check environment variables
2. Verify database connectivity
3. Check MCP server configurations
4. Review log files for specific errors

#### MCP Server Connection Issues
1. Verify environment variable substitution
2. Check network connectivity to HTTP-based servers
3. Validate Docker container access for command-based servers
4. Review MCP server authorization headers

#### Memory/Database Issues
1. Check PostgreSQL connection
2. Verify database schema
3. Monitor memory usage patterns
4. Review compression settings

#### Performance Issues
1. Monitor resource usage
2. Check tool execution times
3. Review database query performance
4. Analyze streaming response latency

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m app.main --debug
```

### Support Resources
- **Health endpoint**: Real-time system status
- **API documentation**: `/docs` and `/redoc` endpoints
- **Log analysis**: Structured JSON logs for debugging
- **Configuration validation**: Startup validation of all configurations