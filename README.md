# Agent Service

A modern, lightweight FastAPI microservice template that provides. Built with Docker, comprehensive logging, and production-ready features.

## ✨ Features

- **FastAPI Framework** - Modern, fast web framework with automatic API documentation
- **Health Check Endpoints** - Built-in monitoring and status endpoints
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
git clone <your-repo-url>
cd agent-service

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

## 📁 Project Structure 

```
agent-service/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── health.py         # Health check endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Application configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── health.py             # Pydantic models
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logging.py            # Logging configuration
│   ├── __init__.py
│   └── main.py                   # FastAPI application entry point
├── docker/
│   ├── Dockerfile                # Multi-stage Docker build
│   ├── docker-compose.yml        # Development orchestration
│   └── .dockerignore
├── tests/
│   └── test_health.py            # Test suite
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## 🔧 Configuration

The application uses environment-based configuration with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | "Agent Service" | Name of the service |
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
  "service": "Agent Service",
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
docker build -f docker/Dockerfile --target production -t agent-service:latest .
docker run -p 8000:8000 agent-service:latest
```

### Docker Environment Variables
```bash
# Override default port
docker run -e PORT=8001 -p 8001:8001 agent-service:latest
```

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
  "service": "Agent Service", 
  "version": "1.0.0"
}
```

## 🛠️ Development

### Adding New Endpoints

1. Create a new route file in `app/api/routes/`
2. Define your Pydantic models in `app/models/`
3. Include the router in `app/main.py`

Example:
```python
# app/api/routes/example.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/example")
async def example_endpoint():
    return {"message": "Hello World"}

# app/main.py
from app.api.routes.example import router as example_router
app.include_router(example_router)
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