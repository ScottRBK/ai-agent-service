# Multi-stage build for Python FastAPI application
FROM python:3.12-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Production stage
FROM python:3.12-slim as production

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Copy application code
COPY app/ /app/app/
COPY pyproject.toml /app/
COPY ai_agent_service.egg-info/ /app/ai_agent_service.egg-info/

# Install the package in editable mode
RUN pip install -e /app

# Create config directory for mounted files
RUN mkdir -p /app/config

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM production as development

# Install development dependencies
RUN pip install pytest pytest-asyncio httpx

# Enable hot reload and development features
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

