"""
FastAPI application entry point for Agent Service.
"""
from app.config.settings import settings
from app.utils.logging import logger
from app.api.routes.health import router as health_router
from fastapi import FastAPI
app = FastAPI(
    title="Agent Service",
    description="A lightweight microservice that provides AI Agent Capabilities through multiple providers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health_router)

@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    logger.info("Root endpoint accessed")
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Agent Service", 
                host=settings.HOST, 
                port=settings.PORT, 
                debug=settings.DEBUG)
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    ) 