"""
FastAPI application entry point for Agent Service.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config.settings import settings
from app.utils.logging import logger
from app.api.routes.health import router as health_router
from app.api.routes.agents import router as agents_router
from app.api.routes.openai_compatible import router as openai_router
from app.core.resources.manager import ResourceManager
from app.core.providers.manager import ProviderManager

# Global resource managers
resource_manager = ResourceManager()
provider_manager = ProviderManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting AI Agent Service...")
    
    # Initialize any global resources here if needed
    # For now, we'll just log startup
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent Service...")
    
    # Cleanup all resources
    await resource_manager.cleanup_all()
    
    # Clear provider caches
    for provider_info in provider_manager.providers.values():
        if hasattr(provider_info, 'cleanup'):
            try:
                await provider_info.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up provider: {e}")
    
    logger.info("AI Agent Service shutdown complete")

app = FastAPI(
    title=settings.SERVICE_NAME,
    description=settings.SERVICE_DESCRIPTION,
    version=settings.SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.include_router(health_router)
app.include_router(agents_router)
app.include_router(openai_router)

@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    logger.info("Root endpoint accessed")
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "agents": "/agents",
            "openai_compatible": "/v1",
            "health": "/health"
        }
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