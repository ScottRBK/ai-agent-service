"""
Health check endpoints for monitoring service status.
"""
from fastapi import APIRouter
from app.models.health import HealthStatus
from app.utils.logging import logger
from datetime import datetime, timezone
from app.config.settings import settings
router = APIRouter()

@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint to verify the service is running.
    """
    health_status = HealthStatus(
        status="healthy",
        timestamp=datetime.now(tz=timezone.utc),
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION
    )
    logger.info(f"Health check response: {health_status}")
    return health_status