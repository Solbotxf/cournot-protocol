"""
Module 09D - Health Check Route

Simple health check endpoint for liveness probes.
"""

from fastapi import APIRouter

from api.models.responses import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status for liveness probes.
    """
    return HealthResponse(
        ok=True,
        service="cournot-protocol-api",
        version="v1",
    )


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """
    Root endpoint - same as health check.
    """
    return HealthResponse(
        ok=True,
        service="cournot-protocol-api",
        version="v1",
    )
