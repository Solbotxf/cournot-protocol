"""
Module 09D - FastAPI Application

Main application setup and configuration.

Usage:
    uvicorn api.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, run, verify, replay
from api.errors import APIError, api_error_handler, generic_error_handler


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Cournot Protocol API",
        description="HTTP API for the Cournot prediction market resolution protocol",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(run.router)
    app.include_router(verify.router)
    app.include_router(replay.router)
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)