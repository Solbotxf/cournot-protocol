"""
Module 09D - FastAPI Application

Main application setup and configuration.

Usage:
    uvicorn api.app:app --reload
    
    # Or run directly
    python -m api.app
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, run, verify, replay, steps
from api.errors import APIError, api_error_handler, generic_error_handler


# Configure logging — respects COURNOT_LOG_LEVEL env var and cournot.json log_level
def _resolve_log_level() -> int:
    """Resolve log level from env var or cournot.json, defaulting to INFO."""
    raw = os.getenv("COURNOT_LOG_LEVEL")
    if raw is None:
        try:
            import json
            from pathlib import Path
            cfg_path = Path.cwd() / "cournot.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    raw = json.load(f).get("log_level")
        except Exception:
            pass
    return getattr(logging, (raw or "INFO").upper(), logging.INFO)


logging.basicConfig(
    level=_resolve_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Cournot Protocol API",
        description="""
HTTP API for the Cournot prediction market resolution protocol.

## Endpoints

- **POST /run** - Execute the full pipeline on a query
- **POST /verify** - Verify an uploaded artifact pack
- **POST /replay** - Replay evidence and verify against original
- **GET /health** - Health check

## Response Formats

The `/run` endpoint supports two response formats:
- `json` - Structured JSON with artifacts and verification results
- `pack_zip` - ZIP file containing the artifact pack

## Execution Modes

- `production` - Fail-closed, requires all capabilities
- `development` - Uses fallback agents when needed (default)
- `test` - Deterministic mock agents only
        """,
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
    app.include_router(steps.router)
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)