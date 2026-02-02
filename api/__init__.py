"""
Module 09D - Minimal API (FastAPI)

HTTP API for the Cournot protocol:
- POST /run - Execute pipeline
- POST /verify - Verify uploaded pack
- POST /replay - Replay evidence and verify
- GET /health - Health check

Usage:
    uvicorn api.app:app --reload
"""

__version__ = "0.1.0"
