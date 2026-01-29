"""
HTTP Client Module

Provider-agnostic HTTP client with receipt recording.
"""

from .client import HttpClient, HttpResponse

__all__ = [
    "HttpClient",
    "HttpResponse",
]
