"""
Module 09D - API Error Handling

Standardized error handling for the API.
"""

from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from api.models.responses import ErrorResponse, ErrorDetail


class APIError(Exception):
    """Base API error with structured response."""
    
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            ok=False,
            error=ErrorDetail(
                code=self.code,
                message=self.message,
                details=self.details,
            ),
        )


class InvalidRequestError(APIError):
    """Invalid request parameters."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="INVALID_REQUEST",
            message=message,
            status_code=400,
            details=details,
        )


class MissingFileError(APIError):
    """Required file not provided."""
    
    def __init__(self, message: str = "Required file not provided"):
        super().__init__(
            code="MISSING_FILE",
            message=message,
            status_code=400,
        )


class InvalidPackError(APIError):
    """Invalid or corrupted pack file."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="INVALID_PACK",
            message=message,
            status_code=400,
            details=details,
        )


class VerificationFailedError(APIError):
    """Verification failed."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            code="VERIFICATION_FAILED",
            message=message,
            status_code=400,
            details=details,
        )


class InternalError(APIError):
    """Internal server error."""
    
    def __init__(self, message: str = "Internal server error", details: dict[str, Any] | None = None):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            status_code=500,
            details=details,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response().model_dump(),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            ok=False,
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred",
                details={"type": type(exc).__name__},
            ),
        ).model_dump(),
    )