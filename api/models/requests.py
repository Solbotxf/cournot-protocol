"""
Module 09D - API Request Models

Pydantic models for API request validation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Request body for POST /run endpoint."""
    
    user_input: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The prediction market query to resolve",
    )
    strict_mode: bool = Field(
        default=True,
        description="Enable strict mode for deterministic hashing",
    )
    enable_sentinel_verify: bool = Field(
        default=True,
        description="Enable sentinel verification after pipeline execution",
    )
    enable_replay: bool = Field(
        default=False,
        description="Enable replay mode for sentinel verification",
    )
    return_format: Literal["json", "pack_zip"] = Field(
        default="json",
        description="Response format: 'json' for structured data, 'pack_zip' for artifact pack",
    )
    include_checks: bool = Field(
        default=False,
        description="Include detailed verification checks in response",
    )


class VerifyRequest(BaseModel):
    """Query parameters for POST /verify endpoint."""
    
    include_checks: bool = Field(
        default=False,
        description="Include detailed verification checks in response",
    )
    enable_sentinel: bool = Field(
        default=True,
        description="Enable sentinel verification",
    )


class ReplayRequest(BaseModel):
    """Query parameters for POST /replay endpoint."""
    
    timeout_s: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout in seconds for replay operations",
    )
    include_checks: bool = Field(
        default=False,
        description="Include detailed verification checks in response",
    )