"""
Receipt Models

Schemas for recording external interactions. Receipts provide audit trails
for LLM calls, HTTP requests, and other external operations.

Key Design Principles:
1. request_hash and response_hash enable deterministic verification
2. Timestamps are in non-committed metadata (timing dict)
3. Receipts are optionally committed based on provenance tier
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.crypto.hashing import hash_canonical, to_hex


ReceiptKind = Literal["llm", "http", "rpc", "websocket", "other"]


class ReceiptRef(BaseModel):
    """
    Lightweight reference to a receipt.
    
    Used when full receipt data is stored separately (e.g., in checks/).
    """
    
    model_config = ConfigDict(extra="forbid")
    
    receipt_id: str = Field(
        ...,
        description="Unique receipt identifier",
    )
    kind: ReceiptKind = Field(
        ...,
        description="Type of receipt",
    )
    request_hash: str = Field(
        ...,
        description="Hash of the request (0x-prefixed)",
    )
    response_hash: str = Field(
        ...,
        description="Hash of the response (0x-prefixed)",
    )


class ReceiptTiming(BaseModel):
    """
    Timing information for a receipt.
    
    This is NON-COMMITTED metadata - excluded from hashing for determinism.
    """
    
    model_config = ConfigDict(extra="forbid")
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="When the operation started",
    )
    ended_at: Optional[datetime] = Field(
        default=None,
        description="When the operation completed",
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Duration in milliseconds",
    )


class Receipt(BaseModel):
    """
    Base receipt for any external interaction.
    
    Every external call (LLM, HTTP, RPC) produces a receipt that records:
    - What was requested (request dict + hash)
    - What was returned (response dict + hash)
    - Timing metadata (non-committed)
    """
    
    model_config = ConfigDict(extra="forbid")
    
    receipt_id: str = Field(
        ...,
        description="Unique identifier for this receipt",
    )
    kind: ReceiptKind = Field(
        ...,
        description="Type of external interaction",
    )
    request: dict[str, Any] = Field(
        ...,
        description="Request parameters/payload",
    )
    response: dict[str, Any] = Field(
        default_factory=dict,
        description="Response data",
    )
    request_hash: Optional[str] = Field(
        default=None,
        description="Hash of canonical request (0x-prefixed)",
    )
    response_hash: Optional[str] = Field(
        default=None,
        description="Hash of canonical response (0x-prefixed)",
    )
    timing: ReceiptTiming = Field(
        default_factory=ReceiptTiming,
        description="Timing metadata (non-committed)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if operation failed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    
    def compute_hashes(self) -> "Receipt":
        """Compute request and response hashes if not already set."""
        if self.request_hash is None:
            self.request_hash = to_hex(hash_canonical(self.request))
        if self.response_hash is None and self.response:
            self.response_hash = to_hex(hash_canonical(self.response))
        return self
    
    def to_ref(self) -> ReceiptRef:
        """Convert to a lightweight reference."""
        self.compute_hashes()
        return ReceiptRef(
            receipt_id=self.receipt_id,
            kind=self.kind,
            request_hash=self.request_hash or "",
            response_hash=self.response_hash or "",
        )
    
    @property
    def is_successful(self) -> bool:
        """Check if the operation completed successfully."""
        return self.error is None and self.response is not None


class LLMReceipt(Receipt):
    """
    Receipt specifically for LLM API calls.
    
    Captures model, provider, token usage, and response content.
    """
    
    kind: Literal["llm"] = "llm"
    
    # LLM-specific fields
    provider: str = Field(
        ...,
        description="LLM provider (openai, anthropic, google, etc.)",
    )
    model: str = Field(
        ...,
        description="Model identifier",
    )
    input_tokens: Optional[int] = Field(
        default=None,
        description="Number of input tokens",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Number of output tokens",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Why generation stopped",
    )


class HTTPReceipt(Receipt):
    """
    Receipt for HTTP requests.
    
    Captures method, URL, headers, status, and response body.
    """
    
    kind: Literal["http"] = "http"
    
    # HTTP-specific fields
    method: str = Field(
        ...,
        description="HTTP method (GET, POST, etc.)",
    )
    url: str = Field(
        ...,
        description="Request URL",
    )
    status_code: Optional[int] = Field(
        default=None,
        description="HTTP status code",
    )
    response_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Response headers",
    )


class RPCReceipt(Receipt):
    """
    Receipt for RPC calls (blockchain, custom protocols).
    """
    
    kind: Literal["rpc"] = "rpc"
    
    # RPC-specific fields
    endpoint: str = Field(
        ...,
        description="RPC endpoint URL",
    )
    rpc_method: str = Field(
        ...,
        description="RPC method name",
    )
    chain_id: Optional[str] = Field(
        default=None,
        description="Blockchain chain ID if applicable",
    )
