"""
Receipt Recorder

Provides a unified interface for recording external interactions.
Used by AgentContext to track all LLM, HTTP, and RPC calls.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    HTTPReceipt,
    LLMReceipt,
    Receipt,
    ReceiptKind,
    ReceiptRef,
    ReceiptTiming,
    RPCReceipt,
)


def generate_receipt_id(kind: ReceiptKind, request_data: dict[str, Any]) -> str:
    """
    Generate a deterministic receipt ID from kind and request data.
    
    Format: rc_{kind}_{hash_prefix}
    """
    # Create a stable string representation
    stable_str = f"{kind}|{sorted(request_data.items())}"
    hash_hex = hashlib.sha256(stable_str.encode()).hexdigest()[:12]
    return f"rc_{kind}_{hash_hex}"


class ReceiptRecorder:
    """
    Records receipts for external interactions.
    
    Usage:
        recorder = ReceiptRecorder()
        
        # Start recording an operation
        receipt = recorder.start_llm_receipt(provider="openai", model="gpt-4", ...)
        
        # ... perform operation ...
        
        # Complete the receipt
        recorder.complete(receipt, response={...})
        
        # Get all receipts
        receipts = recorder.get_receipts()
    """
    
    def __init__(self) -> None:
        self._receipts: list[Receipt] = []
        self._in_progress: dict[str, Receipt] = {}
    
    def start_llm_receipt(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMReceipt:
        """Start recording an LLM call."""
        request = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        
        receipt = LLMReceipt(
            receipt_id=generate_receipt_id("llm", request),
            provider=provider,
            model=model,
            request=request,
            timing=ReceiptTiming(started_at=datetime.now(timezone.utc)),
        )
        
        self._in_progress[receipt.receipt_id] = receipt
        return receipt
    
    def start_http_receipt(
        self,
        *,
        method: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        body: Optional[Any] = None,
        **kwargs: Any,
    ) -> HTTPReceipt:
        """Start recording an HTTP request."""
        request = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "params": params or {},
            "body": body,
            **kwargs,
        }
        
        receipt = HTTPReceipt(
            receipt_id=generate_receipt_id("http", request),
            method=method,
            url=url,
            request=request,
            timing=ReceiptTiming(started_at=datetime.now(timezone.utc)),
        )
        
        self._in_progress[receipt.receipt_id] = receipt
        return receipt
    
    def start_rpc_receipt(
        self,
        *,
        endpoint: str,
        rpc_method: str,
        params: Optional[list[Any]] = None,
        chain_id: Optional[str] = None,
        **kwargs: Any,
    ) -> RPCReceipt:
        """Start recording an RPC call."""
        request = {
            "endpoint": endpoint,
            "method": rpc_method,
            "params": params or [],
            "chain_id": chain_id,
            **kwargs,
        }
        
        receipt = RPCReceipt(
            receipt_id=generate_receipt_id("rpc", request),
            endpoint=endpoint,
            rpc_method=rpc_method,
            chain_id=chain_id,
            request=request,
            timing=ReceiptTiming(started_at=datetime.now(timezone.utc)),
        )
        
        self._in_progress[receipt.receipt_id] = receipt
        return receipt
    
    def complete(
        self,
        receipt: Receipt,
        *,
        response: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        **extra_fields: Any,
    ) -> Receipt:
        """
        Complete a receipt with response data or error.
        
        Args:
            receipt: The receipt to complete
            response: Response data (dict)
            error: Error message if failed
            **extra_fields: Additional fields to set on the receipt
        
        Returns:
            The completed receipt
        """
        # Update timing
        now = datetime.now(timezone.utc)
        receipt.timing.ended_at = now
        if receipt.timing.started_at:
            delta = now - receipt.timing.started_at
            receipt.timing.duration_ms = delta.total_seconds() * 1000
        
        # Set response or error
        if response is not None:
            receipt.response = response
        if error is not None:
            receipt.error = error
        
        # Set any extra fields (e.g., input_tokens for LLM)
        for key, value in extra_fields.items():
            if hasattr(receipt, key):
                setattr(receipt, key, value)
        
        # Compute hashes
        receipt.compute_hashes()
        
        # Move from in-progress to completed
        if receipt.receipt_id in self._in_progress:
            del self._in_progress[receipt.receipt_id]
        
        self._receipts.append(receipt)
        return receipt
    
    def record_complete(
        self,
        kind: ReceiptKind,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        error: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Receipt:
        """
        Record a complete receipt in one call (for cases where timing is known).
        """
        receipt_id = generate_receipt_id(kind, request)
        
        timing = ReceiptTiming(
            started_at=started_at,
            ended_at=ended_at or datetime.now(timezone.utc),
        )
        if timing.started_at and timing.ended_at:
            timing.duration_ms = (timing.ended_at - timing.started_at).total_seconds() * 1000
        
        receipt = Receipt(
            receipt_id=receipt_id,
            kind=kind,
            request=request,
            response=response,
            timing=timing,
            error=error,
            metadata=kwargs,
        )
        
        receipt.compute_hashes()
        self._receipts.append(receipt)
        return receipt
    
    def get_receipts(self) -> list[Receipt]:
        """Get all completed receipts."""
        return list(self._receipts)
    
    def get_receipt_refs(self) -> list[ReceiptRef]:
        """Get lightweight references to all completed receipts."""
        return [r.to_ref() for r in self._receipts]
    
    def get_in_progress(self) -> list[Receipt]:
        """Get receipts that haven't been completed yet."""
        return list(self._in_progress.values())
    
    def clear(self) -> None:
        """Clear all receipts."""
        self._receipts.clear()
        self._in_progress.clear()
    
    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert all receipts to JSON-serializable dicts."""
        return [r.model_dump(mode="json", exclude_none=True) for r in self._receipts]
