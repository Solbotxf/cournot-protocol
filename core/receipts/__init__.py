"""
Core Receipts Module

Provides receipt recording for all external interactions (LLM, HTTP, RPC).
Receipts enable reproducibility, auditing, and replay verification.
"""

from .models import (
    Receipt,
    ReceiptKind,
    LLMReceipt,
    HTTPReceipt,
    RPCReceipt,
    ReceiptRef,
)
from .recorder import ReceiptRecorder

__all__ = [
    "Receipt",
    "ReceiptKind",
    "LLMReceipt",
    "HTTPReceipt",
    "RPCReceipt",
    "ReceiptRef",
    "ReceiptRecorder",
]
