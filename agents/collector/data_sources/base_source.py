"""
Module 05 - Base Source Interface

Defines the adapter protocol for data sources and the FetchedArtifact dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from core.schemas import SourceTarget


@dataclass
class FetchedArtifact:
    """
    Result of a data fetch operation.
    
    Contains both raw and parsed content along with metadata
    about the fetch operation.
    """
    
    # Raw response content
    raw_bytes: bytes
    
    # Content type (json/text/html/bytes)
    content_type: str
    
    # Parsed content (dict/list/str or None if parsing failed)
    parsed: Optional[Any] = None
    
    # Response headers
    response_headers: dict[str, str] = field(default_factory=dict)
    
    # HTTP status code (if applicable)
    status_code: Optional[int] = None
    
    # Final URL after any redirects
    final_url: Optional[str] = None
    
    # Error message if fetch failed
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if fetch was successful."""
        return self.error is None and self.raw_bytes is not None
    
    @property
    def is_json(self) -> bool:
        """Check if content is JSON."""
        return self.content_type == "json" and self.parsed is not None


@runtime_checkable
class SourceProtocol(Protocol):
    """Protocol defining the source adapter interface."""
    
    source_id: str
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data from the source.
        
        Args:
            target: Source target specification
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with the result
        """
        ...


class BaseSource(ABC):
    """
    Abstract base class for data source adapters.
    
    Provides common functionality for fetching data from
    various sources with consistent error handling.
    """
    
    source_id: str = "base"
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the source adapter.
        
        Args:
            config: Optional configuration for the adapter
        """
        self.config = config or {}
    
    @abstractmethod
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data from the source.
        
        Args:
            target: Source target specification
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with the result
        """
        pass
    
    def _create_error_artifact(
        self,
        error: str,
        content_type: str = "bytes"
    ) -> FetchedArtifact:
        """
        Create an artifact representing a failed fetch.
        
        Args:
            error: Error message
            content_type: Expected content type
            
        Returns:
            FetchedArtifact with error set
        """
        return FetchedArtifact(
            raw_bytes=b"",
            content_type=content_type,
            parsed=None,
            error=error
        )
    
    def _parse_content(
        self,
        raw_bytes: bytes,
        content_type: str
    ) -> Any:
        """
        Parse raw bytes based on content type.
        
        Args:
            raw_bytes: Raw response bytes
            content_type: Expected content type
            
        Returns:
            Parsed content or None if parsing fails
        """
        import json
        
        try:
            if content_type == "json":
                return json.loads(raw_bytes.decode("utf-8"))
            elif content_type in ("text", "html"):
                return raw_bytes.decode("utf-8")
            else:
                return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    
    def supports_target(self, target: SourceTarget) -> bool:
        """
        Check if this adapter supports the given target.
        
        Args:
            target: Source target to check
            
        Returns:
            True if supported
        """
        return target.source_id == self.source_id
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source_id={self.source_id})"