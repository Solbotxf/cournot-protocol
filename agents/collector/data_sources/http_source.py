"""
Module 05 - HTTP Source Adapter

HTTP/HTTPS data source for fetching data via GET/POST requests.
"""

import json
from typing import Any, Optional

from core.schemas import SourceTarget

from agents.collector.data_sources.base_source import BaseSource, FetchedArtifact


class HTTPSource(BaseSource):
    """
    HTTP/HTTPS data source adapter.
    
    Supports GET and POST requests with configurable headers,
    parameters, and body.
    """
    
    source_id: str = "http"
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        session: Optional[Any] = None
    ):
        """
        Initialize HTTP source.
        
        Args:
            config: Configuration options
            session: Optional requests session for connection pooling
        """
        super().__init__(config)
        self._session = session
    
    def _get_session(self) -> Any:
        """Get or create HTTP session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data via HTTP request.
        
        Args:
            target: Source target specification
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with the result
        """
        import requests
        
        try:
            session = self._get_session()
            
            # Prepare request kwargs
            kwargs: dict[str, Any] = {
                "timeout": timeout_s,
                "headers": dict(target.headers) if target.headers else {},
            }
            
            # Add params for GET requests
            if target.params:
                kwargs["params"] = dict(target.params)
            
            # Add body for POST requests
            if target.method == "POST" and target.body:
                if isinstance(target.body, dict):
                    kwargs["json"] = target.body
                else:
                    kwargs["data"] = target.body
            
            # Make request
            if target.method == "GET":
                response = session.get(target.uri, **kwargs)
            elif target.method == "POST":
                response = session.post(target.uri, **kwargs)
            else:
                return self._create_error_artifact(
                    f"Unsupported HTTP method: {target.method}",
                    target.expected_content_type
                )
            
            # Get raw content
            raw_bytes = response.content
            
            # Parse content based on expected type
            parsed = self._parse_content(
                raw_bytes,
                target.expected_content_type
            )
            
            # Extract response headers
            response_headers = dict(response.headers)
            
            return FetchedArtifact(
                raw_bytes=raw_bytes,
                content_type=target.expected_content_type,
                parsed=parsed,
                response_headers=response_headers,
                status_code=response.status_code,
                final_url=response.url,
                error=None if response.ok else f"HTTP {response.status_code}"
            )
            
        except requests.Timeout:
            return self._create_error_artifact(
                f"Request timeout after {timeout_s}s",
                target.expected_content_type
            )
        except requests.ConnectionError as e:
            return self._create_error_artifact(
                f"Connection error: {e}",
                target.expected_content_type
            )
        except requests.RequestException as e:
            return self._create_error_artifact(
                f"Request failed: {e}",
                target.expected_content_type
            )
        except Exception as e:
            return self._create_error_artifact(
                f"Unexpected error: {e}",
                target.expected_content_type
            )
    
    def supports_target(self, target: SourceTarget) -> bool:
        """Check if this adapter supports the target."""
        return (
            target.source_id in ("http", "https", "api") and
            target.method in ("GET", "POST")
        )


class APISource(HTTPSource):
    """
    API-specific HTTP source.
    
    Extends HTTPSource with API-specific defaults like
    JSON content type and authentication handling.
    """
    
    source_id: str = "api"
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        session: Optional[Any] = None,
        default_headers: Optional[dict[str, str]] = None
    ):
        """
        Initialize API source.
        
        Args:
            config: Configuration options
            session: Optional requests session
            default_headers: Default headers to include in all requests
        """
        super().__init__(config, session)
        self.default_headers = default_headers or {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch data from API endpoint.
        
        Merges default headers with target headers.
        """
        # Create modified target with default headers
        merged_headers = {**self.default_headers, **target.headers}
        
        modified_target = SourceTarget(
            source_id=target.source_id,
            uri=target.uri,
            method=target.method,
            expected_content_type=target.expected_content_type,
            headers=merged_headers,
            params=target.params,
            body=target.body,
            auth_ref=target.auth_ref,
            cache_ttl_seconds=target.cache_ttl_seconds,
            notes=target.notes
        )
        
        return super().fetch(modified_target, timeout_s=timeout_s)