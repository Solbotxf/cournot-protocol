"""
HTTP Client

Provides a unified HTTP client with receipt recording for auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.receipts import ReceiptRecorder


@dataclass
class HttpResponse:
    """
    Response from an HTTP request.
    """
    status_code: int
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""
    elapsed_ms: float = 0.0
    receipt_id: Optional[str] = None
    
    @property
    def ok(self) -> bool:
        """Check if request was successful (2xx status)."""
        return 200 <= self.status_code < 300
    
    @property
    def text(self) -> str:
        """Get response content as text."""
        return self.content.decode("utf-8", errors="replace")
    
    def json(self) -> Any:
        """Parse response as JSON."""
        import json
        return json.loads(self.content)
    
    def raise_for_status(self) -> None:
        """Raise exception if status is not 2xx."""
        if not self.ok:
            raise HttpError(
                f"HTTP {self.status_code}",
                status_code=self.status_code,
                response=self,
            )


class HttpError(Exception):
    """HTTP request error."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[HttpResponse] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class HttpClient:
    """
    HTTP client with receipt recording.
    
    Usage:
        client = HttpClient(recorder=receipt_recorder)
        
        response = client.get("https://api.example.com/data")
        if response.ok:
            data = response.json()
    """
    
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        recorder: Optional["ReceiptRecorder"] = None,
        default_headers: Optional[dict[str, str]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        """
        Initialize HTTP client.

        Args:
            timeout: Default request timeout in seconds
            recorder: Receipt recorder for audit logging
            default_headers: Headers to include in all requests
            proxy: Proxy URL (e.g., "http://user:pass@host:port")
        """
        self.timeout = timeout
        self.recorder = recorder
        self.default_headers = default_headers or {}
        self.proxy = proxy
        self._session = None
    
    def _get_session(self):
        """Lazy-load requests session."""
        if self._session is None:
            try:
                import requests
            except ImportError:
                raise ImportError("requests package required: pip install requests")
            self._session = requests.Session()
            self._session.headers.update(self.default_headers)
            if self.proxy:
                self._session.proxies = {
                    "http": self.proxy,
                    "https": self.proxy,
                }
        return self._session
    
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """
        Make an HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Additional headers
            params: Query parameters
            data: Request body (form data)
            json: Request body (JSON)
            timeout: Request timeout
        
        Returns:
            HttpResponse with status, content, and headers
        """
        import requests
        
        session = self._get_session()
        effective_timeout = timeout or self.timeout
        
        # Merge headers
        request_headers = dict(self.default_headers)
        if headers:
            request_headers.update(headers)
        
        # Start receipt
        receipt = None
        if self.recorder:
            receipt = self.recorder.start_http_receipt(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                body=json or data,
            )
        
        try:
            response = session.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                data=data,
                json=json,
                timeout=effective_timeout,
            )
            
            result = HttpResponse(
                status_code=response.status_code,
                content=response.content,
                headers=dict(response.headers),
                url=str(response.url),
                elapsed_ms=response.elapsed.total_seconds() * 1000,
            )
            
            # Complete receipt
            if receipt and self.recorder:
                self.recorder.complete(
                    receipt,
                    response={
                        "status_code": response.status_code,
                        "content_length": len(response.content),
                        "content_type": response.headers.get("content-type"),
                    },
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                )
                result.receipt_id = receipt.receipt_id
            
            return result
            
        except requests.RequestException as e:
            if receipt and self.recorder:
                self.recorder.complete(receipt, error=str(e))
            raise HttpError(str(e)) from e
    
    def get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Make a GET request."""
        return self.request(
            "GET", url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    
    def post(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Make a POST request."""
        return self.request(
            "POST", url,
            headers=headers,
            params=params,
            data=data,
            json=json,
            timeout=timeout,
        )
    
    def put(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Make a PUT request."""
        return self.request(
            "PUT", url,
            headers=headers,
            params=params,
            data=data,
            json=json,
            timeout=timeout,
        )
    
    def delete(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Make a DELETE request."""
        return self.request(
            "DELETE", url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self) -> "HttpClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
