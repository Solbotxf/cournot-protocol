"""
Image deliverable validator.
Pre-checks image URLs before/alongside PoR pipeline evaluation.
"""
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("cournot.image_validator")

ALLOWED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/webp", "image/svg+xml", "image/bmp", "image/tiff",
}

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB


@dataclass
class ImageCheckResult:
    url: str
    reachable: bool
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    is_valid_image: bool = False
    content_hash: Optional[str] = None  # SHA-256 of first 4KB for fingerprint
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def evidence_summary(self) -> str:
        if not self.reachable:
            return f"URL unreachable: {self.error}"
        parts = [
            f"HTTP {self.http_status}",
            f"Content-Type: {self.content_type}",
            f"Size: {self.content_length or 'unknown'} bytes",
            f"Valid image: {self.is_valid_image}",
            f"Latency: {self.latency_ms:.0f}ms" if self.latency_ms else None,
        ]
        return " | ".join(p for p in parts if p)


def check_image_url(url: str, timeout: float = 15.0) -> ImageCheckResult:
    """
    Validate an image URL:
    1. URL format check
    2. HTTP HEAD request (reachability + status + Content-Type)
    3. Partial GET for content hash fingerprint
    """
    # URL format check
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ImageCheckResult(
            url=url, reachable=False,
            error=f"Invalid scheme: {parsed.scheme}"
        )

    result = ImageCheckResult(url=url, reachable=False)

    try:
        start = time.monotonic()

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            # HEAD request first (lightweight)
            head_resp = client.head(url)
            result.http_status = head_resp.status_code
            result.content_type = head_resp.headers.get("content-type", "").split(";")[0].strip().lower()
            
            cl = head_resp.headers.get("content-length")
            if cl and cl.isdigit():
                result.content_length = int(cl)

            result.latency_ms = (time.monotonic() - start) * 1000
            result.reachable = True

            # Check if valid image
            if head_resp.status_code == 200 and result.content_type in ALLOWED_IMAGE_TYPES:
                result.is_valid_image = True

                # Partial GET for content fingerprint (first 4KB)
                try:
                    get_resp = client.get(url, headers={"Range": "bytes=0-4095"})
                    chunk = get_resp.content[:4096]
                    result.content_hash = hashlib.sha256(chunk).hexdigest()
                except Exception:
                    pass  # fingerprint is optional

            elif head_resp.status_code == 200:
                # Got 200 but wrong content type - might still be image
                # Some CDNs return generic content-type
                result.error = f"Unexpected Content-Type: {result.content_type}"
            else:
                result.error = f"HTTP {head_resp.status_code}"

    except httpx.TimeoutException:
        result.error = "Timeout"
    except httpx.ConnectError as e:
        result.error = f"Connection failed: {str(e)[:100]}"
    except Exception as e:
        result.error = f"Error: {str(e)[:100]}"

    return result


def build_image_evidence(check: ImageCheckResult) -> dict:
    """
    Convert image check result into evidence format
    compatible with Cournot's evaluation pipeline.
    """
    return {
        "source": "image_url_validation",
        "url": check.url,
        "reachable": check.reachable,
        "http_status": check.http_status,
        "content_type": check.content_type,
        "content_length": check.content_length,
        "is_valid_image": check.is_valid_image,
        "content_hash": check.content_hash,
        "latency_ms": check.latency_ms,
        "error": check.error,
        "evidence_summary": check.evidence_summary,
    }
