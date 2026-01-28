"""
Module 05 - Web Source Adapter

Web scraping data source for fetching HTML content.
Currently implemented as HTTP-based fetch; browser automation
can be added later.
"""

from typing import Any, Optional

from core.schemas import SourceTarget

from agents.collector.data_sources.base_source import BaseSource, FetchedArtifact
from agents.collector.data_sources.http_source import HTTPSource


class WebSource(BaseSource):
    """
    Web scraping data source adapter.
    
    For now, implements simple HTTP-based web fetching.
    Browser automation (e.g., playwright) can be integrated later.
    """
    
    source_id: str = "web"
    
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        use_browser: bool = False
    ):
        """
        Initialize web source.
        
        Args:
            config: Configuration options
            use_browser: Whether to use browser automation (not implemented)
        """
        super().__init__(config)
        self.use_browser = use_browser
        
        # Use HTTP source as fallback
        self._http_source = HTTPSource(config)
    
    def fetch(
        self,
        target: SourceTarget,
        *,
        timeout_s: int = 20
    ) -> FetchedArtifact:
        """
        Fetch web page content.
        
        Args:
            target: Source target specification
            timeout_s: Request timeout in seconds
            
        Returns:
            FetchedArtifact with the result
        """
        if self.use_browser:
            # Browser automation not yet implemented
            return self._create_error_artifact(
                "Browser automation not implemented. "
                "Set use_browser=False to use HTTP fetch.",
                target.expected_content_type
            )
        
        # Use HTTP source with web-appropriate headers
        web_headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; CournotCollector/1.0; "
                "+https://cournot.ai)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        # Merge with target headers (target takes precedence)
        merged_headers = {**web_headers, **target.headers}
        
        # Create modified target
        modified_target = SourceTarget(
            source_id="http",  # Use HTTP source
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
        
        return self._http_source.fetch(modified_target, timeout_s=timeout_s)
    
    def supports_target(self, target: SourceTarget) -> bool:
        """Check if this adapter supports the target."""
        return (
            target.source_id == "web" and
            target.method in ("GET", "POST")
        )
    
    def extract_text(self, artifact: FetchedArtifact) -> Optional[str]:
        """
        Extract text content from HTML artifact.
        
        Args:
            artifact: Fetched HTML artifact
            
        Returns:
            Extracted text or None if extraction fails
        """
        if not artifact.success or artifact.content_type != "html":
            return None
        
        try:
            # Simple text extraction (can be enhanced with BeautifulSoup)
            html_content = artifact.parsed or artifact.raw_bytes.decode("utf-8")
            
            # Basic HTML stripping (placeholder - use BeautifulSoup in production)
            import re
            
            # Remove script and style elements
            text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            
            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()
            
            return text
        except Exception:
            return None