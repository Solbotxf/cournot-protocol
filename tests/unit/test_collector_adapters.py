"""
Tests for collector adapters: operation (fetch/search), evidence_id, and HttpAdapter branching.
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from core.schemas import EvidenceItem, SourceTarget
from agents.collector.adapters import HttpAdapter, MockAdapter, get_adapter
from agents.context import AgentContext, FrozenClock


def _minimal_ctx() -> AgentContext:
    """Minimal context for adapter tests (no HTTP needed for MockAdapter)."""
    return AgentContext(
        http=None,
        clock=FrozenClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
    )


class TestEvidenceIdWithOperation:
    """evidence_id must differ when operation or search_query differ (same uri)."""

    def test_evidence_id_different_for_same_uri_different_operation(self):
        """Same requirement_id, source_id, uri but different operation -> different evidence_id."""
        ctx = _minimal_ctx()
        adapter = MockAdapter()
        req_id = "req_001"
        uri = "https://example.org/page"
        target_fetch = SourceTarget(source_id="http", uri=uri)
        target_search = SourceTarget(
            source_id="http",
            uri=uri,
            operation="search",
            search_query="site:example.org query",
        )
        ev_fetch = adapter.fetch(ctx, target_fetch, req_id)
        ev_search = adapter.fetch(ctx, target_search, req_id)
        assert ev_fetch.evidence_id != ev_search.evidence_id

    def test_evidence_id_different_for_same_operation_different_search_query(self):
        """Same operation 'search' but different search_query -> different evidence_id."""
        ctx = _minimal_ctx()
        adapter = MockAdapter()
        req_id = "req_001"
        target1 = SourceTarget(
            source_id="http",
            uri="https://example.org",
            operation="search",
            search_query="query A",
        )
        target2 = SourceTarget(
            source_id="http",
            uri="https://example.org",
            operation="search",
            search_query="query B",
        )
        ev1 = adapter.fetch(ctx, target1, req_id)
        ev2 = adapter.fetch(ctx, target2, req_id)
        assert ev1.evidence_id != ev2.evidence_id

    def test_evidence_id_same_for_same_target(self):
        """Same target (including operation/search_query) -> same evidence_id."""
        ctx = _minimal_ctx()
        adapter = MockAdapter()
        target = SourceTarget(
            source_id="http",
            uri="https://example.org",
            operation="search",
            search_query="same query",
        )
        ev1 = adapter.fetch(ctx, target, "req_001")
        ev2 = adapter.fetch(ctx, target, "req_001")
        assert ev1.evidence_id == ev2.evidence_id


class TestHttpAdapterOperationBranch:
    """HttpAdapter branches on operation: fetch vs search."""

    def test_operation_fetch_uses_target_uri(self):
        """When operation is None or 'fetch', request uses target.uri."""
        mock_http = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = '{"data": 1}'
        mock_response.content = b'{"data": 1}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.receipt_id = None
        mock_http.get.return_value = mock_response
        ctx = AgentContext(
            http=mock_http,
            clock=FrozenClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
        )
        adapter = HttpAdapter()
        target = SourceTarget(
            source_id="http",
            uri="https://api.example.com/data",
            method="GET",
        )
        ev = adapter.fetch(ctx, target, "req_001")
        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert call_args[0][0] == "https://api.example.com/data"
        assert ev.success is True

    def test_operation_search_uses_effective_uri(self):
        """When operation is 'search' and search_query is set, request uses Google search URL."""
        mock_http = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.text = "<html>search results</html>"
        mock_response.content = b"<html>search results</html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.receipt_id = None
        mock_http.get.return_value = mock_response
        ctx = AgentContext(
            http=mock_http,
            clock=FrozenClock(datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)),
        )
        adapter = HttpAdapter()
        target = SourceTarget(
            source_id="http",
            uri="https://www.example.org",
            operation="search",
            search_query="site:example.org exact phrase",
        )
        ev = adapter.fetch(ctx, target, "req_001")
        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        effective_url = call_args[0][0]
        assert "google.com/search" in effective_url
        assert "site%3Aexample.org" in effective_url or "site:example.org" in effective_url
        assert ev.provenance.source_uri == "https://www.example.org"
