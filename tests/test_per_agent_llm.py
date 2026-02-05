"""Tests for per-agent LLM configuration."""

from __future__ import annotations

import copy
from unittest.mock import patch, MagicMock

import pytest

from core.config.runtime import AgentConfig, AgentsConfig, LLMConfig, RuntimeConfig
from agents.context import AgentContext


# ── from_dict parsing ────────────────────────────────────────────────────────


class TestFromDictParsesLLMOverride:
    def test_converts_nested_dict_to_llm_config(self):
        data = {
            "agents": {
                "collector": {
                    "name": "CollectorLLM",
                    "llm_override": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-20250514",
                    },
                }
            }
        }
        cfg = RuntimeConfig.from_dict(data)
        override = cfg.agents.collector.llm_override
        assert isinstance(override, LLMConfig)
        assert override.provider == "anthropic"
        assert override.model == "claude-sonnet-4-20250514"

    def test_no_override_stays_none(self):
        data = {
            "agents": {
                "collector": {
                    "name": "CollectorNetwork",
                }
            }
        }
        cfg = RuntimeConfig.from_dict(data)
        assert cfg.agents.collector.llm_override is None

    def test_multiple_agents_with_overrides(self):
        data = {
            "agents": {
                "collector": {
                    "name": "CollectorLLM",
                    "llm_override": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                    },
                },
                "auditor": {
                    "name": "AuditorLLM",
                    "llm_override": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-20250514",
                    },
                },
            }
        }
        cfg = RuntimeConfig.from_dict(data)
        assert cfg.agents.collector.llm_override.provider == "openai"
        assert cfg.agents.auditor.llm_override.provider == "anthropic"
        # Agents without override should remain None
        assert cfg.agents.judge.llm_override is None


# ── with_llm_override ───────────────────────────────────────────────────────


class TestWithLLMOverride:
    @patch("core.llm.create_provider")
    @patch("core.llm.LLMClient")
    def test_returns_new_ctx_with_different_llm(self, MockLLMClient, mock_create_provider):
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        mock_llm = MagicMock()
        MockLLMClient.return_value = mock_llm

        original_ctx = AgentContext.create_minimal()
        original_llm = original_ctx.llm

        override_config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514", api_key="test-key")
        new_ctx = original_ctx.with_llm_override(override_config)

        assert new_ctx is not original_ctx
        assert new_ctx.llm is mock_llm
        assert new_ctx.llm is not original_llm

        mock_create_provider.assert_called_once_with(
            "anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            base_url=None,
            proxy=None,
        )

    @patch("core.llm.create_provider")
    @patch("core.llm.LLMClient")
    def test_shares_recorder(self, MockLLMClient, mock_create_provider):
        mock_create_provider.return_value = MagicMock()
        MockLLMClient.return_value = MagicMock()

        original_ctx = AgentContext.create_minimal()
        override_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key")
        new_ctx = original_ctx.with_llm_override(override_config)

        assert new_ctx.recorder is original_ctx.recorder

    @patch("core.llm.create_provider")
    @patch("core.llm.LLMClient")
    def test_shares_clock_and_cache(self, MockLLMClient, mock_create_provider):
        mock_create_provider.return_value = MagicMock()
        MockLLMClient.return_value = MagicMock()

        original_ctx = AgentContext.create_minimal()
        override_config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key")
        new_ctx = original_ctx.with_llm_override(override_config)

        assert new_ctx.clock is original_ctx.clock
        assert new_ctx.cache is original_ctx.cache


# ── _resolve_ctx ─────────────────────────────────────────────────────────────


class TestResolveCtx:
    def _make_pipeline(self):
        from orchestrator.pipeline import Pipeline, PipelineConfig
        return Pipeline(config=PipelineConfig())

    def test_no_override_returns_same_ctx(self):
        from agents.base import AgentStep

        pipeline = self._make_pipeline()
        cfg = RuntimeConfig.from_dict({})
        ctx = AgentContext.create_minimal()
        ctx.config = cfg

        result = pipeline._resolve_ctx(ctx, AgentStep.COLLECTOR)
        assert result is ctx

    @patch("core.llm.create_provider")
    @patch("core.llm.LLMClient")
    def test_with_override_returns_new_ctx(self, MockLLMClient, mock_create_provider):
        from agents.base import AgentStep

        mock_create_provider.return_value = MagicMock()
        mock_llm = MagicMock()
        MockLLMClient.return_value = mock_llm

        pipeline = self._make_pipeline()
        cfg = RuntimeConfig.from_dict({
            "agents": {
                "collector": {
                    "name": "CollectorLLM",
                    "llm_override": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-20250514",
                        "api_key": "test-key",
                    },
                }
            }
        })
        ctx = AgentContext.create_minimal()
        ctx.config = cfg

        result = pipeline._resolve_ctx(ctx, AgentStep.COLLECTOR)
        assert result is not ctx
        assert result.llm is mock_llm
        assert result.recorder is ctx.recorder

    def test_no_config_returns_same_ctx(self):
        from agents.base import AgentStep

        pipeline = self._make_pipeline()
        ctx = AgentContext.create_minimal()
        ctx.config = None

        result = pipeline._resolve_ctx(ctx, AgentStep.COLLECTOR)
        assert result is ctx
