"""
Module 01 - Schemas & Canonicalization
File: transport.py

Purpose: Tool call logs and tool planning schemas.
These models capture the planning and execution of data retrieval tools.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallRecord(BaseModel):
    """
    Record of a single tool call execution.

    Captures the inputs, outputs, timing, and any errors from tool execution.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(
        ...,
        description="Name/identifier of the tool that was called",
        min_length=1,
    )
    input: dict[str, Any] = Field(
        ...,
        description="Input parameters passed to the tool",
    )
    output: dict[str, Any] | None = Field(
        default=None,
        description="Output returned by the tool (None if not yet completed or failed)",
    )
    started_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when the tool call started",
    )
    ended_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when the tool call completed",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the tool call failed",
    )

    @property
    def is_completed(self) -> bool:
        """Check if the tool call has completed (with or without error)."""
        return self.ended_at is not None

    @property
    def is_successful(self) -> bool:
        """Check if the tool call completed successfully."""
        return self.is_completed and self.error is None

    @property
    def is_failed(self) -> bool:
        """Check if the tool call failed."""
        return self.error is not None


class ToolPlan(BaseModel):
    """
    Plan for tool execution to gather required evidence.

    This is created by the Prompt Engineer and executed by the Collector.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(
        ...,
        description="Unique identifier for this tool plan",
        min_length=1,
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="List of data requirements to fulfill",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of source IDs to query",
    )
    min_provenance_tier: int = Field(
        default=0,
        description="Minimum provenance tier required for evidence",
        ge=0,
    )
    allow_fallbacks: bool = Field(
        default=True,
        description="Whether to allow fallback sources if primary sources fail",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional plan parameters",
    )

    def has_source(self, source_id: str) -> bool:
        """Check if a source is in this plan."""
        return source_id in self.sources

    def meets_tier_requirement(self, tier: int) -> bool:
        """Check if a provenance tier meets the minimum requirement."""
        return tier >= self.min_provenance_tier


class ToolExecutionLog(BaseModel):
    """
    Log of all tool executions for a collection session.

    This aggregates multiple ToolCallRecords for audit purposes.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(
        ...,
        description="ID of the ToolPlan being executed",
    )
    calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="List of tool calls made",
    )
    started_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when execution started",
    )
    ended_at: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when execution completed",
    )

    @property
    def total_calls(self) -> int:
        """Total number of tool calls."""
        return len(self.calls)

    @property
    def successful_calls(self) -> int:
        """Number of successful tool calls."""
        return sum(1 for c in self.calls if c.is_successful)

    @property
    def failed_calls(self) -> int:
        """Number of failed tool calls."""
        return sum(1 for c in self.calls if c.is_failed)

    def add_call(self, record: ToolCallRecord) -> None:
        """Add a tool call record to the log."""
        self.calls.append(record)

    def get_calls_for_tool(self, tool: str) -> list[ToolCallRecord]:
        """Get all calls for a specific tool."""
        return [c for c in self.calls if c.tool == tool]