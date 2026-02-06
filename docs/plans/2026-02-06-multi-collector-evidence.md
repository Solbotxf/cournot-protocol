# Multi-Collector Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multiple collectors to gather evidence in parallel, with the auditor and judge evaluating all evidence bundles together.

**Architecture:** Add `collector_name` and `weight` fields to `EvidenceBundle`. Change auditor/judge/pipeline to accept `list[EvidenceBundle]` instead of single bundle. API `/step/collect` accepts list of collectors; API `/step/audit` and `/step/judge` accept list of evidence bundles.

**Tech Stack:** Python, Pydantic, FastAPI

---

## Task 1: Add collector_name and weight to EvidenceBundle

**Files:**
- Modify: `core/schemas/evidence.py:101-130`

**Step 1: Add new fields to EvidenceBundle**

```python
# In EvidenceBundle class, add after plan_id field (line ~113):
    collector_name: str | None = Field(
        default=None,
        description="Name of the collector agent that produced this bundle",
    )
    weight: float = Field(
        default=1.0,
        description="Weight for evidence aggregation (1.0 = equal weight)",
        ge=0.0,
        le=10.0,
    )
```

**Step 2: Run test to verify schema validates**

Run: `python -c "from core.schemas.evidence import EvidenceBundle; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add core/schemas/evidence.py
git commit -m "feat(schema): add collector_name and weight to EvidenceBundle"
```

---

## Task 2: Update LLMReasoner to handle list of EvidenceBundle

**Files:**
- Modify: `agents/auditor/llm_reasoner.py:57-178`

**Step 1: Update reason() method signature**

Change `evidence_bundle: EvidenceBundle` to `evidence_bundles: list[EvidenceBundle]`:

```python
def reason(
    self,
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],  # Changed from single bundle
) -> ReasoningTrace:
```

**Step 2: Update _prepare_evidence_json to handle multiple bundles**

Replace the method to aggregate multiple bundles:

```python
def _prepare_evidence_json(
    self,
    bundles: list[EvidenceBundle],
    *,
    max_chars: int = DEFAULT_MAX_AUDIT_EVIDENCE_CHARS,
    max_items: int = DEFAULT_MAX_AUDIT_EVIDENCE_ITEMS,
    ctx: "AgentContext | None" = None,
) -> str:
    """Prepare multiple evidence bundles as JSON for LLM."""
    all_items = []
    bundle_info = []

    for bundle in bundles:
        bundle_info.append({
            "bundle_id": bundle.bundle_id,
            "collector_name": bundle.collector_name or "unknown",
            "weight": bundle.weight,
            "item_count": len(bundle.items),
        })
        for item in bundle.items:
            # Add collector context to each item
            safe_extracted = {
                k: self._safe_value(v) for k, v in item.extracted_fields.items()
            }
            all_items.append({
                "evidence_id": item.evidence_id,
                "requirement_id": item.requirement_id,
                "source_id": item.provenance.source_id,
                "source_uri": item.provenance.source_uri,
                "provenance_tier": item.provenance.tier,
                "success": item.success,
                "error": item.error,
                "status_code": item.status_code,
                "extracted_fields": safe_extracted,
                "parsed_value": self._safe_value(
                    item.parsed_value,
                    max_str_chars=MAX_PRIMARY_EVIDENCE_CHARS,
                ),
                "from_collector": bundle.collector_name or "unknown",
                "bundle_weight": bundle.weight,
            })

    # Apply item cap
    total_items = len(all_items)
    items_to_serialize = all_items[:max_items]
    truncated_by_count = total_items > max_items

    payload: dict[str, Any] = {
        "bundles": bundle_info,
        "total_bundles": len(bundles),
        "market_id": bundles[0].market_id if bundles else "unknown",
        "total_items": total_items,
        "items": items_to_serialize,
    }
    if truncated_by_count:
        payload["_truncation_note"] = (
            f"Only first {max_items} of {total_items} evidence items shown."
        )

    evidence_json = json.dumps(payload, indent=2)
    truncated_by_size = False

    # Truncate by size if needed
    while len(evidence_json) > max_chars and len(items_to_serialize) > 1:
        truncated_by_size = True
        items_to_serialize = items_to_serialize[:-1]
        payload["items"] = items_to_serialize
        payload["_truncation_note"] = (
            f"Evidence truncated to {len(items_to_serialize)} items "
            f"(total {total_items}) to fit context limit ({max_chars} chars)."
        )
        evidence_json = json.dumps(payload, indent=2)

    if ctx and (truncated_by_count or truncated_by_size):
        ctx.info(
            f"Audit evidence truncated: {len(items_to_serialize)} items shown, "
            f"total_items={total_items}, json_len={len(evidence_json)}"
        )

    return evidence_json
```

**Step 3: Update reason() method to use list**

Update the calls inside reason() to pass `evidence_bundles` to `_prepare_evidence_json`:

```python
# Change line ~98-99 from:
evidence_json = self._prepare_evidence_json(
    evidence_bundle, max_chars=max_chars, max_items=max_items, ctx=ctx
)
# To:
evidence_json = self._prepare_evidence_json(
    evidence_bundles, max_chars=max_chars, max_items=max_items, ctx=ctx
)
```

Also update the secondary truncation loop (~131-132) similarly.

**Step 4: Update _build_trace to use first bundle for IDs**

Update line ~379 and ~384:
```python
# Use first bundle's ID for trace generation
trace_id = data.get("trace_id") or self._generate_trace_id(evidence_bundles[0].bundle_id)
# ...
bundle_id=evidence_bundles[0].bundle_id,  # Use first bundle's ID
```

**Step 5: Commit**

```bash
git add agents/auditor/llm_reasoner.py
git commit -m "feat(auditor): update LLMReasoner to handle list of EvidenceBundle"
```

---

## Task 3: Update Auditor agents to accept list of bundles

**Files:**
- Modify: `agents/auditor/agent.py:65-119` (AuditorLLM.run)
- Modify: `agents/auditor/agent.py:220-268` (AuditorRuleBased.run)
- Modify: `agents/auditor/agent.py:334-354` (audit_evidence function)

**Step 1: Update AuditorLLM.run signature and implementation**

```python
def run(
    self,
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: EvidenceBundle | list[EvidenceBundle],  # Accept both
) -> AgentResult:
    """
    Generate reasoning trace from evidence.

    Args:
        ctx: Agent context with LLM client
        prompt_spec: The prompt specification
        evidence_bundles: Single bundle or list of bundles

    Returns:
        AgentResult with ReasoningTrace as output
    """
    # Normalize to list
    if isinstance(evidence_bundles, EvidenceBundle):
        evidence_bundles = [evidence_bundles]

    total_items = sum(len(b.items) for b in evidence_bundles)
    ctx.info(f"AuditorLLM analyzing {total_items} evidence items from {len(evidence_bundles)} bundles")

    # ... rest of implementation, pass evidence_bundles to reasoner
```

Update the call to reasoner:
```python
trace = self.reasoner.reason(ctx, prompt_spec, evidence_bundles)
```

Update _validate_trace call:
```python
verification = self._validate_trace(trace, evidence_bundles)
```

**Step 2: Update _validate_trace to handle list**

```python
def _validate_trace(
    self,
    trace: ReasoningTrace,
    evidence_bundles: list[EvidenceBundle],
) -> VerificationResult:
    """Validate the reasoning trace."""
    checks: list[CheckResult] = []

    # ... existing checks ...

    # Check 3: References evidence - collect from all bundles
    ref_ids = set(trace.get_evidence_refs())
    evidence_ids = set()
    for bundle in evidence_bundles:
        evidence_ids.update(e.evidence_id for e in bundle.items)

    # ... rest unchanged ...
```

**Step 3: Update AuditorRuleBased similarly**

Same pattern - accept `EvidenceBundle | list[EvidenceBundle]`, normalize to list.

**Step 4: Update audit_evidence function**

```python
def audit_evidence(
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    *,
    prefer_llm: bool = True,
) -> AgentResult:
```

**Step 5: Commit**

```bash
git add agents/auditor/agent.py
git commit -m "feat(auditor): accept list of EvidenceBundle in auditor agents"
```

---

## Task 4: Update RuleBasedReasoner for list of bundles

**Files:**
- Modify: `agents/auditor/reasoner.py`

**Step 1: Update reason() signature and implementation**

Similar pattern - accept list, combine items from all bundles.

```python
def reason(
    self,
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> ReasoningTrace:
    # Combine all items
    all_items = []
    for bundle in evidence_bundles:
        all_items.extend(bundle.items)
    # ... use all_items instead of evidence_bundle.items
```

**Step 2: Commit**

```bash
git add agents/auditor/reasoner.py
git commit -m "feat(auditor): update RuleBasedReasoner for list of bundles"
```

---

## Task 5: Update Judge agents to accept list of bundles

**Files:**
- Modify: `agents/judge/agent.py:71-143` (JudgeLLM.run)
- Modify: `agents/judge/agent.py:344-405` (JudgeRuleBased.run)
- Modify: `agents/judge/verdict_builder.py`

**Step 1: Update JudgeLLM.run signature**

```python
def run(
    self,
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    reasoning_trace: ReasoningTrace,
) -> AgentResult:
    # Normalize to list
    if isinstance(evidence_bundles, EvidenceBundle):
        evidence_bundles = [evidence_bundles]
```

**Step 2: Update _get_llm_review and _validate_verdict signatures**

Pass `evidence_bundles: list[EvidenceBundle]` through the call chain.

**Step 3: Update VerdictBuilder.build to accept list**

```python
def build(
    self,
    ctx: "AgentContext",
    prompt_spec: PromptSpec,
    evidence_bundles: EvidenceBundle | list[EvidenceBundle],
    trace: ReasoningTrace,
    ...
) -> DeterministicVerdict:
    # Normalize and use first bundle for hashing (or combine)
    if isinstance(evidence_bundles, EvidenceBundle):
        evidence_bundles = [evidence_bundles]
```

**Step 4: Update JudgeRuleBased similarly**

**Step 5: Update judge_verdict convenience function**

**Step 6: Commit**

```bash
git add agents/judge/agent.py agents/judge/verdict_builder.py
git commit -m "feat(judge): accept list of EvidenceBundle in judge agents"
```

---

## Task 6: Update Pipeline to handle multiple collectors

**Files:**
- Modify: `orchestrator/pipeline.py:491-532` (_step_evidence_collection)
- Modify: `orchestrator/pipeline.py:534-577` (_step_audit)
- Modify: `orchestrator/pipeline.py:579-624` (_step_judge)
- Modify: `orchestrator/sop_executor.py:49-54` (PipelineState)

**Step 1: Update PipelineState to hold list of bundles**

```python
# In PipelineState, change:
evidence_bundle: Optional[EvidenceBundle] = None
# To:
evidence_bundles: list[EvidenceBundle] = field(default_factory=list)
```

**Step 2: Update _step_evidence_collection**

For now, keep single collector but set collector_name on bundle:

```python
evidence_bundle, execution_log = result.output
evidence_bundle.collector_name = agent.name  # Track which collector
state.evidence_bundles = [evidence_bundle]
```

**Step 3: Update _step_audit to pass list**

```python
result: AgentResult = agent.run(ctx, state.prompt_spec, state.evidence_bundles)
```

**Step 4: Update _step_judge to pass list**

```python
result: AgentResult = agent.run(
    ctx, state.prompt_spec, state.evidence_bundles, state.audit_trace
)
```

**Step 5: Update _step_build_por_bundle**

Use first bundle for PoR (or create combined bundle):
```python
if not all([state.prompt_spec, state.evidence_bundles, state.audit_trace, state.verdict]):
    ...
# Use first bundle for PoR computation
evidence_bundle = state.evidence_bundles[0] if state.evidence_bundles else None
```

**Step 6: Update RunResult**

```python
evidence_bundles: list[EvidenceBundle] = field(default_factory=list)
# Keep evidence_bundle for backward compat as property
@property
def evidence_bundle(self) -> Optional[EvidenceBundle]:
    return self.evidence_bundles[0] if self.evidence_bundles else None
```

**Step 7: Commit**

```bash
git add orchestrator/pipeline.py orchestrator/sop_executor.py
git commit -m "feat(pipeline): support list of evidence bundles in pipeline state"
```

---

## Task 7: Update /step/collect API to accept multiple collectors

**Files:**
- Modify: `api/routes/steps.py:106-141` (CollectRequest, CollectResponse)
- Modify: `api/routes/steps.py:361-434` (run_collect)

**Step 1: Update CollectRequest to accept list of collectors**

```python
class CollectRequest(BaseModel):
    """Request for evidence collection step."""

    prompt_spec: dict[str, Any] = Field(...)
    tool_plan: dict[str, Any] = Field(...)
    collectors: list[Literal["CollectorLLM", "CollectorHyDE", "CollectorHTTP", "CollectorMock"]] = Field(
        default=["CollectorLLM"],
        description="Which collector agents to use (runs all in sequence)",
        min_length=1,
    )
    execution_mode: Literal["production", "development", "test"] = Field(default="development")
    include_raw_content: bool = Field(default=False)
```

**Step 2: Update CollectResponse to return list of bundles**

```python
class CollectResponse(BaseModel):
    """Response from evidence collection step."""

    ok: bool = Field(...)
    collectors_used: list[str] = Field(default_factory=list)
    evidence_bundles: list[dict[str, Any]] = Field(
        default_factory=list, description="Evidence bundles from each collector"
    )
    execution_logs: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution logs from each collector"
    )
    errors: list[str] = Field(default_factory=list)
```

**Step 3: Update run_collect to iterate collectors**

```python
@router.post("/collect", response_model=CollectResponse)
async def run_collect(request: CollectRequest) -> CollectResponse:
    try:
        logger.info(f"Running collect step with collectors: {request.collectors}")

        from core.schemas.prompts import PromptSpec
        from core.schemas.transport import ToolPlan
        from agents.registry import get_registry

        # Parse inputs
        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        try:
            tool_plan = ToolPlan(**request.tool_plan)
        except Exception as e:
            return CollectResponse(ok=False, errors=[f"Invalid tool_plan: {e}"])

        ctx = get_agent_context(with_llm=True, with_http=True)
        registry = get_registry()

        bundles = []
        logs = []
        collectors_used = []
        errors = []

        for collector_name in request.collectors:
            try:
                collector = registry.get_agent_by_name(collector_name, ctx)
            except ValueError:
                errors.append(f"Collector not found: {collector_name}")
                continue

            logger.info(f"Running collector: {collector.name}")
            result = collector.run(ctx, prompt_spec, tool_plan)

            if not result.success:
                errors.append(f"{collector_name}: {result.error or 'Collection failed'}")
                continue

            evidence_bundle, execution_log = result.output
            evidence_bundle.collector_name = collector.name

            # Optionally strip raw_content
            eb_dict = evidence_bundle.model_dump(mode="json")
            if not request.include_raw_content:
                for item in eb_dict.get("items", []):
                    item["raw_content"] = None
                    item["parsed_value"] = None

            bundles.append(eb_dict)
            logs.append(execution_log.model_dump(mode="json") if execution_log else {})
            collectors_used.append(collector.name)

        return CollectResponse(
            ok=len(bundles) > 0,
            collectors_used=collectors_used,
            evidence_bundles=bundles,
            execution_logs=logs,
            errors=errors,
        )

    except Exception as e:
        logger.exception("Collect step failed")
        raise InternalError(f"Collect step failed: {str(e)}")
```

**Step 4: Commit**

```bash
git add api/routes/steps.py
git commit -m "feat(api): /step/collect accepts multiple collectors"
```

---

## Task 8: Update /step/audit and /step/judge APIs for list of bundles

**Files:**
- Modify: `api/routes/steps.py:147-169` (AuditRequest, AuditResponse)
- Modify: `api/routes/steps.py:176-203` (JudgeRequest, JudgeResponse)
- Modify: `api/routes/steps.py:437-484` (run_audit)
- Modify: `api/routes/steps.py:487-542` (run_judge)

**Step 1: Update AuditRequest**

```python
class AuditRequest(BaseModel):
    """Request for audit/reasoning step."""

    prompt_spec: dict[str, Any] = Field(...)
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
    )
    execution_mode: Literal["production", "development", "test"] = Field(default="development")
```

**Step 2: Update run_audit**

```python
@router.post("/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest) -> AuditResponse:
    try:
        logger.info("Running audit step...")

        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle

        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return AuditResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        evidence_bundles = []
        for i, eb_dict in enumerate(request.evidence_bundles):
            try:
                evidence_bundles.append(EvidenceBundle(**eb_dict))
            except Exception as e:
                return AuditResponse(ok=False, errors=[f"Invalid evidence_bundle[{i}]: {e}"])

        if not evidence_bundles:
            return AuditResponse(ok=False, errors=["No evidence bundles provided"])

        ctx = get_agent_context(with_llm=True)

        from agents.auditor import get_auditor
        auditor = get_auditor(ctx)
        logger.info(f"Using auditor: {auditor.name}")

        result = auditor.run(ctx, prompt_spec, evidence_bundles)

        if not result.success:
            return AuditResponse(ok=False, errors=[result.error or "Audit failed"])

        trace = result.output

        return AuditResponse(
            ok=True,
            reasoning_trace=trace.model_dump(mode="json"),
        )

    except Exception as e:
        logger.exception("Audit step failed")
        raise InternalError(f"Audit step failed: {str(e)}")
```

**Step 3: Update JudgeRequest**

```python
class JudgeRequest(BaseModel):
    """Request for judge/verdict step."""

    prompt_spec: dict[str, Any] = Field(...)
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
    )
    reasoning_trace: dict[str, Any] = Field(...)
    execution_mode: Literal["production", "development", "test"] = Field(default="development")
```

**Step 4: Update run_judge similarly**

**Step 5: Update BundleRequest**

```python
class BundleRequest(BaseModel):
    """Request for PoR bundle step."""

    prompt_spec: dict[str, Any] = Field(...)
    evidence_bundles: list[dict[str, Any]] = Field(
        ..., description="Evidence bundles (from /step/collect)"
    )
    reasoning_trace: dict[str, Any] = Field(...)
    verdict: dict[str, Any] = Field(...)
```

**Step 6: Update run_bundle to use first bundle for PoR**

**Step 7: Commit**

```bash
git add api/routes/steps.py
git commit -m "feat(api): /step/audit, /step/judge, /step/bundle accept list of evidence bundles"
```

---

## Task 9: Update tests

**Files:**
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_auditor.py` (if exists)
- Modify: `tests/test_api_steps.py` (if exists)

**Step 1: Update test helpers to pass list of bundles**

**Step 2: Add test for multiple collectors**

```python
def test_collect_multiple_collectors():
    """Test collecting with multiple collectors."""
    # Create mock context
    ctx = AgentContext.create_mock(llm_responses=[...])

    # Test that collectors are run and bundles aggregated
    ...
```

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update tests for multi-collector evidence"
```

---

## Task 10: Update /step/resolve for backward compatibility

**Files:**
- Modify: `api/routes/steps.py:285-354` (run_resolve)

**Step 1: Update run_resolve to return list of bundles in artifacts**

The pipeline internally uses `evidence_bundles` list now, so update the artifacts dict:

```python
if result.evidence_bundles:
    artifacts["evidence_bundles"] = [
        self._strip_raw_content(eb.model_dump(mode="json"))
        for eb in result.evidence_bundles
    ]
    # Also include first bundle as evidence_bundle for backward compat
    if result.evidence_bundle:
        eb = result.evidence_bundle.model_dump(mode="json")
        if not request.include_raw_content:
            for item in eb.get("items", []):
                item["raw_content"] = None
                item["parsed_value"] = None
        artifacts["evidence_bundle"] = eb
```

**Step 2: Commit**

```bash
git add api/routes/steps.py
git commit -m "feat(api): update /step/resolve for multi-bundle support"
```

---

## Verification

Run full test suite:
```bash
pytest tests/ -v
```

Test API manually:
```bash
# Start server
uvicorn api.main:app --reload

# Test multi-collector
curl -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d '{"prompt_spec": {...}, "tool_plan": {...}, "collectors": ["CollectorLLM", "CollectorHyDE"]}'
```
