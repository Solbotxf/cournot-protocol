# Cournot Protocol API

FastAPI-based HTTP API for the Cournot prediction market resolution protocol.

## Installation

```bash
# Install dependencies
pip install -r requirement.txt
```

## Running the API

```bash
# Development mode
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Or directly
python -m api.app
```

The API will be available at `http://localhost:8000`

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```bash
GET /health
```

Returns service status.

### Run Pipeline

```bash
POST /run
```

Execute the full pipeline on a query.

**Request Body:**
```json
{
  "user_input": "Will BTC be above $100k by end of 2025?",
  "strict_mode": true,
  "enable_sentinel_verify": true,
  "enable_replay": false,
  "return_format": "json",
  "include_checks": false,
  "include_artifacts": true,
  "execution_mode": "development"
}
```

**Response (JSON format):**
```json
{
  "ok": true,
  "summary": {
    "market_id": "mk_abc123",
    "outcome": "YES",
    "confidence": 0.85,
    "por_root": "0x...",
    "execution_mode": "development"
  },
  "artifacts": { ... },
  "verification": {
    "sentinel_ok": true,
    "total_checks": 20,
    "passed_checks": 19,
    "failed_checks": 1
  },
  "errors": []
}
```

**Response (pack_zip format):**

Returns a ZIP file with all artifacts. Headers include:
- `X-Market-Id`
- `X-Por-Root`
- `X-Outcome`
- `X-Confidence`

### Verify Pack

```bash
POST /verify
Content-Type: multipart/form-data
```

Verify an uploaded artifact pack.

**Parameters:**
- `file`: ZIP file (required)
- `include_checks`: Include detailed checks (default: false)
- `enable_sentinel`: Enable sentinel verification (default: true)

**Response:**
```json
{
  "ok": true,
  "hashes_ok": true,
  "semantic_ok": true,
  "sentinel_ok": true,
  "por_root": "0x...",
  "market_id": "mk_abc123",
  "outcome": "YES",
  "checks": [],
  "challenges": [],
  "errors": []
}
```

### Replay Evidence

```bash
POST /replay
Content-Type: multipart/form-data
```

Replay evidence collection and verify against original pack.

**Parameters:**
- `file`: ZIP file (required)
- `timeout_s`: Timeout in seconds (default: 30)
- `include_checks`: Include detailed checks (default: false)

**Response:**
```json
{
  "ok": true,
  "replay_ok": true,
  "sentinel_ok": true,
  "por_root": "0x...",
  "market_id": "mk_abc123",
  "divergence": null,
  "checks": [],
  "challenges": [],
  "errors": []
}
```

### Capabilities

```bash
GET /capabilities
```

Discover available agents and configured LLM providers. Only providers with API keys set in the server environment are returned — no secrets are exposed. Use this to populate provider/model dropdowns in a frontend or to check what collectors are available before calling `/step/collect`.

**Response:**
```json
{
  "ok": true,
  "providers": [
    { "provider": "openai", "default_model": "gpt-4o" },
    { "provider": "anthropic", "default_model": "claude-sonnet-4-20250514" },
    { "provider": "google", "default_model": "gemini-2.5-flash" },
    { "provider": "grok", "default_model": "grok-3" }
  ],
  "steps": [
    {
      "step": "prompt_engineer",
      "agents": [
        { "name": "PromptEngineerLLM", "description": "...", "version": "v1", "capabilities": ["llm"], "priority": 100, "is_fallback": false },
        { "name": "PromptEngineerFallback", "description": "...", "version": "v1", "capabilities": ["deterministic"], "priority": 0, "is_fallback": true }
      ]
    },
    {
      "step": "collector",
      "agents": [
        { "name": "CollectorWebPageReader", "description": "...", "version": "v1", "capabilities": ["llm", "network"], "priority": 180, "is_fallback": false },
        { "name": "CollectorCRP", "description": "...", "version": "v1", "capabilities": ["llm", "network"], "priority": 195, "is_fallback": false },
        { "name": "CollectorHyDE", "description": "...", "version": "v1", "capabilities": ["llm", "network"], "priority": 190, "is_fallback": false }
      ]
    },
    { "step": "auditor", "agents": [ "..." ] },
    { "step": "judge", "agents": [ "..." ] },
    { "step": "sentinel", "agents": [ "..." ] }
  ]
}
```

### Step: Prompt Engineer

```bash
POST /step/prompt
```

Compiles a natural language query into a structured `PromptSpec` and `ToolPlan`.

**Request Body:**
```json
{
  "user_input": "Will BTC be above $100k by end of 2025?",
  "strict_mode": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4o"
}
```

**Temporal constraint auto-detection:** When the query involves a scheduled event with a specific date/time, the LLM compiler automatically populates `prompt_spec.extra.temporal_constraint`. The frontend should extract this and pass it back to `/step/audit` and `/step/judge`.

```json
{
  "ok": true,
  "prompt_spec": {
    "extra": {
      "temporal_constraint": {
        "enabled": true,
        "event_time": "2027-05-31T00:00:00Z",
        "reason": "Champions League final scheduled for May 31 2027"
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | bool | Always `true` when present |
| `event_time` | string (ISO 8601 UTC) | When the event is scheduled to occur or conclude |
| `reason` | string | Why temporal awareness matters for this query |

When `temporal_constraint` is absent from `prompt_spec.extra`, the query has no temporal signal — no special handling needed.

### Step: Collect

```bash
POST /step/collect
```

Run evidence collection using one or more collectors. Pass the `prompt_spec` and `tool_plan` from the prompt step.

Available collectors: `CollectorWebPageReader`, `CollectorHyDE`, `CollectorHTTP`, `CollectorMock`, `CollectorAgenticRAG`, `CollectorGraphRAG`, `CollectorPAN`, `CollectorOpenSearch`, `CollectorCRP`.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "tool_plan": { "..." : "..." },
  "collectors": ["CollectorAgenticRAG"],
  "include_raw_content": false,
  "llm_provider": "grok",
  "llm_model": "grok-4-fast"
}
```

### Step: Quality Check

```bash
POST /step/quality_check
```

Evaluate evidence quality before proceeding to audit. Returns a scorecard with quality signals and retry hints. If quality is below threshold, retry `/step/collect` with the `quality_feedback` field.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "evidence_bundles": [ { "..." : "..." } ]
}
```

**Response:**
```json
{
  "ok": true,
  "scorecard": {
    "overall_score": 0.65,
    "meets_threshold": false,
    "retry_hints": {
      "search_queries": ["more specific query"],
      "required_domains": ["reuters.com"],
      "skip_domains": [],
      "data_type_hint": null,
      "focus_requirements": ["req_001"],
      "collector_guidance": "Try broader search terms"
    }
  },
  "meets_threshold": false,
  "errors": []
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt_spec` | object | yes | Compiled prompt specification (from `/step/prompt`) |
| `evidence_bundles` | list[object] | yes | Evidence bundles (from `/step/collect`) |

**Frontend flow:** Call quality check after collect. If `meets_threshold` is `false` and `retry_hints` is non-empty, retry `/step/collect` with `quality_feedback` set to the `retry_hints` object. Repeat up to 2 times.

### Step: Audit

```bash
POST /step/audit
```

Analyze evidence and generate a reasoning trace. Pass the `prompt_spec` and `evidence_bundles` from the collect step.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "evidence_bundles": [ { "..." : "..." } ],
  "quality_scorecard": { "..." : "..." },
  "temporal_constraint": {
    "enabled": true,
    "event_time": "2027-05-31T00:00:00Z",
    "reason": "Champions League final scheduled for May 31 2027"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt_spec` | object | yes | Compiled prompt specification |
| `evidence_bundles` | list[object] | yes | Evidence bundles (from `/step/collect`) |
| `execution_mode` | string | no | `"production"`, `"development"` (default), or `"test"` |
| `llm_provider` | string | no | LLM provider override |
| `llm_model` | string | no | LLM model override |
| `quality_scorecard` | object | no | Quality scorecard from `/step/quality_check`. Informs the auditor about evidence quality issues. |
| `temporal_constraint` | object | no | Temporal constraint from `prompt_spec.extra.temporal_constraint`. When provided, the auditor computes temporal status (FUTURE/ACTIVE/PAST) at resolution time and may force INVALID for future events. |

**Response:**
```json
{
  "ok": true,
  "reasoning_trace": {
    "trace_id": "trace_mk_...",
    "steps": [ { "step_type": "evidence_review", "..." : "..." } ],
    "preliminary_outcome": "YES"
  },
  "errors": []
}
```

### Step: Judge

```bash
POST /step/judge
```

Review reasoning and produce a final verdict. Pass the `prompt_spec`, `evidence_bundles`, and `reasoning_trace` from previous steps.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "evidence_bundles": [ { "..." : "..." } ],
  "reasoning_trace": { "..." : "..." },
  "quality_scorecard": { "..." : "..." },
  "temporal_constraint": {
    "enabled": true,
    "event_time": "2027-05-31T00:00:00Z",
    "reason": "Champions League final scheduled for May 31 2027"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt_spec` | object | yes | Compiled prompt specification |
| `evidence_bundles` | list[object] | yes | Evidence bundles (from `/step/collect`) |
| `reasoning_trace` | object | yes | Reasoning trace (from `/step/audit`) |
| `execution_mode` | string | no | `"production"`, `"development"` (default), or `"test"` |
| `llm_provider` | string | no | LLM provider override |
| `llm_model` | string | no | LLM model override |
| `quality_scorecard` | object | no | Quality scorecard from `/step/quality_check`. Informs the judge about evidence quality issues. |
| `temporal_constraint` | object | no | Temporal constraint from `prompt_spec.extra.temporal_constraint`. When provided, the judge computes temporal status (FUTURE/ACTIVE/PAST) at resolution time and may force INVALID for future events. |

**Temporal status computation:** Both audit and judge compute temporal status at resolution time by comparing `event_time` against the current clock:

| Condition | Status | Effect |
|-----------|--------|--------|
| `event_time > now` | `FUTURE` | Forces outcome to `INVALID` — event hasn't happened yet |
| `now - event_time < 24h` | `ACTIVE` | Forces `INVALID` unless evidence shows event has concluded |
| `now - event_time >= 24h` | `PAST` | Normal resolution — no temporal override |
| Parse failure | `UNKNOWN` | Normal resolution — temporal system disengaged |

**Response:**
```json
{
  "ok": true,
  "verdict": {
    "market_id": "mk_...",
    "outcome": "YES",
    "confidence": 0.85,
    "resolution_rule_id": "R_THRESHOLD"
  },
  "outcome": "YES",
  "confidence": 0.85,
  "errors": []
}
```

### Step: Bundle

```bash
POST /step/bundle
```

Build a cryptographically verifiable Proof of Reasoning bundle from all prior artifacts.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "evidence_bundles": [ { "..." : "..." } ],
  "reasoning_trace": { "..." : "..." },
  "verdict": { "..." : "..." }
}
```

### Step: Resolve (all-in-one)

```bash
POST /step/resolve
```

Run the full resolution pipeline (collect → quality check → audit → judge → PoR bundle) in a single call. Quality check and temporal constraint are handled automatically — `temporal_constraint` is extracted from `prompt_spec.extra` and quality check runs with a retry loop by default.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "tool_plan": { "..." : "..." },
  "collectors": ["CollectorOpenSearch"],
  "execution_mode": "development",
  "enable_quality_check": true,
  "max_quality_retries": 2
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt_spec` | object | yes | — | Compiled prompt specification (from `/step/prompt`) |
| `tool_plan` | object | yes | — | Tool execution plan (from `/step/prompt`) |
| `collectors` | list[string] | no | `["CollectorWebPageReader"]` | Which collectors to use |
| `execution_mode` | string | no | `"development"` | `"production"`, `"development"`, or `"test"` |
| `include_raw_content` | bool | no | `false` | Include raw_content in evidence items |
| `llm_provider` | string | no | — | LLM provider override |
| `llm_model` | string | no | — | LLM model override |
| `enable_quality_check` | bool | no | `true` | Run quality check after collection with retry loop. Set `false` to skip. |
| `max_quality_retries` | int | no | `2` | Max quality check retry iterations (0–5). Only used when `enable_quality_check` is `true`. |

**What happens internally:**
1. Collect evidence using specified collectors
2. If `enable_quality_check` is `true`: run quality check, retry collection with feedback if below threshold (up to `max_quality_retries` times)
3. Extract `temporal_constraint` from `prompt_spec.extra` (if auto-detected by prompt engineer)
4. Run auditor with quality scorecard + temporal constraint injected
5. Run judge with quality scorecard + temporal constraint injected
6. Build PoR bundle

No extra fields needed from the frontend — quality check and temporal constraint are fully automatic in this endpoint.

### Dispute

```bash
POST /dispute
```

Stateless dispute-driven rerun of audit/judge steps. The frontend provides all context artifacts and a structured dispute request. Use this when the UI builds the full `DisputeRequest` directly.

**Request Body:**
```json
{
  "mode": "reasoning_only",
  "reason_code": "EVIDENCE_MISREAD",
  "message": "The evidence was misinterpreted — the deal was announced on Feb 25, not signed on Apr 30",
  "target": {
    "artifact": "evidence_bundle",
    "leaf_path": "items[0].extracted_fields.outcome"
  },
  "prompt_spec": { "..." : "..." },
  "evidence_bundle": { "..." : "..." },
  "reasoning_trace": { "..." : "..." },
  "patch": {
    "evidence_items_append": [
      {
        "evidence_id": "new-item-1",
        "requirement_id": "req_001",
        "provenance": { "source_id": "manual", "source_uri": "https://example.com" },
        "raw_content": "...",
        "extracted_fields": { "outcome": "Yes", "reason": "..." }
      }
    ]
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | `"reasoning_only"` \| `"full_rerun"` | no (default: `reasoning_only`) | `reasoning_only` reruns audit/judge with provided evidence. `full_rerun` re-collects evidence first. |
| `case_id` | string | no | Optional dashboard correlation ID (stateless, no lookup) |
| `reason_code` | enum | yes | `REASONING_ERROR`, `LOGIC_GAP`, `EVIDENCE_MISREAD`, `EVIDENCE_INSUFFICIENT`, `OTHER` |
| `message` | string | yes | Dispute message (1–8000 chars) |
| `target` | object | no | Which artifact/field is disputed |
| `prompt_spec` | object | yes | Full PromptSpec from the previous run |
| `evidence_bundle` | object | no | EvidenceBundle from the previous run |
| `reasoning_trace` | object | no | ReasoningTrace (if present, allows judge-only rerun) |
| `tool_plan` | object | no | Required for `full_rerun` mode |
| `collectors` | list[string] | no | Required for `full_rerun` mode |
| `patch` | object | no | `evidence_items_append` and/or `prompt_spec_override` |

**Response:**
```json
{
  "ok": true,
  "case_id": null,
  "rerun_plan": ["audit", "judge"],
  "artifacts": {
    "prompt_spec": { "..." : "..." },
    "evidence_bundle": { "..." : "..." },
    "evidence_bundles": [ { "..." : "..." } ],
    "reasoning_trace": { "..." : "..." },
    "verdict": { "..." : "..." }
  },
  "diff": {
    "steps_rerun": ["audit", "judge"],
    "verdict_changed": null
  }
}
```

### Dispute (LLM-Assisted)

```bash
POST /dispute/llm
```

Simplified dispute endpoint that accepts 3 user inputs and uses an LLM to translate natural language into a structured `DisputeRequest`, then delegates to the existing dispute logic. Returns the same response as `POST /dispute`.

**Request Body:**
```json
{
  "reason_code": "EVIDENCE_MISREAD",
  "message": "Wikipedia shows PM Shmyhal announced a preliminary agreement on Feb 25 2025",
  "evidence_urls": ["https://en.wikipedia.org/wiki/Ukraine%E2%80%93United_States_Mineral_Resources_Agreement"],
  "prompt_spec": { "..." : "..." },
  "evidence_bundle": { "..." : "..." },
  "reasoning_trace": { "..." : "..." }
}
```

| Field | Type | Required | UI Control | Description |
|-------|------|----------|------------|-------------|
| `reason_code` | enum | yes | Dropdown | `EVIDENCE_MISREAD`, `EVIDENCE_INSUFFICIENT`, `REASONING_ERROR`, `LOGIC_GAP`, `OTHER` |
| `message` | string | yes | Textarea | Free-text dispute message (1–4000 chars) |
| `evidence_urls` | list[string] | no | Repeatable URL input | Up to 5 URLs to fetch as supporting evidence |
| `prompt_spec` | object | yes | auto | Full PromptSpec from the previous run |
| `evidence_bundle` | object | no | auto | EvidenceBundle from the previous run |
| `reasoning_trace` | object | no | auto | ReasoningTrace from the previous run |
| `tool_plan` | object | no | auto | Only needed if LLM decides `full_rerun` |
| `collectors` | list[string] | no | auto | Only needed if LLM decides `full_rerun` |

The context fields (`prompt_spec`, `evidence_bundle`, `reasoning_trace`, `tool_plan`, `collectors`) are attached automatically by the frontend from the current case's artifacts — the user never configures these.

**Response:** Same as `POST /dispute`.

### Validate Market

```bash
POST /validate
```

Validate and compile a market query in a single call. Replaces the need to call both `/step/prompt` and a separate validation endpoint. Runs in parallel:
1. **LLM validation** — classify market type, validate required fields, assess resolvability
2. **Prompt compilation** — compile into PromptSpec + ToolPlan (same as `/step/prompt`)
3. **Source reachability** — probes any data source URLs mentioned in the query to check if they are accessible by the AI (detects Cloudflare blocks, paywalls, timeouts)

**Request Body:**
```json
{
  "user_input": "Highest temperature in Buenos Aires on March 1? Resolves using Wunderground data for Minister Pistarini Intl Airport, °C. Resolution deadline: March 2, 2026, 12:00 PM ET.",
  "strict_mode": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4o"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_input` | string | yes | The prediction market query to validate and compile (1–8000 chars) |
| `strict_mode` | bool | no | Enable strict mode for deterministic hashing (default: true) |
| `llm_provider` | string | no | LLM provider override (e.g., `"openai"`, `"anthropic"`) |
| `llm_model` | string | no | LLM model override (e.g., `"gpt-4o"`) |

**Response:**
```json
{
  "ok": true,
  "classification": {
    "market_type": "TEMPERATURE",
    "confidence": 0.95,
    "detection_rationale": "Contains 'temperature', city name, and date"
  },
  "validation": {
    "checks_passed": ["U-02", "U-03", "TEMP-01", "TEMP-02", "TEMP-03"],
    "checks_failed": [
      {
        "check_id": "TEMP-04",
        "severity": "warning",
        "message": "No fallback data source specified.",
        "suggestion": "If data from Wunderground is unavailable, which alternative source should be used?"
      }
    ]
  },
  "resolvability": {
    "score": 35,
    "level": "MEDIUM",
    "risk_factors": [
      {"factor": "Single data source with no fallback", "points": 30},
      {"factor": "No fallback rule for cancelled/postponed events", "points": 10}
    ],
    "recommendation": "Consider adding fallback data sources."
  },
  "source_reachability": [
    {
      "url": "https://www.wunderground.com",
      "reachable": true,
      "status_code": 200,
      "error": null
    }
  ],
  "market_id": "mk_6d3dc65b5bdabbe0",
  "prompt_spec": { "..." : "..." },
  "tool_plan": { "..." : "..." },
  "prompt_metadata": { "compiler": "llm", "question_type": "temperature" },
  "errors": []
}
```

The `prompt_spec` and `tool_plan` fields can be passed directly to `/step/collect` or `/step/resolve` to continue the pipeline.

**Market Types:** `FINANCIAL_PRICE`, `TEMPERATURE`, `CRYPTO_THRESHOLD`, `SPORTS_MATCH`, `SPORTS_EXACT_SCORE`, `SPORTS_PLAYER_PROP`, `SPEECH_CONTENT`, `GEOPOLITICAL_EVENT`, `BINARY_EVENT`, `MULTI_CHOICE_EVENT`, `OPINION_NOVELTY`, `UNKNOWN`.

**Risk Levels:**

| Score | Level | Meaning |
|-------|-------|---------|
| 0–15 | LOW | Approved for automated resolution |
| 16–35 | MEDIUM | May have difficulty resolving automatically |
| 36–55 | HIGH | High risk of failing automated resolution |
| 56+ | VERY_HIGH | Unlikely to resolve automatically |

## Configuration

### Environment Variables

```bash
# Execution mode
export COURNOT_EXECUTION_MODE=development  # production, development, test

# LLM configuration (optional)
export COURNOT_LLM_PROVIDER=anthropic
export COURNOT_LLM_API_KEY=sk-ant-...
export COURNOT_LLM_MODEL=claude-sonnet-4-20250514
```

### LLM Override Resolution

Each step endpoint resolves the LLM provider/model using a 3-tier fallback:

1. **Per-request** — `llm_provider` / `llm_model` fields in the request body (highest priority).
2. **Per-agent config** — `agents.<agent>.llm_override` in `cournot.json` (e.g. `agents.collector.llm_override`).
3. **Server default** — top-level `llm` section in `cournot.json`.

Example `cournot.json` that uses Grok for collectors and OpenAI for everything else:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4o"
  },
  "agents": {
    "collector": {
      "name": "CollectorWebPageReader",
      "llm_override": {
        "provider": "grok",
        "model": "grok-4-fast",
        "base_url": "https://api.x.ai/v1"
      }
    }
  }
}
```

With this config, calling `POST /step/collect` without `llm_provider`/`llm_model` in the request body will automatically use `grok / grok-4-fast`. Passing `"llm_provider": "anthropic"` in the request body overrides the config for that call.

## Execution Modes

| Mode | Description |
|------|-------------|
| `production` | Fail-closed, requires all capabilities |
| `development` | Uses fallback agents when needed (default) |
| `test` | Deterministic mock agents only |

## Error Responses

All errors return a structured response:

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Human-readable error message",
    "details": {}
  }
}
```

Error codes:
- `INVALID_REQUEST` - Invalid request parameters
- `MISSING_FILE` - Required file not provided
- `INVALID_PACK` - Invalid or corrupted pack file
- `VERIFICATION_FAILED` - Verification failed
- `PIPELINE_ERROR` - Pipeline execution failed
- `INTERNAL_ERROR` - Internal server error

## Examples

### cURL - Capabilities

```bash
curl -s http://localhost:8000/capabilities | python3 -m json.tool
```

### cURL - Run Pipeline

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will BTC be above 100k?",
    "return_format": "json"
  }'
```

### cURL - Download Pack

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will BTC be above 100k?",
    "return_format": "pack_zip"
  }' \
  -o result.zip
```

### cURL - Verify Pack

```bash
curl -X POST http://localhost:8000/verify \
  -F "file=@result.zip" \
  -F "enable_sentinel=true"
```

### cURL - Two-step flow (prompt → resolve)

This is the simplest multi-step flow. `/step/resolve` runs collect, audit, judge, and PoR bundle in one call.

```bash
# Step 1: Compile the prompt
PROMPT=$(curl -s -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Will BTC be above $100k by end of 2025?", "strict_mode": true}')

# Step 2: Resolve (collect → audit → judge → bundle)
curl -X POST http://localhost:8000/step/resolve \
  -H "Content-Type: application/json" \
  -d "$(echo $PROMPT | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(json.dumps({
    'prompt_spec': r['prompt_spec'],
    'tool_plan': r['tool_plan'],
    'collectors': ['CollectorAgenticRAG'],
    'execution_mode': 'development'
}))
")"
```

### cURL - Step-by-step flow (prompt → collect → audit → judge → bundle)

Call each pipeline step individually. This gives full control over each stage — you can inspect intermediate outputs, swap collectors, or retry a single step.

**Step 1: Prompt** — compile the query into a PromptSpec and ToolPlan.

```bash
curl -s -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will BTC be above $100k by end of 2025?",
    "strict_mode": true
  }' -o prompt_out.json
```

Response (`prompt_out.json`):
```json
{
  "ok": true,
  "market_id": "mk_6d3dc65b5bdabbe0",
  "prompt_spec": { "..." : "..." },
  "tool_plan": { "plan_id": "plan_mk_6d3dc65b5bdabbe0", "requirements": ["req_001"], "..." : "..." },
  "metadata": { "compiler": "fallback", "question_type": "crypto_price", "num_requirements": 2 },
  "error": null
}
```

**Step 2: Collect** — gather evidence from external sources. You can choose one or more collectors.

```bash
curl -s -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
r = json.load(open('prompt_out.json'))
print(json.dumps({
    'prompt_spec': r['prompt_spec'],
    'tool_plan': r['tool_plan'],
    'collectors': ['CollectorAgenticRAG'],
    'include_raw_content': False
}))
")" -o collect_out.json
```

Response (`collect_out.json`):
```json
{
  "ok": true,
  "collectors_used": ["CollectorAgenticRAG"],
  "evidence_bundles": [
    {
      "bundle_id": "agentic_rag_plan_mk_6d3dc65b5bdabbe0",
      "market_id": "mk_6d3dc65b5bdabbe0",
      "items": [ { "evidence_id": "a1b2c3...", "success": true, "..." : "..." } ],
      "requirements_fulfilled": ["req_001"],
      "requirements_unfulfilled": [],
      "..." : "..."
    }
  ],
  "execution_logs": [ { "plan_id": "plan_mk_6d3dc65b5bdabbe0", "calls": ["..."], "..." : "..." } ],
  "errors": []
}
```

You can also run multiple collectors in one call (e.g. compare AgenticRAG vs GraphRAG):

```bash
# Run two collectors and get two evidence bundles back
curl -s -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
r = json.load(open('prompt_out.json'))
print(json.dumps({
    'prompt_spec': r['prompt_spec'],
    'tool_plan': r['tool_plan'],
    'collectors': ['CollectorAgenticRAG', 'CollectorGraphRAG']
}))
")"
```

**Step 2.5: Quality Check** _(optional but recommended)_ — evaluate evidence quality and retry if needed.

```bash
curl -s -X POST http://localhost:8000/step/quality_check \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
print(json.dumps({
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles']
}))
")" -o quality_out.json
```

If `meets_threshold` is false and `retry_hints` is non-empty, retry collect with feedback:

```bash
curl -s -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
qc = json.load(open('quality_out.json'))
print(json.dumps({
    'prompt_spec': prompt['prompt_spec'],
    'tool_plan': prompt['tool_plan'],
    'collectors': ['CollectorOpenSearch'],
    'quality_feedback': qc['scorecard']['retry_hints']
}))
")" -o collect_retry_out.json
```

**Step 3: Audit** — analyze evidence and generate a reasoning trace. Pass `quality_scorecard` and `temporal_constraint` when available.

```bash
curl -s -X POST http://localhost:8000/step/audit \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
# Extract optional fields
quality_sc = json.load(open('quality_out.json')).get('scorecard')
temporal = (prompt['prompt_spec'].get('extra') or {}).get('temporal_constraint')
payload = {
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles'],
}
if quality_sc:
    payload['quality_scorecard'] = quality_sc
if temporal:
    payload['temporal_constraint'] = temporal
print(json.dumps(payload))
")" -o audit_out.json
```

Response (`audit_out.json`):
```json
{
  "ok": true,
  "reasoning_trace": {
    "trace_id": "trace_mk_6d3dc65b5bdabbe0",
    "steps": [
      { "step_type": "evidence_review", "description": "...", "..." : "..." },
      { "step_type": "rule_application", "description": "...", "..." : "..." }
    ],
    "preliminary_outcome": "YES",
    "..." : "..."
  },
  "errors": []
}
```

**Step 4: Judge** — produce a final verdict from the evidence and reasoning. Pass the same `quality_scorecard` and `temporal_constraint`.

```bash
curl -s -X POST http://localhost:8000/step/judge \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
audit = json.load(open('audit_out.json'))
quality_sc = json.load(open('quality_out.json')).get('scorecard')
temporal = (prompt['prompt_spec'].get('extra') or {}).get('temporal_constraint')
payload = {
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles'],
    'reasoning_trace': audit['reasoning_trace'],
}
if quality_sc:
    payload['quality_scorecard'] = quality_sc
if temporal:
    payload['temporal_constraint'] = temporal
print(json.dumps(payload))
")" -o judge_out.json
```

Response (`judge_out.json`):
```json
{
  "ok": true,
  "verdict": {
    "market_id": "mk_6d3dc65b5bdabbe0",
    "outcome": "YES",
    "confidence": 0.85,
    "resolution_rule_id": "R_THRESHOLD",
    "..." : "..."
  },
  "outcome": "YES",
  "confidence": 0.85,
  "errors": []
}
```

**Step 5: Bundle** — build the cryptographic Proof of Reasoning bundle.

```bash
curl -s -X POST http://localhost:8000/step/bundle \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
audit = json.load(open('audit_out.json'))
judge = json.load(open('judge_out.json'))
print(json.dumps({
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles'],
    'reasoning_trace': audit['reasoning_trace'],
    'verdict': judge['verdict']
}))
")" -o bundle_out.json
```

Response (`bundle_out.json`):
```json
{
  "ok": true,
  "por_bundle": {
    "por_root": "0xf6ef1f33...",
    "prompt_spec_hash": "0x...",
    "evidence_root": "0x...",
    "reasoning_root": "0x...",
    "..." : "..."
  },
  "por_root": "0xf6ef1f33...",
  "roots": {
    "prompt_spec_hash": "0x...",
    "evidence_root": "0x...",
    "reasoning_root": "0x...",
    "por_root": "0xf6ef1f33..."
  },
  "errors": []
}
```

### Python - Capabilities

```python
import httpx

resp = httpx.get("http://localhost:8000/capabilities").json()

# List configured LLM providers
for p in resp["providers"]:
    print(f"  {p['provider']} (default: {p['default_model']})")

# List available collectors
for step in resp["steps"]:
    if step["step"] == "collector":
        for agent in step["agents"]:
            print(f"  {agent['name']} (priority={agent['priority']}, fallback={agent['is_fallback']})")
```

### Python - Step-by-step flow

```python
import httpx

BASE = "http://localhost:8000"

# Step 1: Prompt
prompt_resp = httpx.post(f"{BASE}/step/prompt", json={
    "user_input": "Will BTC be above $100k by end of 2025?",
    "strict_mode": True,
}).json()
prompt_spec = prompt_resp["prompt_spec"]
tool_plan = prompt_resp["tool_plan"]
print(f"Market: {prompt_resp['market_id']}")

# Step 2: Collect (using GraphRAG collector)
collect_resp = httpx.post(f"{BASE}/step/collect", json={
    "prompt_spec": prompt_spec,
    "tool_plan": tool_plan,
    "collectors": ["CollectorGraphRAG"],
}).json()
evidence_bundles = collect_resp["evidence_bundles"]
print(f"Collectors used: {collect_resp['collectors_used']}")
print(f"Bundles collected: {len(evidence_bundles)}")

# Step 2.5: Quality check (optional)
qc_resp = httpx.post(f"{BASE}/step/quality_check", json={
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
}).json()
quality_scorecard = qc_resp.get("scorecard") if qc_resp.get("ok") else None

# Extract temporal_constraint from prompt_spec.extra (auto-detected by PE)
temporal_constraint = (prompt_spec.get("extra") or {}).get("temporal_constraint")

# Step 3: Audit (with quality scorecard + temporal constraint)
audit_payload = {
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
}
if quality_scorecard:
    audit_payload["quality_scorecard"] = quality_scorecard
if temporal_constraint:
    audit_payload["temporal_constraint"] = temporal_constraint

audit_resp = httpx.post(f"{BASE}/step/audit", json=audit_payload).json()
reasoning_trace = audit_resp["reasoning_trace"]
print(f"Preliminary outcome: {reasoning_trace.get('preliminary_outcome')}")

# Step 4: Judge (with quality scorecard + temporal constraint)
judge_payload = {
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
    "reasoning_trace": reasoning_trace,
}
if quality_scorecard:
    judge_payload["quality_scorecard"] = quality_scorecard
if temporal_constraint:
    judge_payload["temporal_constraint"] = temporal_constraint

judge_resp = httpx.post(f"{BASE}/step/judge", json=judge_payload).json()
verdict = judge_resp["verdict"]
print(f"Outcome: {judge_resp['outcome']} (confidence: {judge_resp['confidence']})")

# Step 5: Bundle
bundle_resp = httpx.post(f"{BASE}/step/bundle", json={
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
    "reasoning_trace": reasoning_trace,
    "verdict": verdict,
}).json()
print(f"PoR root: {bundle_resp['por_root']}")
```

### Python - Run Pipeline (single call)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/run",
    json={
        "user_input": "Will ETH reach $5000?",
        "return_format": "json",
    },
)
data = response.json()
print(f"Outcome: {data['summary']['outcome']}")
print(f"PoR Root: {data['summary']['por_root']}")
```

### Python - Verify Pack

```python
import httpx

with open("result.zip", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/verify",
        files={"file": ("pack.zip", f, "application/zip")},
        data={"enable_sentinel": "true"},
    )
data = response.json()
print(f"Verified: {data['ok']}")
```

## Development

### Running Tests

```bash
# Test API imports
python -c "from api.app import app; print('OK')"

# Test with FastAPI TestClient
python -c "
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)
response = client.get('/health')
print(response.json())
"
```

### Project Structure

```
api/
├── __init__.py        # Package init
├── app.py             # FastAPI application
├── deps.py            # Dependency injection
├── errors.py          # Error handling
├── models/
│   ├── __init__.py
│   ├── requests.py    # Request models
│   └── responses.py   # Response models
└── routes/
    ├── __init__.py
    ├── health.py      # Health check
    ├── run.py         # Pipeline execution
    ├── steps.py       # Individual step execution (prompt, collect, audit, judge, bundle, resolve)
    ├── verify.py      # Pack verification
    ├── replay.py      # Evidence replay
    ├── capabilities.py # Agent/provider discovery
    ├── dispute.py     # Dispute endpoints (structured + LLM-assisted)
    └── validate.py    # Market creation validator (classify + validate + resolvability)
```