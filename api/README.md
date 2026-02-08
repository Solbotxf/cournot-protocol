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

### Step: Collect

```bash
POST /step/collect
```

Run evidence collection using one or more collectors. Pass the `prompt_spec` and `tool_plan` from the prompt step.

Available collectors: `CollectorLLM`, `CollectorHyDE`, `CollectorHTTP`, `CollectorMock`, `CollectorAgenticRAG`, `CollectorGraphRAG`.

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

### Step: Audit

```bash
POST /step/audit
```

Analyze evidence and generate a reasoning trace. Pass the `prompt_spec` and `evidence_bundles` from the collect step.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "evidence_bundles": [ { "..." : "..." } ]
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
  "reasoning_trace": { "..." : "..." }
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

Run the full resolution pipeline (collect → audit → judge → PoR bundle) in a single call.

**Request Body:**
```json
{
  "prompt_spec": { "..." : "..." },
  "tool_plan": { "..." : "..." },
  "collectors": ["CollectorLLM"],
  "execution_mode": "development"
}
```

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
      "name": "CollectorLLM",
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

**Step 3: Audit** — analyze evidence and generate a reasoning trace.

```bash
curl -s -X POST http://localhost:8000/step/audit \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
print(json.dumps({
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles']
}))
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

**Step 4: Judge** — produce a final verdict from the evidence and reasoning.

```bash
curl -s -X POST http://localhost:8000/step/judge \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
prompt = json.load(open('prompt_out.json'))
collect = json.load(open('collect_out.json'))
audit = json.load(open('audit_out.json'))
print(json.dumps({
    'prompt_spec': prompt['prompt_spec'],
    'evidence_bundles': collect['evidence_bundles'],
    'reasoning_trace': audit['reasoning_trace']
}))
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

# Step 3: Audit
audit_resp = httpx.post(f"{BASE}/step/audit", json={
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
}).json()
reasoning_trace = audit_resp["reasoning_trace"]
print(f"Preliminary outcome: {reasoning_trace.get('preliminary_outcome')}")

# Step 4: Judge
judge_resp = httpx.post(f"{BASE}/step/judge", json={
    "prompt_spec": prompt_spec,
    "evidence_bundles": evidence_bundles,
    "reasoning_trace": reasoning_trace,
}).json()
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
    └── replay.py      # Evidence replay
```