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

### cURL - Step check (for example prompt engineering)

```bash
curl -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will BTC be above $100k by end of 2025?",
    "strict_mode": true
  }'
```

### Python - Run Pipeline

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
    ├── verify.py      # Pack verification
    └── replay.py      # Evidence replay
```