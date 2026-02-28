# Handoff: Generalized Site Extractor Registry

## Goal
Pluggable extractor registry for structured HTTP extraction (Phase 1.5), with domain-specific prompt hints for Cloudflare-guarded sites in Phase 2.

## Current State
- **Strict collector** with three-phase approach:
  1. **Phase 1: Serper pre-search** — uses LLM-generated discovery query (`_generate_discovery_query`) to find top 3 URLs on the required domain via `google.serper.dev/search`.
  2. **Phase 1.5: Extractor registry dispatch + LLM resolution** — loops through discovered URLs, finds a matching `SiteExtractor` via `find_extractor()`, fetches structured data, builds a text summary, then asks Gemini to resolve. Currently supports **FotMob**. (FBRef was removed — see below.)
  3. **Phase 2: Gemini UrlContext + GoogleSearch** — fallback if Phase 1.5 is not applicable or fails. Passes discovered URLs to Gemini via `UrlContext` tool for full page ingestion.
- **Tests pass** across 4 test files:
  - `test_fotmob.py` — FotMob data extraction
  - `test_collector_source_pinned.py` — strict agent behavior
  - `test_extractor_registry.py` — registry + FotMobExtractor
  - `test_fotmob_live.py` — live FotMob fetch
- **End-to-end API pipeline verified** for both stat-based and event-based markets.
- **Branch:** `feat/extractor-registry` (6 commits)

## What Changed This Session: Extractor Registry

### Problem
Phase 1.5 was hardcoded to FotMob only (`_try_fotmob_direct_extraction`). Adding new data sources required modifying the strict agent directly. The "Next Steps" from the prior session called for generalizing beyond FotMob.

### Solution
Introduced a `SiteExtractor` ABC and a registry pattern:

1. **`agents/collector/extractors/base.py`** — `SiteExtractor` ABC with `can_handle(url)`, `extract_and_summarize(url, http_client)`, and `source_id` property. Plus `ExtractionError` exception.

2. **`agents/collector/extractors/__init__.py`** — Registry with `find_extractor(url)` that loops through registered extractors and returns the first match, or `None`.

3. **`agents/collector/extractors/fotmob_ext.py`** — `FotMobExtractor` wrapping existing `fotmob.py` (`fetch_match_stats` + `summarize_for_llm`). Returns `(summary_text, metadata_dict)`.

4. **`agents/collector/source_pinned_agent.py`** — Replaced `_try_fotmob_direct_extraction` with `_try_direct_extraction` that uses `find_extractor()` dispatch. Removed direct fotmob imports.

### Architecture

```
agents/collector/extractors/
├── __init__.py          # Registry: find_extractor(url) -> SiteExtractor | None
├── base.py              # SiteExtractor ABC, ExtractionError
└── fotmob_ext.py        # FotMobExtractor (wraps fotmob.py)
```

**Note on FBRef:** `FBRefExtractor` was removed because it was a thin Gemini UrlContext wrapper — functionally identical to Phase 2. Cloudflare-guarded sites like FBRef are now handled by Phase 2 with domain-specific prompt hints (`_DOMAIN_PROMPT_HINTS` in `source_pinned_agent.py`). The extractor registry is reserved for structured HTTP extraction (like FotMob's `__NEXT_DATA__` JSON).

The registry is a simple list of `SiteExtractor` instances. `find_extractor(url)` iterates and returns the first extractor whose `can_handle(url)` returns `True`.

### How to Add a New Extractor

1. Create `agents/collector/extractors/mysite_ext.py`:
   ```python
   from .base import SiteExtractor, ExtractionError

   class MySiteExtractor(SiteExtractor):
       source_id = "mysite"

       def can_handle(self, url: str) -> bool:
           return "mysite.com/" in url

       def extract_and_summarize(self, url, http_client):
           # Fetch data, build text summary
           summary = "..."
           metadata = {"source_url": url, ...}
           return summary, metadata
   ```

2. Register in `agents/collector/extractors/__init__.py`:
   ```python
   from .mysite_ext import MySiteExtractor
   _REGISTRY.append(MySiteExtractor())
   ```

3. Add tests in `tests/test_mysite_extractor.py`.

### Files Changed
- **New:** `agents/collector/extractors/` — package with `base.py`, `__init__.py`, `fotmob_ext.py`
- **Modified:** `agents/collector/source_pinned_agent.py` — replaced `_try_fotmob_direct_extraction` → `_try_direct_extraction`; imports changed from `fotmob` to `extractors`
- **Modified:** `tests/test_collector_source_pinned.py` — updated `TestFotMobDirectExtraction` to mock `find_extractor`; added `TestGenericDirectExtraction` (4 tests)
- **New:** `tests/test_extractor_registry.py` — tests for registry + FotMobExtractor

## Key Findings (carried forward)

### `__NEXT_DATA__` Solves JS-Rendering Problem
FotMob is a Next.js app. The HTML page embeds ALL match data (496KB JSON) in a `<script id="__NEXT_DATA__">` tag. A plain HTTP GET with browser headers returns this data — no Cloudflare challenge, no JS rendering needed.

**Important:** The `Accept-Encoding` header must NOT include `br` (brotli) when using the `requests` library without the brotli package installed.

### FotMob Two-Leg Match Pages
For Champions League knockout ties, FotMob uses a **single URL** for both legs. The SSR `__NEXT_DATA__` always contains the **second leg** data. First-leg markets fall through to Phase 2 Gemini.

## How to Run a Market (step-by-step via API)

```bash
# 1. Start the API server
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# 2. Prompt — compile the market query
curl -s -X POST http://localhost:8000/step/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Will Arsenal score from a corner against Brentford on Feb 12?\n\nThis market will resolve to YES if Arsenal scores at least one goal officially designated as originating from a corner kick situation in their Premier League match against Brentford scheduled for February 12, 2026, 20:00 UTC. Otherwise, this market will resolve to NO.\n\nThe goal must be recorded in Fotmob'\''s shot map data with the situation labeled as from corner and the result as goal.",
    "strict_mode": true
  }' -o prompt_out.json

# 3. Collect
curl -s -X POST http://localhost:8000/step/collect \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
r = json.load(open('prompt_out.json'))
print(json.dumps({
    'prompt_spec': r['prompt_spec'],
    'tool_plan': r['tool_plan'],
    'collectors': ['CollectorSourcePinned'],
    'include_raw_content': False
}))
")" -o collect_out.json

# 4. Check collect output
python3 -c "
import json
r = json.load(open('collect_out.json'))
for bundle in r.get('evidence_bundles', []):
    for item in bundle.get('items', []):
        ef = item.get('extracted_fields', {})
        print(f'outcome: {ef.get(\"outcome\")}')
        print(f'direct_extraction: {ef.get(\"direct_extraction\")}')
        print(f'resolution_method: {ef.get(\"resolution_method\")}')
        print(f'extractor_source_id: {ef.get(\"extractor_source_id\")}')
        print(f'confidence: {ef.get(\"confidence_score\")}')
"

# 5. Audit → Judge → Bundle (same pattern as before)
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

## Key Requirements for Phase 1.5 to Activate

1. A discovered URL matches a registered `SiteExtractor` via `find_extractor(url)`
2. The extractor's `extract_and_summarize()` succeeds (no `ExtractionError`)
3. Gemini LLM call succeeds and returns valid `{"outcome": "Yes/No", "reason": "..."}`

If any step fails, Phase 1.5 returns `None` and the collector falls through to Phase 2 Gemini.

## Remaining Issues
1. **Two-leg matches** — FotMob only serves second-leg data via SSR `__NEXT_DATA__`. First-leg markets cannot use Phase 1.5.

## Next Steps
- **Add more extractors** — the registry pattern makes it straightforward to add extractors for other domains (e.g. Transfermarkt, ESPN, CoinGecko for non-football markets).
- **Add more domain hints** — add `_DOMAIN_PROMPT_HINTS` entries for other Cloudflare-guarded or JS-heavy sites to improve Phase 2 extraction quality.
