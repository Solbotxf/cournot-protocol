# Cournot Module Interface Stub Pack

This pack contains **interfaces and typed contracts** for implementing the Cournot PoR (Proof of Reasoning)
pipeline: Prompt Engineer -> Collector -> Auditor -> Judge -> (optional) Sentinel Validator.

**Notes**
- These are **stubs**: signatures + docstrings + minimal types. Business logic is intentionally absent.
- Designed for multi-agent (multiple engineers / AIs) parallel implementation.
- Uses Pydantic v2 style where possible; adjust if your repo uses v1.



## Run Test:
python -m pytest -q

## Code to prompt:
code2prompt --exclude="*/venv/*" --exclude-from-tree .

## Run pipeline and save pack
python -m cournot_cli run "Will BTC exceed $100k?" --out pack.zip --json

## Verify pack offline
python -m cournot_cli verify pack.zip --json

## Replay evidence (online)
python -m cournot_cli replay pack.zip --json

## Create pack from artifact files
python -m cournot_cli pack --from-dir ./artifacts --out pack.zip