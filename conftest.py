"""Pytest configuration.

This repository uses flat top-level modules (e.g. `agents`, `core`, `orchestrator`).
When running tests from a venv without installing the project as a package,
Python may not include the repo root on `sys.path`, causing import errors.

We explicitly add the repository root to `sys.path` so tests can import
`agents`, `core`, etc. deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
