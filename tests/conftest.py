"""Pytest configuration.

This repo uses top-level packages (core/, agents/, orchestrator/, ...)
without an installed wheel. Add repository root to sys.path so tests can
import modules consistently when executed via `pytest`.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
