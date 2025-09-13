"""Pytest configuration ensuring local `src` package is importable.

Adds project root (the directory containing `pyproject.toml`) to sys.path
at test collection time so tests can `import src.*` regardless of how
pytest was invoked (e.g., via `uv run pytest`).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
