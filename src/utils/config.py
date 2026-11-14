"""Load the YAML files in config/ into one dict, keyed by file stem
(``data_sources`` is shortened to ``sources``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_FILES = {
    "pipeline": "config/pipeline.yaml",
    "sources": "config/data_sources.yaml",
    "events": "config/events.yaml",
    "features": "config/features.yaml",
    "analysis": "config/analysis.yaml",
}


def get_project_root() -> Path:
    """Walk up from this file until a pyproject.toml is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("No pyproject.toml above src/utils/config.py")


def resolve_path(relative_path: str) -> Path:
    return get_project_root() / relative_path


def load_config() -> dict[str, Any]:
    root = get_project_root()
    cfg = {}
    for key, rel in _CONFIG_FILES.items():
        with open(root / rel, encoding="utf-8") as f:
            cfg[key] = yaml.safe_load(f) or {}
    return cfg
