"""
config.py — single config loader for FinGraph Intelligence System.

Usage (everywhere):
    from config import get_config
    cfg = get_config()
    top_k = cfg["advanced_rag"]["top_k_final"]

Rules:
- Import this module; never open config.yaml directly.
- Secrets (API keys, passwords, URIs with credentials) stay in .env — never in config.yaml.
- lru_cache ensures the file is read exactly once per process lifetime.
"""

import yaml
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load config.yaml once and cache for process lifetime."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
