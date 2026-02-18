"""
Config loader for the Site Sentinel project.

All configuration lives in the configs/ directory as YAML files.
Scripts and the Streamlit app load their settings through this module
so there's one place to look when a value needs changing.

Usage:

    from site_sentinel.config import load_config

    cfg = load_config("pipeline")
    raw_dir = cfg["data"]["raw_trajectory_dir"]

    app_cfg = load_config("app")
    precision = app_cfg["model_performance"]["precision"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Resolve the configs/ directory relative to this file so the package works
# regardless of the working directory the caller uses.
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    """
    Load a named YAML config file from the configs/ directory.

    Args:
        name: Config file name without the .yaml extension.
              Valid options: "pipeline", "model_training", "app".

    Returns:
        The parsed YAML contents as a nested dictionary.

    Raises:
        FileNotFoundError: If configs/<name>.yaml does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected one of: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}"
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
