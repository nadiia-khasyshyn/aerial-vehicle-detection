"""Load configs/pipeline.yaml once and read it by dotted path."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"


def _load():
    with open(CONFIG_PATH) as handle:
        return yaml.safe_load(handle)


CONFIG = _load()


def get(path, default=None):
    """get("tiling.tile") -> 512. Missing keys return `default`."""
    node = CONFIG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
