"""Small IO helpers shared across data-preparation scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info("wrote %s", path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_json_section(path: Path, key: str, value: Any) -> None:
    """Merge-write a top-level key into a JSON file, creating it if missing.

    Used by multi-stage pipelines where each stage owns one key in a shared
    stats file and must not clobber keys written by sibling stages.
    """
    payload: dict[str, Any] = load_json(path) if path.exists() else {}
    payload[key] = value
    dump_json(path, payload)
