"""Build type hierarchy sidecar: base_type -> implementing classes."""

import json
import logging
from pathlib import Path

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def build_type_hierarchy(records: list[tuple[str, str, str, str, str, list[str]]]) -> dict[str, list[dict]]:
    """Build reverse map: base_type -> [{class, file, module, namespace}].

    Args:
        records: List of (chunk_type, class_name, file_path, module, namespace, base_types) tuples.

    Returns:
        {"IFoo": [{"class": "FooImpl", "file": "path/to/FooImpl.cs", "module": "...", "namespace": "..."}]}
    """
    hierarchy: dict[str, list[dict]] = {}
    for chunk_type, class_name, file_path, module, namespace, base_types in records:
        for bt in base_types:
            hierarchy.setdefault(bt, []).append({
                "class": class_name,
                "file": file_path,
                "module": module,
                "namespace": namespace,
            })
    return hierarchy


def save_type_hierarchy(repo_name: str, records: list[tuple[str, str, str, str, str, list[str]]]) -> None:
    """Build and save type hierarchy sidecar to disk."""
    hierarchy = build_type_hierarchy(records)
    path = DATA_DIR / f"{repo_name}_type_hierarchy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")
    total_impls = sum(len(v) for v in hierarchy.values())
    logger.info("Saved type hierarchy for %s: %d base types, %d implementations",
                repo_name, len(hierarchy), total_impls)


def load_type_hierarchy(repo_name: str) -> dict[str, list[dict]]:
    """Load type hierarchy from disk."""
    path = DATA_DIR / f"{repo_name}_type_hierarchy.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load type hierarchy for %s: %s", repo_name, e)
        return {}
