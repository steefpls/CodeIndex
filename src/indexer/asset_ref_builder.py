"""Build asset references sidecar: script class -> [asset files that reference it].

The canonical store is GUID-keyed (GUID -> {class_name, script_path, files}).
The materialized JSON remains class_name-keyed for backward compatibility with
the find_asset_references tool, but includes disambiguation metadata when
multiple scripts share the same filename-derived class name.
"""

import json
import logging
import re
from pathlib import Path

from src.config import DATA_DIR, RepoConfig

logger = logging.getLogger(__name__)

_SCRIPT_GUID_RE = re.compile(r"m_Script:.*?guid:\s*([0-9a-f]{32})")


def build_asset_references(
    source_files: list[tuple[Path, object]],
    guid_map: dict[str, tuple[str, str]],
    config: RepoConfig,
) -> dict[str, list[str]]:
    """Build reverse map: class_name -> [asset file paths].

    Scans prefab, scene, and asset files for m_Script GUID references,
    resolves to class names, and builds reverse lookup.
    """
    refs: dict[str, list[str]] = {}
    unity_extensions = {".prefab", ".unity", ".asset"}

    for file_path, sd_config in source_files:
        if file_path.suffix not in unity_extensions:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        rel_path = str(file_path.relative_to(config.root)).replace("\\", "/")
        seen_classes: set[str] = set()

        for match in _SCRIPT_GUID_RE.finditer(text):
            guid = match.group(1)
            entry = guid_map.get(guid)
            if entry is None:
                continue
            class_name, _ = entry
            if class_name not in seen_classes:
                seen_classes.add(class_name)
                refs.setdefault(class_name, []).append(rel_path)

    return refs


def build_asset_references_by_guid(
    source_files: list[tuple[Path, object]],
    guid_map: dict[str, tuple[str, str]],
    config: RepoConfig,
) -> dict[str, dict]:
    """Build GUID-keyed asset reference map.

    Returns:
        {guid: {"class_name": str, "script_path": str, "files": [rel_path, ...]}}
    """
    refs: dict[str, dict] = {}
    unity_extensions = {".prefab", ".unity", ".asset"}

    for file_path, sd_config in source_files:
        if file_path.suffix not in unity_extensions:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        rel_path = str(file_path.relative_to(config.root)).replace("\\", "/")
        seen_guids: set[str] = set()

        for match in _SCRIPT_GUID_RE.finditer(text):
            guid = match.group(1)
            if guid in seen_guids:
                continue
            seen_guids.add(guid)
            entry = guid_map.get(guid)
            if entry is None:
                continue
            class_name, script_path = entry
            if guid not in refs:
                refs[guid] = {
                    "class_name": class_name,
                    "script_path": script_path,
                    "files": [],
                }
            refs[guid]["files"].append(rel_path)

    return refs


def save_asset_references(
    repo_name: str,
    source_files: list[tuple[Path, object]],
    guid_map: dict[str, tuple[str, str]],
    config: RepoConfig,
) -> None:
    """Build and save asset references sidecar to disk.

    Saves two files:
    - <repo>_asset_references.json: class_name -> [files] (backward compatible)
    - <repo>_asset_references_by_guid.json: guid -> {class_name, script_path, files}
    """
    # GUID-keyed (canonical)
    by_guid = build_asset_references_by_guid(source_files, guid_map, config)
    guid_path = DATA_DIR / f"{repo_name}_asset_references_by_guid.json"
    guid_path.parent.mkdir(parents=True, exist_ok=True)
    guid_path.write_text(json.dumps(by_guid, indent=2), encoding="utf-8")

    # Class-name-keyed (backward compatible, materialized from GUID data)
    refs = _materialize_class_keyed(by_guid)
    path = DATA_DIR / f"{repo_name}_asset_references.json"
    path.write_text(json.dumps(refs, indent=2), encoding="utf-8")

    total_refs = sum(len(v["files"]) for v in by_guid.values())
    logger.info("Saved asset references for %s: %d GUIDs, %d total file refs",
                repo_name, len(by_guid), total_refs)


def _materialize_class_keyed(by_guid: dict[str, dict]) -> dict[str, list[str]]:
    """Convert GUID-keyed refs to class_name-keyed for backward compatibility."""
    refs: dict[str, list[str]] = {}
    for guid, entry in by_guid.items():
        class_name = entry["class_name"]
        for f in entry["files"]:
            refs.setdefault(class_name, []).append(f)
    return refs


def load_asset_references(repo_name: str) -> dict[str, list[str]]:
    """Load class-name-keyed asset references from disk."""
    path = DATA_DIR / f"{repo_name}_asset_references.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load asset references for %s: %s", repo_name, e)
        return {}


def load_asset_references_by_guid(repo_name: str) -> dict[str, dict]:
    """Load GUID-keyed asset references from disk."""
    path = DATA_DIR / f"{repo_name}_asset_references_by_guid.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load GUID asset references for %s: %s", repo_name, e)
        return {}
