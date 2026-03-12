"""Resolve Unity script GUIDs to C# class names by scanning .meta files."""

import logging
import re
from pathlib import Path

from src.models.unity_script_ref import UnityScriptRef

logger = logging.getLogger(__name__)

# Regex to extract guid from .meta file content (line 2: "guid: <hex>")
_GUID_PATTERN = re.compile(r"^guid:\s*([0-9a-f]{32})\s*$", re.MULTILINE)

# Light namespace extraction from .cs files (first "namespace X.Y.Z" line)
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([\w.]+)", re.MULTILINE)


def _extract_namespace(cs_path: Path) -> str:
    """Best-effort namespace extraction from a .cs file."""
    try:
        # Read only the first 4KB — namespace declarations are always near the top
        with cs_path.open("r", encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
        match = _NAMESPACE_RE.search(head)
        if match:
            return match.group(1)
    except OSError:
        pass
    return ""


def build_guid_map(asset_root: Path, repo_root: Path | None = None) -> dict[str, tuple[str, str]]:
    """Scan all .cs.meta files under asset_root and build a guid -> (class_name, file_path) map.

    This is the backward-compatible entry point. Callers that only need
    (class_name, script_path) tuples can use this directly.

    Args:
        asset_root: Path to the Unity Assets directory (or a subdirectory).
        repo_root: Repo root for computing relative paths. If None, paths are relative to asset_root.

    Returns:
        Dict mapping GUID hex strings to (class_name, relative_file_path) tuples.
        E.g., {"abc123...": ("RobotDriver", "UnityProject/Assets/Scripts/RobotDriver.cs")}
    """
    rich_map = build_rich_guid_map(asset_root, repo_root)
    return {guid: ref.to_tuple() for guid, ref in rich_map.items()}


def build_rich_guid_map(asset_root: Path, repo_root: Path | None = None) -> dict[str, UnityScriptRef]:
    """Scan all .cs.meta files and build a GUID -> UnityScriptRef map.

    Like build_guid_map but returns rich identity objects with namespace
    and assembly information for disambiguation.
    """
    guid_map: dict[str, UnityScriptRef] = {}
    meta_files = list(asset_root.rglob("*.cs.meta"))
    base = repo_root if repo_root is not None else asset_root

    for meta_path in meta_files:
        try:
            content = meta_path.read_text(encoding="utf-8", errors="replace")
            match = _GUID_PATTERN.search(content)
            if match:
                guid = match.group(1)
                class_name = meta_path.name.removesuffix(".cs.meta")
                cs_path = meta_path.with_name(meta_path.name.removesuffix(".meta"))
                rel_path = str(cs_path.relative_to(base)).replace("\\", "/")
                namespace = _extract_namespace(cs_path) if cs_path.exists() else ""
                guid_map[guid] = UnityScriptRef(
                    guid=guid,
                    class_name=class_name,
                    script_path=rel_path,
                    namespace=namespace,
                )
        except (OSError, ValueError):
            continue

    logger.info("Built GUID map: %d script GUIDs from %d .meta files under %s",
                len(guid_map), len(meta_files), asset_root)
    return guid_map
