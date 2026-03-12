"""Asset reference lookup: find prefabs/scenes that reference a script class."""

import json
import logging
from difflib import get_close_matches

from src.config import REPOS, resolve_repo
from src.indexer.asset_ref_builder import load_asset_references, load_asset_references_by_guid

logger = logging.getLogger(__name__)

# In-memory cache: repo -> refs dict
_asset_ref_cache: dict[str, dict] = {}
_asset_ref_guid_cache: dict[str, dict] = {}


def find_asset_references(class_name: str, repo: str = "mainapp",
                          output_format: str = "text") -> str:
    """Find all prefabs, scenes, and asset files that reference a script class.

    Args:
        class_name: Script class name (e.g., "PlayerController", "LeanTouchCameraController").
        repo: Which repo to search.
        output_format: "text" (default) or "json".

    Returns:
        Asset reference matches in text or JSON format.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    refs = _get_refs(resolved)

    if not refs:
        return (f"No asset reference data for '{resolved}'. "
                f"Run reindex('{resolved}', incremental=False) to build it.")

    # Exact match first
    file_list = refs.get(class_name)

    # Fuzzy match
    if file_list is None:
        lower = class_name.lower()
        matches: list[tuple[str, list[str]]] = []
        for key, files in refs.items():
            if key.lower() == lower:
                matches.append((key, files))
            elif lower in key.lower():
                matches.append((key, files))

        if len(matches) == 1:
            class_name, file_list = matches[0]
        elif len(matches) > 1:
            if output_format == "json":
                return json.dumps({
                    "status": "disambiguation",
                    "repo": resolved,
                    "query": class_name,
                    "matches": [
                        {"class_name": key, "references": len(files)}
                        for key, files in sorted(matches, key=lambda x: x[0])
                    ],
                }, indent=2)
            lines = [f"Multiple classes match '{class_name}':"]
            for key, files in sorted(matches, key=lambda x: x[0]):
                lines.append(f"  {key} ({len(files)} references)")
            return "\n".join(lines)
        else:
            total_classes = len(refs)
            suggestion = _suggest_similar_classes(class_name, refs)
            if output_format == "json":
                payload = {
                    "status": "not_found",
                    "repo": resolved,
                    "query": class_name,
                    "classes_indexed": total_classes,
                    "message": f"No asset references found for '{class_name}' in {resolved} ({total_classes} classes indexed).",
                }
                if suggestion:
                    payload["suggestions"] = suggestion
                return json.dumps(payload, indent=2)
            msg = f"No asset references found for '{class_name}' in {resolved} ({total_classes} classes indexed)."
            if suggestion:
                msg += "\n" + suggestion
            return msg

    # Check GUID data for disambiguation
    guid_refs = _get_guid_refs(resolved)
    disambiguation = _check_disambiguation(class_name, guid_refs)

    # Categorize by type
    prefabs = [f for f in file_list if f.endswith(".prefab")]
    scenes = [f for f in file_list if f.endswith(".unity")]
    others = [f for f in file_list if not f.endswith(".prefab") and not f.endswith(".unity")]

    if output_format == "json":
        return _asset_refs_json_payload(
            resolved=resolved,
            class_name=class_name,
            file_list=file_list,
            prefabs=prefabs,
            scenes=scenes,
            others=others,
            disambiguation=disambiguation,
        )

    lines = [f"'{class_name}' is referenced in {len(file_list)} asset file(s):\n"]

    # Add disambiguation warning if multiple scripts share the same class name
    if disambiguation:
        lines.append(f"  NOTE: {len(disambiguation)} scripts share this class name:")
        for guid, info in disambiguation.items():
            lines.append(f"    {info['script_path']} (GUID: {guid[:8]}..., {len(info['files'])} refs)")
        lines.append("")

    if prefabs:
        lines.append(f"  Prefabs ({len(prefabs)}):")
        for f in sorted(prefabs):
            lines.append(f"    {f}")
    if scenes:
        lines.append(f"  Scenes ({len(scenes)}):")
        for f in sorted(scenes):
            lines.append(f"    {f}")
    if others:
        lines.append(f"  Other ({len(others)}):")
        for f in sorted(others):
            lines.append(f"    {f}")

    return "\n".join(lines)


def _suggest_similar_classes(query: str, refs: dict) -> str:
    """Suggest similar class names from asset refs when an exact lookup finds nothing."""
    all_names = sorted(refs.keys())
    if not all_names:
        return ""

    matches = [m for m in get_close_matches(query, all_names, n=6, cutoff=0.5)
               if m != query][:5]

    # Also try substring matching as a fallback
    if len(matches) < 3:
        lower = query.lower()
        substring_matches = [n for n in all_names
                             if lower in n.lower() and n not in matches
                             and n != query][:5 - len(matches)]
        matches.extend(substring_matches)

    if not matches:
        return ""

    return "Did you mean: " + ", ".join(matches) + "?"


def _asset_refs_json_payload(resolved: str, class_name: str, file_list: list[str],
                             prefabs: list[str], scenes: list[str], others: list[str],
                             disambiguation: dict[str, dict]) -> str:
    payload = {
        "status": "found",
        "repo": resolved,
        "class_name": class_name,
        "total_files": len(file_list),
        "prefabs": sorted(prefabs),
        "scenes": sorted(scenes),
        "others": sorted(others),
        "disambiguation": [
            {
                "guid": guid,
                "class_name": info.get("class_name", ""),
                "script_path": info.get("script_path", ""),
                "reference_count": len(info.get("files", [])),
                "files": sorted(info.get("files", [])),
            }
            for guid, info in sorted(disambiguation.items())
        ],
    }
    return json.dumps(payload, indent=2)


def _check_disambiguation(class_name: str, guid_refs: dict[str, dict]) -> dict[str, dict]:
    """Check if multiple GUIDs resolve to the same class name.

    Returns entries only when there are 2+ GUIDs with the same class_name.
    """
    if not isinstance(guid_refs, dict):
        return {}

    matching = {}
    for guid, entry in guid_refs.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("class_name") == class_name:
            matching[guid] = entry
    if len(matching) >= 2:
        return matching
    return {}


def invalidate_asset_ref_cache(repo: str | None = None) -> None:
    """Clear cached refs after reindex."""
    if repo is None:
        _asset_ref_cache.clear()
        _asset_ref_guid_cache.clear()
    else:
        _asset_ref_cache.pop(repo, None)
        _asset_ref_guid_cache.pop(repo, None)


def _get_refs(repo: str) -> dict:
    """Load class-name-keyed refs with caching."""
    if repo not in _asset_ref_cache:
        _asset_ref_cache[repo] = load_asset_references(repo)
    return _asset_ref_cache[repo]


def _get_guid_refs(repo: str) -> dict:
    """Load GUID-keyed refs with caching."""
    if repo not in _asset_ref_guid_cache:
        _asset_ref_guid_cache[repo] = load_asset_references_by_guid(repo)
    return _asset_ref_guid_cache[repo]
