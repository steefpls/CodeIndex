"""Per-file sidecar contribution state for incremental sidecar maintenance.

Stores per-file contributions for type hierarchy, dependency graph, and asset
references so that sidecars can be updated file-by-file without full rebuilds.

State is persisted as a single JSON file per repo: data/<repo>_sidecar_state.json.
On each incremental or single-file operation, the pipeline:
  1. Removes old contributions for changed/deleted files
  2. Adds new contributions for changed/new files
  3. Materializes the global sidecar JSONs from all contributions

The materialized JSONs (type_hierarchy, dep_graph, asset_references) are the
same format consumed by tools; this module only manages the contribution store.
"""

import json
import logging
from pathlib import Path

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_STATE_VERSION = 2


def _state_path(repo_name: str) -> Path:
    return DATA_DIR / f"{repo_name}_sidecar_state.json"


def load_state(repo_name: str) -> dict:
    """Load sidecar state from disk. Returns empty state if not found."""
    path = _state_path(repo_name)
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _STATE_VERSION:
            logger.warning("Sidecar state version mismatch for %s, resetting", repo_name)
            return _empty_state()
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load sidecar state for %s: %s", repo_name, e)
        return _empty_state()


def save_state(repo_name: str, state: dict) -> None:
    """Persist sidecar state to disk."""
    path = _state_path(repo_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")


def _empty_state() -> dict:
    return {
        "version": _STATE_VERSION,
        "hierarchy": {},  # rel_path -> [{class,file,module,namespace,base_types}]
        "dep_graph": {},  # rel_path -> [{class,file,module,namespace,refs:[...]}]
        # rel_path -> {guid: {"class_name": "...", "script_path": "..."}}
        "asset_refs": {},
    }


# --- Per-file contribution updates ---

def set_hierarchy_contributions(state: dict, rel_path: str, records: list[tuple]) -> None:
    """Set hierarchy contributions for a file.

    Args:
        state: The sidecar state dict.
        rel_path: File path relative to repo root.
        records: List of (chunk_type, class_name, file_path, module, namespace, base_types) tuples.
    """
    entries = []
    for chunk_type, class_name, file_path, module, namespace, base_types in records:
        entries.append({
            "class": class_name,
            "file": file_path,
            "module": module,
            "namespace": namespace,
            "base_types": list(base_types),
        })
    if entries:
        state["hierarchy"][rel_path] = entries
    else:
        state["hierarchy"].pop(rel_path, None)


def set_dep_graph_contributions(state: dict, rel_path: str,
                                class_refs: dict[tuple[str, str], dict]) -> None:
    """Set dependency graph contributions for a file.

    Args:
        state: The sidecar state dict.
        rel_path: File path relative to repo root.
        class_refs: Dict of (class_name, file_path) -> {"module", "namespace", "refs": set[str]}.
    """
    entries = []
    for (class_name, file_path), info in class_refs.items():
        entries.append({
            "class": class_name,
            "file": file_path,
            "module": info["module"],
            "namespace": info["namespace"],
            "refs": sorted(info["refs"]),
        })
    if entries:
        state["dep_graph"][rel_path] = entries
    else:
        state["dep_graph"].pop(rel_path, None)


def set_asset_ref_contributions(state: dict, rel_path: str,
                                guid_entries: dict[str, dict]) -> None:
    """Set GUID-keyed asset reference contributions for an asset file.

    Args:
        state: The sidecar state dict.
        rel_path: File path relative to repo root.
        guid_entries: Dict of guid -> {"class_name": str, "script_path": str}.
    """
    if not guid_entries:
        state["asset_refs"].pop(rel_path, None)
        return

    normalized: dict[str, dict] = {}
    for guid, info in guid_entries.items():
        class_name = info.get("class_name", "")
        script_path = info.get("script_path", "")
        if not class_name:
            continue
        normalized[guid] = {
            "class_name": class_name,
            "script_path": script_path,
        }

    if normalized:
        state["asset_refs"][rel_path] = normalized
    else:
        state["asset_refs"].pop(rel_path, None)


def remove_file_contributions(state: dict, rel_path: str) -> None:
    """Remove all contributions for a file."""
    state["hierarchy"].pop(rel_path, None)
    state["dep_graph"].pop(rel_path, None)
    state["asset_refs"].pop(rel_path, None)


# --- Materialization: contribution store -> global sidecar JSONs ---

def materialize_hierarchy(state: dict) -> dict[str, list[dict]]:
    """Materialize type hierarchy from all per-file contributions.

    Returns the same format as hierarchy_builder.build_type_hierarchy:
        {"IFoo": [{"class": "FooImpl", "file": "...", "module": "...", "namespace": "..."}]}
    """
    hierarchy: dict[str, list[dict]] = {}
    for entries in state["hierarchy"].values():
        for entry in entries:
            for bt in entry["base_types"]:
                hierarchy.setdefault(bt, []).append({
                    "class": entry["class"],
                    "file": entry["file"],
                    "module": entry["module"],
                    "namespace": entry["namespace"],
                })
    return hierarchy


def _dep_node_key(class_name: str, namespace: str, module: str, file_path: str) -> str:
    """Build canonical dep node identity for V2 graph schema."""
    if namespace:
        return f"{namespace}.{class_name}"
    return f"{class_name}@@{module}@@{file_path}"


def materialize_dep_graph(state: dict) -> dict:
    """Materialize dependency graph from all per-file contributions.

    V2 shape:
    {
      "schema_version": 2,
      "nodes": {
        "<node_key>": {
          "class_name": "...",
          "namespace": "...",
          "module": "...",
          "files": ["..."],
          "deps": ["<node_key>", ...]
        }
      },
      "name_index": {
        "ClassName": ["<node_key>", ...]
      }
    }
    """
    node_records: dict[str, dict] = {}
    name_index_sets: dict[str, set[str]] = {}

    for entries in state["dep_graph"].values():
        for entry in entries:
            class_name = entry["class"]
            namespace = entry.get("namespace", "") or ""
            module = entry.get("module", "") or ""
            file_path = entry.get("file", "")
            refs = set(entry.get("refs", []))

            node_key = _dep_node_key(class_name, namespace, module, file_path)
            node = node_records.setdefault(node_key, {
                "class_name": class_name,
                "namespace": namespace,
                "module": module,
                "files": set(),
                "raw_refs": set(),
            })
            node["files"].add(file_path)
            node["raw_refs"].update(refs)
            if not node["module"] and module:
                node["module"] = module
            if not node["namespace"] and namespace:
                node["namespace"] = namespace

            name_index_sets.setdefault(class_name, set()).add(node_key)

    name_index = {name: sorted(keys) for name, keys in sorted(name_index_sets.items())}
    nodes: dict[str, dict] = {}
    for node_key, node in node_records.items():
        deps = set()
        for ref_name in node["raw_refs"]:
            for dep_key in name_index.get(ref_name, []):
                if dep_key != node_key:
                    deps.add(dep_key)

        nodes[node_key] = {
            "class_name": node["class_name"],
            "namespace": node["namespace"],
            "module": node["module"],
            "files": sorted(node["files"]),
            "deps": sorted(deps),
        }

    return {
        "schema_version": 2,
        "nodes": dict(sorted(nodes.items())),
        "name_index": name_index,
    }


def materialize_asset_refs_by_guid(state: dict) -> dict[str, dict]:
    """Materialize GUID-keyed asset refs from all per-file contributions.

    Returns:
        {guid: {"class_name": str, "script_path": str, "files": [rel_path, ...]}}
    """
    refs: dict[str, dict] = {}
    for rel_path, guid_entries in state["asset_refs"].items():
        for guid, info in guid_entries.items():
            class_name = info.get("class_name", "")
            script_path = info.get("script_path", "")
            if not class_name:
                continue

            if guid not in refs:
                refs[guid] = {
                    "class_name": class_name,
                    "script_path": script_path,
                    "files": [],
                }

            if rel_path not in refs[guid]["files"]:
                refs[guid]["files"].append(rel_path)

    normalized: dict[str, dict] = {}
    for guid in sorted(refs):
        entry = refs[guid]
        normalized[guid] = {
            "class_name": entry["class_name"],
            "script_path": entry["script_path"],
            "files": sorted(entry["files"]),
        }
    return normalized


def materialize_asset_refs(state: dict) -> dict[str, list[str]]:
    """Materialize class-keyed asset refs for backward compatibility.

    Returns:
        {"ClassName": ["path/to/prefab.prefab", "path/to/scene.unity"]}
    """
    refs: dict[str, list[str]] = {}
    by_guid = materialize_asset_refs_by_guid(state)
    for entry in by_guid.values():
        class_name = entry["class_name"]
        for rel_path in entry["files"]:
            refs.setdefault(class_name, []).append(rel_path)

    normalized: dict[str, list[str]] = {}
    for class_name in sorted(refs):
        normalized[class_name] = sorted(set(refs[class_name]))
    return normalized


def materialize_and_save_all(repo_name: str, state: dict) -> None:
    """Materialize all sidecar JSONs from state and save to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Type hierarchy
    hierarchy = materialize_hierarchy(state)
    hierarchy_path = DATA_DIR / f"{repo_name}_type_hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")
    total_impls = sum(len(v) for v in hierarchy.values())
    logger.info("Materialized type hierarchy for %s: %d base types, %d implementations",
                repo_name, len(hierarchy), total_impls)

    # Dependency graph
    dep_graph = materialize_dep_graph(state)
    dep_path = DATA_DIR / f"{repo_name}_dep_graph.json"
    dep_path.write_text(json.dumps(dep_graph, indent=2), encoding="utf-8")
    dep_nodes = dep_graph.get("nodes", {})
    total_edges = sum(len(v.get("deps", [])) for v in dep_nodes.values())
    logger.info("Materialized dep graph for %s: %d nodes, %d edges",
                repo_name, len(dep_nodes), total_edges)

    # Asset references (GUID-keyed canonical + class-keyed compatibility)
    asset_refs_by_guid = materialize_asset_refs_by_guid(state)
    guid_asset_path = DATA_DIR / f"{repo_name}_asset_references_by_guid.json"
    guid_asset_path.write_text(json.dumps(asset_refs_by_guid, indent=2), encoding="utf-8")

    asset_refs = materialize_asset_refs(state)
    asset_path = DATA_DIR / f"{repo_name}_asset_references.json"
    asset_path.write_text(json.dumps(asset_refs, indent=2), encoding="utf-8")
    total_refs = sum(len(v.get("files", [])) for v in asset_refs_by_guid.values())
    logger.info("Materialized asset refs for %s: %d GUIDs, %d total file refs",
                repo_name, len(asset_refs_by_guid), total_refs)

    # Save contribution state itself
    save_state(repo_name, state)
