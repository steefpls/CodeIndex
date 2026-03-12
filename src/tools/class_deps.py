"""Class dependency graph lookup: what a class depends on and what depends on it."""

import json
import logging
from collections import Counter

from src.config import REPOS, resolve_repo
from src.indexer.dep_graph_builder import load_dep_graph

logger = logging.getLogger(__name__)

# In-memory cache: repo -> normalized graph:
# {"schema_version": 2, "nodes": {...}, "name_index": {...}}
_dep_cache: dict[str, dict] = {}
# Reverse index cache: repo -> {node_key: [dependent_node_key, ...]}
_reverse_cache: dict[str, dict[str, list[str]]] = {}


def get_class_dependencies(class_name: str | None = None, repo: str = "mainapp",
                           output_format: str = "text") -> str:
    """Get class dependency information.

    If class_name provided: show what it depends on + what depends on it.
    If class_name is None: show summary stats.

    Args:
        class_name: Class or node key (e.g., "IRobotDriver", "MyNs.PathOptimizerController").
        repo: Which repo to query.

    Args:
        output_format: "text" (default) or "json".

    Returns:
        Dependency information in text or JSON format.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    graph = _get_graph(resolved)
    nodes = graph.get("nodes", {})

    if not nodes:
        return (f"No dependency graph data for '{resolved}'. "
                f"Run reindex('{resolved}', incremental=False) to build it.")

    if class_name is None:
        if output_format == "json":
            return _summary_payload_json(graph, resolved)
        return _format_summary(graph, resolved)

    if output_format == "json":
        return _class_payload_json(graph, class_name, resolved)
    return _format_class(graph, class_name, resolved)


def invalidate_dep_cache(repo: str | None = None) -> None:
    """Clear cached graph after reindex."""
    if repo is None:
        _dep_cache.clear()
        _reverse_cache.clear()
    else:
        _dep_cache.pop(repo, None)
        _reverse_cache.pop(repo, None)


def _get_graph(repo: str) -> dict:
    """Load and normalize graph with caching."""
    if repo not in _dep_cache:
        _dep_cache[repo] = load_dep_graph(repo)

    cached = _dep_cache[repo]
    if not _is_normalized_graph(cached):
        cached = _normalize_graph(cached)
        _dep_cache[repo] = cached
        _reverse_cache.pop(repo, None)
    return cached


def _is_normalized_graph(graph: dict) -> bool:
    return (
        isinstance(graph, dict)
        and graph.get("schema_version") == 2
        and isinstance(graph.get("nodes"), dict)
        and isinstance(graph.get("name_index"), dict)
    )


def _infer_class_name_from_key(node_key: str) -> str:
    if "@@" in node_key:
        return node_key.split("@@", 1)[0]
    if "." in node_key:
        return node_key.rsplit(".", 1)[-1]
    return node_key


def _normalize_graph(raw_graph: dict) -> dict:
    """Normalize either V1 or V2 dep graph to a V2 in-memory shape."""
    if not isinstance(raw_graph, dict) or not raw_graph:
        return {"schema_version": 2, "nodes": {}, "name_index": {}}

    # V2 from sidecar materialization
    if raw_graph.get("schema_version") == 2 and isinstance(raw_graph.get("nodes"), dict):
        nodes: dict[str, dict] = {}
        for node_key, raw_node in raw_graph["nodes"].items():
            if not isinstance(raw_node, dict):
                continue
            class_name = raw_node.get("class_name", "") or _infer_class_name_from_key(node_key)
            namespace = raw_node.get("namespace", "") or ""
            module = raw_node.get("module", "") or ""
            files = raw_node.get("files", [])
            deps = raw_node.get("deps", [])

            if not isinstance(files, list):
                files = [files] if files else []
            if not isinstance(deps, list):
                deps = []

            nodes[node_key] = {
                "class_name": class_name,
                "namespace": namespace,
                "module": module,
                "files": sorted({str(f) for f in files if f}),
                "deps": sorted({str(d) for d in deps if d and d != node_key}),
            }

        name_index_sets: dict[str, set[str]] = {}
        raw_index = raw_graph.get("name_index")
        if isinstance(raw_index, dict):
            for class_name, keys in raw_index.items():
                if not isinstance(keys, list):
                    continue
                for key in keys:
                    if key in nodes:
                        name_index_sets.setdefault(class_name, set()).add(key)

        for node_key, node in nodes.items():
            name_index_sets.setdefault(node["class_name"], set()).add(node_key)

        name_index = {name: sorted(keys) for name, keys in sorted(name_index_sets.items())}
        return {
            "schema_version": 2,
            "nodes": dict(sorted(nodes.items())),
            "name_index": name_index,
        }

    # V1 legacy shape: {"ClassA": {"file": "...", "module": "...", "namespace": "...", "deps": ["ClassB"]}}
    nodes: dict[str, dict] = {}
    name_index_sets: dict[str, set[str]] = {}
    for class_name, info in raw_graph.items():
        if not isinstance(info, dict):
            continue
        file_path = info.get("file", "")
        module = info.get("module", "") or ""
        namespace = info.get("namespace", "") or ""
        deps = info.get("deps", [])
        if not isinstance(deps, list):
            deps = []

        node_key = class_name
        nodes[node_key] = {
            "class_name": class_name,
            "namespace": namespace,
            "module": module,
            "files": [file_path] if file_path else [],
            "deps": [str(d) for d in deps if d],
        }
        name_index_sets.setdefault(class_name, set()).add(node_key)

    name_index = {name: sorted(keys) for name, keys in sorted(name_index_sets.items())}

    # Resolve legacy dep names through class-name index so callers always work with node keys
    for node_key, node in nodes.items():
        resolved = set()
        for dep_name in node["deps"]:
            for dep_key in name_index.get(dep_name, []):
                if dep_key != node_key:
                    resolved.add(dep_key)
        node["deps"] = sorted(resolved)

    return {
        "schema_version": 2,
        "nodes": dict(sorted(nodes.items())),
        "name_index": name_index,
    }


def _get_reverse(repo: str, graph: dict) -> dict[str, list[str]]:
    """Build or return cached reverse index: node -> [nodes that depend on it]."""
    if repo not in _reverse_cache:
        normalized = graph if _is_normalized_graph(graph) else _normalize_graph(graph)
        reverse: dict[str, list[str]] = {}
        for node_key, info in normalized.get("nodes", {}).items():
            for dep in info.get("deps", []):
                reverse.setdefault(dep, []).append(node_key)

        for dep in reverse:
            reverse[dep] = sorted(set(reverse[dep]))
        _reverse_cache[repo] = reverse

    return _reverse_cache[repo]


def _format_summary(graph: dict, repo: str) -> str:
    """Format graph summary with stats."""
    nodes = graph.get("nodes", {})
    total_edges = sum(len(info.get("deps", [])) for info in nodes.values())
    reverse = _get_reverse(repo, graph)

    dependent_counts = Counter({node_key: len(dependents) for node_key, dependents in reverse.items()})
    top_depended = dependent_counts.most_common(15)

    dep_counts = Counter({node_key: len(info.get("deps", [])) for node_key, info in nodes.items()})
    top_deps = dep_counts.most_common(15)

    lines = [f"=== Dependency Graph: {repo} ({len(nodes)} classes, {total_edges} edges) ===\n"]

    lines.append("Most depended-on:")
    for node_key, count in top_depended:
        lines.append(f"  {_display_node(node_key, nodes)} ({count} dependents)")

    lines.append("")
    lines.append("Most dependencies:")
    for node_key, count in top_deps:
        if count > 0:
            lines.append(f"  {_display_node(node_key, nodes)} ({count} deps)")

    return "\n".join(lines)


def _summary_payload_json(graph: dict, repo: str) -> str:
    """Return graph summary as structured JSON."""
    nodes = graph.get("nodes", {})
    total_edges = sum(len(info.get("deps", [])) for info in nodes.values())
    reverse = _get_reverse(repo, graph)

    dependent_counts = Counter({node_key: len(dependents) for node_key, dependents in reverse.items()})
    dep_counts = Counter({node_key: len(info.get("deps", [])) for node_key, info in nodes.items()})

    payload = {
        "status": "summary",
        "repo": repo,
        "nodes": len(nodes),
        "edges": total_edges,
        "most_depended_on": [
            {
                "node_key": node_key,
                "class_name": nodes.get(node_key, {}).get("class_name", _infer_class_name_from_key(node_key)),
                "dependents": count,
            }
            for node_key, count in dependent_counts.most_common(15)
        ],
        "most_dependencies": [
            {
                "node_key": node_key,
                "class_name": nodes.get(node_key, {}).get("class_name", _infer_class_name_from_key(node_key)),
                "deps": count,
            }
            for node_key, count in dep_counts.most_common(15) if count > 0
        ],
    }
    return json.dumps(payload, indent=2)


def _node_ref_payload(nodes: dict[str, dict], node_key: str) -> dict:
    node = nodes.get(node_key, {})
    files = node.get("files", [])
    return {
        "node_key": node_key,
        "class_name": node.get("class_name", _infer_class_name_from_key(node_key)),
        "namespace": node.get("namespace", ""),
        "module": node.get("module", ""),
        "file": files[0] if files else "",
    }


def _node_payload(graph: dict, node_key: str, repo: str) -> dict:
    nodes = graph.get("nodes", {})
    node = nodes.get(node_key, {})
    reverse = _get_reverse(repo, graph)
    dependents = sorted(reverse.get(node_key, []))
    deps = node.get("deps", [])

    return {
        "status": "found",
        "repo": repo,
        "query": node_key,
        "node_key": node_key,
        "class_name": node.get("class_name", _infer_class_name_from_key(node_key)),
        "namespace": node.get("namespace", ""),
        "module": node.get("module", ""),
        "files": node.get("files", []),
        "depends_on": [_node_ref_payload(nodes, dep) for dep in deps],
        "depended_on_by": [_node_ref_payload(nodes, dep) for dep in dependents],
    }


def _aggregate_payload(graph: dict, node_keys: list[str], repo: str) -> dict:
    nodes = graph.get("nodes", {})
    reverse = _get_reverse(repo, graph)
    files = sorted({f for key in node_keys for f in nodes.get(key, {}).get("files", [])})
    deps = sorted({
        dep
        for key in node_keys
        for dep in nodes.get(key, {}).get("deps", [])
        if dep not in node_keys
    })
    dependents = sorted({
        dep
        for key in node_keys
        for dep in reverse.get(key, [])
        if dep not in node_keys
    })

    return {
        "files": files,
        "depends_on": [_node_ref_payload(nodes, dep) for dep in deps],
        "depended_on_by": [_node_ref_payload(nodes, dep) for dep in dependents],
    }


def _disambiguation_payload(graph: dict, class_query: str, node_keys: list[str], repo: str) -> dict:
    nodes = graph.get("nodes", {})
    payload = {
        "status": "disambiguation",
        "repo": repo,
        "query": class_query,
        "matches": [_node_ref_payload(nodes, node_key) for node_key in sorted(node_keys)],
    }
    aggregate_key = _aggregate_identity(nodes, node_keys)
    if aggregate_key:
        payload["aggregate"] = {
            "identity": aggregate_key,
            **_aggregate_payload(graph, node_keys, repo),
        }
    return payload


def _class_payload_json(graph: dict, class_query: str, repo: str) -> str:
    """Return class dependency lookup as structured JSON."""
    nodes = graph.get("nodes", {})
    name_index = graph.get("name_index", {})

    if class_query in nodes:
        return json.dumps(_node_payload(graph, class_query, repo), indent=2)

    keys = name_index.get(class_query, [])
    if len(keys) == 1:
        return json.dumps(_node_payload(graph, keys[0], repo), indent=2)
    if len(keys) > 1:
        return json.dumps(_disambiguation_payload(graph, class_query, keys, repo), indent=2)

    matches = _fuzzy_match_nodes(graph, class_query)
    if len(matches) == 1:
        return json.dumps(_node_payload(graph, matches[0], repo), indent=2)
    if len(matches) > 1:
        return json.dumps(_disambiguation_payload(graph, class_query, matches, repo), indent=2)

    return json.dumps({
        "status": "not_found",
        "repo": repo,
        "query": class_query,
        "message": f"Class '{class_query}' not found in {repo} dependency graph.",
    }, indent=2)


def _format_class(graph: dict, class_query: str, repo: str) -> str:
    """Format dependencies for a specific class or node key."""
    nodes = graph.get("nodes", {})
    name_index = graph.get("name_index", {})

    # Exact node key match
    if class_query in nodes:
        return _format_node(graph, class_query, repo)

    # Exact class-name match via index
    keys = name_index.get(class_query, [])
    if len(keys) == 1:
        return _format_node(graph, keys[0], repo)
    if len(keys) > 1:
        return _format_disambiguation(graph, class_query, keys, repo)

    # Fuzzy match across class names and node keys
    matches = _fuzzy_match_nodes(graph, class_query)
    if len(matches) == 1:
        return _format_node(graph, matches[0], repo)
    if len(matches) > 1:
        return _format_disambiguation(graph, class_query, matches, repo)

    return f"Class '{class_query}' not found in {repo} dependency graph."


def _fuzzy_match_nodes(graph: dict, query: str) -> list[str]:
    lower = query.lower()
    nodes = graph.get("nodes", {})
    name_index = graph.get("name_index", {})

    matches: set[str] = set()
    for class_name, keys in name_index.items():
        if class_name.lower() == lower or lower in class_name.lower():
            matches.update(keys)

    if not matches:
        for node_key in nodes:
            if node_key.lower() == lower or lower in node_key.lower():
                matches.add(node_key)

    return sorted(matches)


def _format_disambiguation(graph: dict, class_query: str, node_keys: list[str], repo: str) -> str:
    nodes = graph.get("nodes", {})
    lines = [f"Multiple classes match '{class_query}':"]
    for node_key in sorted(node_keys):
        node = nodes.get(node_key, {})
        files = node.get("files", [])
        if not files:
            file_info = "no file info"
        elif len(files) == 1:
            file_info = files[0]
        else:
            file_info = f"{files[0]} (+{len(files) - 1} more files)"
        lines.append(f"  {node_key}  ({file_info})")

    aggregate_key = _aggregate_identity(nodes, node_keys)
    if aggregate_key:
        lines.append("")
        lines.append(f"Aggregate view for partial class '{aggregate_key}':")
        lines.extend(_format_aggregate(graph, node_keys, repo))

    lines.append("")
    lines.append("Tip: pass a full node key for an exact view.")
    return "\n".join(lines)


def _aggregate_identity(nodes: dict[str, dict], node_keys: list[str]) -> str | None:
    if len(node_keys) < 2:
        return None
    class_names = {nodes.get(k, {}).get("class_name", "") for k in node_keys}
    namespaces = {nodes.get(k, {}).get("namespace", "") for k in node_keys}
    if len(class_names) != 1 or len(namespaces) != 1:
        return None

    class_name = next(iter(class_names))
    namespace = next(iter(namespaces))
    if not class_name:
        return None
    if namespace:
        return f"{namespace}.{class_name}"
    return class_name


def _format_aggregate(graph: dict, node_keys: list[str], repo: str) -> list[str]:
    nodes = graph.get("nodes", {})
    reverse = _get_reverse(repo, graph)

    files = sorted({f for key in node_keys for f in nodes.get(key, {}).get("files", [])})
    deps = sorted({
        dep
        for key in node_keys
        for dep in nodes.get(key, {}).get("deps", [])
        if dep not in node_keys
    })
    dependents = sorted({
        dep
        for key in node_keys
        for dep in reverse.get(key, [])
        if dep not in node_keys
    })

    lines: list[str] = []
    lines.append(f"  Files ({len(files)}):")
    if files:
        for file_path in files:
            lines.append(f"    {file_path}")
    else:
        lines.append("    (none)")

    lines.append(f"  Depends on ({len(deps)}):")
    if deps:
        for dep in deps:
            lines.append(f"    {dep}{_node_suffix(nodes, dep)}")
    else:
        lines.append("    (none)")

    lines.append(f"  Depended on by ({len(dependents)}):")
    if dependents:
        for dep in dependents:
            lines.append(f"    {dep}{_node_suffix(nodes, dep)}")
    else:
        lines.append("    (none)")

    return lines


def _display_node(node_key: str, nodes: dict[str, dict]) -> str:
    node = nodes.get(node_key, {})
    class_name = node.get("class_name", node_key)
    return class_name if node_key == class_name else node_key


def _node_suffix(nodes: dict[str, dict], node_key: str) -> str:
    node = nodes.get(node_key, {})
    module = node.get("module", "")
    files = node.get("files", [])
    file_path = files[0] if files else ""
    if module and file_path:
        return f"  {module} / {file_path}"
    if module:
        return f"  {module}"
    if file_path:
        return f"  {file_path}"
    return ""


def _format_node(graph: dict, node_key: str, repo: str) -> str:
    nodes = graph.get("nodes", {})
    node = nodes.get(node_key, {})
    reverse = _get_reverse(repo, graph)

    dependents = sorted(reverse.get(node_key, []))
    deps = node.get("deps", [])
    files = node.get("files", [])

    lines = [f"=== {node_key} ({node.get('module', '') or 'no module'}) ==="]
    lines.append(f"Class: {node.get('class_name', _infer_class_name_from_key(node_key))}")
    if node.get("namespace"):
        lines.append(f"Namespace: {node['namespace']}")

    if len(files) == 1:
        lines.append(f"File: {files[0]}")
    elif files:
        lines.append(f"Files ({len(files)}):")
        for file_path in files:
            lines.append(f"  {file_path}")
    else:
        lines.append("File: unknown")

    lines.append("")
    lines.append(f"Depends on ({len(deps)}):")
    if deps:
        for dep in deps:
            lines.append(f"  {dep}{_node_suffix(nodes, dep)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"Depended on by ({len(dependents)}):")
    if dependents:
        for dep in dependents:
            lines.append(f"  {dep}{_node_suffix(nodes, dep)}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)
