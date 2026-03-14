"""Spreading activation and graph algorithms on the class dependency graph.

Operates on the dep_graph sidecar (V2 schema) by converting it to a NetworkX
DiGraph for algorithm support. Results are cached per-repo.
"""

import logging

import networkx as nx

from src.config import resolve_repo
from src.indexer.dep_graph_builder import load_dep_graph

logger = logging.getLogger(__name__)

# Cache: repo -> NetworkX DiGraph
_nx_cache: dict[str, nx.DiGraph] = {}


def invalidate_graph_cache(repo: str | None = None) -> None:
    """Clear cached NetworkX graph (e.g. after reindex)."""
    if repo is None:
        _nx_cache.clear()
    else:
        _nx_cache.pop(repo, None)


def _get_nx_graph(repo: str) -> nx.DiGraph:
    """Load dep_graph sidecar and convert to NetworkX DiGraph with caching."""
    if repo in _nx_cache:
        return _nx_cache[repo]

    raw = load_dep_graph(repo)
    g = nx.DiGraph()

    # Handle V2 schema
    nodes = raw.get("nodes", {})
    if not nodes and raw and "schema_version" not in raw:
        # V1 fallback: {ClassName: {file, module, namespace, deps}}
        nodes = {}
        for class_name, info in raw.items():
            if isinstance(info, dict) and "deps" in info:
                node_key = class_name
                nodes[node_key] = {
                    "class_name": class_name,
                    "namespace": info.get("namespace", ""),
                    "module": info.get("module", ""),
                    "files": [info["file"]] if "file" in info else info.get("files", []),
                    "deps": info.get("deps", []),
                }

    # Add nodes and edges
    for node_key, node_data in nodes.items():
        g.add_node(node_key, **{
            "class_name": node_data.get("class_name", node_key),
            "namespace": node_data.get("namespace", ""),
            "module": node_data.get("module", ""),
        })

    # Build name index for dep resolution
    name_index = raw.get("name_index", {})
    # Also build a reverse lookup: node_key -> node_key (identity) for direct matches
    all_keys = set(nodes.keys())

    for node_key, node_data in nodes.items():
        for dep in node_data.get("deps", []):
            # Resolve dep to node key(s)
            if dep in all_keys:
                g.add_edge(node_key, dep)
            elif dep in name_index:
                for target_key in name_index[dep]:
                    if target_key in all_keys:
                        g.add_edge(node_key, target_key)

    _nx_cache[repo] = g
    logger.info("Built NetworkX graph for '%s': %d nodes, %d edges",
                repo, g.number_of_nodes(), g.number_of_edges())
    return g


def spread_activation(repo: str, seed_class_names: set[str],
                      decay: float = 0.7, max_hops: int = 2,
                      top_k: int = 10) -> dict[str, float]:
    """Spreading activation from seed classes through the dependency graph.

    Args:
        repo: Repo name.
        seed_class_names: Class names (or node keys) from vector search results.
        decay: Energy decay per hop (default 0.7).
        max_hops: Max traversal depth (default 2).
        top_k: Keep only top-K activated nodes per hop (default 10).

    Returns:
        Dict mapping node_key -> activation energy (excludes seeds).
    """
    graph = _get_nx_graph(repo)
    if graph.number_of_nodes() == 0:
        return {}

    # Resolve class names to node keys
    seed_keys = set()
    for name in seed_class_names:
        if graph.has_node(name):
            seed_keys.add(name)
        else:
            # Try matching by class_name attribute
            for nk, data in graph.nodes(data=True):
                if data.get("class_name") == name:
                    seed_keys.add(nk)

    if not seed_keys:
        return {}

    activation = {sid: 1.0 for sid in seed_keys}
    total_energy: dict[str, float] = dict(activation)

    frontier = dict(activation)
    for hop in range(max_hops):
        next_frontier: dict[str, float] = {}

        for node_id, energy in frontier.items():
            # Outgoing (dependencies)
            for target in graph.successors(node_id):
                next_frontier[target] = next_frontier.get(target, 0.0) + energy * decay

            # Incoming (dependents)
            for source in graph.predecessors(node_id):
                next_frontier[source] = next_frontier.get(source, 0.0) + energy * decay

        if not next_frontier:
            break

        # Lateral inhibition
        sorted_next = sorted(next_frontier.items(), key=lambda x: -x[1])[:top_k]
        frontier = dict(sorted_next)

        for node_id, energy in frontier.items():
            total_energy[node_id] = total_energy.get(node_id, 0.0) + energy

    # Remove seeds
    for sid in seed_keys:
        total_energy.pop(sid, None)

    return total_energy


def compute_pagerank(repo: str, top_n: int = 20,
                     alpha: float = 0.85) -> list[dict]:
    """Compute PageRank over the dependency graph.

    Args:
        repo: Repo name.
        top_n: Number of top-ranked classes to return.
        alpha: Damping factor.

    Returns:
        List of dicts with node_key, class_name, namespace, module, pagerank.
    """
    graph = _get_nx_graph(repo)
    if graph.number_of_nodes() == 0:
        return []

    try:
        pr = nx.pagerank(graph, alpha=alpha)
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(graph, alpha=0.85, max_iter=200)

    sorted_pr = sorted(pr.items(), key=lambda x: -x[1])[:top_n]
    results = []
    for node_key, score in sorted_pr:
        data = graph.nodes[node_key]
        results.append({
            "node_key": node_key,
            "class_name": data.get("class_name", node_key),
            "namespace": data.get("namespace", ""),
            "module": data.get("module", ""),
            "pagerank": round(score, 6),
        })
    return results


def detect_communities(repo: str) -> list[list[dict]]:
    """Detect communities in the dependency graph using Louvain.

    Returns:
        List of communities, each a list of dicts with node_key, class_name,
        module. Sorted by size descending.
    """
    graph = _get_nx_graph(repo)
    if graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected()
    try:
        communities = nx.community.louvain_communities(undirected)
    except Exception as e:
        logger.warning("Community detection failed for '%s': %s", repo, e)
        return []

    result = []
    for community in sorted(communities, key=len, reverse=True):
        members = []
        for node_key in community:
            data = graph.nodes.get(node_key, {})
            members.append({
                "node_key": node_key,
                "class_name": data.get("class_name", node_key),
                "module": data.get("module", ""),
            })
        result.append(members)
    return result
