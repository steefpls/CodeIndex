"""Graph analysis tool — PageRank, community detection on the dependency graph."""

import json
import logging

from src.config import REPOS, resolve_repo
from src.graph.activation import compute_pagerank, detect_communities

logger = logging.getLogger(__name__)


def analyze_codebase(repo: str = "mainapp", top_n: int = 20,
                     output_format: str = "text") -> str:
    """Analyze the class dependency graph: PageRank centrality and community detection.

    Args:
        repo: Which repo to analyze (default "mainapp").
        top_n: Number of top PageRank results (default 20, max 50).
        output_format: "text" (default) or "json".

    Returns:
        Analysis results.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    top_n = min(max(top_n, 1), 50)

    # PageRank
    pr_results = compute_pagerank(resolved, top_n=top_n)

    # Communities
    communities = detect_communities(resolved)
    # Cap at 15 communities for output
    communities = communities[:15]

    if output_format == "json":
        return json.dumps({
            "repo": resolved,
            "pagerank": pr_results,
            "communities": communities,
            "community_count": len(communities),
        }, indent=2)

    # Text format
    lines = [f"Codebase Analysis: {resolved}", ""]

    # PageRank section
    if pr_results:
        lines.append(f"PageRank — Top {len(pr_results)} most central classes:")
        for i, item in enumerate(pr_results, 1):
            mod = f" [{item['module']}]" if item.get("module") else ""
            ns = f" ({item['namespace']})" if item.get("namespace") else ""
            lines.append(f"  {i}. {item['class_name']}{ns}{mod} — PR: {item['pagerank']:.6f}")
    else:
        lines.append("PageRank: no dependency graph data. Run reindex() first.")
    lines.append("")

    # Communities section
    if communities:
        lines.append(f"Communities — {len(communities)} detected (Louvain):")
        for i, members in enumerate(communities, 1):
            # Group by module for readability
            modules = {}
            for m in members:
                mod = m.get("module") or "(no module)"
                modules.setdefault(mod, []).append(m["class_name"])

            lines.append(f"  Community {i} ({len(members)} classes):")
            for mod, classes in sorted(modules.items()):
                shown = classes[:6]
                extra = f", ...+{len(classes) - 6}" if len(classes) > 6 else ""
                lines.append(f"    [{mod}] {', '.join(shown)}{extra}")
    else:
        lines.append("Communities: no data.")

    return "\n".join(lines)
