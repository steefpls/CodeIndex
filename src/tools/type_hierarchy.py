"""Type hierarchy lookup: find all implementations of an interface or base class."""

import logging

from src.config import REPOS, resolve_repo
from src.indexer.hierarchy_builder import load_type_hierarchy

logger = logging.getLogger(__name__)

# In-memory cache: repo -> hierarchy dict
_hierarchy_cache: dict[str, dict] = {}


def find_implementations(type_name: str, repo: str = "mainapp",
                         max_results: int = 50, offset: int = 0) -> str:
    """Find all classes that implement an interface or extend a base class.

    Args:
        type_name: Interface or base class name (e.g., "IRobotDriver", "MonoBehaviour").
        repo: Which repo to search.
        max_results: Maximum results to return (default 50). Use 0 for unlimited.
        offset: Skip the first N results (for pagination). Default 0.

    Returns:
        Formatted list of implementing classes with file paths.
    """
    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    hierarchy = _get_hierarchy(resolved)

    if not hierarchy:
        return (f"No type hierarchy data for '{resolved}'. "
                f"Run reindex('{resolved}', incremental=False) to build it.")

    # Exact match first
    implementations = hierarchy.get(type_name)

    # Fuzzy match if no exact hit
    if not implementations:
        lower = type_name.lower()
        matches: list[tuple[str, list[dict]]] = []
        for key, impls in hierarchy.items():
            if key.lower() == lower:
                matches.append((key, impls))
            elif lower in key.lower():
                matches.append((key, impls))

        if len(matches) == 1:
            type_name, implementations = matches[0]
        elif len(matches) > 1:
            lines = [f"Multiple types match '{type_name}':"]
            for key, impls in sorted(matches, key=lambda x: x[0]):
                lines.append(f"  {key} ({len(impls)} implementations)")
            return "\n".join(lines)
        else:
            return f"No implementations found for '{type_name}' in {resolved}."

    total = len(implementations)

    # Sort by module grouping (then class name within each module) so the first
    # page surfaces logically grouped results instead of an alphabetical list.
    sorted_impls = sorted(implementations, key=lambda x: (x.get("module") or "", x["class"]))

    # Apply offset
    if offset > 0:
        sorted_impls = sorted_impls[offset:]

    # Apply max_results cap (0 = unlimited)
    showing_count = len(sorted_impls)
    if max_results > 0 and showing_count > max_results:
        sorted_impls = sorted_impls[:max_results]
        showing_count = max_results

    # Collect distinct modules for summary
    all_modules = sorted({i.get("module") or "(no module)" for i in implementations})

    # Build header
    if offset > 0 or (max_results > 0 and total > offset + showing_count):
        range_start = offset + 1
        range_end = offset + showing_count
        lines = [f"Found {total} implementation(s) of '{type_name}' "
                 f"across {len(all_modules)} module(s) "
                 f"(showing {range_start}-{range_end} of {total}):\n"]
    else:
        lines = [f"Found {total} implementation(s) of '{type_name}' "
                 f"across {len(all_modules)} module(s):\n"]

    current_module = None
    for impl in sorted_impls:
        mod = impl.get("module") or "(no module)"
        if mod != current_module:
            current_module = mod
            lines.append(f"  [{mod}]")
        lines.append(f"    {impl['class']}")
        lines.append(f"      File: {impl['file']}")
        if impl.get("namespace"):
            lines.append(f"      Namespace: {impl['namespace']}")
        lines.append("")

    # Add pagination hint if there are more results
    remaining = total - offset - showing_count
    if remaining > 0:
        lines.append(f"--- {remaining} more. Use offset={offset + showing_count} to see next page. ---")

    return "\n".join(lines).rstrip()


def invalidate_hierarchy_cache(repo: str | None = None) -> None:
    """Clear cached hierarchy after reindex."""
    if repo is None:
        _hierarchy_cache.clear()
    else:
        _hierarchy_cache.pop(repo, None)


def _get_hierarchy(repo: str) -> dict:
    """Load hierarchy with caching."""
    if repo not in _hierarchy_cache:
        _hierarchy_cache[repo] = load_type_hierarchy(repo)
    return _hierarchy_cache[repo]
