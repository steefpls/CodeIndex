"""Unity assembly definition (.asmdef) dependency graph."""

import json
import logging
from pathlib import Path

from src.config import REPOS, resolve_repo

logger = logging.getLogger(__name__)

# In-memory cache: repo -> graph dict
_graph_cache: dict[str, dict] = {}


def get_assembly_graph(repo: str = "mainapp", assembly: str | None = None,
                       top_n: int = 20, min_references: int = 0) -> str:
    """Get the Unity assembly definition dependency graph.

    Shows which assemblies reference which other assemblies, based on .asmdef files.

    Args:
        repo: Which repo to query.
        assembly: Optional assembly name to focus on. If None, returns a summary
                  of the top assemblies by dependent count.
        top_n: When showing the full graph (no assembly filter), limit to the top N
               assemblies by number of dependents. Default 20. Use 0 for unlimited.
        min_references: When showing the full graph, only include assemblies with at
                        least this many dependents. Default 0 (no filter).

    Returns:
        Formatted assembly dependency information.
    """
    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    config = REPOS[resolved]
    graph = _get_graph(resolved, config)

    if not graph:
        return f"No .asmdef files found in '{resolved}'."

    if assembly is not None:
        return _format_single(graph, assembly)
    return _format_full(graph, top_n=top_n, min_references=min_references)


def invalidate_graph_cache(repo: str | None = None) -> None:
    """Clear cached graph."""
    if repo is None:
        _graph_cache.clear()
    else:
        _graph_cache.pop(repo, None)


def _get_graph(repo: str, config) -> dict:
    """Build or return cached assembly graph."""
    if repo in _graph_cache:
        return _graph_cache[repo]

    # Try common Unity project layouts:
    # 1. root IS the UnityProject (root/Assets/)
    # 2. root contains a UnityProject subfolder (root/UnityProject/Assets/)
    assets_dir = config.root / "Assets"
    if not assets_dir.exists():
        assets_dir = config.root / "UnityProject" / "Assets"
    if not assets_dir.exists():
        _graph_cache[repo] = {}
        return {}

    # Find all .asmdef files
    asmdef_files = list(assets_dir.rglob("*.asmdef"))
    if not asmdef_files:
        _graph_cache[repo] = {}
        return {}

    # Build GUID -> assembly name map from .asmdef.meta files
    guid_to_name: dict[str, str] = {}
    assemblies: dict[str, dict] = {}  # name -> {path, references_guids, references, referenced_by}

    for asmdef_path in asmdef_files:
        try:
            data = json.loads(asmdef_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("name", asmdef_path.stem)
        refs = data.get("references", [])

        # Read .meta file for this assembly's GUID
        meta_path = Path(str(asmdef_path) + ".meta")
        guid = ""
        if meta_path.exists():
            try:
                meta_text = meta_path.read_text(encoding="utf-8", errors="replace")
                for line in meta_text.split("\n"):
                    if line.startswith("guid:"):
                        guid = line.split(":", 1)[1].strip()
                        break
            except OSError:
                pass
        if guid:
            guid_to_name[guid] = name

        assemblies[name] = {
            "path": str(asmdef_path),
            "references_raw": refs,
            "references": [],
            "referenced_by": [],
        }

    # Resolve GUID references to assembly names
    for name, info in assemblies.items():
        resolved_refs = []
        for ref in info["references_raw"]:
            if ref.startswith("GUID:"):
                guid = ref[5:]
                resolved_name = guid_to_name.get(guid)
                if resolved_name:
                    resolved_refs.append(resolved_name)
                else:
                    resolved_refs.append(f"[{guid[:8]}...]")
            else:
                resolved_refs.append(ref)
        info["references"] = sorted(resolved_refs)
        del info["references_raw"]

    # Build reverse references
    for name, info in assemblies.items():
        for ref in info["references"]:
            if ref in assemblies:
                assemblies[ref]["referenced_by"].append(name)

    for info in assemblies.values():
        info["referenced_by"] = sorted(info["referenced_by"])

    _graph_cache[repo] = assemblies
    return assemblies


def _format_full(graph: dict, top_n: int = 20, min_references: int = 0) -> str:
    """Format assembly graph summary, sorted by number of dependents.

    By default shows the top 20 assemblies to avoid overwhelming output.
    Use top_n=0 for unlimited, or min_references to filter by dependent count.
    """
    sorted_assemblies = sorted(graph.items(),
                                key=lambda x: len(x[1].get("referenced_by", [])),
                                reverse=True)

    # Apply min_references filter
    if min_references > 0:
        sorted_assemblies = [(n, i) for n, i in sorted_assemblies
                             if len(i.get("referenced_by", [])) >= min_references]

    total = len(graph)
    showing = len(sorted_assemblies)

    # Apply top_n cap
    if top_n > 0 and showing > top_n:
        sorted_assemblies = sorted_assemblies[:top_n]
        showing = top_n

    # Header
    if showing < total:
        lines = [f"=== Assembly Graph (showing {showing} of {total} assemblies, "
                 f"sorted by dependent count) ===\n"]
    else:
        lines = [f"=== Assembly Graph ({total} assemblies) ===\n"]

    for name, info in sorted_assemblies:
        refs = info.get("references", [])
        rev_refs = info.get("referenced_by", [])
        resolved_refs = [r for r in refs if not r.startswith("[")]
        unresolved_count = len(refs) - len(resolved_refs)
        lines.append(f"{name}")
        ref_str = ', '.join(resolved_refs) if resolved_refs else '(none)'
        if unresolved_count:
            ref_str += f" (+{unresolved_count} unresolved GUIDs)"
        lines.append(f"  References ({len(resolved_refs)}): {ref_str}")
        lines.append(f"  Referenced by ({len(rev_refs)}): {', '.join(rev_refs) if rev_refs else '(none)'}")
        lines.append("")

    # Pagination hint
    omitted = total - len(sorted_assemblies)
    if omitted > 0:
        lines.append(f"--- {omitted} more assemblies omitted. "
                     f"Use top_n=0 for all, or assembly='<name>' to focus on one. ---")

    return "\n".join(lines).rstrip()


def _format_single(graph: dict, assembly: str) -> str:
    """Format a single assembly's dependencies."""
    info = graph.get(assembly)

    # Fuzzy match
    if info is None:
        lower = assembly.lower()
        matches = [(k, v) for k, v in graph.items() if lower in k.lower()]
        if len(matches) == 1:
            assembly, info = matches[0]
        elif 2 <= len(matches) <= 3:
            # Few enough to show inline — return full details for all matches
            sections = [f"Multiple assemblies match '{assembly}' — showing all {len(matches)}:\n"]
            for k, v in sorted(matches, key=lambda x: x[0]):
                refs = v.get("references", [])
                rev_refs = v.get("referenced_by", [])
                resolved_refs = [r for r in refs if not r.startswith("[")]
                unresolved_refs = [r for r in refs if r.startswith("[")]
                sections.append(f"Assembly: {k}")
                sections.append(f"  References (can use): {', '.join(resolved_refs) if resolved_refs else '(none)'}")
                if unresolved_refs:
                    sections.append(f"  Unresolved GUIDs ({len(unresolved_refs)}): likely Unity packages outside Assets/")
                sections.append(f"  Referenced by (uses this): {', '.join(rev_refs) if rev_refs else '(none)'}")
                sections.append(f"  Full path: {v.get('path', 'unknown')}")
                sections.append("")
            return "\n".join(sections).rstrip()
        elif len(matches) > 3:
            lines = [f"Multiple assemblies match '{assembly}' ({len(matches)} matches, be more specific):"]
            for k, _ in sorted(matches):
                lines.append(f"  {k}")
            return "\n".join(lines)
        else:
            return f"Assembly '{assembly}' not found."

    refs = info.get("references", [])
    rev_refs = info.get("referenced_by", [])

    # Separate resolved refs from unresolved GUIDs
    resolved_refs = [r for r in refs if not r.startswith("[")]
    unresolved_refs = [r for r in refs if r.startswith("[")]

    lines = [f"Assembly: {assembly}"]
    lines.append(f"  References (can use): {', '.join(resolved_refs) if resolved_refs else '(none)'}")
    if unresolved_refs:
        lines.append(f"  Unresolved GUIDs ({len(unresolved_refs)}): likely Unity packages outside Assets/")
    lines.append(f"  Referenced by (uses this): {', '.join(rev_refs) if rev_refs else '(none)'}")
    lines.append(f"\n  Full path: {info.get('path', 'unknown')}")

    return "\n".join(lines)
