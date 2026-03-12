"""lookup_binding tool implementation."""

import json
import logging
from difflib import get_close_matches
from pathlib import Path

from src.config import DATA_DIR, REPOS, REPO_ALIASES, resolve_repo
from src.indexer.zenject_parser import parse_all_installers
from src.models.binding import ZenjectBinding

logger = logging.getLogger(__name__)

_bindings_cache: list[ZenjectBinding] | None = None
BINDINGS_FILE = DATA_DIR / "zenject_bindings.json"


def _get_unity_repo_name() -> str | None:
    """Find the canonical repo name that has Zenject bindings (Unity/MainApp).

    Checks for 'mainapp' directly, then resolves it via aliases, then falls back
    to finding any repo with csharp source dirs containing Zenject installers.
    """
    # Direct match
    if "mainapp" in REPOS:
        return "mainapp"
    # Check if "mainapp" is an alias
    resolved = resolve_repo("mainapp")
    if resolved in REPOS:
        return resolved
    # Fallback: find any repo that looks like a Unity project (has .asmdef or Zenject)
    for name, config in REPOS.items():
        if any(sd.language == "unity" for sd in config.source_dirs):
            return name
    return None


def _load_bindings() -> list[ZenjectBinding]:
    """Load bindings from cache file or parse installers."""
    global _bindings_cache
    if _bindings_cache is not None:
        return _bindings_cache

    if BINDINGS_FILE.exists():
        try:
            data = json.loads(BINDINGS_FILE.read_text(encoding="utf-8"))
            _bindings_cache = [ZenjectBinding.from_dict(d) for d in data]
            return _bindings_cache
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Parse fresh from MainApp installers
    _bindings_cache = _rebuild_bindings()
    return _bindings_cache


def _rebuild_bindings() -> list[ZenjectBinding]:
    """Parse all installer files and save binding data."""
    global _bindings_cache
    unity_repo = _get_unity_repo_name()
    if unity_repo is None:
        return []
    config = REPOS[unity_repo]

    # Zenject installers are C# — use the first csharp source dir
    source_dir = None
    for sd in config.source_dirs:
        if sd.language == "csharp" and sd.path.exists():
            source_dir = sd.path
            break
    if source_dir is None:
        return []

    bindings = parse_all_installers(source_dir)

    # Save to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BINDINGS_FILE.write_text(
        json.dumps([b.to_dict() for b in bindings], indent=2),
        encoding="utf-8",
    )

    _bindings_cache = bindings
    return bindings


def lookup_binding(interface_name: str, repo: str = "mainapp") -> str:
    """Look up Zenject DI bindings by interface or concrete type name.

    Args:
        interface_name: The interface or type to look up (e.g., "IRobotDriver", "NetworkHandler").
        repo: Which repo (currently only the Unity/MainApp repo has Zenject bindings).

    Returns:
        Formatted binding information including concrete type, scope, installer file, and line number.
    """
    resolved = resolve_repo(repo)
    # Find the canonical name of the Unity repo (the one with Zenject bindings)
    unity_repo = _get_unity_repo_name()
    if unity_repo is None:
        return "No Unity/MainApp repo configured. Add one with add_repo() first."
    if resolved != unity_repo:
        return f"Zenject bindings are only available for the '{unity_repo}' repo."

    bindings = _load_bindings()
    if not bindings:
        return f"No Zenject bindings indexed. Run reindex('{unity_repo}') first, or source dir not found."

    # Exact match first
    exact = [b for b in bindings if
             b.interface_name == interface_name or
             b.concrete_type == interface_name]

    # Fuzzy match (case-insensitive substring)
    if not exact:
        query_lower = interface_name.lower()
        exact = [b for b in bindings if
                 query_lower in b.interface_name.lower() or
                 query_lower in b.concrete_type.lower()]

    if not exact:
        suggestion = _suggest_similar_bindings(interface_name, bindings)
        msg = f"No bindings found for '{interface_name}' ({len(bindings)} bindings indexed)."
        if suggestion:
            msg += "\n" + suggestion
        return msg

    lines = [f"Found {len(exact)} binding(s) for '{interface_name}':\n"]
    for b in exact:
        lines.append(f"  {b.binding_type}: {b.interface_name} -> {b.concrete_type}")
        scope_str = f" ({b.scope})" if b.scope else ""
        lines.append(f"  Scope{scope_str}")
        lines.append(f"  Installer: {b.installer_file}:{b.line_number}")
        lines.append("")

    return "\n".join(lines)


def _suggest_similar_bindings(query: str, bindings: list[ZenjectBinding]) -> str:
    """Suggest similar binding names when an exact lookup finds nothing.

    Prioritizes: token overlap > substring > edit distance, so that
    'IRobotDriver' suggests names containing 'Robot' and 'Driver' before
    unrelated names that happen to have a short edit distance.
    """
    all_names: set[str] = set()
    for b in bindings:
        all_names.add(b.interface_name)
        all_names.add(b.concrete_type)

    if not all_names:
        return ""

    # Split query into meaningful tokens (split on camelCase / PascalCase / underscores)
    import re
    query_tokens = set(t.lower() for t in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', query) if len(t) > 1)

    # Score each name by token overlap
    scored: list[tuple[int, str]] = []
    for name in sorted(all_names):
        name_tokens = set(t.lower() for t in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', name) if len(t) > 1)
        overlap = len(query_tokens & name_tokens)
        if overlap > 0:
            scored.append((overlap, name))

    # Sort by overlap descending, take top matches
    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [name for _, name in scored if name != query][:5]

    # Fill remaining slots with substring matches
    if len(matches) < 5:
        lower = query.lower()
        for name in sorted(all_names):
            if name not in matches and name != query and lower in name.lower():
                matches.append(name)
                if len(matches) >= 5:
                    break

    # Fall back to edit distance if still short
    if len(matches) < 3:
        edit_matches = get_close_matches(query, sorted(all_names), n=6 - len(matches), cutoff=0.4)
        for m in edit_matches:
            if m not in matches and m != query:
                matches.append(m)

    if not matches:
        return ""

    return "Did you mean: " + ", ".join(matches[:5]) + "?"


def rebuild_zenject_bindings() -> str:
    """Force rebuild of Zenject binding index."""
    bindings = _rebuild_bindings()
    return f"Rebuilt Zenject bindings: {len(bindings)} bindings found."
