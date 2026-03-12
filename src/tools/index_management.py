"""reindex, index_status, and repo management tool implementations."""

import json
import logging
from collections.abc import Callable
from datetime import datetime

from src.config import (
    REPOS, REPO_ALIASES, DATA_DIR, resolve_repo,
    add_repo_config, remove_repo_config,
)
from src.indexer.embedder import get_collection, get_active_backend
from src.indexer.pipeline import index_repo
from src.indexer.calibration import load_calibration

logger = logging.getLogger(__name__)

# Type alias for progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None] | None


def reindex(repo: str, incremental: bool = True,
            progress_callback: ProgressCallback = None) -> str:
    """Trigger re-indexing of a repo.

    Args:
        repo: Which repo to reindex ("perception", "mainapp", or "all").
        incremental: If True, only re-index changed files. If False, full rebuild.
        progress_callback: Optional (current, total, message) callback for progress updates.

    Returns:
        Summary of indexing results.
    """
    if repo == "all":
        results = []
        for repo_name in REPOS:
            result = index_repo(repo_name, incremental=incremental,
                                progress_callback=progress_callback)
            results.append(_format_index_result(result))
        return "\n\n".join(results)

    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Unknown repo '{repo}'. Valid: {list(REPOS.keys())} or 'all'."

    result = index_repo(repo, incremental=incremental,
                        progress_callback=progress_callback)
    return _format_index_result(result)


def index_status() -> str:
    """Health check: embedding backend, chunk counts per collection, last index time.

    Returns:
        Formatted status report.
    """
    lines = ["=== Code Index Status ===\n"]

    # Embedding backends by role (do not force initialization)
    index_backend = get_active_backend(role="index")
    search_backend = get_active_backend(role="search")
    if "not initialized" in index_backend:
        lines.append(f"Embedding backend (index role): {index_backend} "
                     f"(search-only mode — use start_reindex to enable indexing)")
    else:
        lines.append(f"Embedding backend (index role): {index_backend}")
    lines.append(f"Embedding backend (search role): {search_backend}")

    # Hardware profile
    try:
        from src.hardware import get_hardware_profile
        hw = get_hardware_profile()
        lines.append(f"Hardware: {hw.summary()}")
    except Exception:
        pass

    # Collection stats
    seen_collections = set()
    for repo_name, config in REPOS.items():
        try:
            collection = get_collection(config.collection_name)
            count = collection.count()
        except Exception:
            count = "N/A (not initialized)"

        manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                file_count = len(manifest)
                mtime = manifest_path.stat().st_mtime
                last_indexed = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                file_count = "?"
                last_indexed = "?"
        else:
            file_count = 0
            last_indexed = "never"

        # Only show chunk count once per collection
        if config.collection_name not in seen_collections:
            lines.append(f"\n{repo_name} ({config.collection_name}):")
            lines.append(f"  Chunks indexed: {count}")
            seen_collections.add(config.collection_name)
        else:
            lines.append(f"\n{repo_name} ({config.collection_name}):")

        # Show all source directories
        for sd in config.source_dirs:
            lines.append(f"  Source dir [{sd.language}]: {sd.path}")
            lines.append(f"    Exists: {sd.path.exists()}")
        lines.append(f"  Files in manifest: {file_count}")
        lines.append(f"  Last indexed: {last_indexed}")

        # Calibration info
        cal = load_calibration(repo_name)
        if cal and "thresholds" in cal:
            t = cal["thresholds"]
            cal_time = cal.get("computed_at", "?")
            # Trim seconds from ISO timestamp for brevity
            if "T" in cal_time:
                cal_time = cal_time.rsplit(":", 1)[0].replace("T", " ")
            lines.append(f"  Calibration: {cal_time} | HIGH<{t['HIGH']} MEDIUM<{t['MEDIUM']} LOW<{t['LOW']}")
        else:
            lines.append("  Calibration: not computed (run full reindex)")

    # Unity coverage check
    lines.append("\n--- Unity Coverage ---")
    for repo_name, config in REPOS.items():
        has_unity = any(sd.language == "unity" for sd in config.source_dirs)
        if has_unity:
            unity_dirs = [sd for sd in config.source_dirs if sd.language == "unity"]
            existing = [sd for sd in unity_dirs if sd.path.exists()]
            lines.append(f"{repo_name}: unity dirs configured ({len(existing)}/{len(unity_dirs)} exist on disk)")
            # Check if unity chunks are actually indexed
            try:
                collection = get_collection(config.collection_name)
                unity_results = collection.get(
                    where={"chunk_type": {"$in": ["prefab_summary", "gameobject", "scriptable_object"]}},
                    limit=1,
                    include=[],
                )
                unity_count = len(unity_results.get("ids", []))
                if unity_count == 0:
                    lines.append(f"  WARNING: No Unity chunks indexed. Run reindex('{repo_name}', incremental=False) to index Unity assets.")
                else:
                    lines.append(f"  Unity chunks present in index.")
            except Exception:
                lines.append(f"  Could not check Unity chunk count.")
        else:
            lines.append(f"{repo_name}: no unity source dirs configured")

    return "\n".join(lines)


def list_repos() -> str:
    """List all configured repos with their paths and status.

    Returns:
        Formatted list of repos showing name, root, source dirs, and path existence.
    """
    lines = ["=== Configured Repos ===\n"]

    for name, config in REPOS.items():
        root_exists = config.root.exists()
        lines.append(f"{name}:")
        lines.append(f"  Root: {config.root}")
        lines.append(f"    Exists: {root_exists}")
        lines.append(f"  Collection: {config.collection_name}")
        for sd in config.source_dirs:
            sd_exists = sd.path.exists()
            lines.append(f"  Source dir [{sd.language}]: {sd.path}")
            lines.append(f"    Exists: {sd_exists}")
            if sd.exclude_patterns:
                lines.append(f"    Excludes: {', '.join(sd.exclude_patterns)}")
        if config.strip_prefixes:
            lines.append(f"  Strip prefixes: {config.strip_prefixes}")

        # Show aliases that point to this repo
        aliases = [a for a, t in REPO_ALIASES.items() if t == name]
        if aliases:
            lines.append(f"  Aliases: {', '.join(aliases)}")
        lines.append("")

    if not REPOS:
        lines.append("(no repos configured)")

    return "\n".join(lines)


def add_repo(name: str, root: str, source_dirs_json: str | list | None = None,
             aliases: str | None = None,
             strip_prefixes: str | None = None) -> str:
    """Add a new repo to the code index.

    Args:
        name: Repo name (e.g. "weld-detect"). Used in search_code(repo=...) and reindex(repo=...).
        root: Root path of the repo. Supports ~ and env vars (e.g. "~/Documents/MyProject").
        source_dirs_json: Optional JSON array of source directories. Each entry needs at minimum
            a "path" field. If omitted, the server auto-detects source directories by scanning
            the repo for known file types (C#, C++, Python, JS, HTML, Unity).
            Example: [{"path": "~/Documents/MyProject/src", "language": "csharp", "exclude_patterns": ["bin/", "obj/"]}]
            Supported languages: "csharp", "cpp", "python", "javascript", "html", "unity".
        aliases: Optional comma-separated alias names (e.g. "wd,weld").
        strip_prefixes: Optional comma-separated .asmdef prefixes to strip (e.g. "MyCompany.,MyCompany.App.").

    Returns:
        Success or error message.
    """
    source_dirs = None
    if source_dirs_json is not None:
        # Handle both str (raw JSON) and list (pre-parsed by MCP framework)
        if isinstance(source_dirs_json, list):
            source_dirs = source_dirs_json
        elif isinstance(source_dirs_json, str):
            try:
                source_dirs = json.loads(source_dirs_json)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON in source_dirs_json: {e}"
        else:
            return f"Error: source_dirs_json must be a JSON string or list, got {type(source_dirs_json).__name__}"

        if not isinstance(source_dirs, list):
            return "Error: source_dirs_json must be a JSON array."

    alias_list = [a.strip() for a in aliases.split(",")] if aliases else None
    prefix_list = [p.strip() for p in strip_prefixes.split(",")] if strip_prefixes else None

    return add_repo_config(name, root, source_dirs, alias_list, prefix_list)


def remove_repo(name: str) -> str:
    """Remove a repo from the code index.

    Deletes the ChromaDB collection, manifest, calibration files, and removes
    the repo from config.local.json.

    Args:
        name: Repo name to remove.

    Returns:
        Success or error message.
    """
    return remove_repo_config(name)


def reindex_file(repo: str, file_path: str) -> str:
    """Re-index a single file: delete old chunks and re-chunk/re-embed it.

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        file_path: Path to the file, relative to the repo root (e.g. "UnityProject/Assets/Scripts/Foo.cs").

    Returns:
        Summary of chunks deleted/added.
    """
    from src.indexer.pipeline import reindex_single_file
    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Error: Unknown repo '{repo}'. Valid: {list(REPOS.keys())}"
    try:
        result = reindex_single_file(repo, file_path)
        return (f"Reindexed '{file_path}' in {repo}:\n"
                f"  Chunks deleted: {result['chunks_deleted']}\n"
                f"  Chunks added: {result['chunks_added']}")
    except Exception as e:
        logger.exception("reindex_file failed")
        return f"Error: {e}"


def remove_file(repo: str, file_path: str) -> str:
    """Remove a single file from the index (delete its chunks from ChromaDB and manifest).

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        file_path: Path to the file, relative to the repo root.

    Returns:
        Summary of chunks removed.
    """
    from src.indexer.pipeline import remove_single_file
    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Error: Unknown repo '{repo}'. Valid: {list(REPOS.keys())}"
    try:
        result = remove_single_file(repo, file_path)
        return (f"Removed '{file_path}' from {repo} index:\n"
                f"  Chunks deleted: {result['chunks_deleted']}")
    except Exception as e:
        logger.exception("remove_file failed")
        return f"Error: {e}"


def rebuild_sidecars(repo: str, skip_unity: bool = False, progress_callback=None) -> str:
    """Rebuild sidecar data (type hierarchy, dep graph, asset refs) from indexed files.

    Re-scans all files in the manifest, re-extracts sidecar contributions, and
    materializes fresh sidecar JSONs. Useful for repair or after schema changes.
    Does NOT re-embed — only rebuilds sidecar data.

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        skip_unity: If True, skip Unity YAML files and only rebuild from code files.
        progress_callback: Optional (current, total, message) callback for progress updates.

    Returns:
        Summary of rebuild results.
    """
    from src.indexer.pipeline import rebuild_sidecars as _rebuild
    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Error: Unknown repo '{repo}'. Valid: {list(REPOS.keys())}"
    try:
        return _rebuild(repo, skip_unity=skip_unity, progress_callback=progress_callback)
    except Exception as e:
        logger.exception("rebuild_sidecars failed")
        return f"Error: {e}"


def _format_index_result(result: dict) -> str:
    if "error" in result:
        return f"Error indexing {result.get('repo', '?')}: {result['error']}"

    setup_s = result.get('setup_seconds', 0)
    index_s = result['elapsed_seconds']
    total_s = round(setup_s + index_s, 1)
    lines = [
        f"=== Indexing complete: {result['repo']} ===",
        f"  Files scanned: {result['files_scanned']}",
        f"  Files indexed: {result['files_indexed']}",
        f"  Files skipped (unchanged): {result['files_skipped']}",
        f"  Chunks added: {result['chunks_added']}",
        f"  Chunks deleted: {result['chunks_deleted']}",
        f"  Total chunks in collection: {result['total_chunks']}",
        f"  Time: {total_s}s (setup {setup_s}s + indexing {index_s}s)",
    ]
    if result.get("errors"):
        lines.append(f"  Errors ({len(result['errors'])}):")
        for err in result["errors"]:
            lines.append(f"    - {err}")
    return "\n".join(lines)
