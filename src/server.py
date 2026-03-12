"""FastMCP server entry point with tool definitions."""

import logging
import os
import sys
from pathlib import Path

# Add project root to path so imports work when run directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import LOG_FILE, DATA_DIR
from src.tools.search import search_code as _search_code, get_file_chunks as _get_file_chunks, invalidate_calibration_cache, invalidate_modules_cache, start_search_init
from src.tools.zenject import lookup_binding as _lookup_binding, rebuild_zenject_bindings
from src.tools.index_management import (
    reindex as _reindex, index_status as _index_status,
    list_repos as _list_repos, add_repo as _add_repo, remove_repo as _remove_repo,
    reindex_file as _reindex_file, remove_file as _remove_file,
    rebuild_sidecars as _rebuild_sidecars,
)
from src.tools.references import find_references as _find_references
from src.tools.type_hierarchy import find_implementations as _find_implementations, invalidate_hierarchy_cache
from src.tools.assembly_graph import get_assembly_graph as _get_assembly_graph, invalidate_graph_cache
from src.tools.asset_references import find_asset_references as _find_asset_references, invalidate_asset_ref_cache
from src.tools.project_info import get_project_info as _get_project_info
from src.tools.class_deps import get_class_dependencies as _get_class_dependencies, invalidate_dep_cache
from src.tools.unity_context import get_unity_entity_context as _get_unity_entity_context
from src.indexer.embedder import get_embedding_function, get_active_backend, release_embedding_function

# Configure logging to file (never stdout -- stdio transport)
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("code-index")

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("code-index")


def _startup_check():
    """Validate configuration and optionally prewarm embedding backends."""
    import time
    t0 = time.perf_counter()

    # Validate repo configs early so warnings appear at the top of the log
    from src.config import validate_repos
    validate_repos()

    prewarm = os.environ.get("CODERANK_STARTUP_PREWARM", "0").strip().lower() in {"1", "true", "yes", "on"}
    if prewarm:
        logger.info("Startup prewarm enabled: initializing index and search embedders.")
        index_ef = get_embedding_function(role="index")
        search_ef = get_embedding_function(role="search")
        logger.info("Startup: index backend = %s", index_ef.backend)
        logger.info("Startup: search backend = %s", search_ef.backend)
        # Warm up index backend only; search is CPU-by-default and cheap to lazy-init.
        index_ef.warmup()
        index_ef.warmup_all_shapes()
    else:
        logger.info("Startup prewarm disabled (default). Embedders initialize on-demand.")

    # Start search backend init eagerly in background so it's ready by first query.
    # Without this, the first search_code call triggers init and stalls for ~30-40s.
    start_search_init()

    elapsed = time.perf_counter() - t0
    logger.info("Startup complete in %.1fs", elapsed)


@mcp.tool()
def search_code(query: str, repo: str = "all", n_results: int = 10,
                module: str | None = None, chunk_type: str | None = None,
                offset: int = 0, file_path: str | None = None,
                output_format: str = "text") -> str:
    """Semantic search across indexed C#, C++, Python, JavaScript, HTML code,
    and Unity prefab/scene files.

    Searches code chunks using natural language queries. Returns source snippets
    with file paths, line numbers, and chunk IDs.

    Best for: discovering code by describing what it does (e.g., "hand-eye
    calibration workflow", "TCP pose transformation"). For exact symbol name
    lookups (e.g., finding all usages of "HardwareManager"), use find_references
    instead — it does exact text matching and won't miss results.

    Args:
        query: Natural language description of what you're looking for
               (e.g., "hand-eye calibration error handling", "TCP network connection",
               "camera pivot prefab", "which prefab uses LeanTouchCameraController").
        repo: Which repo to search — defaults to "all" (searches all repos and merges
              results by relevance). Use a specific repo name to narrow the search:
              "perception", "mainapp", "pipeline", etc. Use list_repos() to see
              available repos. "pipeline" is an alias for "perception" (C++ pipeline
              shares the perception_code collection).
        n_results: Number of results to return (1-30, default 10).
        module: Optional module filter (e.g., "Robotics", "Network", "Trajectory").
        chunk_type: Optional filter - "method", "class_summary", "whole_class", "constructor", "property",
                    "function", "component", "template", "prefab_summary", "gameobject".
        offset: Skip the first N results (for pagination). Default 0.
        file_path: Optional path prefix filter (e.g., "UnityProject/Assets/Prefabs/",
                   "UnityProject/Assets/Scripts/Robotics/"). Only results under this
                   path are returned.
        output_format: "text" (default) or "json" for structured results.
    """
    logger.info("search_code(query=%r, repo=%s, n=%d, module=%s, offset=%d, file_path=%s, format=%s)",
                query, repo, n_results, module, offset, file_path, output_format)
    try:
        return _search_code(query, repo, n_results, module, chunk_type, offset, file_path, output_format)
    except Exception as e:
        logger.exception("search_code failed")
        return f"Error: {e}\n\nTip: Has the repo been indexed? Run reindex('{repo}') first."


@mcp.tool()
def get_file_chunks(file_path: str, repo: str = "mainapp",
                    offset: int = 0, limit: int = 20,
                    output_format: str = "text") -> str:
    """List all indexed chunks for a specific file.

    Given a file path (relative to repo root), returns all chunks
    (class summaries, methods, constructors, properties, prefab summaries, etc.)
    indexed for that file (including chunk IDs). Useful for understanding a
    file's full structure after finding it in search results.

    Args:
        file_path: Path relative to repo root (e.g., "UnityProject/Assets/Scripts/Foo.cs").
        repo: Which repo to search (default "mainapp").
        offset: Skip the first N chunks (for pagination). Default 0.
        limit: Maximum number of chunks to return (1-50, default 20).
        output_format: "text" (default) or "json" for structured chunks.
    """
    logger.info("get_file_chunks(file=%r, repo=%s, offset=%d, limit=%d, format=%s)",
                file_path, repo, offset, limit, output_format)
    try:
        return _get_file_chunks(file_path, repo, output_format, offset, limit)
    except Exception as e:
        logger.exception("get_file_chunks failed")
        return f"Error: {e}"


@mcp.tool()
def lookup_binding(interface_name: str, repo: str = "mainapp") -> str:
    """Look up Zenject dependency injection bindings. Unity/MainApp only.

    Given an interface or type name, returns the concrete implementation,
    scope (Singleton/Transient), installer file, and line number.

    NOTE: This tool only works with the mainapp (Unity) repo, which uses
    Zenject for DI. The perception repo does not use Zenject.

    Args:
        interface_name: Interface or type to look up (e.g., "IRobotDriver", "NetworkHandler").
        repo: Only "mainapp" is supported. Other repos will return an error.
    """
    logger.info("lookup_binding(interface=%r)", interface_name)
    try:
        return _lookup_binding(interface_name, repo)
    except Exception as e:
        logger.exception("lookup_binding failed")
        return f"Error looking up binding: {e}"


@mcp.tool()
async def reindex(repo: str = "perception", incremental: bool = True,
                  ctx: Context | None = None) -> str:
    """Reindex a codebase (simple version). Prefer start_reindex for job tracking.

    Blocks until complete. Use this for quick, one-off reindexing.
    For job tracking (status checks via get_reindex_status/list_reindex_jobs),
    use start_reindex instead.

    Args:
        repo: Which repo to index. Use list_repos() to see available repos, or "all".
              "pipeline" is an alias for "perception".
        incremental: If True (default), only re-index changed files. False for full rebuild.
    """
    import anyio

    from src.config import resolve_repo
    logger.info("reindex(repo=%s, incremental=%s)", repo, incremental)
    resolved = resolve_repo(repo) if repo != "all" else "all"
    try:
        _progress = {"current": 0, "total": 0, "msg": "Starting..."}
        _stderr_state = {"enabled": True}

        def _do_reindex():
            def _progress_cb(current: int, total: int, message: str):
                _progress["current"] = current
                _progress["total"] = total
                _progress["msg"] = message
            return _reindex(resolved, incremental, progress_callback=_progress_cb)

        async def _reporter(*, task_status=anyio.TASK_STATUS_IGNORED):
            task_status.started()
            while True:
                await anyio.sleep(0.1)
                msg = _progress["msg"]
                if _progress["total"] > 0:
                    _render_progress_to_stderr(msg, _stderr_state)
                    if ctx:
                        try:
                            await ctx.report_progress(
                                _progress["current"], message=msg
                            )
                        except Exception:
                            pass

        async with anyio.create_task_group() as tg:
            await tg.start(_reporter)
            result = await anyio.to_thread.run_sync(_do_reindex)
            tg.cancel_scope.cancel()
        _finish_progress_line(_stderr_state)  # newline after progress bar

        # Invalidate caches so next queries use fresh data
        invalidate_calibration_cache(None if resolved == "all" else resolved)
        invalidate_hierarchy_cache(None if resolved == "all" else resolved)
        invalidate_asset_ref_cache(None if resolved == "all" else resolved)
        invalidate_graph_cache(None if resolved == "all" else resolved)
        invalidate_dep_cache(None if resolved == "all" else resolved)
        invalidate_modules_cache(None if resolved == "all" else resolved)
        # Also rebuild Zenject bindings when indexing mainapp
        if resolved in ("mainapp", "all"):
            logger.info("Rebuilding Zenject bindings...")
            zenject_result = rebuild_zenject_bindings()
            result += "\n\n" + zenject_result
        return result + _gpu_warning()
    except Exception as e:
        logger.exception("reindex failed")
        return f"Reindex error: {e}\n\nTip: Run index_status() to check the embedding backend."
    finally:
        # Reindexing is the only GPU-heavy path; release when done.
        release_embedding_function(role="index")


@mcp.tool()
def index_status() -> str:
    """Health check for the code index.

    Reports: embedding backend (GPU/CPU), chunk counts per collection,
    last index time, and whether source directories exist.
    """
    logger.info("index_status()")
    try:
        return _index_status()
    except Exception as e:
        logger.exception("index_status failed")
        return f"Error checking status: {e}"


@mcp.tool()
def list_repos() -> str:
    """List all configured repos with their paths, source directories, and status.

    Shows each repo's name, root path, source dirs (with language), collection name,
    whether paths exist on disk, and any aliases.
    """
    logger.info("list_repos()")
    try:
        return _list_repos()
    except Exception as e:
        logger.exception("list_repos failed")
        return f"Error: {e}"


@mcp.tool()
def add_repo(name: str, root: str, source_dirs_json: str | list | None = None,
             aliases: str | None = None,
             strip_prefixes: str | None = None) -> str:
    """Add a new repo to the code index. Persisted to config.local.json.

    After adding, run reindex(repo_name) to index the new repo.

    If source_dirs_json is omitted, the server auto-detects source directories
    by scanning the repo root for known file types (C#, C++, Python, JS, HTML,
    Unity prefab/scene). This makes it easy to add any project with just a name
    and path: add_repo(name="my-project", root="~/code/my-project")

    Args:
        name: Repo name (e.g. "weld-detect"). Used in search_code(repo=...) and reindex(repo=...).
        root: Root path of the repo. Supports ~ and env vars (e.g. "~/Documents/MyProject").
        source_dirs_json: Optional JSON array of source directories. Each entry needs at minimum
            a "path" field. If omitted, source dirs are auto-detected.
            Example: [{"path": "~/Documents/MyProject/src", "language": "csharp", "exclude_patterns": ["bin/", "obj/"]}]
            Supported languages: "csharp", "cpp", "python", "javascript", "html", "unity".
        aliases: Optional comma-separated alias names (e.g. "wd,weld").
        strip_prefixes: Optional comma-separated .asmdef prefixes to strip (e.g. "MyCompany.,MyCompany.App.").
    """
    logger.info("add_repo(name=%r, root=%r)", name, root)
    try:
        return _add_repo(name, root, source_dirs_json, aliases, strip_prefixes)
    except Exception as e:
        logger.exception("add_repo failed")
        return f"Error: {e}"


@mcp.tool()
def remove_repo(name: str) -> str:
    """Remove a repo from the code index.

    Deletes the ChromaDB collection, manifest, calibration files, and removes from config.

    Args:
        name: Repo name to remove.
    """
    logger.info("remove_repo(name=%r)", name)
    try:
        return _remove_repo(name)
    except Exception as e:
        logger.exception("remove_repo failed")
        return f"Error: {e}"


@mcp.tool()
def reindex_file(repo: str, file_path: str) -> str:
    """Re-index a single file: delete old chunks and re-chunk/re-embed it.

    Useful for updating the index after editing a specific file without
    running a full incremental reindex.

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        file_path: Path to the file, relative to the repo root
                   (e.g. "UnityProject/Assets/Scripts/Foo.cs").
    """
    logger.info("reindex_file(repo=%r, file=%r)", repo, file_path)
    try:
        return _reindex_file(repo, file_path)
    except Exception as e:
        logger.exception("reindex_file failed")
        return f"Error: {e}"
    finally:
        release_embedding_function(role="index")


@mcp.tool()
def remove_file(repo: str, file_path: str) -> str:
    """Remove a single file from the index (delete its chunks).

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        file_path: Path to the file, relative to the repo root.
    """
    logger.info("remove_file(repo=%r, file=%r)", repo, file_path)
    try:
        return _remove_file(repo, file_path)
    except Exception as e:
        logger.exception("remove_file failed")
        return f"Error: {e}"


@mcp.tool()
async def rebuild_sidecars(repo: str = "mainapp",
                           skip_unity: bool = False,
                           ctx: Context | None = None) -> str:
    """Rebuild sidecar data (type hierarchy, dep graph, asset refs) without re-embedding.

    Re-scans all indexed files, re-extracts sidecar contributions, and
    materializes fresh sidecar JSONs. Useful for repair after schema changes
    or when sidecar data seems stale.

    Args:
        repo: Repo name (e.g. "mainapp", "perception").
        skip_unity: If True, skip Unity YAML files (prefabs/scenes/assets) and only
            rebuild from code files. Much faster for large Unity repos. Existing
            unity sidecar data is preserved from the previous state.
    """
    import anyio

    logger.info("rebuild_sidecars(repo=%r, skip_unity=%s)", repo, skip_unity)
    try:
        _progress = {"current": 0, "total": 0, "msg": "Starting..."}
        _stderr_state = {"enabled": True}

        def _do_rebuild():
            def _progress_cb(current: int, total: int, message: str):
                _progress["current"] = current
                _progress["total"] = total
                _progress["msg"] = message
            return _rebuild_sidecars(repo, skip_unity=skip_unity,
                                     progress_callback=_progress_cb)

        async def _reporter(*, task_status=anyio.TASK_STATUS_IGNORED):
            task_status.started()
            while True:
                await anyio.sleep(0.1)
                msg = _progress["msg"]
                if _progress["total"] > 0:
                    _render_progress_to_stderr(msg, _stderr_state)
                    if ctx:
                        try:
                            await ctx.report_progress(
                                _progress["current"], message=msg
                            )
                        except Exception:
                            pass

        async with anyio.create_task_group() as tg:
            await tg.start(_reporter)
            result = await anyio.to_thread.run_sync(_do_rebuild)
            tg.cancel_scope.cancel()
        _finish_progress_line(_stderr_state)

        # Invalidate caches so next queries use fresh data
        from src.config import resolve_repo
        resolved = resolve_repo(repo)
        invalidate_hierarchy_cache(resolved)
        invalidate_asset_ref_cache(resolved)
        invalidate_dep_cache(resolved)
        return result
    except Exception as e:
        logger.exception("rebuild_sidecars failed")
        return f"Error: {e}"


@mcp.tool()
def find_references(symbol: str, repo: str, file_pattern: str | None = None,
                    whole_word: bool = True, max_results: int = 50) -> str:
    """Find all references to a symbol across the codebase.

    Performs exact text search (not semantic) across all source files on disk.
    Use this for precise symbol lookups — it searches live files, so results
    include recently added/modified code without needing a reindex.

    Prefer this over search_code when you know the exact symbol name.
    Use search_code when you need to describe what code does in natural language.

    Args:
        symbol: The symbol to search for (e.g., "IRobotDriver", "HandleCalibration").
        repo: Which repo to search (required). Use "all" to search all repos.
              Use list_repos() to see available repo names.
        file_pattern: Optional glob pattern for files (e.g., "*.cs", "*.prefab").
        whole_word: If True (default), only match whole words. Set False for partial matches.
        max_results: Maximum results to return (default 50). When results are capped,
                     the total match count is shown so you know if you need more.
    """
    logger.info("find_references(symbol=%r, repo=%s)", symbol, repo)
    try:
        return _find_references(symbol, repo, file_pattern, whole_word, max_results)
    except Exception as e:
        logger.exception("find_references failed")
        return f"Error: {e}"


@mcp.tool()
def find_implementations(type_name: str, repo: str = "mainapp",
                         max_results: int = 50, offset: int = 0) -> str:
    """Find all classes that implement an interface or extend a base class.

    Uses pre-built type hierarchy data (generated during full reindex).

    Args:
        type_name: Interface or base class name (e.g., "IRobotDriver", "MonoBehaviour").
        repo: Which repo to search (default "mainapp").
        max_results: Maximum results to return (default 50). Use 0 for unlimited.
                     When capped, shows total count so you know the full scope.
        offset: Skip the first N results (for pagination). Default 0.
    """
    logger.info("find_implementations(type=%r, repo=%s, max=%d, offset=%d)", type_name, repo, max_results, offset)
    try:
        return _find_implementations(type_name, repo, max_results, offset)
    except Exception as e:
        logger.exception("find_implementations failed")
        return f"Error: {e}"


@mcp.tool()
def get_assembly_graph(repo: str = "mainapp", assembly: str | None = None,
                       top_n: int = 20, min_references: int = 0) -> str:
    """Get the Unity assembly definition (.asmdef) dependency graph.

    Shows which assemblies reference which other assemblies. Useful for understanding
    project structure and module boundaries.

    Without an assembly filter, returns a summary of the top assemblies by dependent
    count (default top 20) to avoid excessive output. Use assembly='<name>' to drill
    into a specific assembly's full dependency details.

    Args:
        repo: Which repo to query (default "mainapp").
        assembly: Optional assembly name to focus on. Supports fuzzy matching.
                  If None, returns a summary sorted by dependent count.
        top_n: When no assembly filter, limit to the top N assemblies (default 20).
               Use 0 to show all assemblies.
        min_references: When no assembly filter, only show assemblies with at least
                        this many dependents. Default 0.
    """
    logger.info("get_assembly_graph(repo=%s, assembly=%s, top_n=%d, min_refs=%d)",
                repo, assembly, top_n, min_references)
    try:
        return _get_assembly_graph(repo, assembly, top_n, min_references)
    except Exception as e:
        logger.exception("get_assembly_graph failed")
        return f"Error: {e}"


@mcp.tool()
def find_asset_references(class_name: str, repo: str = "mainapp",
                          output_format: str = "text") -> str:
    """Find all prefabs, scenes, and asset files that reference a script class.

    Uses pre-built asset reference data (generated during full reindex).

    Args:
        class_name: Script class name (e.g., "PlayerController", "LeanTouchCameraController").
        repo: Which repo to search (default "mainapp").
        output_format: "text" (default) or "json" for structured references.
    """
    logger.info("find_asset_references(class=%r, repo=%s, format=%s)", class_name, repo, output_format)
    try:
        return _find_asset_references(class_name, repo, output_format)
    except Exception as e:
        logger.exception("find_asset_references failed")
        return f"Error: {e}"


@mcp.tool()
def get_project_info(repo: str = "mainapp") -> str:
    """Get Unity project metadata: version, packages, build scenes, scripting defines.

    Reads ProjectVersion.txt, manifest.json, ProjectSettings.asset, and
    EditorBuildSettings.asset directly from disk.

    Args:
        repo: Which repo to query (default "mainapp").
    """
    logger.info("get_project_info(repo=%s)", repo)
    try:
        return _get_project_info(repo)
    except Exception as e:
        logger.exception("get_project_info failed")
        return f"Error: {e}"


@mcp.tool()
def get_class_dependencies(class_name: str | None = None, repo: str = "mainapp",
                           output_format: str = "text") -> str:
    """Get class dependency information: what a class depends on and what depends on it.

    Shows intra-project class-to-class dependencies based on type references.
    If class_name is provided: shows dependencies and reverse dependencies.
    If class_name is None: shows graph summary with most-depended-on classes.
    Accepts either class names or canonical node keys; disambiguates duplicates.

    Uses pre-built dependency graph data (generated during full reindex).

    Args:
        class_name: Class name or node key (e.g., "IRobotDriver",
                    "Augmentus.Controller.OfflineMode.PathOptimizerController").
                    If None, returns a graph summary with stats.
        repo: Which repo to query (default "mainapp").
        output_format: "text" (default) or "json" for structured output.
    """
    logger.info("get_class_dependencies(class=%r, repo=%s, format=%s)", class_name, repo, output_format)
    try:
        return _get_class_dependencies(class_name, repo, output_format)
    except Exception as e:
        logger.exception("get_class_dependencies failed")
        return f"Error: {e}"


@mcp.tool()
def get_unity_entity_context(repo: str = "mainapp", chunk_id: str | None = None,
                              file_path: str | None = None,
                              entity_name: str | None = None,
                              output_format: str = "text") -> str:
    """Get full parsed context for a Unity entity without token blowup.

    Retrieves the complete (un-truncated) source for a Unity chunk.
    Use after finding a relevant chunk via search_code to get expanded detail.

    Args:
        repo: Repo name (e.g. "mainapp").
        chunk_id: Direct chunk ID from search results (preferred).
        file_path: File path (relative to repo root) to search within.
        entity_name: Entity name (GO name, script name) to find in the file.
        output_format: "text" (default) or "json" for structured output.
    """
    logger.info("get_unity_entity_context(repo=%r, chunk_id=%r, file=%r, entity=%r, format=%s)",
                repo, chunk_id, file_path, entity_name, output_format)
    try:
        return _get_unity_entity_context(repo, chunk_id, file_path, entity_name, output_format)
    except Exception as e:
        logger.exception("get_unity_entity_context failed")
        return f"Error: {e}"


@mcp.tool()
def get_chunk(chunk_id: str, repo: str = "mainapp",
              output_format: str = "text") -> str:
    """Get the full source of any chunk by its chunk ID.

    Use after search_code returns a truncated result (source_truncated: true)
    to retrieve the complete un-truncated source. Works for all chunk types
    (code methods, class summaries, Unity entities, etc.).

    The chunk_id is returned in search_code results. Copy it exactly.

    Args:
        chunk_id: Chunk ID from search results (e.g., "Source_Network_Driver_ClientTcpCore.cs__ClientTcpCore__Connect__L119-202").
        repo: Repo name (default "mainapp").
        output_format: "text" (default) or "json" for structured output.
    """
    import json as _json
    logger.info("get_chunk(chunk_id=%r, repo=%s, format=%s)", chunk_id, repo, output_format)
    try:
        from src.config import REPOS, resolve_repo
        from src.indexer.chunk_payload_store import get_payload
        from src.indexer.embedder import get_collection

        output_format = (output_format or "text").lower()
        resolved = resolve_repo(repo)
        if resolved not in REPOS:
            msg = f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"
            return _json.dumps({"status": "error", "message": msg}, indent=2) if output_format == "json" else msg

        config = REPOS[resolved]

        # 1. Try full payload store (has un-truncated source for large chunks)
        full_source = get_payload(resolved, chunk_id)
        if full_source:
            if output_format == "json":
                return _json.dumps({
                    "status": "found",
                    "repo": resolved,
                    "chunk_id": chunk_id,
                    "source": full_source,
                    "source_full": True,
                }, indent=2)
            return f"=== {chunk_id} ===\n\n{full_source}"

        # 2. Fall back to ChromaDB metadata (source field — may be truncated for large chunks)
        try:
            collection = get_collection(config.collection_name)
            results = collection.get(ids=[chunk_id], include=["metadatas"])
            if results["ids"]:
                meta = results["metadatas"][0]
                source = meta.get("source", "")
                file_path = meta.get("file_path", "")
                class_name = meta.get("class_name", "")
                method_name = meta.get("method_name", "")
                chunk_type = meta.get("chunk_type", "")
                start_line = meta.get("start_line", "")
                end_line = meta.get("end_line", "")

                if output_format == "json":
                    return _json.dumps({
                        "status": "found",
                        "repo": resolved,
                        "chunk_id": chunk_id,
                        "file_path": file_path,
                        "class_name": class_name,
                        "method_name": method_name,
                        "chunk_type": chunk_type,
                        "start_line": start_line,
                        "end_line": end_line,
                        "source": source,
                        "source_full": True,  # if not in payload store, it wasn't truncated
                    }, indent=2)

                label = method_name or class_name or chunk_id
                header = f"=== [{chunk_type}] {label} ==="
                if file_path:
                    header += f"\n  {file_path}:{start_line}-{end_line}"
                return f"{header}\n\n{source}"
        except Exception:
            pass

        msg = f"Chunk '{chunk_id}' not found in {resolved}."
        return _json.dumps({"status": "not_found", "message": msg}, indent=2) if output_format == "json" else msg

    except Exception as e:
        logger.exception("get_chunk failed")
        return f"Error: {e}"


# --- Background reindex with progress notifications ---

_REINDEX_JOBS: dict[str, dict] = {}  # repo -> {status, progress, result}


def _render_progress_to_stderr(message: str, state: dict[str, bool]) -> None:
    """Render in-place progress to stderr; disable on stream failures."""
    if not state.get("enabled", True):
        return
    try:
        print(f"\r\033[K{message}", end="", file=sys.stderr, flush=True)
    except (OSError, ValueError) as e:
        state["enabled"] = False
        logger.warning("Disabling stderr progress rendering: %s", e)


def _finish_progress_line(state: dict[str, bool]) -> None:
    """Write a trailing newline after progress output; disable on failure."""
    if not state.get("enabled", True):
        return
    try:
        print(file=sys.stderr, flush=True)
    except (OSError, ValueError) as e:
        state["enabled"] = False
        logger.warning("Disabling stderr progress rendering: %s", e)


def _gpu_warning() -> str:
    """Return a warning string if the embedding backend is not using GPU acceleration."""
    backend = get_active_backend()
    if "CUDA" in backend or "DirectML" in backend:
        return ""
    if "not initialized" in backend:
        return ""  # Embedder wasn't needed this run (e.g., all files up-to-date)

    # Detect GPU vendor to give actionable advice
    try:
        from scripts.detect_gpu import detect_gpu_vendor
        vendor = detect_gpu_vendor()
    except Exception:
        vendor = "unknown"

    if "PyTorch" in backend:
        hint = "Run setup.bat to export the ONNX model first."
    elif vendor == "nvidia":
        hint = ("NVIDIA GPU detected but CUDA libraries are missing. "
                "Install: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 "
                "nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 "
                "nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12")
    elif vendor in ("amd", "intel"):
        hint = (f"{vendor.upper()} GPU detected but DirectML is not active. "
                "Install: pip install onnxruntime-directml "
                "(and uninstall onnxruntime / onnxruntime-gpu if present)")
    else:
        hint = "No GPU detected. Install CUDA Toolkit or onnxruntime-directml for GPU acceleration."

    return f"\n\nWARNING: Running on CPU ({backend}). Indexing will be slow. {hint}"


@mcp.tool()
async def start_reindex(repo: str = "mainapp", incremental: bool = True,
                        ctx: Context | None = None) -> str:
    """Reindex a codebase with job tracking and progress updates.

    Blocks until complete. Tracks job state so you can check progress via
    get_reindex_status() or list_reindex_jobs(). Also prevents duplicate
    concurrent reindex runs on the same repo. Prefer this over reindex().

    Args:
        repo: Which repo to index (default "mainapp"). Use "all" for all repos.
        incremental: If True (default), only re-index changed files.
    """
    import anyio

    from src.config import resolve_repo
    resolved = resolve_repo(repo) if repo != "all" else "all"

    if resolved in _REINDEX_JOBS and _REINDEX_JOBS[resolved].get("status") == "running":
        return f"Reindex already running for '{resolved}'."

    _REINDEX_JOBS[resolved] = {"status": "running", "progress": "Starting...", "result": None}
    _progress = {"current": 0, "total": 0, "msg": "Starting..."}
    _stderr_state = {"enabled": True}

    def _progress_cb(current: int, total: int, message: str):
        _progress["current"] = current
        _progress["total"] = total
        _progress["msg"] = message
        _REINDEX_JOBS[resolved]["progress"] = message

    async def _reporter(*, task_status=anyio.TASK_STATUS_IGNORED):
        task_status.started()
        while True:
            await anyio.sleep(0.1)
            msg = _progress["msg"]
            if _progress["total"] > 0:
                _render_progress_to_stderr(msg, _stderr_state)
                if ctx:
                    try:
                        await ctx.report_progress(
                            _progress["current"], message=msg
                        )
                    except Exception:
                        pass

    def _do_reindex():
        return _reindex(resolved, incremental, progress_callback=_progress_cb)

    try:
        async with anyio.create_task_group() as tg:
            await tg.start(_reporter)
            result = await anyio.to_thread.run_sync(_do_reindex)
            tg.cancel_scope.cancel()
        _finish_progress_line(_stderr_state)  # newline after progress bar
        # Post-reindex maintenance
        invalidate_calibration_cache(None if resolved == "all" else resolved)
        invalidate_hierarchy_cache(None if resolved == "all" else resolved)
        invalidate_asset_ref_cache(None if resolved == "all" else resolved)
        invalidate_graph_cache(None if resolved == "all" else resolved)
        invalidate_dep_cache(None if resolved == "all" else resolved)
        invalidate_modules_cache(None if resolved == "all" else resolved)
        if resolved in ("mainapp", "all"):
            zenject_result = rebuild_zenject_bindings()
            result += "\n\n" + zenject_result
        _REINDEX_JOBS[resolved] = {"status": "complete", "progress": "Done", "result": result}
        return result + _gpu_warning()
    except Exception as e:
        _REINDEX_JOBS[resolved] = {"status": "error", "progress": str(e), "result": None}
        logger.exception("start_reindex failed")
        return f"Reindex error: {e}"
    finally:
        release_embedding_function(role="index")


@mcp.tool()
def get_reindex_status(repo: str = "mainapp") -> str:
    """Check the status of a background reindex job.

    Args:
        repo: Which repo to check (default "mainapp").
    """
    from src.config import resolve_repo
    resolved = resolve_repo(repo) if repo != "all" else "all"

    job = _REINDEX_JOBS.get(resolved)
    if not job:
        return f"No reindex job found for '{resolved}'."

    if job["status"] == "running":
        return f"Reindex '{resolved}': RUNNING - {job['progress']}"
    elif job["status"] == "complete":
        return f"Reindex '{resolved}': COMPLETE\n\n{job['result']}"
    elif job["status"] == "error":
        return f"Reindex '{resolved}': ERROR - {job['progress']}"
    return f"Reindex '{resolved}': {job['status']}"


@mcp.tool()
def list_reindex_jobs() -> str:
    """List all active and recent reindex jobs with their status."""
    if not _REINDEX_JOBS:
        return "No reindex jobs."

    lines = ["=== Reindex Jobs ==="]
    for repo, job in _REINDEX_JOBS.items():
        lines.append(f"\n{repo}: {job['status']}")
        lines.append(f"  Progress: {job['progress']}")
    return "\n".join(lines)


if __name__ == "__main__":
    _startup_check()
    logger.info("Starting code-index MCP server")
    mcp.run()
