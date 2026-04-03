"""search_code tool implementation."""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from src.config import REPOS, DATA_DIR, resolve_repo
from src.indexer.embedder import get_collection, get_embedding_function, get_active_backend
from src.indexer.calibration import get_thresholds

logger = logging.getLogger(__name__)

# Cache loaded calibrations to avoid re-reading JSON on every search.
_calibration_cache: dict[str, dict] = {}
_SEARCH_INIT_GUARD_SECONDS = float(os.environ.get("CODERANK_SEARCH_INIT_TIMEOUT_SECONDS", "90"))
_SEARCH_INIT_RETRY_SECONDS = float(os.environ.get("CODERANK_SEARCH_RETRY_TIMEOUT_SECONDS", "3"))
# nonblocking (default): start init in background and return immediately on first call
# blocking: wait up to guard timeout for initialization
_SEARCH_INIT_MODE = os.environ.get("CODERANK_SEARCH_INIT_MODE", "nonblocking").strip().lower()

# Single-flight state for first search backend initialization in each MCP process.
_search_init_lock = threading.Lock()
_search_init_event: threading.Event | None = None
_search_init_thread: threading.Thread | None = None
_search_init_error: Exception | None = None
_search_init_started_at: float | None = None


def invalidate_calibration_cache(repo: str | None = None) -> None:
    """Clear cached calibration thresholds (e.g. after reindex recalibrates)."""
    if repo is None:
        _calibration_cache.clear()
    else:
        _calibration_cache.pop(repo, None)


_STALENESS_THRESHOLD_MINUTES = 30

# Cache for changed file counts: repo -> (manifest_mtime, count, check_time)
_changed_files_cache: dict[str, tuple[float, int, float]] = {}
_CHANGED_FILES_CACHE_TTL = 300  # re-check every 5 minutes


def _count_changed_files(repo: str, manifest_mtime: float) -> int | None:
    """Count source files modified after the manifest mtime. Cached for performance."""
    now = time.time()

    # Check cache
    if repo in _changed_files_cache:
        cached_mtime, cached_count, cached_at = _changed_files_cache[repo]
        if cached_mtime == manifest_mtime and (now - cached_at) < _CHANGED_FILES_CACHE_TTL:
            return cached_count

    if repo not in REPOS:
        return None

    config = REPOS[repo]
    count = 0
    try:
        for sd in config.source_dirs:
            if not sd.path.exists():
                continue
            count += _count_newer_files(sd.path, manifest_mtime, sd.extensions, sd.exclude_patterns)
    except Exception:
        return None

    _changed_files_cache[repo] = (manifest_mtime, count, now)
    return count


def _count_newer_files(directory: Path, threshold: float, extensions: frozenset[str],
                        exclude_patterns: list[str]) -> int:
    """Fast count of files newer than threshold using os.scandir recursion."""
    from src.indexer.file_scanner import _parse_exclude_patterns
    excluded_dirs, excluded_extensions, other_patterns = _parse_exclude_patterns(exclude_patterns)

    count = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in excluded_dirs:
                    stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in extensions or ext in excluded_extensions:
                    continue
                if other_patterns:
                    path_fwd = entry.path.replace("\\", "/")
                    if any(p in path_fwd for p in other_patterns):
                        continue
                try:
                    if entry.stat().st_mtime > threshold:
                        count += 1
                except OSError:
                    pass
    return count


def _get_staleness_warning(repo: str) -> str:
    """Return a staleness warning if the index is older than the threshold, else empty string."""
    manifest_path = DATA_DIR / f"{repo}_manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        mtime = manifest_path.stat().st_mtime
        age_minutes = (datetime.now().timestamp() - mtime) / 60
        if age_minutes > _STALENESS_THRESHOLD_MINUTES:
            last = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            hours = age_minutes / 60
            if hours >= 1:
                age_str = f"{hours:.1f}h"
            else:
                age_str = f"{age_minutes:.0f}m"

            # Count changed files for actionable context
            changed = _count_changed_files(repo, mtime)
            changed_str = ""
            if changed is not None and changed > 0:
                changed_str = f" ({changed} file{'s' if changed != 1 else ''} changed since)"
            elif changed == 0:
                changed_str = " (no files changed since)"

            return (f"\n\nNote: Index last updated {age_str} ago ({last}).{changed_str} "
                    "Recently added/modified files may be missing. "
                    f"Run reindex('{repo}') or reindex_file('{repo}', '<path>') to update.")
    except OSError:
        pass
    return ""


def search_code(query: str, repo: str = "all", n_results: int = 10,
                module: str | None = None, chunk_type: str | None = None,
                offset: int = 0, file_path: str | None = None,
                strategy: str = "associative",
                output_format: str = "text") -> str:
    """Semantic code search across indexed C#, C++, Python, JavaScript, HTML,
    and Unity prefab/scene codebases.

    Default strategy uses spreading activation to explore the dependency graph
    outward from vector search hits, then merges both rankings via RRF.

    Args:
        query: Natural language description of what you're looking for.
        repo: Which repo to search. Use list_repos() to see available repos.
        n_results: Number of results to return (default 10, max 30).
        module: Optional module filter (e.g., "Robotics", "Network").
        chunk_type: Optional chunk type filter ("method", "class_summary", "whole_class",
                    "constructor", "property", "function", "component", "template",
                    "prefab_summary", "gameobject").
        offset: Skip the first N results (for pagination). Default 0.
        file_path: Optional path prefix filter. Only results whose file_path starts
                   with this string are returned (e.g., "UnityProject/Assets/Prefabs/").
        strategy: "associative" (default — spreading activation + RRF fusion)
                  or "semantic" (vector-only, no graph expansion).
        output_format: "text" (default) or "json".

    Returns:
        Search results in text or JSON format.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    # Cross-repo search: merge results from all repos sorted by distance
    if repo == "all":
        return _search_all_repos(query, n_results, module, chunk_type, offset, file_path, output_format)

    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Error: Unknown repo '{repo}'. Valid repos: {list(REPOS.keys())} or 'all'"

    config = REPOS[repo]
    collection = get_collection(config.collection_name)

    n_results = min(max(n_results, 1), 30)
    offset = max(offset, 0)

    # Build optional where clause
    where = None
    where_conditions = []
    if module:
        where_conditions.append({"module": module})
    if chunk_type:
        where_conditions.append({"chunk_type": chunk_type})

    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}

    # Kick off backend init in background and wait for it to complete.
    # Previous approach returned "retry later" immediately, but LLM-based MCP clients
    # (Claude Code) auto-retry soft failures, creating an infinite retry loop where
    # the client never waits long enough for init to finish.
    if _SEARCH_INIT_MODE != "blocking" and get_active_backend(role="search") == "not initialized":
        event = _ensure_search_backend_init_started()
        if not event.wait(timeout=_SEARCH_INIT_GUARD_SECONDS):
            elapsed = int(time.monotonic() - (_search_init_started_at or time.monotonic()))
            return (f"Search backend initialization timed out ({elapsed}s). "
                    "The ONNX model may be very slow to load. Check server logs for details.")
        if _search_init_error is not None:
            return (f"Search backend initialization failed: {_search_init_error}. "
                    "Retry to attempt re-initialization.")
        # Init completed successfully — fall through to actual search

    try:
        # CodeRankEmbed needs a query prefix - embed manually and pass pre-computed vectors.
        # Guard first-time search initialization so MCP callers don't hit 120s tool deadlines.
        query_embeddings = _get_query_embeddings_with_guard(query)

        # When file_path filter is active, over-fetch 3x to compensate for post-query filtering
        if file_path:
            file_path = file_path.replace("\\", "/")
            fetch_count = (n_results + offset) * 3
        else:
            fetch_count = n_results + offset

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=fetch_count,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except TimeoutError as e:
        logger.warning("search_code init guard triggered: %s", e)
        # Quick one-shot retry of the same semantic query.
        try:
            query_embeddings = _get_query_embeddings_with_guard(
                query, guard_seconds=_SEARCH_INIT_RETRY_SECONDS
            )
            if file_path:
                file_path = file_path.replace("\\", "/")
                fetch_count = (n_results + offset) * 3
            else:
                fetch_count = n_results + offset
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=fetch_count,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            logger.info("search_code quick-retry succeeded after init timeout")
        except TimeoutError:
            return (f"Search backend is still initializing (>{_SEARCH_INIT_GUARD_SECONDS:.0f}s). "
                    "Please retry this query in a few seconds.")
        except Exception as retry_exc:
            return f"Search error: {retry_exc}"
    except Exception as e:
        err_str = str(e)
        if "Error finding id" in err_str or "Internal error" in err_str:
            return (f"Search error: ChromaDB index is corrupted (stale chunk IDs). "
                    f"Run reindex('{repo}') with a full (non-incremental) rebuild to fix this.")
        return f"Search error: {e}"

    if not results["ids"] or not results["ids"][0]:
        hint = ""
        if module:
            available = _get_available_modules(config.collection_name)
            if available:
                hint = f"\n\nAvailable modules for '{repo}': {', '.join(sorted(available))}"
        return f"No results found for '{query}' in {repo}.{hint}"

    # Apply file_path prefix filter (post-query)
    if file_path:
        results = _filter_by_path_prefix(results, file_path)

        # If not enough results after filtering, try a larger fetch (up to 100)
        if len(results["ids"][0]) < n_results + offset:
            try:
                bigger_results = collection.query(
                    query_embeddings=query_embeddings,
                    n_results=100,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                results = _filter_by_path_prefix(bigger_results, file_path)
            except Exception:
                pass  # use what we have

    if not results["ids"][0]:
        filter_msg = f" under '{file_path}'" if file_path else ""
        hint = ""
        if module:
            available = _get_available_modules(config.collection_name)
            if available:
                hint = f"\n\nAvailable modules for '{repo}': {', '.join(sorted(available))}"
        return f"No results found for '{query}' in {repo}{filter_msg}.{hint}"

    # Apply offset by slicing
    if offset > 0:
        results = {
            "ids": [results["ids"][0][offset:]],
            "metadatas": [results["metadatas"][0][offset:]],
            "documents": [results["documents"][0][offset:]],
            "distances": [results["distances"][0][offset:]],
        }

    if not results["ids"][0]:
        return f"No more results for '{query}' in {repo} at offset {offset}."

    # Trim to n_results
    if len(results["ids"][0]) > n_results:
        results = {
            "ids": [results["ids"][0][:n_results]],
            "metadatas": [results["metadatas"][0][:n_results]],
            "documents": [results["documents"][0][:n_results]],
            "distances": [results["distances"][0][:n_results]],
        }

    # Filter out NO MATCH results to avoid wasting context on noise
    results = _filter_no_match(results, repo)
    if not results["ids"][0]:
        hint = ""
        if module:
            available = _get_available_modules(config.collection_name)
            if available:
                hint = f"\n\nAvailable modules for '{repo}': {', '.join(sorted(available))}"
        return f"No confident results for '{query}' in {repo} (all results below relevance threshold).{hint}"

    # Graph expansion via spreading activation + RRF fusion
    strategy = (strategy or "associative").lower()
    if strategy == "associative" and results["ids"][0]:
        results = _expand_via_graph(results, repo, collection, n_results)

    staleness = _get_staleness_warning(repo)
    if output_format == "json":
        return _format_results_json(results, repo, offset) + staleness
    return _format_results(results, repo, offset) + staleness


def _search_all_repos(query: str, n_results: int, module: str | None,
                       chunk_type: str | None, offset: int,
                       file_path: str | None, output_format: str) -> str:
    """Search across all repos, merge results by distance, return top N."""
    # Deduplicate collections (e.g., perception + pipeline share the same collection)
    seen_collections: set[str] = set()
    repo_names: list[str] = []
    for repo_name, config in REPOS.items():
        if config.collection_name not in seen_collections:
            seen_collections.add(config.collection_name)
            repo_names.append(repo_name)

    if not repo_names:
        return "No repos configured. Use add_repo() first."

    # Get query embeddings once (shared across all collections)
    try:
        query_embeddings = _get_query_embeddings_with_guard(query)
    except TimeoutError:
        return ("Search backend is still initializing. "
                "Please retry this query in a few seconds.")
    except Exception as e:
        return f"Search error: {e}"

    # Collect results from each repo's collection
    all_items: list[tuple[float, str, str, dict, str]] = []  # (distance, chunk_id, repo, meta, doc)

    for repo_name in repo_names:
        config = REPOS[repo_name]
        try:
            collection = get_collection(config.collection_name)
        except Exception:
            continue

        where = None
        where_conditions = []
        if module:
            where_conditions.append({"module": module})
        if chunk_type:
            where_conditions.append({"chunk_type": chunk_type})
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        fetch_count = (n_results + offset) * 2  # over-fetch to allow merging
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=fetch_count,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            continue

        if not results["ids"] or not results["ids"][0]:
            continue

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            meta = metadatas[i]
            # Apply file_path filter
            if file_path:
                fp = meta.get("file_path", "")
                if not fp.startswith(file_path.replace("\\", "/")):
                    continue
            all_items.append((distances[i], ids[i], repo_name, meta, documents[i]))

    if not all_items:
        return f"No results found for '{query}' across all repos."

    # Filter out NO MATCH results to avoid wasting context on noise
    all_items = [item for item in all_items
                 if _confidence_label(item[0], item[2]) != "NO MATCH"]
    if not all_items:
        return f"No confident results for '{query}' across all repos (all below relevance threshold)."

    # Sort by normalized score (ascending = most relevant first)
    # Normalized score makes distances comparable across repos with different calibrations
    all_items.sort(key=lambda x: _normalized_score(x[0], x[2]), reverse=True)

    # Apply offset and limit
    page = all_items[offset:offset + n_results]
    if not page:
        return f"No more results for '{query}' at offset {offset}."

    # Format results with repo labels
    staleness_parts: list[str] = []
    seen_repos: set[str] = set()

    if output_format == "json":
        payload = {"repo": "all", "offset": offset, "results": []}
        for i, (dist, chunk_id, repo_name, meta, doc) in enumerate(page):
            seen_repos.add(repo_name)
            source = meta.get("source", doc)
            source_lines = source.split("\n")
            truncated = len(source_lines) > 20
            snippet = "\n".join(source_lines[:20]) if truncated else source

            payload["results"].append({
                "result_num": offset + i + 1,
                "chunk_id": chunk_id,
                "repo": repo_name,
                "distance": round(float(dist), 4),
                "relevance_pct": _normalized_score(dist, repo_name),
                "confidence": _confidence_label(dist, repo_name),
                "file_path": meta.get("file_path", ""),
                "class_name": meta.get("class_name", ""),
                "method_name": meta.get("method_name", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "module": meta.get("module", ""),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "source_snippet": snippet,
                "source_truncated": truncated,
            })

        result = json.dumps(payload, indent=2)
    else:
        lines = []
        for i, (dist, chunk_id, repo_name, meta, doc) in enumerate(page):
            seen_repos.add(repo_name)
            confidence = _confidence_label(dist, repo_name)
            score = _normalized_score(dist, repo_name)
            result_num = offset + i + 1
            header = f"--- Result {result_num} (relevance: {score}%, confidence: {confidence}, repo: {repo_name}) ---"
            file_path_str = meta.get("file_path", "?")
            class_name = meta.get("class_name", "?")
            method_name = meta.get("method_name", "")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            ct = meta.get("chunk_type", "?")
            mod = meta.get("module", "")

            location = f"{file_path_str}:{start_line}-{end_line}"
            name = f"{class_name}.{method_name}" if method_name else class_name
            module_str = f" [{mod}]" if mod else ""

            lines.append(header)
            lines.append(f"  {name}{module_str} ({ct})")
            lines.append(f"  {location}")
            lines.append(f"  Chunk ID: {chunk_id}")
            lines.append("")

            source = meta.get("source", doc)
            source_lines = source.split("\n")
            if len(source_lines) > 20:
                snippet = "\n".join(source_lines[:20]) + f"\n  ... ({len(source_lines) - 20} more lines)"
            else:
                snippet = source
            lines.append(snippet)
            lines.append("")

        result = "\n".join(lines)

    # Append consolidated staleness warning for repos that had results
    stale_repos: list[str] = []
    for repo_name in sorted(seen_repos):
        warning = _get_staleness_warning(repo_name)
        if warning:
            stale_repos.append(repo_name)

    if stale_repos:
        # Build a compact single-line summary instead of repeating the full warning per repo
        parts = []
        for repo_name in stale_repos:
            manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
            try:
                mtime = manifest_path.stat().st_mtime
                age_minutes = (datetime.now().timestamp() - mtime) / 60
                hours = age_minutes / 60
                age_str = f"{hours:.1f}h" if hours >= 1 else f"{age_minutes:.0f}m"
                changed = _count_changed_files(repo_name, mtime)
                if changed is not None and changed > 0:
                    parts.append(f"{repo_name} ({age_str}, {changed} file{'s' if changed != 1 else ''} changed)")
                elif changed == 0:
                    parts.append(f"{repo_name} ({age_str}, no changes)")
                else:
                    parts.append(f"{repo_name} ({age_str})")
            except OSError:
                parts.append(repo_name)
        staleness_parts.append(
            f"\n\nNote: Stale indexes: {', '.join(parts)}. "
            "Run reindex('<repo>') to update."
        )

    return result + "".join(staleness_parts)


def _expand_via_graph(results: dict, repo: str, collection, n_results: int) -> dict:
    """Expand search results via spreading activation on the dependency graph.

    Extracts class names from vector results, runs spreading activation to find
    related classes, fetches their chunks, and merges via RRF.
    """
    try:
        from src.graph.activation import spread_activation
    except Exception:
        return results  # graceful fallback if graph module fails

    # Extract unique class names from vector results
    seed_classes = set()
    for meta in results["metadatas"][0]:
        cn = meta.get("class_name", "")
        if cn:
            seed_classes.add(cn)

    if not seed_classes:
        return results

    # Run spreading activation
    try:
        activated = spread_activation(repo, seed_classes, decay=0.7, max_hops=2, top_k=10)
    except Exception as e:
        logger.debug("Graph expansion failed for '%s': %s", repo, e)
        return results

    if not activated:
        return results

    # Get activated class names (excluding seeds)
    activated_classes = set()
    for node_key in activated:
        # Extract class name from node key
        if "@@" in node_key:
            activated_classes.add(node_key.split("@@", 1)[0])
        elif "." in node_key:
            activated_classes.add(node_key.rsplit(".", 1)[-1])
        else:
            activated_classes.add(node_key)

    activated_classes -= seed_classes
    if not activated_classes:
        return results

    # Fetch chunks for activated classes (batch query by class_name)
    graph_chunks: list[tuple[str, dict, str, float]] = []  # (id, meta, doc, energy)
    existing_ids = set(results["ids"][0])

    for class_name in list(activated_classes)[:8]:  # cap to avoid over-fetching
        energy = 0.0
        # Find the energy for this class (may match multiple node keys)
        for nk, e in activated.items():
            nk_class = nk.split("@@", 1)[0] if "@@" in nk else (nk.rsplit(".", 1)[-1] if "." in nk else nk)
            if nk_class == class_name:
                energy = max(energy, e)

        try:
            class_results = collection.get(
                where={"class_name": class_name},
                include=["metadatas", "documents"],
                limit=3,  # top 3 chunks per activated class
            )
        except Exception:
            continue

        if not class_results["ids"]:
            continue

        for i, cid in enumerate(class_results["ids"]):
            if cid in existing_ids:
                continue
            graph_chunks.append((
                cid,
                class_results["metadatas"][i],
                class_results["documents"][i],
                energy,
            ))

    if not graph_chunks:
        return results

    # RRF fusion: rank vector results by position, graph results by energy
    vector_ids = results["ids"][0]
    graph_chunks.sort(key=lambda x: -x[3])  # sort by energy desc

    rrf_scores: dict[str, float] = {}
    k = 60

    for rank, cid in enumerate(vector_ids, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, (cid, _, _, _) in enumerate(graph_chunks, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    # Merge all items
    all_ids = list(vector_ids)
    all_metas = list(results["metadatas"][0])
    all_docs = list(results["documents"][0])
    all_dists = list(results["distances"][0])

    # Add graph chunks with worst vector distance as baseline
    worst_dist = max(all_dists) if all_dists else 1000.0
    for cid, meta, doc, energy in graph_chunks:
        if cid in existing_ids:
            continue
        existing_ids.add(cid)
        all_ids.append(cid)
        all_metas.append(meta)
        all_docs.append(doc)
        all_dists.append(worst_dist)

    # Sort by RRF score
    indices = list(range(len(all_ids)))
    indices.sort(key=lambda i: rrf_scores.get(all_ids[i], 0.0), reverse=True)

    # Rebuild results in RRF order, trimmed to n_results
    indices = indices[:n_results]
    return {
        "ids": [[all_ids[i] for i in indices]],
        "metadatas": [[all_metas[i] for i in indices]],
        "documents": [[all_docs[i] for i in indices]],
        "distances": [[all_dists[i] for i in indices]],
    }


def _get_query_embeddings_with_guard(query: str, guard_seconds: float | None = None) -> list[list[float]]:
    """Get query embeddings while preventing first-call deadlocks from blocking tool calls."""
    guard = _SEARCH_INIT_GUARD_SECONDS if guard_seconds is None else max(0.0, float(guard_seconds))

    # Warm path: avoid thread overhead when already initialized.
    if get_active_backend(role="search") != "not initialized":
        ef = get_embedding_function(role="search")
        return ef.embed_queries([query])

    event = _ensure_search_backend_init_started()
    elapsed_now = time.monotonic() - (_search_init_started_at or time.monotonic())
    # If initialization has already exceeded the guard, fail fast on retries
    # instead of blocking another full timeout window each call.
    if elapsed_now >= guard and not event.is_set():
        raise TimeoutError(
            f"search embedder initialization still in progress ({int(elapsed_now)}s elapsed)"
        )

    remaining = max(0.0, guard - elapsed_now)
    if not event.wait(remaining):
        elapsed = int((time.monotonic() - (_search_init_started_at or time.monotonic())))
        raise TimeoutError(
            f"search embedder initialization still in progress ({elapsed}s elapsed)"
        )
    if _search_init_error is not None:
        raise _search_init_error

    ef = get_embedding_function(role="search")
    return ef.embed_queries([query])


def _ensure_search_backend_init_started() -> threading.Event:
    """Start search backend init once per process; retries piggyback on same job."""
    global _search_init_event, _search_init_thread, _search_init_error, _search_init_started_at

    with _search_init_lock:
        # If already initialized, expose a set event so callers proceed immediately.
        if get_active_backend(role="search") != "not initialized":
            if _search_init_event is None:
                _search_init_event = threading.Event()
            _search_init_event.set()
            return _search_init_event

        # Reuse in-flight initialization.
        if _search_init_event is not None and _search_init_thread is not None and _search_init_thread.is_alive():
            return _search_init_event

        # If previous init failed, clear state and allow a fresh retry.
        _search_init_error = None
        _search_init_event = threading.Event()
        _search_init_started_at = time.monotonic()

        def _worker() -> None:
            global _search_init_error
            try:
                get_embedding_function(role="search")
                logger.info("search backend initialization complete")
            except Exception as exc:  # pragma: no cover - surfaced on caller thread
                _search_init_error = exc
                logger.exception("search backend initialization failed")
            finally:
                if _search_init_event is not None:
                    _search_init_event.set()

        _search_init_thread = threading.Thread(
            target=_worker, daemon=True, name="code-index-search-init"
        )
        _search_init_thread.start()
        logger.info("search backend initialization started (single-flight)")
        return _search_init_event


def start_search_init() -> None:
    """Start search backend initialization eagerly (called at server startup).

    By starting init early, the embedding model is likely ready by the time
    the first search_code call arrives, avoiding cold-start stalls.
    """
    if get_active_backend(role="search") == "not initialized":
        _ensure_search_backend_init_started()
        logger.info("Eager search init: background thread started at server startup")


def _filter_by_path_prefix(results: dict, prefix: str) -> dict:
    """Filter ChromaDB results to only include files matching a path prefix."""
    filtered = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    for i, meta in enumerate(results["metadatas"][0]):
        fp = meta.get("file_path", "")
        if fp.startswith(prefix):
            filtered["ids"][0].append(results["ids"][0][i])
            filtered["metadatas"][0].append(meta)
            filtered["documents"][0].append(results["documents"][0][i])
            filtered["distances"][0].append(results["distances"][0][i])
    return filtered


# Cache available modules per collection to avoid repeated ChromaDB scans.
_modules_cache: dict[str, set[str]] = {}


def _get_available_modules(collection_name: str) -> set[str]:
    """Get the set of unique module names indexed in a collection.

    Paginates through ChromaDB metadata (batch_size=5000) to avoid
    SQLite variable limits on large collections. Caches the result.
    """
    if collection_name in _modules_cache:
        return _modules_cache[collection_name]

    try:
        collection = get_collection(collection_name)
        modules: set[str] = set()
        batch_size = 5000
        offset = 0
        while True:
            result = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
            metas = result.get("metadatas", [])
            if not metas:
                break
            for meta in metas:
                m = meta.get("module", "")
                if m:
                    modules.add(m)
            offset += len(metas)
            if len(metas) < batch_size:
                break
        _modules_cache[collection_name] = modules
        return modules
    except Exception:
        return set()


def invalidate_modules_cache(repo: str | None = None) -> None:
    """Clear cached module list (e.g. after reindex)."""
    if repo is None:
        _modules_cache.clear()
    else:
        # Remove entries matching the repo's collection name
        from src.config import REPOS
        config = REPOS.get(repo)
        if config:
            _modules_cache.pop(config.collection_name, None)


def _get_thresholds_cached(repo: str) -> dict:
    """Get thresholds for a repo, with in-memory caching."""
    if repo not in _calibration_cache:
        _calibration_cache[repo] = get_thresholds(repo)
    return _calibration_cache[repo]


def _confidence_label(distance: float, repo: str) -> str:
    """Return a confidence label based on L2 distance and per-repo thresholds."""
    thresholds = _get_thresholds_cached(repo)
    if distance < thresholds["HIGH"]:
        return "HIGH"
    elif distance < thresholds["MEDIUM"]:
        return "MEDIUM"
    elif distance < thresholds["LOW"]:
        return "LOW"
    else:
        return "NO MATCH"


def _normalized_score(distance: float, repo: str) -> float:
    """Return a normalized relevance percentage (100% = perfect, 0% = irrelevant).

    Uses a piecewise scale based on per-repo calibration thresholds:
      distance 0        → 100%
      distance = HIGH   → 85%   (knowledge p25 — top-quartile match)
      distance = MEDIUM → 55%   (knowledge p75 — typical match)
      distance = LOW    → 15%   (nonsense p25  — noise floor)
      distance > floor  →  0%
    """
    thresholds = _get_thresholds_cached(repo)
    high = thresholds.get("HIGH", 700)
    med = thresholds.get("MEDIUM", 800)
    low = thresholds.get("LOW", 900)
    floor = low * 1.2  # below this = 0%

    if distance <= 0:
        return 100.0
    elif distance <= high:
        return round(100.0 - (distance / high) * 15.0, 1)
    elif distance <= med:
        t = (distance - high) / (med - high) if med > high else 0
        return round(85.0 - t * 30.0, 1)
    elif distance <= low:
        t = (distance - med) / (low - med) if low > med else 0
        return round(55.0 - t * 40.0, 1)
    elif distance <= floor:
        t = (distance - low) / (floor - low) if floor > low else 0
        return round(max(0.0, 15.0 - t * 15.0), 1)
    else:
        return 0.0


def _filter_no_match(results: dict, repo: str) -> dict:
    """Remove results with NO MATCH confidence to avoid wasting context on noise."""
    filtered = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
    for i, dist in enumerate(results["distances"][0]):
        if _confidence_label(dist, repo) != "NO MATCH":
            filtered["ids"][0].append(results["ids"][0][i])
            filtered["metadatas"][0].append(results["metadatas"][0][i])
            filtered["documents"][0].append(results["documents"][0][i])
            filtered["distances"][0].append(results["distances"][0][i])
    return filtered


def _format_results(results: dict, repo: str, offset: int = 0) -> str:
    """Format ChromaDB results into readable text."""
    lines = []
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    for i, (chunk_id, meta, doc, dist) in enumerate(zip(ids, metadatas, documents, distances)):
        confidence = _confidence_label(dist, repo)
        score = _normalized_score(dist, repo)
        result_num = offset + i + 1
        header = f"--- Result {result_num} (relevance: {score}%, confidence: {confidence}) ---"
        file_path = meta.get("file_path", "?")
        class_name = meta.get("class_name", "?")
        method_name = meta.get("method_name", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        chunk_type = meta.get("chunk_type", "?")
        module = meta.get("module", "")

        location = f"{file_path}:{start_line}-{end_line}"
        name = f"{class_name}.{method_name}" if method_name else class_name
        module_str = f" [{module}]" if module else ""

        lines.append(header)
        lines.append(f"  {name}{module_str} ({chunk_type})")
        lines.append(f"  {location}")
        lines.append(f"  Chunk ID: {chunk_id}")
        lines.append("")

        # Show source snippet (truncated to save context window)
        source = meta.get("source", doc)
        source_lines = source.split("\n")
        if len(source_lines) > 20:
            snippet = "\n".join(source_lines[:20]) + f"\n  ... ({len(source_lines) - 20} more lines)"
        else:
            snippet = source
        lines.append(snippet)
        lines.append("")

    return "\n".join(lines)


def _format_results_json(results: dict, repo: str, offset: int = 0) -> str:
    """Format ChromaDB results into structured JSON."""
    payload = {
        "repo": repo,
        "offset": offset,
        "results": [],
    }

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    for i, (chunk_id, meta, doc, dist) in enumerate(zip(ids, metadatas, documents, distances)):
        source = meta.get("source", doc)
        source_lines = source.split("\n")
        truncated = len(source_lines) > 20
        snippet = "\n".join(source_lines[:20]) if truncated else source

        payload["results"].append({
            "result_num": offset + i + 1,
            "chunk_id": chunk_id,
            "distance": round(float(dist), 4),
            "relevance_pct": _normalized_score(dist, repo),
            "confidence": _confidence_label(dist, repo),
            "file_path": meta.get("file_path", ""),
            "class_name": meta.get("class_name", ""),
            "method_name": meta.get("method_name", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "module": meta.get("module", ""),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "source_snippet": snippet,
            "source_truncated": truncated,
        })

    return json.dumps(payload, indent=2)


# Chunk type display order for get_file_chunks output
_CHUNK_TYPE_ORDER = {
    "class_summary": 0, "whole_class": 1, "prefab_summary": 2,
    "constructor": 3, "property": 4, "method": 5, "function": 6,
    "gameobject": 7, "component": 8, "template": 9,
}


def get_file_chunks(file_path: str, repo: str = "mainapp", output_format: str = "text",
                    offset: int = 0, limit: int = 20) -> str:
    """Retrieve indexed chunks for a specific file with pagination.

    Args:
        file_path: Path relative to repo root (e.g., "UnityProject/Assets/Scripts/Foo.cs").
        repo: Which repo to search (default "mainapp").
        output_format: "text" (default) or "json".
        offset: Skip the first N chunks (default 0).
        limit: Maximum chunks to return (1-50, default 20).

    Returns:
        Paginated file chunk list in text or JSON format.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    repo = resolve_repo(repo)
    if repo not in REPOS:
        return f"Error: Unknown repo '{repo}'. Valid repos: {list(REPOS.keys())}"

    offset = max(offset, 0)
    limit = min(max(limit, 1), 50)

    config = REPOS[repo]
    collection = get_collection(config.collection_name)

    # Normalize path separators
    file_path = file_path.replace("\\", "/")

    try:
        results = collection.get(
            where={"file_path": file_path},
            include=["metadatas", "documents"],
        )
    except Exception as e:
        return f"Error querying chunks: {e}"

    if not results["ids"]:
        return f"No chunks found for '{file_path}' in {repo}. Is the file indexed?"

    # Pair up and sort: class_summary first, then by start_line
    all_items = list(zip(results["ids"], results["metadatas"], results["documents"]))
    all_items.sort(key=lambda x: (
        _CHUNK_TYPE_ORDER.get(x[1].get("chunk_type", ""), 99),
        x[1].get("start_line", 0),
    ))

    total = len(all_items)

    # Apply pagination
    items = all_items[offset:offset + limit]

    if not items:
        return f"No more chunks for '{file_path}' at offset {offset} (total: {total})."

    if output_format == "json":
        return _format_file_chunks_json(file_path, repo, items, total, offset, limit)

    # Header with pagination info
    end_idx = offset + len(items)
    lines = [f"=== Chunks for {file_path} ({total} total, showing {offset + 1}-{end_idx}) ===", ""]

    for chunk_id, meta, doc in items:
        chunk_type = meta.get("chunk_type", "?")
        class_name = meta.get("class_name", "?")
        method_name = meta.get("method_name", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")

        name = f"{class_name}.{method_name}" if method_name else class_name
        lines.append(f"[{chunk_type}] {name}")
        lines.append(f"  Chunk ID: {chunk_id}")
        lines.append(f"  Lines {start_line}-{end_line}")
        lines.append("")

        # Show source snippet (truncated at 60 lines)
        source = meta.get("source", doc)
        source_lines = source.split("\n")
        if len(source_lines) > 60:
            snippet = "\n".join(source_lines[:60]) + f"\n  ... ({len(source_lines) - 60} more lines)"
        else:
            snippet = source
        lines.append(snippet)
        lines.append("")

    if end_idx < total:
        lines.append(f"--- {total - end_idx} more chunks. Use offset={end_idx} to see next page. ---")

    return "\n".join(lines)


def _format_file_chunks_json(file_path: str, repo: str, items: list[tuple],
                             total: int, offset: int, limit: int) -> str:
    """Format get_file_chunks output as structured JSON with pagination info."""
    payload = {
        "repo": repo,
        "file_path": file_path,
        "total_chunks": total,
        "offset": offset,
        "limit": limit,
        "returned": len(items),
        "chunks": [],
    }

    for chunk_id, meta, doc in items:
        source = meta.get("source", doc)
        source_lines = source.split("\n")
        truncated = len(source_lines) > 60
        snippet = "\n".join(source_lines[:60]) if truncated else source

        payload["chunks"].append({
            "chunk_id": chunk_id,
            "chunk_type": meta.get("chunk_type", ""),
            "class_name": meta.get("class_name", ""),
            "method_name": meta.get("method_name", ""),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "source_snippet": snippet,
            "source_truncated": truncated,
        })

    return json.dumps(payload, indent=2)
