"""Orchestrates scan -> chunk -> embed -> store indexing pipeline."""

import json
import logging
import re
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from src.config import REPOS, DATA_DIR, get_embedding_batch_size, RepoConfig, SourceDirConfig
from src.indexer.file_scanner import scan_repo_files
from src.indexer.chunker import chunk_file
from src.indexer.chunker_cpp import chunk_file_cpp
from src.indexer.chunker_python import chunk_file_python
from src.indexer.chunker_js import chunk_file_js
from src.indexer.chunker_html import chunk_file_html
from src.indexer.chunker_unity import chunk_file_unity
from src.indexer.chunker_json import chunk_file_json
from src.indexer.chunker_yaml import chunk_file_yaml
from src.indexer.chunker_markdown import chunk_file_markdown
from src.indexer.chunker_rust import chunk_file_rust
from src.indexer.chunker_ts import chunk_file_ts
from src.indexer.chunker_css import chunk_file_css
from src.indexer.chunker_lua import chunk_file_lua
from src.indexer.chunker_go import chunk_file_go
from src.indexer.guid_resolver import build_guid_map
from src.indexer.metadata import detect_module
from src.indexer.embedder import get_collection, get_embedding_function, release_embedding_function
from src.indexer.dep_graph_builder import extract_type_candidates, CODE_CHUNK_TYPES
from src.indexer.sidecar_state import (
    load_state, _empty_state,
    set_hierarchy_contributions, set_dep_graph_contributions,
    set_asset_ref_contributions, remove_file_contributions,
    materialize_and_save_all,
)
from src.indexer.chunk_payload_store import (
    update_payloads_for_file, remove_payloads_for_file,
    load_payloads, save_payloads,
)
from src.models.chunk import CodeChunk

# Type alias for progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None] | None

logger = logging.getLogger(__name__)

# Language -> chunker function registry
_CHUNKERS = {
    "csharp": chunk_file,
    "cpp": chunk_file_cpp,
    "python": chunk_file_python,
    "javascript": chunk_file_js,
    "html": chunk_file_html,
    "unity": chunk_file_unity,
    "json": chunk_file_json,
    "yaml": chunk_file_yaml,
    "markdown": chunk_file_markdown,
    "rust": chunk_file_rust,
    "typescript": chunk_file_ts,
    "css": chunk_file_css,
    "lua": chunk_file_lua,
    "go": chunk_file_go,
}

# For extracting script GUID references from Unity asset files (asset ref sidecar)
_SCRIPT_GUID_RE = re.compile(r"m_Script:.*?guid:\s*([0-9a-f]{32})")
_UNITY_ASSET_EXTENSIONS = frozenset({".prefab", ".unity", ".asset"})


def _extract_asset_ref_entries(file_path: Path, guid_map: dict[str, tuple[str, str]] | None) -> dict[str, dict]:
    """Extract GUID-keyed script entries referenced via m_Script in a Unity file."""
    if guid_map is None or file_path.suffix not in _UNITY_ASSET_EXTENSIONS:
        return {}
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    entries: dict[str, dict] = {}
    seen_guids: set[str] = set()
    for match in _SCRIPT_GUID_RE.finditer(text):
        guid = match.group(1)
        if guid in seen_guids:
            continue
        seen_guids.add(guid)
        entry = guid_map.get(guid)
        if entry is not None:
            class_name, script_path = entry
            entries[guid] = {
                "class_name": class_name,
                "script_path": script_path,
            }
    return entries


def _update_payloads_in_memory(
    payload_store: dict, payload_file_idx: dict[str, list[str]],
    rel_path: str, chunks: list,
) -> None:
    """Update payload store in memory for a file's chunks (no disk I/O).

    Uses payload_file_idx for O(1) lookup of old chunk IDs instead of
    scanning all keys.
    """
    from src.models.chunk import MAX_EMBED_CHARS
    # Remove old payloads for this file via reverse index (O(1) lookup)
    old_ids = payload_file_idx.pop(rel_path, [])
    for cid in old_ids:
        payload_store.pop(cid, None)
    # Add new payloads and update reverse index
    new_ids = []
    for chunk in chunks:
        if len(chunk.source) > MAX_EMBED_CHARS:
            payload_store[chunk.chunk_id] = chunk.source
            new_ids.append(chunk.chunk_id)
    if new_ids:
        payload_file_idx[rel_path] = new_ids


def _remove_payloads_in_memory(
    payload_store: dict, payload_file_idx: dict[str, list[str]],
    rel_path: str,
) -> None:
    """Remove payloads for a file from the in-memory store (no disk I/O)."""
    old_ids = payload_file_idx.pop(rel_path, [])
    for cid in old_ids:
        payload_store.pop(cid, None)


def index_repo(repo_name: str, incremental: bool = True,
               progress_callback: ProgressCallback = None) -> dict:
    """Index a repo: scan files, chunk, embed, store in ChromaDB.

    Args:
        repo_name: Which repo to index.
        incremental: If True, only re-index changed files.
        progress_callback: Optional (current, total, message) callback for progress updates.

    Returns a summary dict with counts and timing.
    """
    if repo_name not in REPOS:
        return {"error": f"Unknown repo: {repo_name}. Valid: {list(REPOS.keys())}"}

    config = REPOS[repo_name]

    # Full rebuild: wipe the collection to remove any orphaned chunks
    if not incremental:
        from src.indexer.embedder import get_chroma_client
        client = get_chroma_client()
        try:
            client.delete_collection(config.collection_name)
            logger.info("Deleted collection '%s' for full rebuild", config.collection_name)
        except Exception as e:
            if isinstance(e, ValueError) or "NotFoundError" in type(e).__name__ or "does not exist" in str(e):
                pass  # collection didn't exist yet
            else:
                raise

    setup_start = time.time()
    has_unity = any(sd.language == "unity" for sd in config.source_dirs)
    setup_total = 5 if has_unity else 4  # number of setup steps

    def _setup_progress(step: int, label: str):
        """Report setup progress with a bar matching the indexing style."""
        pct = int(100 * step / setup_total)
        bar_w = 20
        filled = int(bar_w * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
        elapsed = time.time() - setup_start
        elapsed_str = f"{elapsed:.1f}s"
        msg = f"[{bar}] Setup {step}/{setup_total} | {label} | {elapsed_str}"
        if progress_callback:
            progress_callback(step, setup_total, msg)
        logger.info("Setup: %s", msg)

    _setup_progress(0, "Loading collection...")
    collection = get_collection(config.collection_name)
    index_embedder = get_embedding_function(role="index")

    manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
    old_manifest = _load_manifest(manifest_path) if incremental else {}

    _setup_progress(1, "Scanning files...")

    # Scan all source directories (C#, C++, etc.)
    try:
        source_files = scan_repo_files(config)
    except FileNotFoundError as e:
        return {"error": str(e)}

    new_manifest = {}
    files_indexed = 0
    files_skipped = 0
    chunks_added = 0
    chunks_deleted = 0
    errors = []

    _setup_progress(2, f"Loading sidecar state... ({len(source_files)} files found)")

    # Load sidecar contribution state (empty for full rebuild, existing for incremental)
    sidecar_state = load_state(repo_name) if incremental else _empty_state()
    force_sidecar_refresh = (
        incremental
        and bool(old_manifest)
        and not sidecar_state.get("hierarchy")
        and not sidecar_state.get("dep_graph")
        and not sidecar_state.get("asset_refs")
    )
    if force_sidecar_refresh:
        logger.info("Sidecar state is empty for %s; rebuilding per-file sidecar contributions.", repo_name)

    # Build GUID map once if any source_dir uses the "unity" language
    guid_map = None
    if has_unity:
        _setup_progress(3, "Building GUID map...")
        assets_dir = config.root / "UnityProject" / "Assets"
        if assets_dir.exists():
            guid_map = build_guid_map(assets_dir, repo_root=config.root)
        else:
            logger.warning("Assets dir not found for GUID resolution: %s", assets_dir)

    # Payload store: empty for full rebuild (old data is stale), load for incremental.
    # Uses a reverse index (file -> chunk_ids) for O(1) removal instead of O(n) scan.
    _setup_progress(setup_total - 1, "Loading payload store...")
    if incremental:
        payload_store = load_payloads(repo_name)
        # Build reverse index from existing manifest so we know which chunk_ids
        # belong to which file (for O(1) removal during re-indexing)
        payload_file_idx: dict[str, list[str]] = {}
        for rp, entry in old_manifest.items():
            cids_in_store = [cid for cid in entry.get("chunk_ids", []) if cid in payload_store]
            if cids_in_store:
                payload_file_idx[rp] = cids_in_store
    else:
        payload_store = {}
        payload_file_idx = {}
    payload_store_dirty = False
    _payload_flush_interval = max(1, len(source_files) // 20)  # every 5%
    _payload_files_since_flush = 0

    setup_elapsed = time.time() - setup_start
    _setup_progress(setup_total, f"Done — {len(source_files)} files to index ({setup_elapsed:.1f}s)")
    logger.info("Setup complete in %.1fs: %d files to process", setup_elapsed, len(source_files))

    # Start the indexing timer AFTER setup is done
    start_time = time.time()
    total_files = len(source_files)
    embed_batch_size = get_embedding_batch_size()
    _last_cb_time = start_time
    _last_log_time = start_time
    for file_idx, (file_path, sd_config) in enumerate(source_files):
        now = time.time()
        if file_idx == 0 or now - _last_cb_time >= 0.1:
            elapsed_so_far = now - start_time
            rate = chunks_added / elapsed_so_far if elapsed_so_far > 0 and chunks_added > 0 else 0
            files_done = file_idx + 1
            files_remaining = total_files - files_done
            pct = int(100 * files_done / total_files) if total_files > 0 else 0
            elapsed_str = f"{elapsed_so_far/60:.1f}min" if elapsed_so_far >= 60 else f"{elapsed_so_far:.0f}s"
            if files_done > 0 and files_done < total_files and elapsed_so_far > 1:
                eta_sec = elapsed_so_far * files_remaining / files_done
                total_est = elapsed_so_far + eta_sec
                total_str = f"{total_est/60:.1f}min" if total_est >= 60 else f"{total_est:.0f}s"
                time_str = f" | {elapsed_str} / ~{total_str}"
            else:
                time_str = f" | {elapsed_str}"
            bar_w = 20
            filled = int(bar_w * pct / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            msg = f"[{bar}] {pct}% | {files_done}/{total_files} files ({rate:.0f} chunks/sec){time_str}"
            if progress_callback:
                progress_callback(files_done, total_files, msg)
            _last_cb_time = now
            # Log less frequently to avoid spam
            if file_idx == 0 or now - _last_log_time >= 2.0:
                logger.info("Progress: %s", msg)
                _last_log_time = now

        rel_path = str(file_path.relative_to(config.root)).replace("\\", "/")
        mtime = file_path.stat().st_mtime

        sidecar_only_refresh = False

        # Incremental: skip unchanged files unless sidecar refresh is required
        if incremental and rel_path in old_manifest:
            old_entry = old_manifest[rel_path]
            if old_entry.get("mtime") == mtime:
                if not force_sidecar_refresh:
                    new_manifest[rel_path] = old_entry
                    files_skipped += 1
                    continue
                sidecar_only_refresh = True

        # File changed or new: re-chunk
        try:
            source = file_path.read_bytes()
            module = detect_module(file_path, sd_config, config.strip_prefixes or None)
            chunker = _CHUNKERS.get(sd_config.language)
            if chunker is None:
                logger.warning("No chunker for language %r, skipping %s", sd_config.language, rel_path)
                continue
            if sd_config.language == "unity" and guid_map is not None:
                chunker = partial(chunker, guid_map=guid_map)
            chunks = chunker(source, rel_path, module)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", rel_path, e)
            errors.append(f"{rel_path}: {e}")
            continue

        # Collect sidecar contributions for this file
        file_hierarchy_records = []
        file_dep_refs: dict[tuple[str, str], dict] = {}
        for c in chunks:
            if c.class_name and c.base_types and c.chunk_type in ("class_summary", "whole_class"):
                file_hierarchy_records.append((
                    c.chunk_type, c.class_name, c.file_path,
                    c.module, c.namespace, list(c.base_types),
                ))
            if c.class_name and c.chunk_type in CODE_CHUNK_TYPES:
                key = (c.class_name, c.file_path)
                if key not in file_dep_refs:
                    file_dep_refs[key] = {"module": c.module, "namespace": c.namespace, "refs": set()}
                file_dep_refs[key]["refs"].update(extract_type_candidates(c.source))

        # Update sidecar state for this file
        set_hierarchy_contributions(sidecar_state, rel_path, file_hierarchy_records)
        set_dep_graph_contributions(sidecar_state, rel_path, file_dep_refs)

        # Asset references: extract GUID-keyed script refs in Unity asset files
        asset_ref_entries = _extract_asset_ref_entries(file_path, guid_map)
        set_asset_ref_contributions(sidecar_state, rel_path, asset_ref_entries)

        # Sidecar-only refresh path for unchanged files after state reset:
        # rebuild contribution state but keep existing embeddings/chunk ids.
        if sidecar_only_refresh:
            new_manifest[rel_path] = old_manifest[rel_path]
            files_skipped += 1
            continue

        # Store full payloads for chunks that exceed embedding cap (in-memory, periodic flush)
        _update_payloads_in_memory(payload_store, payload_file_idx, rel_path, chunks)
        payload_store_dirty = True
        _payload_files_since_flush += 1
        if _payload_files_since_flush >= _payload_flush_interval:
            save_payloads(repo_name, payload_store)
            payload_store_dirty = False
            _payload_files_since_flush = 0

        # Delete old chunks for this file if they existed
        if rel_path in old_manifest:
            old_ids = old_manifest[rel_path].get("chunk_ids", [])
            if old_ids:
                try:
                    collection.delete(ids=old_ids)
                    chunks_deleted += len(old_ids)
                except Exception as e:
                    logger.warning("Failed to delete old chunks for %s: %s", rel_path, e)
                    chunks_deleted += len(old_ids)  # count as deleted for reporting

        # Embed and add new chunks with retry
        chunk_ids = []
        for i in range(0, len(chunks), embed_batch_size):
            batch = chunks[i:i + embed_batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.embedding_text for c in batch]
            metadatas = [c.metadata for c in batch]
            embeddings = index_embedder(documents)
            for attempt in range(3):
                try:
                    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                    chunk_ids.extend(ids)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning("Embed retry %d for %s: %s", attempt + 1, rel_path, e)
                        time.sleep(2 ** attempt)
                    else:
                        logger.error("Failed to embed %s after 3 attempts: %s", rel_path, e)
                        errors.append(f"{rel_path}: embed failed: {e}")

        chunks_added += len(chunks)
        files_indexed += 1
        new_manifest[rel_path] = {
            "mtime": mtime,
            "chunk_ids": chunk_ids,
        }

    # Delete chunks for files that no longer exist
    current_rel_paths = {str(f.relative_to(config.root)).replace("\\", "/") for f, _ in source_files}
    removed_files = set(old_manifest.keys()) - current_rel_paths
    for removed in removed_files:
        old_ids = old_manifest[removed].get("chunk_ids", [])
        if old_ids:
            try:
                collection.delete(ids=old_ids)
            except Exception as e:
                logger.warning("Failed to delete chunks for removed file %s: %s", removed, e)
            chunks_deleted += len(old_ids)
        remove_file_contributions(sidecar_state, removed)
        _remove_payloads_in_memory(payload_store, payload_file_idx, removed)
        payload_store_dirty = True

    # Save updated manifest
    _save_manifest(manifest_path, new_manifest)

    # Flush batched payload store to disk (single write instead of per-file)
    if payload_store_dirty:
        save_payloads(repo_name, payload_store)

    # Materialize sidecar data files from contribution state (both incremental and full)
    try:
        materialize_and_save_all(repo_name, sidecar_state)
    except Exception as e:
        logger.warning("Failed to materialize sidecars for %s: %s", repo_name, e)

    elapsed = time.time() - start_time
    total_chunks = collection.count()

    # Run calibration on full reindex, or incremental if no calibration exists yet
    from src.indexer.calibration import calibrate_collection
    cal_path = DATA_DIR / f"{repo_name}_calibration.json"
    if not incremental or not cal_path.exists():
        try:
            calibrate_collection(collection, repo_name)
        except Exception as e:
            logger.warning("Calibration failed for %s: %s", repo_name, e)

    # Release the GPU embedder to reclaim VRAM accumulated in DirectML/CUDA
    # memory pools during indexing.  The singleton is recreated on next use.
    try:
        release_embedding_function(role="index")
        logger.info("Released index embedder to reclaim GPU memory")
    except Exception as e:
        logger.warning("Failed to release index embedder: %s", e)

    return {
        "repo": repo_name,
        "files_scanned": len(source_files),
        "files_indexed": files_indexed,
        "files_skipped": files_skipped,
        "chunks_added": chunks_added,
        "chunks_deleted": chunks_deleted,
        "total_chunks": total_chunks,
        "errors": errors[:10],  # cap error list
        "setup_seconds": round(setup_elapsed, 2),
        "elapsed_seconds": round(elapsed, 1),
    }


def reindex_single_file(repo_name: str, file_path_str: str) -> dict:
    """Re-index a single file: delete old chunks and re-chunk/re-embed.

    Args:
        repo_name: Resolved repo name.
        file_path_str: Relative path from repo root (forward slashes).

    Returns:
        Dict with chunks_deleted and chunks_added counts.
    """
    if repo_name not in REPOS:
        raise ValueError(f"Unknown repo: {repo_name}")

    config = REPOS[repo_name]
    collection = get_collection(config.collection_name)
    index_embedder = get_embedding_function(role="index")
    manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
    manifest = _load_manifest(manifest_path)

    # Normalize to forward slashes
    rel_path = file_path_str.replace("\\", "/")
    abs_path = config.root / rel_path

    if not abs_path.exists():
        raise FileNotFoundError(f"File not found: {abs_path}")

    # Find which source_dir this file belongs to — match by file extension
    ext = abs_path.suffix.lower()
    sd_config = None
    sd_fallback = None
    for sd in config.source_dirs:
        try:
            abs_path.relative_to(sd.path)
        except ValueError:
            continue
        if ext in sd.extensions:
            sd_config = sd
            break
        elif sd_fallback is None:
            sd_fallback = sd
    if sd_config is None:
        sd_config = sd_fallback

    if sd_config is None:
        raise ValueError(f"File '{rel_path}' is not under any configured source directory for repo '{repo_name}'.")

    # Delete old chunks
    chunks_deleted = 0
    if rel_path in manifest:
        old_ids = manifest[rel_path].get("chunk_ids", [])
        if old_ids:
            collection.delete(ids=old_ids)
            chunks_deleted = len(old_ids)

    # Re-chunk the file
    source = abs_path.read_bytes()
    module = detect_module(abs_path, sd_config, config.strip_prefixes or None)
    chunker = _CHUNKERS.get(sd_config.language)
    if chunker is None:
        raise ValueError(f"No chunker for language '{sd_config.language}'")
    guid_map = None
    if sd_config.language == "unity":
        assets_dir = config.root / "UnityProject" / "Assets"
        guid_map = build_guid_map(assets_dir, repo_root=config.root) if assets_dir.exists() else {}
        chunker = partial(chunker, guid_map=guid_map)
    chunks = chunker(source, rel_path, module)

    # Add new chunks
    chunk_ids = []
    embed_batch_size = get_embedding_batch_size()
    for i in range(0, len(chunks), embed_batch_size):
        batch = chunks[i:i + embed_batch_size]
        ids = [c.chunk_id for c in batch]
        documents = [c.embedding_text for c in batch]
        metadatas = [c.metadata for c in batch]
        embeddings = index_embedder(documents)
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        chunk_ids.extend(ids)

    # Update manifest
    manifest[rel_path] = {
        "mtime": abs_path.stat().st_mtime,
        "chunk_ids": chunk_ids,
    }
    _save_manifest(manifest_path, manifest)

    # Update sidecar contributions for this file
    sidecar_state = load_state(repo_name)
    file_hierarchy = []
    file_dep_refs: dict[tuple[str, str], dict] = {}
    for c in chunks:
        if c.class_name and c.base_types and c.chunk_type in ("class_summary", "whole_class"):
            file_hierarchy.append((
                c.chunk_type, c.class_name, c.file_path,
                c.module, c.namespace, list(c.base_types),
            ))
        if c.class_name and c.chunk_type in CODE_CHUNK_TYPES:
            key = (c.class_name, c.file_path)
            if key not in file_dep_refs:
                file_dep_refs[key] = {"module": c.module, "namespace": c.namespace, "refs": set()}
            file_dep_refs[key]["refs"].update(extract_type_candidates(c.source))

    set_hierarchy_contributions(sidecar_state, rel_path, file_hierarchy)
    set_dep_graph_contributions(sidecar_state, rel_path, file_dep_refs)
    set_asset_ref_contributions(sidecar_state, rel_path,
                                _extract_asset_ref_entries(abs_path, guid_map))
    materialize_and_save_all(repo_name, sidecar_state)

    # Update payload store
    update_payloads_for_file(repo_name, rel_path, chunks)

    return {"chunks_deleted": chunks_deleted, "chunks_added": len(chunks)}


def remove_single_file(repo_name: str, file_path_str: str) -> dict:
    """Remove a single file from the index.

    Args:
        repo_name: Resolved repo name.
        file_path_str: Relative path from repo root (forward slashes).

    Returns:
        Dict with chunks_deleted count.
    """
    if repo_name not in REPOS:
        raise ValueError(f"Unknown repo: {repo_name}")

    config = REPOS[repo_name]
    collection = get_collection(config.collection_name)
    manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
    manifest = _load_manifest(manifest_path)

    rel_path = file_path_str.replace("\\", "/")

    if rel_path not in manifest:
        raise ValueError(f"File '{rel_path}' not found in {repo_name} manifest. Is it indexed?")

    old_ids = manifest[rel_path].get("chunk_ids", [])
    chunks_deleted = 0
    if old_ids:
        collection.delete(ids=old_ids)
        chunks_deleted = len(old_ids)

    del manifest[rel_path]
    _save_manifest(manifest_path, manifest)

    # Remove sidecar contributions and rematerialize
    sidecar_state = load_state(repo_name)
    remove_file_contributions(sidecar_state, rel_path)
    materialize_and_save_all(repo_name, sidecar_state)

    # Remove payloads
    remove_payloads_for_file(repo_name, rel_path)

    return {"chunks_deleted": chunks_deleted}


def _load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _process_file_for_sidecar(
    rel_path: str,
    config: RepoConfig,
    guid_map: dict[str, tuple[str, str]] | None,
    skip_unity: bool,
) -> tuple[str, list, dict[tuple[str, str], dict], dict[str, dict]] | tuple[str, None, None, None]:
    """Process a single file for sidecar extraction (no shared state access).

    Returns (rel_path, hierarchy, dep_refs, asset_refs) on success,
    or (rel_path, None, None, None) on skip/error.
    """
    abs_path = config.root / rel_path
    if not abs_path.exists():
        return (rel_path, None, None, None)

    # Find source_dir config — match by file extension
    ext = abs_path.suffix.lower()
    sd_config = None
    sd_fallback = None
    for sd in config.source_dirs:
        try:
            abs_path.relative_to(sd.path)
        except ValueError:
            continue
        if ext in sd.extensions:
            sd_config = sd
            break
        elif sd_fallback is None:
            sd_fallback = sd
    if sd_config is None:
        sd_config = sd_fallback
    if sd_config is None:
        return (rel_path, None, None, None)

    # Skip unity files if requested
    if skip_unity and sd_config.language == "unity":
        return (rel_path, None, None, None)

    try:
        source = abs_path.read_bytes()
        module = detect_module(abs_path, sd_config, config.strip_prefixes or None)
        chunker = _CHUNKERS.get(sd_config.language)
        if chunker is None:
            return (rel_path, None, None, None)
        if sd_config.language == "unity" and guid_map is not None:
            chunker = partial(chunker, guid_map=guid_map)
        chunks = chunker(source, rel_path, module)
    except Exception:
        return (rel_path, None, None, None)

    file_hierarchy = []
    file_dep_refs: dict[tuple[str, str], dict] = {}
    for c in chunks:
        if c.class_name and c.base_types and c.chunk_type in ("class_summary", "whole_class"):
            file_hierarchy.append((
                c.chunk_type, c.class_name, c.file_path,
                c.module, c.namespace, list(c.base_types),
            ))
        if c.class_name and c.chunk_type in CODE_CHUNK_TYPES:
            key = (c.class_name, c.file_path)
            if key not in file_dep_refs:
                file_dep_refs[key] = {"module": c.module, "namespace": c.namespace, "refs": set()}
            file_dep_refs[key]["refs"].update(extract_type_candidates(c.source))

    asset_refs = _extract_asset_ref_entries(abs_path, guid_map) if not skip_unity else {}
    return (rel_path, file_hierarchy, file_dep_refs, asset_refs)


def rebuild_sidecars(repo_name: str,
                     skip_unity: bool = False,
                     progress_callback: ProgressCallback = None) -> str:
    """Rebuild sidecar data by re-scanning all indexed files.

    Reads the manifest, re-chunks each file (without re-embedding), extracts
    sidecar contributions, and materializes fresh sidecar JSONs. Useful for
    repair/migration after schema changes.

    Args:
        repo_name: Repo name (e.g. "mainapp", "perception").
        skip_unity: If True, skip Unity YAML files (prefabs/scenes/assets) and only
            rebuild sidecars from code files. Much faster for large Unity repos.
        progress_callback: Optional (current, total, message) callback.
    """
    t_total = time.time()

    if repo_name not in REPOS:
        return f"Error: Unknown repo '{repo_name}'. Valid: {list(REPOS.keys())}"

    config = REPOS[repo_name]
    manifest_path = DATA_DIR / f"{repo_name}_manifest.json"
    manifest = _load_manifest(manifest_path)

    if not manifest:
        return f"No manifest found for '{repo_name}'. Run reindex first."

    total_files = len(manifest)

    def _report(current: int, total: int, label: str):
        pct = int(100 * current / total) if total else 0
        bar_w = 20
        filled = int(bar_w * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
        elapsed = time.time() - t_total
        elapsed_str = f"{elapsed/60:.1f}min" if elapsed >= 60 else f"{elapsed:.1f}s"
        if current > 0 and current < total and elapsed > 1:
            eta = elapsed * (total - current) / current
            total_est = elapsed + eta
            total_str = f"{total_est/60:.1f}min" if total_est >= 60 else f"{total_est:.0f}s"
            time_str = f"{elapsed_str} / ~{total_str}"
        else:
            time_str = elapsed_str
        msg = f"[{bar}] {pct}% | {current}/{total} files | {label} | {time_str}"
        if progress_callback:
            progress_callback(current, total, msg)

    # Phase 1: Build GUID map (skip if not needed)
    _report(0, total_files, "Building GUID map...")
    t0 = time.time()
    has_unity = any(sd.language == "unity" for sd in config.source_dirs)

    guid_map = None
    if has_unity and not skip_unity:
        assets_dir = config.root / "UnityProject" / "Assets"
        if assets_dir.exists():
            guid_map = build_guid_map(assets_dir, repo_root=config.root)
    t_guid = time.time() - t0
    logger.info("rebuild_sidecars(%s) GUID map: %.1fs%s",
                repo_name, t_guid, " (skipped)" if skip_unity else "")

    # Phase 2: Process files sequentially
    # (CPU-bound parsing — threading adds GIL contention overhead)
    _report(0, total_files, "Processing files...")
    t0 = time.time()

    # When skipping unity, load existing state to preserve unity contributions
    if skip_unity:
        sidecar_state = load_state(repo_name)
    else:
        sidecar_state = _empty_state()

    files_processed = 0
    files_skipped = 0

    for file_idx, rel_path in enumerate(manifest):
        result_tuple = _process_file_for_sidecar(rel_path, config, guid_map, skip_unity)
        _, hierarchy, dep_refs, asset_refs = result_tuple
        if hierarchy is not None:
            set_hierarchy_contributions(sidecar_state, rel_path, hierarchy)
            set_dep_graph_contributions(sidecar_state, rel_path, dep_refs)
            set_asset_ref_contributions(sidecar_state, rel_path, asset_refs)
            files_processed += 1
        else:
            files_skipped += 1
        if (file_idx + 1) % 200 == 0 or file_idx + 1 == total_files:
            _report(file_idx + 1, total_files, "Processing files...")

    t_process = time.time() - t0
    logger.info("rebuild_sidecars(%s) file processing: %.1fs (%d processed, %d skipped)",
                repo_name, t_process, files_processed, files_skipped)

    # Phase 3: Materialize sidecar JSONs
    _report(total_files, total_files, "Materializing sidecars...")
    t0 = time.time()
    materialize_and_save_all(repo_name, sidecar_state)
    t_materialize = time.time() - t0
    logger.info("rebuild_sidecars(%s) materialize: %.1fs", repo_name, t_materialize)

    t_elapsed = time.time() - t_total
    _report(total_files, total_files, "Done")
    result = (f"Rebuilt sidecars for '{repo_name}': {files_processed} files processed."
              f"{f' ({files_skipped} unity skipped)' if files_skipped else ''}\n"
              f"  Time: {t_elapsed:.1f}s (GUID map {t_guid:.1f}s + "
              f"processing {t_process:.1f}s + materialize {t_materialize:.1f}s)")
    logger.info(result)
    return result


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
