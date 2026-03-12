"""Targeted Unity entity context retrieval tool.

Returns full parsed context for a specific entity (chunk_id or file+entity)
without dumping the entire 70k+ scene summary. Looks up the full payload
from the chunk payload store, falling back to the indexed source.
"""

import json
import logging

from src.config import REPOS, resolve_repo
from src.indexer.chunk_payload_store import get_payload
from src.indexer.embedder import get_collection

logger = logging.getLogger(__name__)


def get_unity_entity_context(repo: str, chunk_id: str | None = None,
                             file_path: str | None = None,
                             entity_name: str | None = None,
                             output_format: str = "text") -> str:
    """Get full parsed context for a Unity entity.

    Retrieves the full (un-truncated) source for a chunk, either by
    chunk_id directly or by searching for a matching chunk in a file.

    Args:
        repo: Repo name (e.g., "mainapp").
        chunk_id: Direct chunk ID from search results.
        file_path: File path (relative to repo root) to search within.
        entity_name: Entity name (GO name, script name) to find in the file.
        output_format: "text" (default) or "json".

    Returns:
        Full entity context or error message, in text or JSON format.
    """
    output_format = (output_format or "text").lower()
    if output_format not in {"text", "json"}:
        return "Error: output_format must be 'text' or 'json'."

    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        msg = f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"
        if output_format == "json":
            return json.dumps({"status": "error", "message": msg}, indent=2)
        return msg

    config = REPOS[resolved]

    if chunk_id:
        # Direct lookup by chunk_id
        full_source = get_payload(resolved, chunk_id)
        if full_source:
            if output_format == "json":
                return json.dumps({
                    "status": "found",
                    "repo": resolved,
                    "chunk_id": chunk_id,
                    "source": full_source,
                    "source_full": True,
                }, indent=2)
            return f"=== Full context for {chunk_id} ===\n\n{full_source}"

        # Fallback: get from ChromaDB (may be truncated)
        try:
            collection = get_collection(config.collection_name)
            results = collection.get(
                ids=[chunk_id],
                include=["metadatas"],
            )
            if results["ids"]:
                meta = results["metadatas"][0]
                source = meta.get("source", "")
                if output_format == "json":
                    return json.dumps({
                        "status": "found",
                        "repo": resolved,
                        "chunk_id": chunk_id,
                        "source": source,
                        "source_full": False,
                        "note": "from index, may be truncated",
                    }, indent=2)
                return (f"=== Context for {chunk_id} ===\n"
                        f"(from index, may be truncated)\n\n{source}")
        except Exception:
            pass

        msg = f"Chunk '{chunk_id}' not found in {resolved}."
        if output_format == "json":
            return json.dumps({
                "status": "not_found",
                "repo": resolved,
                "chunk_id": chunk_id,
                "message": msg,
            }, indent=2)
        return msg

    if file_path and entity_name:
        # Search for matching chunk by file + entity name
        try:
            collection = get_collection(config.collection_name)
            # Get all chunks for this file
            all_results = collection.get(
                where={"file_path": file_path},
                include=["metadatas"],
            )
            if not all_results["ids"]:
                msg = f"No chunks found for '{file_path}' in {resolved}."
                if output_format == "json":
                    return json.dumps({
                        "status": "not_found",
                        "repo": resolved,
                        "file_path": file_path,
                        "entity_name": entity_name,
                        "message": msg,
                    }, indent=2)
                return msg

            # Find chunks matching the entity name
            # Matches against: GO name (method_name), root name (class_name),
            # and component/script names (base_types) for Unity prefab/scene chunks.
            matches = []
            for i, cid in enumerate(all_results["ids"]):
                meta = all_results["metadatas"][i]
                # Check GO name and root name
                name_match = (meta.get("method_name") == entity_name or
                              meta.get("class_name") == entity_name)
                # Check component/script names in base_types
                if not name_match:
                    base_types = meta.get("base_types", "")
                    if base_types and entity_name in base_types.split(","):
                        name_match = True
                if name_match:
                    # Check payload store first
                    full_source = get_payload(resolved, cid)
                    if full_source is None:
                        full_source = meta.get("source", "")
                    matches.append((cid, meta.get("chunk_type", ""), full_source))

            if not matches:
                available_chunks = [
                    all_results["metadatas"][i].get("method_name")
                    or all_results["metadatas"][i].get("class_name", "?")
                    for i in range(len(all_results["ids"]))
                ]
                if output_format == "json":
                    return json.dumps({
                        "status": "not_found",
                        "repo": resolved,
                        "file_path": file_path,
                        "entity_name": entity_name,
                        "available_chunks": available_chunks,
                    }, indent=2)
                return (f"No entity '{entity_name}' found in '{file_path}'. "
                        f"Available chunks: " + ", ".join(available_chunks))

            if output_format == "json":
                return json.dumps({
                    "status": "found",
                    "repo": resolved,
                    "file_path": file_path,
                    "entity_name": entity_name,
                    "matches": [
                        {
                            "chunk_id": cid,
                            "chunk_type": ctype,
                            "source": source,
                        }
                        for cid, ctype, source in matches
                    ],
                }, indent=2)

            lines = [f"=== {len(matches)} match(es) for '{entity_name}' in {file_path} ===\n"]
            for cid, ctype, source in matches:
                lines.append(f"--- {ctype}: {cid} ---")
                lines.append(source)
                lines.append("")
            return "\n".join(lines)

        except Exception as e:
            if output_format == "json":
                return json.dumps({
                    "status": "error",
                    "repo": resolved,
                    "message": str(e),
                }, indent=2)
            return f"Error: {e}"

    if entity_name and not file_path:
        # Search across all chunks in the repo for matching entity name
        try:
            collection = get_collection(config.collection_name)
            # Search by class_name first, then method_name
            matches = []
            for field in ("class_name", "method_name"):
                try:
                    results = collection.get(
                        where={field: entity_name},
                        include=["metadatas"],
                    )
                    if results["ids"]:
                        seen = {m[0] for m in matches}
                        for i, cid in enumerate(results["ids"]):
                            if cid in seen:
                                continue
                            meta = results["metadatas"][i]
                            full_source = get_payload(resolved, cid)
                            if full_source is None:
                                full_source = meta.get("source", "")
                            matches.append((cid, meta.get("chunk_type", ""),
                                            meta.get("file_path", ""), full_source))
                except Exception:
                    continue

            if matches:
                if output_format == "json":
                    return json.dumps({
                        "status": "found",
                        "repo": resolved,
                        "entity_name": entity_name,
                        "matches": [
                            {
                                "chunk_id": cid,
                                "chunk_type": ctype,
                                "file_path": fpath,
                                "source": source,
                            }
                            for cid, ctype, fpath, source in matches
                        ],
                    }, indent=2)

                lines = [f"=== {len(matches)} match(es) for '{entity_name}' across {resolved} ===\n"]
                for cid, ctype, fpath, source in matches:
                    lines.append(f"--- {ctype}: {cid} ---")
                    lines.append(f"  File: {fpath}")
                    lines.append(source)
                    lines.append("")
                return "\n".join(lines)

            msg = f"No entity '{entity_name}' found in {resolved}."
            if output_format == "json":
                return json.dumps({"status": "not_found", "repo": resolved,
                                   "entity_name": entity_name, "message": msg}, indent=2)
            return msg

        except Exception as e:
            if output_format == "json":
                return json.dumps({"status": "error", "repo": resolved, "message": str(e)}, indent=2)
            return f"Error: {e}"

    msg = "Provide either chunk_id, or both file_path and entity_name, or just entity_name to search."
    if output_format == "json":
        return json.dumps({"status": "error", "message": msg}, indent=2)
    return msg
