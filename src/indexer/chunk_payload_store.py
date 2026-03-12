"""Full-payload store for Unity chunks, persisted outside embedding text.

Search returns lightweight snippets (capped at 5000 chars). This store
keeps the full chunk source keyed by chunk_id so that detail tools can
retrieve expanded context on demand without re-parsing the file.

Storage: data/<repo>_chunk_payloads.json (chunk_id -> full source text).
Only stores payloads that exceed MAX_EMBED_CHARS — short chunks don't
need a separate store since their embedding text already contains everything.
"""

import json
import logging
from pathlib import Path

from src.config import DATA_DIR
from src.models.chunk import MAX_EMBED_CHARS

logger = logging.getLogger(__name__)


def _store_path(repo_name: str) -> Path:
    return DATA_DIR / f"{repo_name}_chunk_payloads.json"


def load_payloads(repo_name: str) -> dict[str, str]:
    """Load chunk payloads from disk."""
    path = _store_path(repo_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load chunk payloads for %s: %s", repo_name, e)
        return {}


def save_payloads(repo_name: str, payloads: dict[str, str]) -> None:
    """Save chunk payloads to disk."""
    path = _store_path(repo_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payloads, separators=(",", ":")), encoding="utf-8")


def update_payloads_for_file(repo_name: str, rel_path: str, chunks: list) -> None:
    """Update payload store for a file's chunks.

    Removes old payloads for this file and adds new ones for chunks
    whose source exceeds MAX_EMBED_CHARS.

    Args:
        repo_name: Repo name.
        rel_path: File path relative to repo root.
        chunks: List of CodeChunk objects from this file.
    """
    payloads = load_payloads(repo_name)

    # Remove old payloads for this file (chunk_ids that start with the file path prefix)
    safe_prefix = rel_path.replace("\\", "/").replace("/", "_").replace(":", "")
    to_remove = [cid for cid in payloads if cid.startswith(safe_prefix)]
    for cid in to_remove:
        del payloads[cid]

    # Add new payloads for chunks that exceed the embedding cap
    for chunk in chunks:
        if len(chunk.source) > MAX_EMBED_CHARS:
            payloads[chunk.chunk_id] = chunk.source

    save_payloads(repo_name, payloads)


def remove_payloads_for_file(repo_name: str, rel_path: str) -> None:
    """Remove all payloads for a file."""
    payloads = load_payloads(repo_name)
    safe_prefix = rel_path.replace("\\", "/").replace("/", "_").replace(":", "")
    to_remove = [cid for cid in payloads if cid.startswith(safe_prefix)]
    if to_remove:
        for cid in to_remove:
            del payloads[cid]
        save_payloads(repo_name, payloads)


def get_payload(repo_name: str, chunk_id: str) -> str | None:
    """Get full payload for a chunk_id, or None if not stored."""
    payloads = load_payloads(repo_name)
    return payloads.get(chunk_id)
