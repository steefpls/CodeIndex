"""Per-collection distance calibration for search confidence thresholds.

After a full reindex, runs generic code + nonsense queries against the collection
to derive adaptive L2 distance thresholds based on percentiles.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.indexer.embedder import get_embedding_function

logger = logging.getLogger(__name__)

_CODE_QUERIES = [
    "error handling exception", "file read write", "constructor initialization",
    "string parsing format", "callback event handler", "configuration settings",
    "logging debug trace", "network connection timeout", "data serialization json",
    "input validation check", "cache lookup store", "async await task",
    "interface implementation abstract", "factory pattern create",
    "state machine transition", "message queue dispatch", "retry backoff logic",
    "hash map dictionary lookup", "tree traversal search", "sort comparison algorithm",
    "mutex lock synchronization", "buffer copy memory", "type conversion cast",
    "enum flags bitmask", "null check guard clause", "dispose cleanup resource",
    "inheritance override virtual", "generic type parameter constraint",
    "loop iteration collection filter", "class constructor dependency injection",
]

_NONSENSE_QUERIES = [
    "chocolate cake recipe frosting", "weather forecast tomorrow rain",
    "guitar chord progression blues", "gardening tips growing tomatoes",
    "movie review rating stars",
]

# Fallback thresholds if no calibration file exists
_DEFAULT_THRESHOLDS = {
    "HIGH": 650,
    "MEDIUM": 775,
    "LOW": 875,
}


def calibrate_collection(collection, repo_name: str) -> dict:
    """Run calibration queries and save per-collection thresholds.

    Embeds 30 code + 5 nonsense queries, queries top-1 distances,
    derives thresholds from percentiles, and saves to JSON.
    """
    ef = get_embedding_function(role="index")

    # Embed all queries at once
    all_queries = _CODE_QUERIES + _NONSENSE_QUERIES
    all_embeddings = ef.embed_queries(all_queries)

    code_embeddings = all_embeddings[:len(_CODE_QUERIES)]
    nonsense_embeddings = all_embeddings[len(_CODE_QUERIES):]

    # Batch query top-1 distances for all embeddings at once
    code_result = collection.query(
        query_embeddings=code_embeddings, n_results=1, include=["distances"],
    )
    code_distances = [
        dists[0] for dists in code_result["distances"] if dists
    ]

    nonsense_result = collection.query(
        query_embeddings=nonsense_embeddings, n_results=1, include=["distances"],
    )
    nonsense_distances = [
        dists[0] for dists in nonsense_result["distances"] if dists
    ]

    code_arr = np.array(code_distances)
    nonsense_arr = np.array(nonsense_distances)

    # Derive thresholds from percentiles
    code_p25 = float(np.percentile(code_arr, 25))
    code_p50 = float(np.percentile(code_arr, 50))
    code_p75 = float(np.percentile(code_arr, 75))
    nonsense_p25 = float(np.percentile(nonsense_arr, 25))
    nonsense_p50 = float(np.percentile(nonsense_arr, 50))
    nonsense_p75 = float(np.percentile(nonsense_arr, 75))

    calibration = {
        "collection_name": collection.name,
        "total_chunks": collection.count(),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "code_distances": {
            "min": float(code_arr.min()),
            "p25": round(code_p25, 1),
            "p50": round(code_p50, 1),
            "p75": round(code_p75, 1),
            "max": float(code_arr.max()),
        },
        "nonsense_distances": {
            "min": float(nonsense_arr.min()),
            "p25": round(nonsense_p25, 1),
            "p50": round(nonsense_p50, 1),
            "p75": round(nonsense_p75, 1),
            "max": float(nonsense_arr.max()),
        },
        "thresholds": {
            "HIGH": round(code_p25, 1),
            "MEDIUM": round(code_p75, 1),
            "LOW": round(nonsense_p25, 1),
        },
    }

    # Save to JSON
    cal_path = DATA_DIR / f"{repo_name}_calibration.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    logger.info("Calibration saved for %s: HIGH<%s MEDIUM<%s LOW<%s",
                repo_name, calibration["thresholds"]["HIGH"],
                calibration["thresholds"]["MEDIUM"], calibration["thresholds"]["LOW"])

    return calibration


def load_calibration(repo_name: str) -> dict | None:
    """Load calibration JSON for a repo. Returns None if not found."""
    cal_path = DATA_DIR / f"{repo_name}_calibration.json"
    if not cal_path.exists():
        return None
    try:
        return json.loads(cal_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_thresholds(repo_name: str) -> dict:
    """Get thresholds for a repo, falling back to defaults if uncalibrated."""
    cal = load_calibration(repo_name)
    if cal and "thresholds" in cal:
        return cal["thresholds"]
    return _DEFAULT_THRESHOLDS
