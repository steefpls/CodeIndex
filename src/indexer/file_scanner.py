"""Walk repo directories, find source files, apply exclusion patterns."""

import os
from pathlib import Path

from src.config import RepoConfig, SourceDirConfig


def scan_repo_files(config: RepoConfig) -> list[tuple[Path, SourceDirConfig]]:
    """Scan all source directories in a repo.

    Returns a sorted list of (file_path, source_dir_config) tuples so the
    caller knows which language/chunker to use for each file.
    """
    all_files: list[tuple[Path, SourceDirConfig]] = []
    for sd in config.source_dirs:
        files = _scan_dir(sd.path, list(sd.extensions), sd.exclude_patterns)
        all_files.extend((f, sd) for f in files)
    all_files.sort(key=lambda x: x[0])
    return all_files


def _parse_exclude_patterns(patterns: list[str]) -> tuple[set[str], set[str], list[str]]:
    """Pre-parse exclusion patterns into categories for fast matching.

    Returns:
        (excluded_dirs, excluded_extensions, other_patterns)
    """
    excluded_dirs: set[str] = set()
    excluded_extensions: set[str] = set()
    other_patterns: list[str] = []
    for pattern in patterns:
        if pattern.startswith("*."):
            # Extension pattern (e.g., "*.meta") -> ".meta"
            excluded_extensions.add(pattern[1:])
        elif pattern.endswith("/"):
            # Directory pattern (e.g., "bin/", "obj/") -> "bin"
            excluded_dirs.add(pattern.rstrip("/"))
        else:
            other_patterns.append(pattern)
    return excluded_dirs, excluded_extensions, other_patterns


def _scan_dir(source_dir: Path, extensions: list[str], exclude_patterns: list[str]) -> list[Path]:
    """Find all files matching extensions in a directory, excluding configured patterns.

    Uses os.walk with directory pruning instead of rglob to avoid walking
    into excluded directories (e.g., node_modules/, .venv/).
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    ext_set = frozenset(extensions)
    excluded_dirs, excluded_extensions, other_patterns = _parse_exclude_patterns(exclude_patterns)

    files: list[Path] = []
    source_dir_str = str(source_dir)

    for dirpath, dirnames, filenames in os.walk(source_dir):
        # Prune excluded directories in-place (prevents os.walk from descending)
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

        for fname in filenames:
            _, ext = os.path.splitext(fname)
            ext_lower = ext.lower()

            # Must match a wanted extension
            if ext_lower not in ext_set:
                continue

            # Must not match an excluded extension
            if ext_lower in excluded_extensions:
                continue

            full_path = os.path.join(dirpath, fname)

            # Check other patterns (substring matches on the path)
            if other_patterns:
                path_fwd = full_path.replace("\\", "/")
                if any(p in path_fwd for p in other_patterns):
                    continue

            files.append(Path(full_path))

    files.sort()
    return files


def _is_excluded(path: Path, patterns: list[str]) -> bool:
    """Check if a file path matches any exclusion pattern.

    Kept for backward compatibility (used by external callers).
    """
    path_str = str(path).replace("\\", "/")
    for pattern in patterns:
        if pattern.startswith("*."):
            # File extension pattern (e.g., "*.meta")
            if path.suffix == pattern[1:]:
                return True
        elif pattern.endswith("/"):
            # Directory pattern (e.g., "bin/", "obj/")
            dir_name = pattern.rstrip("/")
            if f"/{dir_name}/" in path_str or path_str.endswith(f"/{dir_name}"):
                return True
        else:
            if pattern in path_str:
                return True
    return False
