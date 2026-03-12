"""Module detection from directory structure and .asmdef files."""

import json
from pathlib import Path

from src.config import SourceDirConfig


# Default strip prefixes (used when no repo-level config is available)
_DEFAULT_STRIP_PREFIXES = ("Augmentus.", "Augmentus.MainApp.")


def detect_module(file_path: Path, sd_config: SourceDirConfig,
                  strip_prefixes: list[str] | None = None) -> str:
    """Detect the module name for a file based on directory structure.

    For MainApp: looks for .asmdef files walking up from the file.
    For Perception: uses the first directory under the source dir.
    Falls back to the first directory component under the source dir.

    Args:
        file_path: Absolute path to the source file.
        sd_config: Source directory config for the file.
        strip_prefixes: Optional list of .asmdef name prefixes to strip.
    """
    try:
        rel = file_path.relative_to(sd_config.path)
    except ValueError:
        return ""

    # Try .asmdef-based detection (walk up from file)
    current = file_path.parent
    while current != sd_config.path and str(current).startswith(str(sd_config.path)):
        asmdef_files = list(current.glob("*.asmdef"))
        if asmdef_files:
            return _parse_asmdef_name(asmdef_files[0], strip_prefixes)
        current = current.parent

    # Fall back to first directory component under source_dir
    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return ""


def _parse_asmdef_name(asmdef_path: Path,
                       strip_prefixes: list[str] | None = None) -> str:
    """Extract the assembly name from a Unity .asmdef file.

    Args:
        asmdef_path: Path to the .asmdef file.
        strip_prefixes: Prefixes to strip from the assembly name.
                        Defaults to _DEFAULT_STRIP_PREFIXES if None.
    """
    if strip_prefixes is None:
        prefixes = _DEFAULT_STRIP_PREFIXES
    else:
        prefixes = tuple(strip_prefixes)
    try:
        data = json.loads(asmdef_path.read_text(encoding="utf-8"))
        name = data.get("name", asmdef_path.stem)
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name
    except (json.JSONDecodeError, OSError):
        return asmdef_path.stem
