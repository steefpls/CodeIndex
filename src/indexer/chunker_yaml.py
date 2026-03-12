"""YAML file chunking for code-index.

Small files (<150 lines) -> single whole_class chunk.
Large files -> split by top-level keys, each becoming a yaml_key chunk.
Falls back to whole_class if top-level is not a mapping or parse fails.
"""

import logging
import re
from pathlib import PurePosixPath

import yaml

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

_TOP_KEY_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_ -]*)\s*:')


def chunk_file_yaml(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total_lines = len(lines) or 1
    file_stem = PurePosixPath(file_path).stem
    namespace = PurePosixPath(file_path).parent.as_posix()
    if namespace == ".":
        namespace = ""

    # Try to parse YAML
    parsed = None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        pass

    is_mapping = isinstance(parsed, dict)
    doc_comment = _extract_doc_comment(parsed) if is_mapping else ""

    # Small file or non-mapping: single whole_class chunk
    if total_lines < SMALL_FILE_LINE_THRESHOLD or not is_mapping or len(parsed) < 2:
        return [CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=text,
            chunk_type="whole_class",
            module=module,
            doc_comment=doc_comment,
        )]

    # Large mapping: split by top-level keys
    key_boundaries = _find_top_level_key_lines(lines)
    if len(key_boundaries) < 2:
        return [CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=text,
            chunk_type="whole_class",
            module=module,
            doc_comment=doc_comment,
        )]

    chunks = []
    for i, (key_name, start_line) in enumerate(key_boundaries):
        if i + 1 < len(key_boundaries):
            end_line = key_boundaries[i + 1][1] - 1
        else:
            end_line = total_lines
        # Trim trailing blank lines
        while end_line > start_line and lines[end_line - 1].strip() == "":
            end_line -= 1

        chunk_source = "".join(lines[start_line - 1:end_line])
        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=key_name,
            namespace=namespace,
            start_line=start_line,
            end_line=end_line,
            source=chunk_source,
            chunk_type="yaml_key",
            module=module,
            doc_comment=doc_comment if i == 0 else "",
        ))

    return chunks or [CodeChunk(
        file_path=file_path,
        class_name=file_stem,
        method_name=None,
        namespace=namespace,
        start_line=1,
        end_line=total_lines,
        source=text,
        chunk_type="whole_class",
        module=module,
        doc_comment=doc_comment,
    )]


def _extract_doc_comment(parsed: dict) -> str:
    for key in ("description", "version"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _find_top_level_key_lines(lines: list[str]) -> list[tuple[str, int]]:
    """Find lines with unindented YAML keys (top-level mapping keys)."""
    boundaries = []
    for i, line in enumerate(lines):
        # Skip comments and blank lines
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        # Top-level key: starts at column 0 (no leading whitespace)
        if line[0] not in (" ", "\t", "-", "#"):
            m = _TOP_KEY_RE.match(line)
            if m:
                boundaries.append((m.group(1), i + 1))  # 1-indexed
    return boundaries
