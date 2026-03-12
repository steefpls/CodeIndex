"""Markdown file chunking for code-index.

Small files (<150 lines) -> single whole_class chunk.
Large files -> split by ## headings (H2), falling back to # (H1).
If no headings at all -> single whole_class chunk.
"""

import logging
import re
from pathlib import PurePosixPath

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)


def chunk_file_markdown(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total_lines = len(lines) or 1
    file_stem = PurePosixPath(file_path).stem
    namespace = PurePosixPath(file_path).parent.as_posix()
    if namespace == ".":
        namespace = ""

    doc_comment = _extract_doc_comment(lines)

    # Small file: single whole_class chunk
    if total_lines < SMALL_FILE_LINE_THRESHOLD:
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

    # Try splitting by H2 first, fall back to H1
    sections = _find_heading_sections(lines, level=2)
    if len(sections) < 2:
        sections = _find_heading_sections(lines, level=1)

    if len(sections) < 2:
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
    for i, (heading, start_line) in enumerate(sections):
        if i + 1 < len(sections):
            end_line = sections[i + 1][1] - 1
        else:
            end_line = total_lines
        # Trim trailing blank lines
        while end_line > start_line and lines[end_line - 1].strip() == "":
            end_line -= 1

        chunk_source = "".join(lines[start_line - 1:end_line])
        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=heading,
            namespace=namespace,
            start_line=start_line,
            end_line=end_line,
            source=chunk_source,
            chunk_type="md_section",
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


def _extract_doc_comment(lines: list[str]) -> str:
    """Extract the first H1 title or first non-empty paragraph."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            return stripped[2:].strip()
        # First non-empty, non-heading line as fallback
        return stripped
    return ""


def _find_heading_sections(lines: list[str], level: int) -> list[tuple[str, int]]:
    """Find heading boundaries at the given level (1 for #, 2 for ##)."""
    prefix = "#" * level + " "
    sections = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(prefix):
            # Make sure it's exactly this level (## not ###)
            if level < 6 and stripped.startswith("#" * (level + 1)):
                continue
            heading_text = stripped[len(prefix):].strip()
            sections.append((heading_text, i + 1))  # 1-indexed
    return sections
