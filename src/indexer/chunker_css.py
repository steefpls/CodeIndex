"""Tree-sitter AST-based CSS code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_css as tscss
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk

logger = logging.getLogger(__name__)

CSS_LANGUAGE = Language(tscss.language())
_parser = Parser(CSS_LANGUAGE)

# Small CSS files are indexed as a single chunk
_SMALL_FILE_LINES = 150

# Maximum length for selector text used as method_name
_MAX_SELECTOR_LENGTH = 100


def chunk_file_css(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a CSS file and return code chunks.

    Small files (< 150 lines) -> one whole_class chunk.
    Large files -> individual css_rule chunks per top-level construct.
    """
    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem
    namespace = PurePosixPath(file_path).parent.as_posix()
    if namespace == ".":
        namespace = ""

    # Small files: single whole_class chunk
    if total_lines < _SMALL_FILE_LINES:
        return [CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="whole_class",
            module=module,
        )]

    # Large files: parse and extract individual rules
    tree = _parser.parse(source)
    root = tree.root_node

    chunks: list[CodeChunk] = []

    for child in root.children:
        if child.type == "rule_set":
            selector = _extract_selector(child)
            doc_comment = _extract_css_comment(child, source)
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_stem,
                method_name=_truncate(selector, _MAX_SELECTOR_LENGTH),
                namespace=namespace,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=_node_text(child),
                chunk_type="css_rule",
                module=module,
                doc_comment=doc_comment,
            ))
        elif child.type == "media_statement":
            condition = _extract_media_condition(child)
            method_name = f"@media {condition}" if condition else "@media"
            doc_comment = _extract_css_comment(child, source)
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_stem,
                method_name=_truncate(method_name, _MAX_SELECTOR_LENGTH),
                namespace=namespace,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=_node_text(child),
                chunk_type="css_rule",
                module=module,
                doc_comment=doc_comment,
            ))
        elif child.type == "keyframes_statement":
            name = _extract_keyframes_name(child)
            method_name = f"@keyframes {name}" if name else "@keyframes"
            doc_comment = _extract_css_comment(child, source)
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_stem,
                method_name=_truncate(method_name, _MAX_SELECTOR_LENGTH),
                namespace=namespace,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=_node_text(child),
                chunk_type="css_rule",
                module=module,
                doc_comment=doc_comment,
            ))
        elif child.type in ("import_statement", "charset_statement",
                             "namespace_statement", "supports_statement",
                             "at_rule"):
            # Other at-rules
            doc_comment = _extract_css_comment(child, source)
            text = _node_text(child)
            first_line = text.split("\n")[0].strip()
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_stem,
                method_name=_truncate(first_line, _MAX_SELECTOR_LENGTH),
                namespace=namespace,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                source=text,
                chunk_type="css_rule",
                module=module,
                doc_comment=doc_comment,
            ))

    # Fallback: if nothing was found, index the whole file
    if not chunks and total_lines > 0:
        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="whole_class",
            module=module,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_selector(rule_node: Node) -> str:
    """Extract the selector text from a rule_set node."""
    for child in rule_node.children:
        if child.type == "selectors":
            return child.text.decode("utf-8", errors="replace").strip()
    # Fallback: text up to first {
    text = _node_text(rule_node)
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip()
    return text.split("\n")[0].strip()


def _extract_media_condition(media_node: Node) -> str:
    """Extract the condition from a @media statement."""
    # Collect all text between @media and the opening {
    text = _node_text(media_node)
    idx = text.find("{")
    if idx > 0:
        header = text[:idx].strip()
        # Strip the @media prefix
        if header.startswith("@media"):
            return header[6:].strip()
        return header
    return ""


def _extract_keyframes_name(keyframes_node: Node) -> str:
    """Extract the animation name from a @keyframes statement."""
    name_node = keyframes_node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace").strip()
    # Fallback: parse from text
    text = _node_text(keyframes_node)
    idx = text.find("{")
    if idx > 0:
        header = text[:idx].strip()
        if header.startswith("@keyframes"):
            return header[10:].strip()
    return ""


def _extract_css_comment(node: Node, source: bytes) -> str:
    """Extract /* ... */ block comment preceding a CSS node."""
    start_line = node.start_point[0]
    source_lines = source.decode("utf-8", errors="replace").split("\n")

    # Look backwards for a block comment
    idx = start_line - 1
    while idx >= 0 and source_lines[idx].strip() == "":
        idx -= 1

    if idx < 0:
        return ""

    line = source_lines[idx].strip()
    if not line.endswith("*/"):
        return ""

    # Collect the full comment block
    comment_lines = []
    while idx >= 0:
        line = source_lines[idx].strip()
        comment_lines.insert(0, line)
        if line.startswith("/*"):
            break
        idx -= 1

    if not comment_lines or not comment_lines[0].strip().startswith("/*"):
        return ""

    # Clean up comment markers
    cleaned = []
    for line in comment_lines:
        line = line.strip()
        if line.startswith("/*"):
            line = line[2:]
        if line.endswith("*/"):
            line = line[:-2]
        line = line.lstrip("* ").strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
