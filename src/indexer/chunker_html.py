"""Tree-sitter AST-based HTML code chunking.

Handles regular HTML files and Vue Single File Components (SFCs).
Script blocks are sub-chunked via the JavaScript chunker.
"""

import logging
from pathlib import PurePosixPath

import tree_sitter_html as tshtml
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

HTML_LANGUAGE = Language(tshtml.language())
_parser = Parser(HTML_LANGUAGE)

# Semantic landmark elements that can serve as split points
_SECTION_ELEMENTS = {"section", "article", "header", "footer", "nav", "main", "aside"}


def chunk_file_html(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse an HTML file and return code chunks.

    Small files -> single template chunk for the whole file.
    Large files -> try splitting at semantic elements, fall back to whole-file.
    Script blocks -> sub-chunked via the JavaScript chunker.
    Vue SFCs -> script block chunked via JS chunker, template as template chunk.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem
    namespace = PurePosixPath(file_path).parent.as_posix()
    if namespace == ".":
        namespace = ""

    # Detect Vue SFC: top-level elements are <template>, <script>, <style> (no <html>/<body>)
    is_vue = _is_vue_sfc(root)

    # Try to extract <title> for class_name
    class_name = _extract_title(root) or file_stem

    chunks = []

    # Extract and sub-chunk <script> blocks via JS chunker
    script_chunks = _extract_script_chunks(root, source, file_path, file_stem, module, is_vue)
    chunks.extend(script_chunks)

    # Handle the markup/template portion
    if is_vue:
        # For Vue SFCs, chunk the <template> block
        template_node = _find_element_by_tag(root, "template")
        if template_node:
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_stem,
                method_name=None,
                namespace=namespace,
                start_line=template_node.start_point[0] + 1,
                end_line=template_node.end_point[0] + 1,
                source=_node_text(template_node),
                chunk_type="template",
                module=module,
            ))
    elif total_lines < SMALL_FILE_LINE_THRESHOLD:
        # Small file: single template chunk
        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=class_name,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="template",
            module=module,
        ))
    else:
        # Large file: try splitting at semantic elements with id attributes
        section_chunks = _split_at_sections(root, source, file_path, class_name, namespace, module)
        if section_chunks:
            chunks.extend(section_chunks)
        else:
            # Fall back to whole-file chunk
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=class_name,
                method_name=None,
                namespace=namespace,
                start_line=1,
                end_line=total_lines,
                source=source.decode("utf-8", errors="replace"),
                chunk_type="template",
                module=module,
            ))

    # Fallback: if nothing was found at all
    if not chunks and total_lines > 0:
        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="template",
            module=module,
        ))

    return chunks


def _is_vue_sfc(root: Node) -> bool:
    """Detect Vue Single File Component: top-level elements are <template>, <script>, <style>."""
    top_tags = set()
    for child in root.children:
        if child.type == "element":
            tag = _get_tag_name(child)
            if tag:
                top_tags.add(tag)

    # Vue SFCs have template/script/style at top level, never html/body/head
    if top_tags & {"html", "body", "head", "DOCTYPE"}:
        return False
    return bool(top_tags & {"template", "script"})


def _get_tag_name(element_node: Node) -> str:
    """Extract the tag name from an element node."""
    start_tag = element_node.child_by_field_name("start_tag") if element_node.type == "element" else None
    if not start_tag:
        # Try direct children for self-closing or script/style elements
        for child in element_node.children:
            if child.type in ("start_tag", "self_closing_tag"):
                start_tag = child
                break
    if not start_tag:
        return ""

    for child in start_tag.children:
        if child.type == "tag_name":
            return child.text.decode("utf-8", errors="replace")
    return ""


def _get_attribute(element_node: Node, attr_name: str) -> str | None:
    """Get the value of a specific attribute from an element node."""
    start_tag = None
    for child in element_node.children:
        if child.type in ("start_tag", "self_closing_tag"):
            start_tag = child
            break
    if not start_tag:
        return None

    for child in start_tag.children:
        if child.type == "attribute":
            name_node = child.child_by_field_name("name") or (child.children[0] if child.children else None)
            if name_node and name_node.text.decode("utf-8", errors="replace") == attr_name:
                value_node = child.child_by_field_name("value")
                if value_node:
                    text = value_node.text.decode("utf-8", errors="replace")
                    # Strip surrounding quotes
                    if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
                        text = text[1:-1]
                    return text
    return None


def _extract_title(root: Node) -> str:
    """Extract <title> text content from the document."""
    for node in _walk_elements(root):
        tag = _get_tag_name(node)
        if tag == "title":
            # Get the text content (everything between start and end tags)
            for child in node.children:
                if child.type == "text":
                    return child.text.decode("utf-8", errors="replace").strip()
    return ""


def _find_element_by_tag(root: Node, tag_name: str) -> Node | None:
    """Find the first element with the given tag name at the top level."""
    for child in root.children:
        if child.type == "element":
            tag = _get_tag_name(child)
            if tag == tag_name:
                return child
    return None


def _walk_elements(node: Node):
    """Recursively yield all element nodes."""
    for child in node.children:
        if child.type == "element":
            yield child
            yield from _walk_elements(child)
        else:
            yield from _walk_elements(child)


def _extract_script_chunks(root: Node, source: bytes, file_path: str,
                            file_stem: str, module: str, is_vue: bool) -> list[CodeChunk]:
    """Extract <script> tag contents and sub-chunk via the JS chunker."""
    from src.indexer.chunker_js import chunk_file_js

    chunks = []
    for node in root.children if is_vue else _walk_elements(root):
        if node.type == "element":
            tag = _get_tag_name(node)
        elif node.type == "script_element":
            tag = "script"
        else:
            continue

        if tag != "script":
            continue

        # Extract the raw text content of the script tag
        script_content = _get_element_text_content(node)
        if not script_content or len(script_content.strip()) < 10:
            continue

        # Sub-chunk the JavaScript content
        script_bytes = script_content.encode("utf-8")
        js_chunks = chunk_file_js(script_bytes, file_path, module)

        # Adjust line numbers: script content starts after the <script> tag line
        script_start_line = node.start_point[0] + 1  # 0-indexed to 1-indexed, +1 for tag line
        for chunk in js_chunks:
            chunk.start_line += script_start_line
            chunk.end_line += script_start_line

        chunks.extend(js_chunks)

    return chunks


def _get_element_text_content(node: Node) -> str:
    """Get the raw text content between start and end tags of an element."""
    # For script_element, the raw_text child contains the content
    if node.type == "script_element":
        for child in node.children:
            if child.type == "raw_text":
                return child.text.decode("utf-8", errors="replace")
        return ""

    # For regular elements, collect text between start_tag and end_tag
    parts = []
    in_content = False
    for child in node.children:
        if child.type == "start_tag":
            in_content = True
            continue
        if child.type == "end_tag":
            break
        if in_content:
            parts.append(child.text.decode("utf-8", errors="replace"))
    return "".join(parts)


def _split_at_sections(root: Node, source: bytes, file_path: str, class_name: str,
                        namespace: str, module: str) -> list[CodeChunk]:
    """Try to split a large HTML file at semantic elements (section, article, etc.) with id attributes."""
    sections = []
    for element in _walk_elements(root):
        tag = _get_tag_name(element)
        if tag in _SECTION_ELEMENTS:
            element_id = _get_attribute(element, "id")
            if element_id:
                sections.append((element, tag, element_id))

    if len(sections) < 2:
        return []  # Not enough sections to warrant splitting

    chunks = []
    for element, tag, element_id in sections:
        text = _node_text(element)
        line_count = text.count("\n") + 1
        if line_count < 3:
            continue

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=class_name,
            method_name=f"{tag}#{element_id}",
            namespace=namespace,
            start_line=element.start_point[0] + 1,
            end_line=element.end_point[0] + 1,
            source=text,
            chunk_type="template",
            module=module,
        ))

    return chunks


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")
