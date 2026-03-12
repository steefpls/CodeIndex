"""Tree-sitter AST-based TypeScript code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
_parser_ts = Parser(TS_LANGUAGE)
_parser_tsx = Parser(TSX_LANGUAGE)

# Average line length above which we skip the file (likely minified/bundled)
_MINIFIED_AVG_LINE_LENGTH = 200


def chunk_file_ts(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a TypeScript/TSX file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk per class.
    Large files -> individual method chunks + one class_summary chunk per class.
    Interfaces -> always whole_class.
    Enums -> always whole_class.
    Type aliases (>= 3 lines) -> function chunks.
    Top-level functions and arrow functions -> function chunks.
    PascalCase functions containing JSX -> component chunks.
    Skips minified files (avg line length > 200).
    """
    # Skip minified/bundled files
    lines_raw = source.split(b"\n")
    if len(lines_raw) > 0:
        avg_len = len(source) / len(lines_raw)
        if avg_len > _MINIFIED_AVG_LINE_LENGTH:
            logger.debug("Skipping likely minified file: %s (avg line len %.0f)", file_path, avg_len)
            return []

    # Select parser based on file extension
    if file_path.endswith(".tsx"):
        parser = _parser_tsx
    else:
        parser = _parser_ts

    tree = parser.parse(source)
    root = tree.root_node

    total_lines = len(lines_raw)
    file_stem = PurePosixPath(file_path).stem
    namespace = PurePosixPath(file_path).parent.as_posix()
    if namespace == ".":
        namespace = ""

    chunks: list[CodeChunk] = []
    seen_nodes: set[int] = set()  # Track node IDs to avoid double-counting from exports

    # --- Class declarations (including abstract) ---
    for class_node in _find_class_declarations(root):
        if id(class_node) in seen_nodes:
            continue
        seen_nodes.add(id(class_node))

        class_name = _get_node_name(class_node)
        if not class_name:
            continue

        base_types = _extract_base_types(class_node)
        doc_comment = _extract_tsdoc(class_node, source)

        if total_lines < SMALL_FILE_LINE_THRESHOLD:
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=class_name,
                method_name=None,
                namespace=namespace,
                start_line=class_node.start_point[0] + 1,
                end_line=class_node.end_point[0] + 1,
                source=_node_text(class_node),
                chunk_type="whole_class",
                module=module,
                doc_comment=doc_comment,
                base_types=base_types,
            ))
        else:
            chunks.append(_make_class_summary(
                class_node, file_path, class_name, namespace, module, doc_comment, base_types,
            ))
            for method_node in _find_class_methods(class_node):
                mc = _make_method_chunk(
                    method_node, source, file_path, class_name, namespace, module, base_types,
                )
                if mc:
                    chunks.append(mc)

    # --- Interface declarations (always whole_class) ---
    for iface_node in _find_interface_declarations(root):
        if id(iface_node) in seen_nodes:
            continue
        seen_nodes.add(id(iface_node))

        iface_name = _get_node_name(iface_node)
        if not iface_name:
            continue

        base_types = _extract_extends_types(iface_node)
        doc_comment = _extract_tsdoc(iface_node, source)

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=iface_name,
            method_name=None,
            namespace=namespace,
            start_line=iface_node.start_point[0] + 1,
            end_line=iface_node.end_point[0] + 1,
            source=_node_text(iface_node),
            chunk_type="whole_class",
            module=module,
            doc_comment=doc_comment,
            base_types=base_types,
        ))

    # --- Enum declarations (always whole_class) ---
    for enum_node in _find_enum_declarations(root):
        if id(enum_node) in seen_nodes:
            continue
        seen_nodes.add(id(enum_node))

        enum_name = _get_node_name(enum_node)
        if not enum_name:
            continue

        doc_comment = _extract_tsdoc(enum_node, source)

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=enum_name,
            method_name=None,
            namespace=namespace,
            start_line=enum_node.start_point[0] + 1,
            end_line=enum_node.end_point[0] + 1,
            source=_node_text(enum_node),
            chunk_type="whole_class",
            module=module,
            doc_comment=doc_comment,
        ))

    # --- Type alias declarations (>= 3 lines -> function chunk) ---
    for type_node in _find_type_alias_declarations(root):
        if id(type_node) in seen_nodes:
            continue
        seen_nodes.add(id(type_node))

        type_text = _node_text(type_node)
        line_count = type_text.count("\n") + 1
        if line_count < 3:
            continue

        type_name = _get_node_name(type_node)
        if not type_name:
            continue

        doc_comment = _extract_tsdoc(type_node, source)

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=type_name,
            namespace=namespace,
            start_line=type_node.start_point[0] + 1,
            end_line=type_node.end_point[0] + 1,
            source=type_text,
            chunk_type="function",
            module=module,
            doc_comment=doc_comment,
        ))

    # --- Top-level functions (function declarations + arrow functions) ---
    for func_node, func_name in _find_top_level_functions(root):
        if id(func_node) in seen_nodes:
            continue
        seen_nodes.add(id(func_node))

        func_chunk = _make_function_chunk(
            func_node, source, file_path, file_stem, func_name, namespace, module,
        )
        if func_chunk:
            chunks.append(func_chunk)

    # --- Fallback: whole file ---
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
# Node finders
# ---------------------------------------------------------------------------

def _find_class_declarations(root: Node) -> list[Node]:
    """Find class_declaration and abstract_class_declaration, unwrapping exports."""
    results = []
    for child in root.children:
        if child.type in ("class_declaration", "abstract_class_declaration"):
            results.append(child)
        elif child.type == "export_statement":
            for inner in child.children:
                if inner.type in ("class_declaration", "abstract_class_declaration"):
                    results.append(inner)
    return results


def _find_interface_declarations(root: Node) -> list[Node]:
    """Find interface_declaration nodes, unwrapping exports."""
    results = []
    for child in root.children:
        if child.type == "interface_declaration":
            results.append(child)
        elif child.type == "export_statement":
            for inner in child.children:
                if inner.type == "interface_declaration":
                    results.append(inner)
    return results


def _find_enum_declarations(root: Node) -> list[Node]:
    """Find enum_declaration nodes, unwrapping exports."""
    results = []
    for child in root.children:
        if child.type == "enum_declaration":
            results.append(child)
        elif child.type == "export_statement":
            for inner in child.children:
                if inner.type == "enum_declaration":
                    results.append(inner)
    return results


def _find_type_alias_declarations(root: Node) -> list[Node]:
    """Find type_alias_declaration nodes, unwrapping exports."""
    results = []
    for child in root.children:
        if child.type == "type_alias_declaration":
            results.append(child)
        elif child.type == "export_statement":
            for inner in child.children:
                if inner.type == "type_alias_declaration":
                    results.append(inner)
    return results


def _find_class_methods(class_node: Node) -> list[Node]:
    """Find method_definition and public_field_definition with arrow functions inside a class body."""
    body = class_node.child_by_field_name("body")
    if not body:
        return []

    methods = []
    for child in body.children:
        if child.type == "method_definition":
            methods.append(child)
    return methods


def _find_top_level_functions(root: Node) -> list[tuple[Node, str]]:
    """Find top-level function declarations and const/let arrow function assignments.

    Returns (node, function_name) tuples. Unwraps export_statement wrappers.
    """
    results = []

    def _process_node(node: Node):
        if node.type == "function_declaration":
            name = _get_node_name(node)
            if name:
                results.append((node, name))
        elif node.type in ("lexical_declaration", "variable_declaration"):
            for declarator in node.children:
                if declarator.type == "variable_declarator":
                    name_node = declarator.child_by_field_name("name")
                    value_node = declarator.child_by_field_name("value")
                    if name_node and value_node and value_node.type in ("arrow_function", "function_expression"):
                        name = name_node.text.decode("utf-8", errors="replace")
                        results.append((node, name))

    for child in root.children:
        if child.type == "export_statement":
            for inner in child.children:
                _process_node(inner)
        else:
            _process_node(child)

    return results


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _get_node_name(node: Node) -> str:
    """Extract the identifier name from a declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    return ""


def _extract_base_types(class_node: Node) -> list[str]:
    """Extract extends and implements types from a class declaration."""
    base_types = []
    for child in class_node.children:
        if child.type == "class_heritage":
            for inner in child.children:
                if inner.type == "extends_clause":
                    for type_node in inner.children:
                        if type_node.type not in ("extends", ","):
                            text = type_node.text.decode("utf-8", errors="replace").strip()
                            if text:
                                base_types.append(text)
                elif inner.type == "implements_clause":
                    for type_node in inner.children:
                        if type_node.type not in ("implements", ","):
                            text = type_node.text.decode("utf-8", errors="replace").strip()
                            if text:
                                base_types.append(text)
    return base_types


def _extract_extends_types(iface_node: Node) -> list[str]:
    """Extract extends types from an interface declaration."""
    base_types = []
    for child in iface_node.children:
        if child.type == "extends_type_clause":
            for inner in child.children:
                if inner.type not in ("extends", ","):
                    text = inner.text.decode("utf-8", errors="replace").strip()
                    if text:
                        base_types.append(text)
    return base_types


def _extract_tsdoc(node: Node, source: bytes) -> str:
    """Extract TSDoc comment (/** ... */) preceding a node."""
    start_line = node.start_point[0]
    source_lines = source.decode("utf-8", errors="replace").split("\n")

    # Look backwards for a comment block ending with */
    idx = start_line - 1
    while idx >= 0 and source_lines[idx].strip() == "":
        idx -= 1

    if idx < 0:
        return ""

    line = source_lines[idx].strip()
    if not line.endswith("*/"):
        return ""

    # Collect the full TSDoc block
    comment_lines = []
    while idx >= 0:
        line = source_lines[idx].strip()
        comment_lines.insert(0, line)
        if line.startswith("/**"):
            break
        idx -= 1

    if not comment_lines or not comment_lines[0].strip().startswith("/**"):
        return ""

    # Clean up TSDoc markers
    cleaned = []
    for line in comment_lines:
        line = line.strip()
        if line.startswith("/**"):
            line = line[3:]
        if line.endswith("*/"):
            line = line[:-2]
        line = line.lstrip("* ").strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _is_react_component(func_node: Node, func_name: str) -> bool:
    """Check if a function is likely a React/TSX component (PascalCase + contains JSX)."""
    if not func_name or not func_name[0].isupper():
        return False

    text = _node_text(func_node)
    return "<" in text and ("/>" in text or "</" in text)


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def _make_class_summary(class_node: Node, file_path: str, class_name: str,
                         namespace: str, module: str, doc_comment: str,
                         base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for a class (signature + method signatures)."""
    summary_lines = []
    source_text = _node_text(class_node)
    brace_idx = source_text.find("{")
    if brace_idx > 0:
        summary_lines.append(source_text[:brace_idx].strip())
    else:
        summary_lines.append(source_text[:200])

    body = class_node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "method_definition":
                sig = _method_signature(child)
                if sig:
                    summary_lines.append("  " + sig)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=class_node.start_point[0] + 1,
        end_line=class_node.end_point[0] + 1,
        source="\n".join(summary_lines),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _method_signature(node: Node) -> str:
    """Extract just the signature of a method (no body)."""
    text = _node_text(node)
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip()
    return text.split("\n")[0].strip()


def _make_method_chunk(method_node: Node, source: bytes, file_path: str,
                        class_name: str, namespace: str, module: str,
                        base_types: list[str]) -> CodeChunk | None:
    """Create a chunk for an individual class method."""
    method_text = _node_text(method_node)
    line_count = method_text.count("\n") + 1
    if line_count < 3:
        return None

    method_name = _get_node_name(method_node)
    if not method_name:
        return None

    chunk_type = "constructor" if method_name == "constructor" else "method"
    doc_comment = _extract_tsdoc(method_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=method_name,
        namespace=namespace,
        start_line=method_node.start_point[0] + 1,
        end_line=method_node.end_point[0] + 1,
        source=method_text,
        chunk_type=chunk_type,
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _make_function_chunk(func_node: Node, source: bytes, file_path: str,
                          file_stem: str, func_name: str,
                          namespace: str, module: str) -> CodeChunk | None:
    """Create a chunk for a top-level function or arrow function assignment."""
    func_text = _node_text(func_node)
    line_count = func_text.count("\n") + 1
    if line_count < 3:
        return None

    doc_comment = _extract_tsdoc(func_node, source)

    # Detect React/TSX components (PascalCase + JSX)
    if _is_react_component(func_node, func_name):
        chunk_type = "component"
    else:
        chunk_type = "function"

    return CodeChunk(
        file_path=file_path,
        class_name=file_stem,
        method_name=func_name,
        namespace=namespace,
        start_line=func_node.start_point[0] + 1,
        end_line=func_node.end_point[0] + 1,
        source=func_text,
        chunk_type=chunk_type,
        module=module,
        doc_comment=doc_comment,
    )
