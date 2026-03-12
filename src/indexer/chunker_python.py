"""Tree-sitter AST-based Python code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def chunk_file_python(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a Python file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk per class.
    Large files -> individual method chunks + one class_summary chunk per class.
    Top-level functions -> function chunks with class_name = file stem.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _derive_module_path(file_path)
    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem

    chunks = []

    # Collect class definitions
    for class_node in _find_class_definitions(root):
        class_name = _get_node_name(class_node)
        if not class_name:
            continue

        base_types = _extract_base_classes(class_node)
        doc_comment = _extract_docstring(class_node)
        decorators = _extract_decorators(class_node)
        if decorators:
            doc_comment = (decorators + " " + doc_comment).strip() if doc_comment else decorators

        if total_lines < SMALL_FILE_LINE_THRESHOLD:
            chunk = CodeChunk(
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
            )
            chunks.append(chunk)
        else:
            # Class summary + individual methods
            chunks.append(_make_class_summary(
                class_node, file_path, class_name, namespace, module, doc_comment, base_types,
            ))
            for method_node in _find_methods(class_node):
                method_chunk = _make_method_chunk(
                    method_node, source, file_path, class_name, namespace, module, base_types,
                )
                if method_chunk:
                    chunks.append(method_chunk)

    # Collect top-level functions (not inside classes)
    for func_node in _find_top_level_functions(root):
        func_chunk = _make_function_chunk(
            func_node, source, file_path, file_stem, namespace, module,
        )
        if func_chunk:
            chunks.append(func_chunk)

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


def _derive_module_path(file_path: str) -> str:
    """Derive a dotted module path from file path (e.g. 'utils/helpers.py' -> 'utils.helpers')."""
    path = PurePosixPath(file_path)
    # Strip .py extension and convert slashes to dots
    parts = list(path.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Remove __init__ from the end (package init files)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _find_class_definitions(root: Node) -> list[Node]:
    """Find all class_definition nodes at the top level (including inside decorated_definition)."""
    results = []
    for child in root.children:
        if child.type == "class_definition":
            results.append(child)
        elif child.type == "decorated_definition":
            # Unwrap: the actual class/function is the last child
            inner = child.children[-1] if child.children else None
            if inner and inner.type == "class_definition":
                results.append(child)  # Include the decorated_definition node (has decorators)
    return results


def _find_methods(class_node: Node) -> list[Node]:
    """Find function_definition nodes within a class body."""
    # If this is a decorated_definition, get the inner class_definition
    actual_class = class_node
    if class_node.type == "decorated_definition":
        for child in class_node.children:
            if child.type == "class_definition":
                actual_class = child
                break

    body = actual_class.child_by_field_name("body")
    if not body:
        return []

    methods = []
    for child in body.children:
        if child.type == "function_definition":
            methods.append(child)
        elif child.type == "decorated_definition":
            inner = child.children[-1] if child.children else None
            if inner and inner.type == "function_definition":
                methods.append(child)  # Keep decorators attached
    return methods


def _find_top_level_functions(root: Node) -> list[Node]:
    """Find function_definition nodes at module level (not inside classes)."""
    results = []
    for child in root.children:
        if child.type == "function_definition":
            results.append(child)
        elif child.type == "decorated_definition":
            inner = child.children[-1] if child.children else None
            if inner and inner.type == "function_definition":
                results.append(child)
    return results


def _get_node_name(node: Node) -> str:
    """Extract the identifier name from a class/function definition.

    Handles both plain definitions and decorated_definition wrappers.
    """
    actual = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("class_definition", "function_definition"):
                actual = child
                break

    name_node = actual.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    return ""


def _extract_base_classes(class_node: Node) -> list[str]:
    """Extract base class names from the argument_list (superclasses) of a class definition."""
    actual = class_node
    if class_node.type == "decorated_definition":
        for child in class_node.children:
            if child.type == "class_definition":
                actual = child
                break

    superclasses = actual.child_by_field_name("superclasses")
    if not superclasses:
        return []

    bases = []
    for child in superclasses.children:
        if child.type in ("identifier", "attribute", "subscript"):
            text = child.text.decode("utf-8", errors="replace").strip()
            if text:
                bases.append(text)
    return bases


def _extract_docstring(node: Node) -> str:
    """Extract the docstring (first expression_statement containing a string) from a class/function body."""
    actual = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("class_definition", "function_definition"):
                actual = child
                break

    body = actual.child_by_field_name("body")
    if not body or not body.children:
        return ""

    first_stmt = body.children[0]
    if first_stmt.type == "expression_statement":
        expr = first_stmt.children[0] if first_stmt.children else None
        if expr and expr.type == "string":
            text = expr.text.decode("utf-8", errors="replace")
            # Strip triple quotes
            for quote in ('"""', "'''"):
                if text.startswith(quote) and text.endswith(quote):
                    text = text[3:-3]
                    break
            return text.strip()
    return ""


def _extract_decorators(node: Node) -> str:
    """Extract decorator text from a decorated_definition node."""
    if node.type != "decorated_definition":
        return ""

    decorators = []
    for child in node.children:
        if child.type == "decorator":
            decorators.append(child.text.decode("utf-8", errors="replace").strip())
    return " ".join(decorators)


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _make_class_summary(class_node: Node, file_path: str, class_name: str,
                         namespace: str, module: str, doc_comment: str, base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for a class (signature + method signatures, no bodies)."""
    actual = class_node
    if class_node.type == "decorated_definition":
        for child in class_node.children:
            if child.type == "class_definition":
                actual = child
                break

    summary_lines = []
    source_text = _node_text(actual)
    # Take the class signature (everything up to the first colon line)
    colon_idx = source_text.find(":")
    if colon_idx > 0:
        summary_lines.append(source_text[:colon_idx + 1].strip())
    else:
        summary_lines.append(source_text[:200])

    # Add method signatures
    body = actual.child_by_field_name("body")
    if body:
        for child in body.children:
            func = None
            if child.type == "function_definition":
                func = child
            elif child.type == "decorated_definition":
                inner = child.children[-1] if child.children else None
                if inner and inner.type == "function_definition":
                    func = inner

            if func:
                sig = _function_signature(func)
                if sig:
                    summary_lines.append("    " + sig)

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


def _function_signature(func_node: Node) -> str:
    """Extract just the signature of a function (def line only)."""
    text = _node_text(func_node)
    # Take up to the first colon that ends the def line
    first_line = text.split("\n")[0].strip()
    return first_line


def _make_method_chunk(method_node: Node, source: bytes, file_path: str, class_name: str,
                        namespace: str, module: str, base_types: list[str]) -> CodeChunk | None:
    """Create a chunk for an individual class method."""
    method_text = _node_text(method_node)
    line_count = method_text.count("\n") + 1
    if line_count < 3:
        return None

    method_name = _get_node_name(method_node)
    if not method_name:
        return None

    # Determine chunk type
    if method_name in ("__init__", "__new__"):
        chunk_type = "constructor"
    elif method_name.startswith("__") and method_name.endswith("__"):
        chunk_type = "method"  # dunder methods
    else:
        chunk_type = "method"

    doc_comment = _extract_docstring(method_node)
    decorators = _extract_decorators(method_node)
    if decorators:
        doc_comment = (decorators + " " + doc_comment).strip() if doc_comment else decorators

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
                          file_stem: str, namespace: str, module: str) -> CodeChunk | None:
    """Create a chunk for a top-level function (not in a class)."""
    func_text = _node_text(func_node)
    line_count = func_text.count("\n") + 1
    if line_count < 3:
        return None

    func_name = _get_node_name(func_node)
    if not func_name:
        return None

    doc_comment = _extract_docstring(func_node)
    decorators = _extract_decorators(func_node)
    if decorators:
        doc_comment = (decorators + " " + doc_comment).strip() if doc_comment else decorators

    return CodeChunk(
        file_path=file_path,
        class_name=file_stem,
        method_name=func_name,
        namespace=namespace,
        start_line=func_node.start_point[0] + 1,
        end_line=func_node.end_point[0] + 1,
        source=func_text,
        chunk_type="function",
        module=module,
        doc_comment=doc_comment,
    )
