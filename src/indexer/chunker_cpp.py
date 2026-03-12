"""Tree-sitter AST-based C++ code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

CPP_LANGUAGE = Language(tscpp.language())
_parser = Parser(CPP_LANGUAGE)

# Top-level type declaration node types
TYPE_DECLARATIONS = {
    "class_specifier",
    "struct_specifier",
    "enum_specifier",
}

# Member-level node types to extract as individual chunks
MEMBER_DECLARATIONS = {
    "function_definition",
}


def chunk_file_cpp(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a C++ file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk per type.
    Large files -> individual method chunks + one class_summary chunk per type.
    Free functions (outside any class) -> method chunks with class_name = filename stem.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _extract_namespace(root)
    total_lines = source.count(b"\n") + 1

    chunks = []
    # Collect type declarations
    for type_node in _find_type_declarations(root):
        class_name = _get_node_name(type_node)
        if not class_name:
            continue

        base_types = _extract_base_types(type_node)
        doc_comment = _extract_doc_comment(type_node, source)

        if type_node.type == "enum_specifier" or total_lines < SMALL_FILE_LINE_THRESHOLD:
            chunk = CodeChunk(
                file_path=file_path,
                class_name=class_name,
                method_name=None,
                namespace=namespace,
                start_line=type_node.start_point[0] + 1,
                end_line=type_node.end_point[0] + 1,
                source=_node_text(type_node),
                chunk_type="whole_class",
                module=module,
                doc_comment=doc_comment,
                base_types=base_types,
            )
            chunks.append(chunk)
        else:
            chunks.append(_make_class_summary(
                type_node, source, file_path, class_name,
                namespace, module, doc_comment, base_types,
            ))
            for member_node in _find_members(type_node):
                member_chunk = _make_member_chunk(
                    member_node, source, file_path, class_name,
                    namespace, module, base_types,
                )
                if member_chunk:
                    chunks.append(member_chunk)

    # Collect free functions (not inside any class/struct)
    free_funcs = _find_free_functions(root)
    if free_funcs:
        file_stem = PurePosixPath(file_path).stem
        for func_node in free_funcs:
            chunk = _make_free_function_chunk(
                func_node, source, file_path, file_stem, namespace, module,
            )
            if chunk:
                chunks.append(chunk)

    # Fallback: if nothing was found, index the whole file
    if not chunks and total_lines > 0:
        file_stem = PurePosixPath(file_path).stem
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


def _extract_namespace(root: Node) -> str:
    """Extract namespace from the first namespace_definition found."""
    for child in root.children:
        if child.type == "namespace_definition":
            name_node = child.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
            # Check for nested namespace inside body
            body = child.child_by_field_name("body")
            if body:
                for inner in body.children:
                    if inner.type == "namespace_definition":
                        inner_name = inner.child_by_field_name("name")
                        if inner_name:
                            return inner_name.text.decode("utf-8", errors="replace")
    return ""


def _find_type_declarations(node: Node, depth: int = 0) -> list[Node]:
    """Recursively find class/struct/enum specifiers."""
    results = []
    for child in node.children:
        if child.type in TYPE_DECLARATIONS:
            # Only include named types with a body (skip forward declarations)
            if _get_node_name(child) and child.child_by_field_name("body"):
                results.append(child)
                if depth < 2:
                    results.extend(_find_type_declarations(child, depth + 1))
        elif child.type in ("namespace_definition", "declaration_list",
                            "translation_unit", "linkage_specification",
                            "field_declaration_list", "field_declaration"):
            body = child.child_by_field_name("body")
            target = body if body else child
            results.extend(_find_type_declarations(target, depth))
    return results


def _find_members(type_node: Node) -> list[Node]:
    """Find function_definition nodes within a class/struct body (field_declaration_list)."""
    members = []
    body = type_node.child_by_field_name("body")
    if not body:
        return members
    for child in body.children:
        if child.type == "function_definition":
            members.append(child)
        elif child.type == "declaration" and _has_function_body(child):
            members.append(child)
        # Handle access specifiers (public:, private:, etc.)
        elif child.type == "access_specifier":
            continue
    return members


def _has_function_body(node: Node) -> bool:
    """Check if a declaration node contains a function body (compound_statement)."""
    for child in node.children:
        if child.type == "compound_statement":
            return True
    return False


def _find_free_functions(root: Node) -> list[Node]:
    """Find function_definition nodes at file scope (outside classes)."""
    results = []

    def _walk(node: Node):
        for child in node.children:
            if child.type == "function_definition" and not _is_inside_type(child):
                results.append(child)
            elif child.type == "namespace_definition":
                body = child.child_by_field_name("body")
                if body:
                    _walk(body)

    def _is_inside_type(node: Node) -> bool:
        parent = node.parent
        while parent:
            if parent.type in ("field_declaration_list",):
                return True
            parent = parent.parent
        return False

    _walk(root)
    return results


def _get_node_name(node: Node) -> str:
    """Extract the identifier name from a declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    return ""


def _get_function_name(node: Node) -> str:
    """Extract function name from a function_definition, handling qualified names."""
    declarator = node.child_by_field_name("declarator")
    if not declarator:
        return ""

    # For function_declarator, the 'declarator' field holds the name
    name_node = declarator.child_by_field_name("declarator")
    if name_node:
        text = name_node.text.decode("utf-8", errors="replace")
        # For qualified names like ClassName::MethodName, extract just the method name
        if "::" in text:
            return text.split("::")[-1]
        return text

    return declarator.text.decode("utf-8", errors="replace").split("(")[0].strip()


def _extract_base_types(type_node: Node) -> list[str]:
    """Extract base class names from base_class_clause."""
    base_types = []
    for child in type_node.children:
        if child.type == "base_class_clause":
            for bc in child.children:
                if bc.type == "type_identifier" or bc.type == "qualified_identifier":
                    text = bc.text.decode("utf-8", errors="replace").strip()
                    if text:
                        base_types.append(text)
    return base_types


def _extract_doc_comment(node: Node, source: bytes) -> str:
    """Extract XML doc comment (/// lines) preceding a node."""
    start_line = node.start_point[0]
    source_lines = source.decode("utf-8", errors="replace").split("\n")
    lines = []
    idx = start_line - 1
    while idx >= 0:
        line = source_lines[idx].strip()
        if line.startswith("///"):
            lines.insert(0, line)
            idx -= 1
        elif line.startswith("//") or line == "" or line.startswith("#"):
            # Skip regular comments, blank lines, preprocessor directives
            idx -= 1
        else:
            break

    # Clean up the doc comment
    cleaned = []
    for line in lines:
        line = line.lstrip("/").strip()
        # Strip XML tags for embedding (keep the text content)
        line = line.replace("<summary>", "").replace("</summary>", "")
        line = line.replace("<param ", "").replace("</param>", "")
        line = line.replace("<returns>", "").replace("</returns>", "")
        line = line.replace("<remarks>", "").replace("</remarks>", "")
        line = line.replace("<c>", "").replace("</c>", "")
        line = line.strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _make_class_summary(type_node: Node, source: bytes, file_path: str, class_name: str,
                         namespace: str, module: str, doc_comment: str, base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for a class (signature + member signatures, no bodies)."""
    summary_lines = []
    source_text = _node_text(type_node)
    # Take the class signature (everything up to the first {)
    brace_idx = source_text.find("{")
    if brace_idx > 0:
        summary_lines.append(source_text[:brace_idx].strip())
    else:
        summary_lines.append(source_text[:200])

    # Add member signatures from body
    body = type_node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                sig = _member_signature(child)
                if sig:
                    summary_lines.append("  " + sig)
            elif child.type in ("field_declaration", "declaration"):
                summary_lines.append("  " + _node_text(child).strip())

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=type_node.start_point[0] + 1,
        end_line=type_node.end_point[0] + 1,
        source="\n".join(summary_lines),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _member_signature(node: Node) -> str:
    """Extract just the signature of a function (no body)."""
    text = _node_text(node)
    # Find the opening brace and truncate
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip() + ";"
    return text.split("\n")[0].strip()


def _make_member_chunk(member_node: Node, source: bytes, file_path: str, class_name: str,
                        namespace: str, module: str, base_types: list[str]) -> CodeChunk | None:
    """Create a chunk for an individual method/constructor/destructor."""
    member_text = _node_text(member_node)
    line_count = member_text.count("\n") + 1
    if line_count < 3:
        return None

    method_name = _get_function_name(member_node)
    if not method_name:
        return None

    # Determine chunk type
    if method_name == class_name:
        chunk_type = "constructor"
    elif method_name.startswith("~"):
        chunk_type = "method"  # destructors as methods
    else:
        chunk_type = "method"

    doc_comment = _extract_doc_comment(member_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=method_name,
        namespace=namespace,
        start_line=member_node.start_point[0] + 1,
        end_line=member_node.end_point[0] + 1,
        source=member_text,
        chunk_type=chunk_type,
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _make_free_function_chunk(func_node: Node, source: bytes, file_path: str,
                               file_stem: str, namespace: str, module: str) -> CodeChunk | None:
    """Create a chunk for a free function (not in a class)."""
    func_text = _node_text(func_node)
    line_count = func_text.count("\n") + 1
    if line_count < 3:
        return None

    method_name = _get_function_name(func_node)
    if not method_name:
        return None

    # For qualified names (ClassName::Method), use the class part as class_name
    declarator = func_node.child_by_field_name("declarator")
    class_name = file_stem
    if declarator:
        name_node = declarator.child_by_field_name("declarator")
        if name_node:
            full_name = name_node.text.decode("utf-8", errors="replace")
            if "::" in full_name:
                class_name = full_name.rsplit("::", 1)[0]

    doc_comment = _extract_doc_comment(func_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=method_name,
        namespace=namespace,
        start_line=func_node.start_point[0] + 1,
        end_line=func_node.end_point[0] + 1,
        source=func_text,
        chunk_type="method",
        module=module,
        doc_comment=doc_comment,
    )
