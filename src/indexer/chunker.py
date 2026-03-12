"""Tree-sitter AST-based C# code chunking."""

import logging
import re

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

# Regex fallback for when tree-sitter fails (e.g., #if/#endif with unbalanced braces)
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([\w.]+)", re.MULTILINE)
_TYPE_DECL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*"
    r"(?:class|struct|interface|record)\s+"
    r"(\w+)"                        # class name
    r"(?:<[^>]+>)?"                 # optional generic params
    r"\s*:\s*"                      # colon
    r"([\w.,\s<>]+?)"              # base types
    r"\s*(?:\{|where\b)",          # opening brace or where clause
    re.MULTILINE,
)

logger = logging.getLogger(__name__)

CS_LANGUAGE = Language(tscsharp.language())
_parser = Parser(CS_LANGUAGE)

# Top-level type declaration node types
TYPE_DECLARATIONS = {
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
}

# Member-level node types to extract as individual chunks
MEMBER_DECLARATIONS = {
    "method_declaration",
    "constructor_declaration",
    "property_declaration",
}


def chunk_file(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a C# file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk per type.
    Large files -> individual method chunks + one class_summary chunk per type.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _extract_namespace(root)
    total_lines = source.count(b"\n") + 1

    chunks = []
    for type_node in _find_type_declarations(root):
        class_name = _get_node_name(type_node)
        if not class_name:
            continue

        # Handle nested classes by prefixing outer class name
        outer = _find_enclosing_type(type_node)
        if outer:
            outer_name = _get_node_name(outer)
            if outer_name:
                class_name = f"{outer_name}.{class_name}"

        base_types = _extract_base_types(type_node)
        doc_comment = _extract_doc_comment(type_node, source)

        if type_node.type == "enum_declaration" or total_lines < SMALL_FILE_LINE_THRESHOLD:
            # Index as single whole_class chunk
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
            # Extract individual members + class summary
            chunks.append(_make_class_summary(type_node, source, file_path, class_name, namespace, module, doc_comment, base_types))
            for member_node in _find_members(type_node):
                member_chunk = _make_member_chunk(member_node, source, file_path, class_name, namespace, module, base_types)
                if member_chunk:
                    chunks.append(member_chunk)

    # If no type declarations found, try regex fallback for files with parse errors
    if not chunks and total_lines > 0:
        source_text = source.decode("utf-8", errors="replace")
        fallback_types = _regex_extract_types(source_text) if root.has_error else []

        if fallback_types:
            if not namespace:
                ns_match = _NAMESPACE_RE.search(source_text)
                namespace = ns_match.group(1) if ns_match else ""
            logger.debug("Regex fallback for %s: %d types extracted", file_path, len(fallback_types))

            for class_name, base_types in fallback_types:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    class_name=class_name,
                    method_name=None,
                    namespace=namespace,
                    start_line=1,
                    end_line=total_lines,
                    source=source_text,
                    chunk_type="whole_class",
                    module=module,
                    base_types=base_types,
                ))
        else:
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".cs", ""),
                method_name=None,
                namespace=namespace,
                start_line=1,
                end_line=total_lines,
                source=source_text,
                chunk_type="whole_class",
                module=module,
            ))

    return chunks


def _regex_extract_types(source_text: str) -> list[tuple[str, list[str]]]:
    """Regex fallback to extract type declarations when tree-sitter fails.

    Returns deduplicated list of (class_name, [base_types]) tuples.
    Keeps the match with the most base types for each class name.
    """
    seen: dict[str, list[str]] = {}
    for match in _TYPE_DECL_RE.finditer(source_text):
        class_name = match.group(1)
        base_str = match.group(2)
        base_types = [b.strip() for b in base_str.split(",") if b.strip()]
        if class_name not in seen or len(base_types) > len(seen[class_name]):
            seen[class_name] = base_types
    return list(seen.items())


def _extract_namespace(root: Node) -> str:
    """Extract namespace from file-scoped or block-scoped namespace declaration."""
    for child in root.children:
        if child.type == "file_scoped_namespace_declaration":
            name_node = child.child_by_field_name("name")
            return _node_text(name_node) if name_node else ""
        if child.type == "namespace_declaration":
            name_node = child.child_by_field_name("name")
            return _node_text(name_node) if name_node else ""
    return ""


def _find_type_declarations(node: Node, depth: int = 0) -> list[Node]:
    """Recursively find all type declarations (classes, structs, interfaces, enums)."""
    results = []
    for child in node.children:
        if child.type in TYPE_DECLARATIONS:
            results.append(child)
            # Also look for nested types
            if depth < 2:
                results.extend(_find_type_declarations(child, depth + 1))
        elif child.type in ("namespace_declaration", "file_scoped_namespace_declaration", "declaration_list"):
            results.extend(_find_type_declarations(child, depth))
    return results


def _find_enclosing_type(node: Node) -> Node | None:
    """Find the enclosing type declaration for a nested type."""
    parent = node.parent
    while parent:
        if parent.type in TYPE_DECLARATIONS:
            return parent
        parent = parent.parent
    return None


def _find_members(type_node: Node) -> list[Node]:
    """Find method/constructor/property declarations within a type."""
    members = []
    body = type_node.child_by_field_name("body")
    if not body:
        return members
    for child in body.children:
        if child.type in MEMBER_DECLARATIONS:
            members.append(child)
    return members


def _get_node_name(node: Node) -> str:
    """Extract the identifier name from a declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    return ""


def _extract_base_types(type_node: Node) -> list[str]:
    """Extract base class / interface names from the base_list."""
    base_types = []
    for child in type_node.children:
        if child.type == "base_list":
            for bc in child.children:
                if bc.type != "," and bc.type != ":":
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
        elif line.startswith("[") or line == "":
            # Skip attributes and blank lines
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
        line = line.strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _is_serialized_field(field_node: Node) -> bool:
    """Detect Unity Inspector-visible fields ([SerializeField] or public without [HideInInspector])."""
    text = _node_text(field_node)
    if "[SerializeField]" in text:
        return True
    if "[HideInInspector]" in text:
        return False
    for child in field_node.children:
        if child.type == "modifier" and child.text == b"public":
            return True
    return False


def _node_text(node: Node) -> str:
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _make_class_summary(type_node: Node, source: bytes, file_path: str, class_name: str,
                         namespace: str, module: str, doc_comment: str, base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for a class (signature + field list, no method bodies)."""
    summary_lines = []
    source_text = _node_text(type_node)
    # Take the class signature (everything up to the first {)
    brace_idx = source_text.find("{")
    if brace_idx > 0:
        summary_lines.append(source_text[:brace_idx].strip())
    else:
        summary_lines.append(source_text[:200])

    # Add field declarations
    body = type_node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "field_declaration":
                field_text = _node_text(child).strip()
                tag = "  // [serialized]" if _is_serialized_field(child) else ""
                summary_lines.append(f"  {field_text}{tag}")
            elif child.type in MEMBER_DECLARATIONS:
                # Add just the signature, not the body
                sig = _member_signature(child)
                if sig:
                    summary_lines.append("  " + sig)

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
    """Extract just the signature of a method/property/constructor (no body)."""
    text = _node_text(node)
    # Find the opening brace or => and truncate
    for marker in ("{", "=>"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx].strip() + ";"
    return text.split("\n")[0].strip()


def _make_member_chunk(member_node: Node, source: bytes, file_path: str, class_name: str,
                        namespace: str, module: str, base_types: list[str]) -> CodeChunk | None:
    """Create a chunk for an individual method/constructor/property."""
    member_text = _node_text(member_node)
    # Skip trivially small members (< 3 lines)
    line_count = member_text.count("\n") + 1
    if line_count < 3 and member_node.type == "property_declaration":
        return None

    if member_node.type == "constructor_declaration":
        method_name = _get_node_name(member_node) or class_name
        chunk_type = "constructor"
    elif member_node.type == "property_declaration":
        method_name = _get_node_name(member_node)
        chunk_type = "property"
    else:
        method_name = _get_node_name(member_node)
        chunk_type = "method"

    if not method_name:
        return None

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
