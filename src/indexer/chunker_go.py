"""Tree-sitter AST-based Go code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

GO_LANGUAGE = Language(tsgo.language())
_parser = Parser(GO_LANGUAGE)


def chunk_file_go(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a Go file and return code chunks.

    Go has no classes. Types (struct/interface/alias) live at package level;
    methods are top-level func declarations with a receiver. We model each
    named type as a "class" and attach its methods by receiver name.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> whole_class per type,
    methods as individual method chunks grouped by receiver.
    Large files -> class_summary per type (signature + method sigs) +
    individual method chunks.
    Top-level functions -> function chunks with class_name = file stem.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _extract_package_name(root) or _namespace_from_path(file_path)
    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem

    # First pass: collect all type declarations and group methods by receiver
    type_specs: dict[str, Node] = {}  # name -> type_spec node
    type_kinds: dict[str, str] = {}  # name -> "struct" / "interface" / "alias"
    methods_by_receiver: dict[str, list[Node]] = {}
    top_functions: list[Node] = []

    for child in root.children:
        if child.type == "type_declaration":
            for spec in child.children:
                if spec.type == "type_spec":
                    name = _type_spec_name(spec)
                    if name:
                        type_specs[name] = spec
                        type_kinds[name] = _type_spec_kind(spec)
        elif child.type == "method_declaration":
            receiver = _method_receiver_type(child)
            if receiver:
                methods_by_receiver.setdefault(receiver, []).append(child)
        elif child.type == "function_declaration":
            top_functions.append(child)

    chunks: list[CodeChunk] = []

    # --- Type declarations (struct, interface, alias) ---
    for type_name, spec in type_specs.items():
        doc_comment = _extract_doc_comment(spec.parent, source)
        methods = methods_by_receiver.get(type_name, [])

        if total_lines < SMALL_FILE_LINE_THRESHOLD:
            # Whole_class for the type itself
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=type_name,
                method_name=None,
                namespace=namespace,
                start_line=spec.parent.start_point[0] + 1,
                end_line=spec.parent.end_point[0] + 1,
                source=_node_text(spec.parent),
                chunk_type="whole_class",
                module=module,
                doc_comment=doc_comment,
            ))
        else:
            # Summary with method signatures
            chunks.append(_make_type_summary(
                spec, methods, file_path, type_name, namespace, module, doc_comment,
            ))

        # Methods attached to this receiver
        for method_node in methods:
            mc = _make_method_chunk(
                method_node, source, file_path, type_name, namespace, module,
            )
            if mc:
                chunks.append(mc)

    # --- Methods with receivers that don't match any local type (cross-file) ---
    for receiver, methods in methods_by_receiver.items():
        if receiver in type_specs:
            continue  # already handled above
        for method_node in methods:
            mc = _make_method_chunk(
                method_node, source, file_path, receiver, namespace, module,
            )
            if mc:
                chunks.append(mc)

    # --- Top-level functions ---
    for func_node in top_functions:
        fc = _make_function_chunk(
            func_node, source, file_path, file_stem, namespace, module,
        )
        if fc:
            chunks.append(fc)

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
# Helpers
# ---------------------------------------------------------------------------

def _extract_package_name(root: Node) -> str:
    """Extract package name from the package_clause."""
    for child in root.children:
        if child.type == "package_clause":
            for c in child.children:
                if c.type == "package_identifier":
                    return c.text.decode("utf-8", errors="replace")
    return ""


def _namespace_from_path(file_path: str) -> str:
    """Fallback namespace derived from directory path."""
    p = PurePosixPath(file_path)
    parts = [pt for pt in p.parent.parts if pt not in (".", "")]
    return "/".join(parts)


def _type_spec_name(spec: Node) -> str:
    """Extract the identifier name from a type_spec node."""
    name_node = spec.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    for child in spec.children:
        if child.type == "type_identifier":
            return child.text.decode("utf-8", errors="replace")
    return ""


def _type_spec_kind(spec: Node) -> str:
    """Classify a type_spec as struct/interface/alias."""
    for child in spec.children:
        if child.type == "struct_type":
            return "struct"
        if child.type == "interface_type":
            return "interface"
    return "alias"


def _method_receiver_type(method_node: Node) -> str:
    """Extract the receiver type name from a method_declaration.

    method_declaration layout:
        func (r *Foo) Name(...) { ... }
    The first parameter_list is the receiver; the receiver's type is either
    a type_identifier or pointer_type -> type_identifier.
    """
    for child in method_node.children:
        if child.type == "parameter_list":
            for pd in child.children:
                if pd.type == "parameter_declaration":
                    return _extract_type_name_from_node(pd)
            return ""
    return ""


def _extract_type_name_from_node(node: Node) -> str:
    """Walk a parameter_declaration / pointer_type / generic_type and return the base type name."""
    for child in node.children:
        if child.type == "type_identifier":
            return child.text.decode("utf-8", errors="replace")
        if child.type == "pointer_type":
            inner = _extract_type_name_from_node(child)
            if inner:
                return inner
        if child.type == "generic_type":
            type_id = child.child_by_field_name("type")
            if type_id:
                return type_id.text.decode("utf-8", errors="replace")
            # fallback
            for c in child.children:
                if c.type == "type_identifier":
                    return c.text.decode("utf-8", errors="replace")
    return ""


def _method_name(method_node: Node) -> str:
    """Extract the method name from a method_declaration.

    The method name is the field_identifier that follows the receiver
    parameter_list (tree-sitter-go exposes it as an unnamed child of type
    field_identifier).
    """
    name_node = method_node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    # Fallback: first field_identifier child
    for child in method_node.children:
        if child.type == "field_identifier":
            return child.text.decode("utf-8", errors="replace")
    return ""


def _func_name(func_node: Node) -> str:
    """Extract the function name from a function_declaration."""
    name_node = func_node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    for child in func_node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return ""


def _fn_signature(node: Node) -> str:
    """Signature of a function/method declaration (everything before the body)."""
    text = _node_text(node)
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip()
    return text.split("\n")[0].strip()


def _extract_doc_comment(node: Node, source: bytes) -> str:
    """Extract // doc comment lines immediately preceding a node.

    Go convention: the doc comment is a contiguous block of // lines directly
    above the declaration (no blank line between).
    """
    start_line = node.start_point[0]
    source_lines = source.decode("utf-8", errors="replace").split("\n")
    lines = []
    idx = start_line - 1
    while idx >= 0:
        line = source_lines[idx].rstrip()
        stripped = line.lstrip()
        if stripped.startswith("//"):
            lines.insert(0, stripped)
            idx -= 1
        else:
            break

    cleaned = []
    for line in lines:
        line = line.lstrip("/").strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def _node_text(node: Node) -> str:
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _make_type_summary(spec: Node, methods: list[Node], file_path: str,
                        class_name: str, namespace: str, module: str,
                        doc_comment: str) -> CodeChunk:
    """class_summary for a Go type: the type declaration text + method signatures."""
    decl = spec.parent  # type_declaration
    decl_text = _node_text(decl)
    summary_lines = [decl_text]
    for m in methods:
        sig = _fn_signature(m)
        if sig:
            summary_lines.append("  " + sig)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=decl.start_point[0] + 1,
        end_line=decl.end_point[0] + 1,
        source="\n".join(summary_lines),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
    )


def _make_method_chunk(method_node: Node, source: bytes, file_path: str,
                        class_name: str, namespace: str, module: str) -> CodeChunk | None:
    """Chunk for a method_declaration attached to a receiver type."""
    method_text = _node_text(method_node)
    line_count = method_text.count("\n") + 1
    if line_count < 3:
        return None

    name = _method_name(method_node)
    if not name:
        return None

    doc_comment = _extract_doc_comment(method_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=name,
        namespace=namespace,
        start_line=method_node.start_point[0] + 1,
        end_line=method_node.end_point[0] + 1,
        source=method_text,
        chunk_type="method",
        module=module,
        doc_comment=doc_comment,
    )


def _make_function_chunk(func_node: Node, source: bytes, file_path: str,
                          file_stem: str, namespace: str, module: str) -> CodeChunk | None:
    """Chunk for a top-level function_declaration."""
    func_text = _node_text(func_node)
    line_count = func_text.count("\n") + 1
    if line_count < 3:
        return None

    name = _func_name(func_node)
    if not name:
        return None

    doc_comment = _extract_doc_comment(func_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=file_stem,
        method_name=name,
        namespace=namespace,
        start_line=func_node.start_point[0] + 1,
        end_line=func_node.end_point[0] + 1,
        source=func_text,
        chunk_type="function",
        module=module,
        doc_comment=doc_comment,
    )
