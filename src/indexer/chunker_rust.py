"""Tree-sitter AST-based Rust code chunking."""

import logging
from pathlib import PurePosixPath

import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

RUST_LANGUAGE = Language(tsrust.language())
_parser = Parser(RUST_LANGUAGE)

# Top-level type declaration node types
TYPE_DECLARATIONS = {
    "struct_item",
    "enum_item",
    "trait_item",
}


def chunk_file_rust(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a Rust file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk per type.
    Large files -> class_summary + method chunks for structs with impl blocks.
    Enums -> always whole_class.
    Traits -> whole_class (small) or class_summary + method signatures (large).
    impl blocks -> methods attached to the implementing type.
    Top-level functions -> function chunks with class_name = filename stem.
    Macros -> function chunks.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _namespace_from_path(file_path)
    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem

    chunks: list[CodeChunk] = []

    # --- Type declarations (struct, enum, trait) ---
    for type_node in _find_type_declarations(root):
        class_name = _get_node_name(type_node)
        if not class_name:
            continue

        doc_comment = _extract_doc_comment(type_node, source)

        if type_node.type == "enum_item":
            # Enums always as whole_class
            chunks.append(CodeChunk(
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
            ))
        elif type_node.type == "trait_item":
            base_types = _extract_supertraits(type_node)
            if total_lines < SMALL_FILE_LINE_THRESHOLD:
                chunks.append(CodeChunk(
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
                ))
            else:
                chunks.append(_make_trait_summary(
                    type_node, file_path, class_name, namespace, module,
                    doc_comment, base_types,
                ))
                for method_node in _find_trait_methods(type_node):
                    mc = _make_method_chunk(
                        method_node, source, file_path, class_name,
                        namespace, module, base_types,
                    )
                    if mc:
                        chunks.append(mc)
        else:
            # struct_item
            if total_lines < SMALL_FILE_LINE_THRESHOLD:
                chunks.append(CodeChunk(
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
                ))
            else:
                chunks.append(_make_struct_summary(
                    type_node, file_path, class_name, namespace, module, doc_comment,
                ))

    # --- impl blocks ---
    for impl_node in _find_impl_blocks(root):
        impl_class, impl_trait = _parse_impl_header(impl_node)
        if not impl_class:
            continue

        base_types = [impl_trait] if impl_trait else []
        doc_comment = _extract_doc_comment(impl_node, source)

        if total_lines < SMALL_FILE_LINE_THRESHOLD:
            # Small file: whole impl block as one chunk
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=impl_class,
                method_name=None,
                namespace=namespace,
                start_line=impl_node.start_point[0] + 1,
                end_line=impl_node.end_point[0] + 1,
                source=_node_text(impl_node),
                chunk_type="whole_class",
                module=module,
                doc_comment=doc_comment,
                base_types=base_types,
            ))
        else:
            # Large file: summary + individual method chunks
            chunks.append(_make_impl_summary(
                impl_node, file_path, impl_class, namespace, module,
                doc_comment, base_types,
            ))
            for fn_node in _find_impl_methods(impl_node):
                mc = _make_method_chunk(
                    fn_node, source, file_path, impl_class,
                    namespace, module, base_types,
                )
                if mc:
                    chunks.append(mc)

    # --- Top-level functions ---
    for func_node in _find_top_level_functions(root):
        func_name = _get_node_name(func_node)
        if not func_name:
            continue

        func_text = _node_text(func_node)
        line_count = func_text.count("\n") + 1
        if line_count < 3:
            continue

        doc_comment = _extract_doc_comment(func_node, source)

        chunks.append(CodeChunk(
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
        ))

    # --- Macro definitions ---
    for macro_node in _find_macro_definitions(root):
        macro_name = _get_macro_name(macro_node)
        if not macro_name:
            continue

        macro_text = _node_text(macro_node)
        line_count = macro_text.count("\n") + 1
        if line_count < 3:
            continue

        doc_comment = _extract_doc_comment(macro_node, source)

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=file_stem,
            method_name=macro_name,
            namespace=namespace,
            start_line=macro_node.start_point[0] + 1,
            end_line=macro_node.end_point[0] + 1,
            source=macro_text,
            chunk_type="function",
            module=module,
            doc_comment=doc_comment,
        ))

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

def _namespace_from_path(file_path: str) -> str:
    """Derive a Rust-style namespace from file path using :: separator.

    e.g. src/network/tcp.rs -> network::tcp
    """
    p = PurePosixPath(file_path)
    parts = list(p.parent.parts)
    # Strip leading src/ if present
    if parts and parts[0] == "src":
        parts = parts[1:]
    # Drop "." (root-level files)
    parts = [pt for pt in parts if pt != "."]
    return "::".join(parts)


def _find_type_declarations(root: Node) -> list[Node]:
    """Find top-level struct_item, enum_item, trait_item nodes."""
    results = []
    for child in root.children:
        if child.type in TYPE_DECLARATIONS:
            results.append(child)
    return results


def _find_impl_blocks(root: Node) -> list[Node]:
    """Find all top-level impl_item nodes."""
    return [c for c in root.children if c.type == "impl_item"]


def _find_top_level_functions(root: Node) -> list[Node]:
    """Find top-level function_item nodes (not inside impl or trait)."""
    return [c for c in root.children if c.type == "function_item"]


def _find_macro_definitions(root: Node) -> list[Node]:
    """Find top-level macro_definition nodes."""
    return [c for c in root.children if c.type == "macro_definition"]


def _find_trait_methods(trait_node: Node) -> list[Node]:
    """Find function declarations inside a trait body."""
    body = trait_node.child_by_field_name("body")
    if not body:
        return []
    results = []
    for child in body.children:
        if child.type == "function_item":
            results.append(child)
        elif child.type == "function_signature_item":
            results.append(child)
    return results


def _find_impl_methods(impl_node: Node) -> list[Node]:
    """Find function_item nodes inside an impl body."""
    body = impl_node.child_by_field_name("body")
    if not body:
        return []
    return [c for c in body.children if c.type == "function_item"]


def _get_node_name(node: Node) -> str:
    """Extract the identifier name from a declaration node."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    return ""


def _get_macro_name(node: Node) -> str:
    """Extract the name of a macro_definition."""
    # macro_definition's first named child is usually the name (identifier)
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")
    # Fallback: first child that is an identifier
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return ""


def _parse_impl_header(impl_node: Node) -> tuple[str, str]:
    """Parse an impl block to extract (class_name, trait_name).

    For `impl Foo` -> ("Foo", "")
    For `impl Trait for Foo` -> ("Foo", "Trait")
    """
    # tree-sitter-rust impl_item fields:
    #   type: the implementing type
    #   trait: the trait being implemented (if any)
    type_node = impl_node.child_by_field_name("type")
    trait_node = impl_node.child_by_field_name("trait")

    class_name = ""
    trait_name = ""

    if type_node:
        class_name = _type_node_name(type_node)
    if trait_node:
        trait_name = _type_node_name(trait_node)

    return class_name, trait_name


def _type_node_name(node: Node) -> str:
    """Extract a readable name from a type node (type_identifier, generic_type, etc.)."""
    if node.type == "type_identifier":
        return node.text.decode("utf-8", errors="replace")
    if node.type == "generic_type":
        # e.g. Vec<T> -> just "Vec"
        type_id = node.child_by_field_name("type")
        if type_id:
            return type_id.text.decode("utf-8", errors="replace")
    if node.type == "scoped_type_identifier":
        # e.g. std::io::Result -> full path
        return node.text.decode("utf-8", errors="replace")
    return node.text.decode("utf-8", errors="replace")


def _extract_supertraits(trait_node: Node) -> list[str]:
    """Extract supertrait names from a trait declaration (trait Foo: Bar + Baz)."""
    base_types = []
    # Look for trait_bounds in the trait header
    for child in trait_node.children:
        if child.type == "trait_bounds":
            for inner in child.children:
                if inner.type == "type_identifier":
                    base_types.append(inner.text.decode("utf-8", errors="replace"))
                elif inner.type == "generic_type":
                    type_id = inner.child_by_field_name("type")
                    if type_id:
                        base_types.append(type_id.text.decode("utf-8", errors="replace"))
    return base_types


def _extract_doc_comment(node: Node, source: bytes) -> str:
    """Extract /// doc comment lines preceding a node."""
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
    """Get the text content of a tree-sitter node."""
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _fn_signature(node: Node) -> str:
    """Extract just the function signature (no body)."""
    text = _node_text(node)
    idx = text.find("{")
    if idx > 0:
        return text[:idx].strip()
    return text.split("\n")[0].strip()


def _make_struct_summary(type_node: Node, file_path: str, class_name: str,
                          namespace: str, module: str, doc_comment: str) -> CodeChunk:
    """Create a summary chunk for a struct (fields, no impl methods)."""
    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=type_node.start_point[0] + 1,
        end_line=type_node.end_point[0] + 1,
        source=_node_text(type_node),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
    )


def _make_trait_summary(trait_node: Node, file_path: str, class_name: str,
                         namespace: str, module: str, doc_comment: str,
                         base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for a trait (signature + method signatures)."""
    summary_lines = []
    text = _node_text(trait_node)
    brace_idx = text.find("{")
    if brace_idx > 0:
        summary_lines.append(text[:brace_idx].strip())
    else:
        summary_lines.append(text[:200])

    body = trait_node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type in ("function_item", "function_signature_item"):
                sig = _fn_signature(child)
                if sig:
                    summary_lines.append("  " + sig)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=trait_node.start_point[0] + 1,
        end_line=trait_node.end_point[0] + 1,
        source="\n".join(summary_lines),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _make_impl_summary(impl_node: Node, file_path: str, class_name: str,
                        namespace: str, module: str, doc_comment: str,
                        base_types: list[str]) -> CodeChunk:
    """Create a summary chunk for an impl block (signature + fn signatures)."""
    summary_lines = []
    text = _node_text(impl_node)
    brace_idx = text.find("{")
    if brace_idx > 0:
        summary_lines.append(text[:brace_idx].strip())
    else:
        summary_lines.append(text[:200])

    for fn_node in _find_impl_methods(impl_node):
        sig = _fn_signature(fn_node)
        if sig:
            summary_lines.append("  " + sig)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=None,
        namespace=namespace,
        start_line=impl_node.start_point[0] + 1,
        end_line=impl_node.end_point[0] + 1,
        source="\n".join(summary_lines),
        chunk_type="class_summary",
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


def _make_method_chunk(method_node: Node, source: bytes, file_path: str,
                        class_name: str, namespace: str, module: str,
                        base_types: list[str]) -> CodeChunk | None:
    """Create a chunk for an individual method inside an impl or trait."""
    method_text = _node_text(method_node)
    line_count = method_text.count("\n") + 1
    if line_count < 3:
        return None

    method_name = _get_node_name(method_node)
    if not method_name:
        return None

    doc_comment = _extract_doc_comment(method_node, source)

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=method_name,
        namespace=namespace,
        start_line=method_node.start_point[0] + 1,
        end_line=method_node.end_point[0] + 1,
        source=method_text,
        chunk_type="method",
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )
