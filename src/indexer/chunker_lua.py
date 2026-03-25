"""Tree-sitter AST-based Lua code chunking.

Handles production game-Lua patterns:
- Module tables (local M = {} ... return M)
- Named function declarations (function, local function, M.func, M:method)
- xLua/ToLua hotfix calls — single-method and batch table forms
- Variable-assigned functions (local f = function() end)
- Assignment-based methods (M.handler = function() end, CS.X.Y = function() end)
- Table field functions (local Config = { onCast = function() end })
- Class-like OOP frameworks (class("Name", Base))
- EmmyLua annotations (---@class Name : Base, ---@param, ---@return)
- Constructor detection (ctor, new, __init)
- Class summary generation for large files with multiple methods
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import tree_sitter_lua as tslua
from tree_sitter import Language, Parser, Node

from src.models.chunk import CodeChunk
from src.config import SMALL_FILE_LINE_THRESHOLD

logger = logging.getLogger(__name__)

LUA_LANGUAGE = Language(tslua.language())
_parser = Parser(LUA_LANGUAGE)

_CONSTRUCTOR_NAMES = frozenset({"ctor", "new", "__init"})

# EmmyLua: ---@class ClassName : BaseClass, Base2
_EMMYLUA_CLASS_RE = re.compile(r"^---@class\s+(\w+)(?:\s*:\s*(.+))?$")

# Recognized xLua hotfix callee names
_HOTFIX_CALLEES = frozenset({"xlua.hotfix", "util.hotfix_ex", "xlua.hotfix_ex"})


@dataclass
class _ClassInfo:
    """Metadata about a Lua class or module table detected during pre-scan."""
    name: str
    base_types: list[str] = field(default_factory=list)
    is_module_table: bool = False
    doc_comment: str = ""


_EMPTY_INFO = _ClassInfo(name="")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_file_lua(source: bytes, file_path: str, module: str) -> list[CodeChunk]:
    """Parse a Lua file and return code chunks.

    Small files (< SMALL_FILE_LINE_THRESHOLD lines) -> one whole_class chunk.
    Large files -> individual chunks per function/method/hotfix, plus class summaries
    for classes or modules with multiple methods.
    """
    tree = _parser.parse(source)
    root = tree.root_node

    namespace = _derive_namespace(file_path)
    total_lines = source.count(b"\n") + 1
    file_stem = PurePosixPath(file_path).stem

    # Pre-scan: build registry of classes, module tables, EmmyLua annotations
    class_registry = _build_class_registry(root, source)
    module_table = _detect_module_table(root)
    default_class = module_table or file_stem

    # Enrich default class with base types if available
    default_base = class_registry[default_class].base_types if default_class in class_registry else []

    # Check for hotfix calls and CS.* assignments — these should always be
    # extracted individually regardless of file size, because the C# class/method
    # metadata is critical for search quality.
    has_hotfix_patterns = any(
        child.type == "function_call" and _get_callee_name(child) in _HOTFIX_CALLEES
        or (child.type == "assignment_statement" and _has_cs_lhs(child))
        for child in root.children
    )

    # Small files without hotfix patterns: single chunk
    if total_lines < SMALL_FILE_LINE_THRESHOLD and not has_hotfix_patterns:
        return [CodeChunk(
            file_path=file_path,
            class_name=default_class,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="whole_class",
            module=module,
            base_types=default_base,
        )]

    # Extract individual chunks from all patterns
    chunks: list[CodeChunk] = []
    is_small = total_lines < SMALL_FILE_LINE_THRESHOLD

    for child in root.children:
        if child.type == "function_declaration":
            # Skip regular function extraction for small files (they're in the whole_class)
            if not is_small:
                chunk = _make_named_function_chunk(
                    child, file_path, namespace, module, default_class, class_registry, root,
                )
                if chunk:
                    chunks.append(chunk)

        elif child.type == "function_call":
            # Always extract hotfix calls — the C# metadata is critical for search
            chunks.extend(_extract_hotfix_chunks(
                child, file_path, namespace, module,
            ))

        elif child.type == "variable_declaration":
            if not is_small:
                chunks.extend(_extract_variable_chunks(
                    child, file_path, namespace, module, default_class, class_registry, root,
                ))

        elif child.type == "assignment_statement":
            if not is_small:
                chunks.extend(_extract_assignment_chunks(
                    child, file_path, namespace, module, class_registry, root,
                ))
            else:
                # For small files, only extract CS.* overrides
                for ac in _extract_assignment_chunks(
                    child, file_path, namespace, module, class_registry, root,
                ):
                    if "Direct CS override" in (ac.doc_comment or ""):
                        chunks.append(ac)

    # For small files with hotfix patterns, also add a whole_class chunk for the
    # surrounding code (require statements, local helpers, etc.)
    if is_small and chunks:
        chunks.insert(0, CodeChunk(
            file_path=file_path,
            class_name=default_class,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=source.decode("utf-8", errors="replace"),
            chunk_type="whole_class",
            module=module,
            base_types=default_base,
        ))

    # Add class summaries for classes/modules with >= 2 methods
    if not is_small:
        _inject_class_summaries(chunks, file_path, namespace, module, class_registry)

    # Fallback: if nothing found, index whole file
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
# Pre-scan: class and module registry
# ---------------------------------------------------------------------------

def _build_class_registry(root: Node, source: bytes) -> dict[str, _ClassInfo]:
    """Detect class registrations, module tables, and EmmyLua annotations.

    Scans for:
    - EmmyLua ---@class Name : Base annotations
    - class("Name", Base) OOP framework calls
    - local M = {} module table pattern
    """
    registry: dict[str, _ClassInfo] = {}

    # --- EmmyLua annotations ---
    # Collect ---@class annotations and associate them with the next declaration
    pending_class: _ClassInfo | None = None
    pending_doc_lines: list[str] = []

    for child in root.children:
        if child.type == "comment":
            text = _node_text(child).strip()
            match = _EMMYLUA_CLASS_RE.match(text)
            if match:
                name = match.group(1)
                bases_str = match.group(2)
                bases = [b.strip() for b in bases_str.split(",")] if bases_str else []
                pending_class = _ClassInfo(name=name, base_types=bases)
                pending_doc_lines = []
            elif text.startswith("---"):
                stripped = text[3:].strip()
                if not stripped.startswith("@") and stripped:
                    pending_doc_lines.append(stripped)
        else:
            if pending_class:
                pending_class.doc_comment = " ".join(pending_doc_lines)
                # Register under the variable name if it's a declaration
                var_name = _get_declaration_name(child)
                if var_name:
                    registry[var_name] = pending_class
                    if pending_class.name != var_name:
                        registry[pending_class.name] = pending_class
                else:
                    registry[pending_class.name] = pending_class
                pending_class = None
                pending_doc_lines = []
            else:
                pending_doc_lines = []

    # --- class() OOP framework calls ---
    for child in root.children:
        if child.type == "variable_declaration":
            var_name = _get_declaration_name(child)
            if not var_name:
                continue
            func_call = _find_rhs_call(child)
            if func_call and _get_callee_name(func_call) == "class":
                args = _get_call_arguments(func_call)
                bases = []
                for arg in args[1:]:
                    if arg.type in ("identifier", "dot_index_expression"):
                        bases.append(_node_text(arg))
                if var_name not in registry:
                    registry[var_name] = _ClassInfo(name=var_name, base_types=bases)
                elif not registry[var_name].base_types and bases:
                    registry[var_name].base_types = bases

    # --- Module table pattern ---
    module_table = _detect_module_table(root)
    if module_table:
        if module_table not in registry:
            registry[module_table] = _ClassInfo(name=module_table, is_module_table=True)
        else:
            registry[module_table].is_module_table = True

    return registry


# ---------------------------------------------------------------------------
# Named function declarations
# ---------------------------------------------------------------------------

def _make_named_function_chunk(node: Node, file_path: str, namespace: str, module: str,
                               default_class: str, class_registry: dict[str, _ClassInfo],
                               root: Node) -> CodeChunk | None:
    """Create a chunk from a function_declaration node.

    Handles: function foo(), function M.foo(), function M:foo(), local function foo()
    """
    owner, func_name, is_method = _parse_function_name(node)
    if not func_name:
        return None

    text = _node_text(node)
    if text.count("\n") + 1 < 3:
        return None

    class_name = owner or default_class
    base_types = class_registry[class_name].base_types if class_name in class_registry else []
    doc_comment = _extract_doc_comment(root, node)

    if func_name in _CONSTRUCTOR_NAMES:
        chunk_type = "constructor"
    elif owner:
        chunk_type = "method"
    else:
        chunk_type = "function"

    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=func_name,
        namespace=namespace,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        source=text,
        chunk_type=chunk_type,
        module=module,
        doc_comment=doc_comment,
        base_types=base_types,
    )


# ---------------------------------------------------------------------------
# xLua / ToLua hotfix extraction
# ---------------------------------------------------------------------------

def _extract_hotfix_chunks(node: Node, file_path: str, namespace: str,
                           module: str) -> list[CodeChunk]:
    """Extract chunks from xLua/ToLua hotfix calls.

    Single method:  xlua.hotfix(CS.X.Y, 'Method', function(self) ... end)
    Batch:          xlua.hotfix(CS.X.Y, { Method1 = function() end, ... })
    """
    callee = _get_callee_name(node)
    if callee not in _HOTFIX_CALLEES:
        return []

    args = _get_call_arguments(node)
    if len(args) < 2:
        return []

    # Parse C# class path from first argument
    cs_path = _flatten_dot_path(args[0])
    if not cs_path:
        return []
    if cs_path[0] == "CS":
        cs_path = cs_path[1:]

    cs_class = cs_path[-1] if cs_path else ""
    cs_namespace = ".".join(cs_path[:-1]) if len(cs_path) > 1 else ""
    full_cs_path = ".".join(cs_path)

    chunks: list[CodeChunk] = []

    if len(args) >= 3 and args[1].type == "string":
        # Single method form
        method_name = _extract_string_content(args[1])
        if method_name:
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=cs_class,
                method_name=method_name,
                namespace=cs_namespace or namespace,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                source=_node_text(node),
                chunk_type="method",
                module=module,
                doc_comment=f"xLua hotfix for {full_cs_path}",
            ))

    elif len(args) >= 2 and args[1].type == "table_constructor":
        # Batch form — one chunk per function field
        for field_node in args[1].children:
            if field_node.type != "field":
                continue
            name_node = field_node.child_by_field_name("name")
            value_node = field_node.child_by_field_name("value")
            if not name_node or not value_node or value_node.type != "function_definition":
                continue

            method_name = _node_text(name_node)
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=cs_class,
                method_name=method_name,
                namespace=cs_namespace or namespace,
                start_line=field_node.start_point[0] + 1,
                end_line=field_node.end_point[0] + 1,
                source=_node_text(field_node),
                chunk_type="method",
                module=module,
                doc_comment=f"xLua hotfix for {full_cs_path}",
            ))

    return chunks


# ---------------------------------------------------------------------------
# Variable declaration chunks (local f = function, local T = { ... })
# ---------------------------------------------------------------------------

def _extract_variable_chunks(node: Node, file_path: str, namespace: str, module: str,
                             default_class: str, class_registry: dict[str, _ClassInfo],
                             root: Node) -> list[CodeChunk]:
    """Extract chunks from variable declarations with function values or table constructors."""
    var_name = _get_declaration_name(node)
    if not var_name:
        return []

    rhs = _find_rhs_expression(node)
    if not rhs:
        return []

    chunks: list[CodeChunk] = []

    if rhs.type == "function_definition":
        # local f = function() end
        text = _node_text(node)
        if text.count("\n") + 1 >= 3:
            doc = _extract_doc_comment(root, node)
            chunks.append(CodeChunk(
                file_path=file_path,
                class_name=default_class,
                method_name=var_name,
                namespace=namespace,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                source=text,
                chunk_type="function",
                module=module,
                doc_comment=doc,
            ))

    elif rhs.type == "table_constructor":
        # local Config = { onCast = function() end, ... }
        chunks.extend(_extract_table_field_functions(
            var_name, rhs, file_path, namespace, module, class_registry,
        ))

    return chunks


def _extract_table_field_functions(table_name: str, table_node: Node,
                                   file_path: str, namespace: str, module: str,
                                   class_registry: dict[str, _ClassInfo]) -> list[CodeChunk]:
    """Extract function-valued fields from a table constructor as method chunks."""
    chunks: list[CodeChunk] = []
    base_types = class_registry[table_name].base_types if table_name in class_registry else []

    for field_node in table_node.children:
        if field_node.type != "field":
            continue
        name_node = field_node.child_by_field_name("name")
        value_node = field_node.child_by_field_name("value")
        if not name_node or not value_node or value_node.type != "function_definition":
            continue

        field_name = _node_text(name_node)
        field_text = _node_text(field_node)
        if field_text.count("\n") + 1 < 3:
            continue

        chunk_type = "constructor" if field_name in _CONSTRUCTOR_NAMES else "method"

        chunks.append(CodeChunk(
            file_path=file_path,
            class_name=table_name,
            method_name=field_name,
            namespace=namespace,
            start_line=field_node.start_point[0] + 1,
            end_line=field_node.end_point[0] + 1,
            source=field_text,
            chunk_type=chunk_type,
            module=module,
            base_types=base_types,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Assignment statement chunks (M.handler = function, CS.X.Y = function)
# ---------------------------------------------------------------------------

def _extract_assignment_chunks(node: Node, file_path: str, namespace: str, module: str,
                               class_registry: dict[str, _ClassInfo],
                               root: Node) -> list[CodeChunk]:
    """Extract chunks from root-level assignment statements with function values."""
    lhs_node = None
    rhs_node = None
    for sub in node.children:
        if sub.type == "variable_list":
            for inner in sub.children:
                if inner.type in ("dot_index_expression", "identifier"):
                    lhs_node = inner
                    break
        elif sub.type == "expression_list":
            for inner in sub.children:
                if inner.type == "function_definition":
                    rhs_node = inner
                    break

    if not lhs_node or not rhs_node:
        return []

    text = _node_text(node)
    if text.count("\n") + 1 < 3:
        return []

    doc = _extract_doc_comment(root, node)

    if lhs_node.type == "dot_index_expression":
        path = _flatten_dot_path(lhs_node)

        if len(path) >= 3 and path[0] == "CS":
            # CS.Game.Player.Method = function() → hotfix-style
            cs_path = path[1:]
            cs_class = cs_path[-2] if len(cs_path) >= 2 else cs_path[0]
            method_name = cs_path[-1]
            cs_ns = ".".join(cs_path[:-2]) if len(cs_path) > 2 else ""
            full_cs = ".".join(cs_path)
            return [CodeChunk(
                file_path=file_path,
                class_name=cs_class,
                method_name=method_name,
                namespace=cs_ns or namespace,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                source=text,
                chunk_type="method",
                module=module,
                doc_comment=f"Direct CS override: {full_cs}" + (f". {doc}" if doc else ""),
            )]

        elif len(path) >= 2:
            # M.handler = function() → method chunk
            owner = path[-2]
            method_name = path[-1]
            base_types = class_registry[owner].base_types if owner in class_registry else []
            chunk_type = "constructor" if method_name in _CONSTRUCTOR_NAMES else "method"
            return [CodeChunk(
                file_path=file_path,
                class_name=owner,
                method_name=method_name,
                namespace=namespace,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                source=text,
                chunk_type=chunk_type,
                module=module,
                doc_comment=doc,
                base_types=base_types,
            )]

    return []


# ---------------------------------------------------------------------------
# Class summary injection
# ---------------------------------------------------------------------------

def _inject_class_summaries(chunks: list[CodeChunk], file_path: str, namespace: str,
                            module: str, class_registry: dict[str, _ClassInfo]) -> None:
    """Add class_summary chunks for classes/modules with multiple methods."""
    methods_by_class: dict[str, list[CodeChunk]] = defaultdict(list)
    for c in chunks:
        if c.chunk_type in ("method", "function", "constructor") and c.method_name:
            methods_by_class[c.class_name].append(c)

    summaries: list[CodeChunk] = []
    for class_name, methods in methods_by_class.items():
        if len(methods) < 2:
            continue

        info = class_registry.get(class_name, _EMPTY_INFO)

        # Build summary source
        lines: list[str] = []
        if info.is_module_table:
            lines.append(f"local {class_name} = {{}}")
        elif info.base_types:
            lines.append(f"local {class_name} = class(\"{class_name}\", {', '.join(info.base_types)})")
        else:
            lines.append(f"-- {class_name}")

        for m in methods:
            sig = _function_signature(m)
            lines.append(f"  {sig}")

        all_starts = [m.start_line for m in methods]
        all_ends = [m.end_line for m in methods]

        summaries.append(CodeChunk(
            file_path=file_path,
            class_name=class_name,
            method_name=None,
            namespace=namespace,
            start_line=min(all_starts),
            end_line=max(all_ends),
            source="\n".join(lines),
            chunk_type="class_summary",
            module=module,
            base_types=info.base_types,
            doc_comment=info.doc_comment,
        ))

    # Insert summaries at the front (before their methods)
    chunks[0:0] = summaries


def _function_signature(chunk: CodeChunk) -> str:
    """Build a one-line function signature from a chunk."""
    first_line = chunk.source.split("\n")[0].strip()
    # For table field functions like "onCast = function(self, target)"
    if "= function" in first_line:
        return first_line.rstrip(",")
    # For named declarations
    if first_line.startswith("function ") or first_line.startswith("local function "):
        return first_line
    return f"function {chunk.class_name}:{chunk.method_name}(...)"


# ---------------------------------------------------------------------------
# Module table and function name detection
# ---------------------------------------------------------------------------

def _detect_module_table(root: Node) -> str | None:
    """Detect the Lua module pattern: local M = {} ... return M."""
    # Look for `local X = {}` near the top.
    # AST: variable_declaration -> [local, assignment_statement]
    #   assignment_statement -> [variable_list -> identifier, =, expression_list -> table_constructor]
    table_candidates: set[str] = set()
    for child in root.children:
        if child.type == "variable_declaration":
            assign = None
            for sub in child.children:
                if sub.type == "assignment_statement":
                    assign = sub
                    break
            if assign is None:
                continue
            has_table = False
            for sub in assign.children:
                if sub.type == "expression_list":
                    for expr in sub.children:
                        if expr.type == "table_constructor":
                            has_table = True
                            break
            if not has_table:
                continue
            for sub in assign.children:
                if sub.type == "variable_list":
                    for inner in sub.children:
                        if inner.type == "identifier":
                            table_candidates.add(_node_text(inner))
                            break

    if not table_candidates:
        return None

    # Look for `return X` at the end.
    # AST: return_statement -> [return, expression_list -> identifier]
    for child in reversed(root.children):
        if child.type == "return_statement":
            for sub in child.children:
                if sub.type == "expression_list":
                    for expr in sub.children:
                        if expr.type == "identifier":
                            name = _node_text(expr)
                            if name in table_candidates:
                                return name
                elif sub.type == "identifier":
                    name = _node_text(sub)
                    if name in table_candidates:
                        return name
            break

    return None


def _parse_function_name(node: Node) -> tuple[str | None, str | None, bool]:
    """Extract (owner, func_name, is_method) from a function declaration.

    function foo()       -> (None, "foo", False)
    function M.foo()     -> ("M", "foo", False)
    function M:bar()     -> ("M", "bar", True)
    local function foo() -> (None, "foo", False)
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return (None, None, False)

    if name_node.type == "identifier":
        return (None, _node_text(name_node), False)

    if name_node.type == "dot_index_expression":
        table_node = name_node.child_by_field_name("table")
        field_node = name_node.child_by_field_name("field")
        owner = _node_text(table_node) if table_node else None
        func = _node_text(field_node) if field_node else None
        return (owner, func, False)

    if name_node.type == "method_index_expression":
        table_node = name_node.child_by_field_name("table")
        method_node = name_node.child_by_field_name("method")
        owner = _node_text(table_node) if table_node else None
        method = _node_text(method_node) if method_node else None
        return (owner, method, True)

    return (None, _node_text(name_node), False)


# ---------------------------------------------------------------------------
# Comment / doc extraction
# ---------------------------------------------------------------------------

def _extract_doc_comment(root: Node, target_node: Node) -> str:
    """Extract doc comments preceding a node.

    Handles:
    - Single line: -- comment
    - EmmyLua: ---@param name type, --- description text
    - Separates descriptions from @annotations for cleaner output
    """
    target_line = target_node.start_point[0]

    # Collect all comment nodes before the target
    comments: list[tuple[int, str]] = []
    for child in root.children:
        if child.start_point[0] >= target_line:
            break
        if child.type == "comment":
            comments.append((child.end_point[0], _node_text(child).strip()))

    # Keep only contiguous comments immediately before the target
    result: list[str] = []
    for end_line, text in reversed(comments):
        if end_line == target_line - 1 - len(result):
            result.insert(0, text)
        else:
            break

    if not result:
        return ""

    descriptions: list[str] = []
    annotations: list[str] = []
    for line in result:
        stripped = line
        if stripped.startswith("---"):
            stripped = stripped[3:].strip()
        elif stripped.startswith("--"):
            stripped = stripped[2:].strip()

        if stripped.startswith("@class"):
            continue  # skip @class — already in class_registry
        elif stripped.startswith("@"):
            annotations.append(stripped)
        elif stripped:
            descriptions.append(stripped)

    parts: list[str] = []
    if descriptions:
        parts.append(" ".join(descriptions))
    if annotations:
        parts.append(" ".join(annotations))
    return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node) -> str:
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _derive_namespace(file_path: str) -> str:
    """Derive a dotted namespace from file path."""
    path = PurePosixPath(file_path)
    parts = list(path.parts)
    if parts and parts[-1].endswith(".lua"):
        parts[-1] = parts[-1][:-4]
    return ".".join(parts)


def _get_declaration_name(node: Node) -> str | None:
    """Extract the variable name from a variable_declaration or function_declaration."""
    if node.type == "variable_declaration":
        for sub in node.children:
            if sub.type == "assignment_statement":
                for inner in sub.children:
                    if inner.type == "variable_list":
                        for ident in inner.children:
                            if ident.type == "identifier":
                                return _node_text(ident)
    elif node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        if name_node and name_node.type == "identifier":
            return _node_text(name_node)
    return None


def _find_rhs_call(node: Node) -> Node | None:
    """Find a function_call on the RHS of a variable_declaration."""
    for sub in node.children:
        if sub.type == "assignment_statement":
            for inner in sub.children:
                if inner.type == "expression_list":
                    for expr in inner.children:
                        if expr.type == "function_call":
                            return expr
    return None


def _find_rhs_expression(node: Node) -> Node | None:
    """Find the first meaningful expression on the RHS of a variable_declaration."""
    for sub in node.children:
        if sub.type == "assignment_statement":
            for inner in sub.children:
                if inner.type == "expression_list":
                    for expr in inner.children:
                        if expr.type not in (",",):
                            return expr
    return None


def _get_callee_name(node: Node) -> str:
    """Get the dotted name of a function call's callee.

    xlua.hotfix(...) -> "xlua.hotfix"
    print(...)       -> "print"
    """
    if node.type != "function_call":
        return ""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child)
        if child.type == "dot_index_expression":
            return _node_text(child)
        if child.type == "method_index_expression":
            return _node_text(child)
    return ""


def _get_call_arguments(node: Node) -> list[Node]:
    """Get argument nodes from a function call (excluding parens and commas)."""
    for child in node.children:
        if child.type == "arguments":
            return [c for c in child.children if c.type not in ("(", ")", ",")]
    return []


def _flatten_dot_path(node: Node) -> list[str]:
    """Flatten nested dot_index_expression into identifier parts.

    CS.Game.Combat.DamageSystem -> ["CS", "Game", "Combat", "DamageSystem"]
    """
    if node.type == "identifier":
        return [_node_text(node)]
    if node.type == "dot_index_expression":
        table = node.child_by_field_name("table")
        field_node = node.child_by_field_name("field")
        parts = _flatten_dot_path(table) if table else []
        if field_node:
            parts.append(_node_text(field_node))
        return parts
    return [_node_text(node)]


def _has_cs_lhs(node: Node) -> bool:
    """Check if an assignment statement has a CS.* dotted path on the left-hand side."""
    for sub in node.children:
        if sub.type == "variable_list":
            for inner in sub.children:
                if inner.type == "dot_index_expression":
                    path = _flatten_dot_path(inner)
                    return len(path) >= 3 and path[0] == "CS"
    return False


def _extract_string_content(node: Node) -> str:
    """Extract text content from a string literal node (strip quotes)."""
    for child in node.children:
        if child.type == "string_content":
            return _node_text(child)
    text = _node_text(node)
    if len(text) >= 2 and text[0] in ("'", '"') and text[-1] in ("'", '"'):
        return text[1:-1]
    return text
