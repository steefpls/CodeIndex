"""Build class dependency graph sidecar: class -> [classes it references]."""

import json
import logging
import re
from pathlib import Path

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# PascalCase identifier: starts uppercase, has at least one lowercase letter,
# minimum 3 chars. Filters ALL_CAPS, single letters, and short tokens like "On".
_TYPE_CANDIDATE_RE = re.compile(r'\b([A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*)\b')
_MIN_NAME_LENGTH = 3

# Patterns to strip before extraction (reduce false positives)
_SINGLE_LINE_COMMENT_RE = re.compile(r'//.*$', re.MULTILINE)
_MULTI_LINE_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRING_LITERAL_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"')
_ATTRIBUTE_RE = re.compile(r'\[\s*\w+(?:\s*\([^)]*\))?\s*\]')

# Python import patterns: capture module paths and imported names.
# The from-import pattern handles both single-line and parenthesized multi-line imports.
_PYTHON_FROM_IMPORT_RE = re.compile(
    r'^from\s+([\w.]+)\s+import\s+\(([^)]+)\)', re.MULTILINE | re.DOTALL
)
_PYTHON_FROM_IMPORT_SINGLE_RE = re.compile(
    r'^from\s+([\w.]+)\s+import\s+([^(\n]+)$', re.MULTILINE
)
_PYTHON_IMPORT_RE = re.compile(
    r'^import\s+([\w.,\s]+)$', re.MULTILINE
)

# JS/TS import patterns: ESM and CommonJS
# import { Foo, bar } from './module'  AND  import type { Foo } from './module'
_JS_NAMED_IMPORT_RE = re.compile(
    r"""import\s+(?:type\s+)?\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]""",
)
# import Foo from './module'
_JS_DEFAULT_IMPORT_RE = re.compile(
    r"""import\s+(\w+)\s+from\s+['"]([^'"]+)['"]""",
)
# import * as Foo from './module'
_JS_NAMESPACE_IMPORT_RE = re.compile(
    r"""import\s+\*\s+as\s+(\w+)\s+from\s+['"]([^'"]+)['"]""",
)
# const Foo = require('./module')
_JS_REQUIRE_RE = re.compile(
    r"""(?:const|let|var)\s+(?:\{([^}]+)\}|(\w+))\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
)
# export { Foo } from './module' (re-exports)
_JS_REEXPORT_RE = re.compile(
    r"""export\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]""",
)

# Chunk types that represent actual code classes (not prefab/scene elements)
CODE_CHUNK_TYPES = frozenset({
    "whole_class", "class_summary", "method", "constructor", "property", "function",
})

# Type alias for lightweight records from pipeline
DepRecord = tuple[str, str, str, str, set[str]]
# (class_name, file_path, module, namespace, type_candidates)


def _clean_source(source_text: str) -> str:
    """Strip comments, string literals, and attributes to reduce noise."""
    text = _SINGLE_LINE_COMMENT_RE.sub('', source_text)
    text = _MULTI_LINE_COMMENT_RE.sub('', text)
    text = _STRING_LITERAL_RE.sub('', text)
    text = _ATTRIBUTE_RE.sub('', text)
    return text


def _extract_python_import_names(source_text: str) -> set[str]:
    """Extract dependency names from Python import statements.

    Captures:
      - ``from a.b.module import X, Y`` → {module, X, Y}
      - ``import a.b.module`` → {module}
      - Handles 'as' aliases by capturing the original name, not the alias.
      - Strips parenthesized multi-line imports.

    These names are intersected with known class names by the caller,
    so false positives (e.g., stdlib modules) are filtered out.
    """
    candidates: set[str] = set()

    def _process_from_import(module_path: str, imports_str: str) -> None:
        # Add last component of the module path (e.g., "embedder" from "src.indexer.embedder")
        last_segment = module_path.rsplit(".", 1)[-1]
        if len(last_segment) >= _MIN_NAME_LENGTH:
            candidates.add(last_segment)
        # Add each imported name (strip "as alias")
        for part in imports_str.split(","):
            token = part.strip().split(" as ")[0].strip()
            if not token:
                continue
            name = token.split()[0]
            if name and len(name) >= _MIN_NAME_LENGTH and name.isidentifier():
                candidates.add(name)

    # Parenthesized multi-line imports: from X import (A, B, C)
    for match in _PYTHON_FROM_IMPORT_RE.finditer(source_text):
        _process_from_import(match.group(1), match.group(2))

    # Single-line imports: from X import A, B
    for match in _PYTHON_FROM_IMPORT_SINGLE_RE.finditer(source_text):
        _process_from_import(match.group(1), match.group(2))

    for match in _PYTHON_IMPORT_RE.finditer(source_text):
        for module in match.group(1).split(","):
            module = module.strip().split(" as ")[0].strip()
            # Take last component of dotted path
            last_segment = module.rsplit(".", 1)[-1]
            if len(last_segment) >= _MIN_NAME_LENGTH and last_segment.isidentifier():
                candidates.add(last_segment)

    return candidates


def _extract_js_import_names(source_text: str) -> set[str]:
    """Extract dependency names from JavaScript/TypeScript import statements.

    Captures:
      - ``import { Foo, bar } from './module'`` → {Foo, bar, module}
      - ``import Foo from './module'`` → {Foo, module}
      - ``import * as Foo from './module'`` → {Foo, module}
      - ``const { Foo } = require('./module')`` → {Foo, module}
      - ``export { Foo } from './module'`` → {Foo, module}
      - Handles 'as' aliases by capturing the original name.

    Module paths are resolved to their last segment (e.g., './utils/api' → 'api').
    These names are intersected with known class names by the caller.
    """
    candidates: set[str] = set()

    def _add_module_name(module_path: str) -> None:
        """Add the last path segment as a candidate (strip extension)."""
        # './components/MyWidget' → 'MyWidget', '@/utils/api' → 'api'
        segment = module_path.rsplit("/", 1)[-1]
        # Strip common extensions
        for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            if segment.endswith(ext):
                segment = segment[: -len(ext)]
                break
        if len(segment) >= _MIN_NAME_LENGTH and segment.isidentifier():
            candidates.add(segment)

    def _add_named_imports(names_str: str) -> None:
        """Parse comma-separated import names, handling 'as' aliases."""
        for part in names_str.split(","):
            token = part.strip().split(" as ")[0].strip()
            if not token:
                continue
            # Handle 'type Foo' in TS: import { type Foo } from ...
            if token.startswith("type "):
                token = token[5:].strip()
            if token and len(token) >= _MIN_NAME_LENGTH and token.isidentifier():
                candidates.add(token)

    # import { Foo, bar } from './module'
    for match in _JS_NAMED_IMPORT_RE.finditer(source_text):
        _add_named_imports(match.group(1))
        _add_module_name(match.group(2))

    # import Foo from './module'
    for match in _JS_DEFAULT_IMPORT_RE.finditer(source_text):
        name = match.group(1)
        if len(name) >= _MIN_NAME_LENGTH and name.isidentifier():
            candidates.add(name)
        _add_module_name(match.group(2))

    # import * as Foo from './module'
    for match in _JS_NAMESPACE_IMPORT_RE.finditer(source_text):
        name = match.group(1)
        if len(name) >= _MIN_NAME_LENGTH and name.isidentifier():
            candidates.add(name)
        _add_module_name(match.group(2))

    # const { Foo } = require('./module')  OR  const Foo = require('./module')
    for match in _JS_REQUIRE_RE.finditer(source_text):
        named = match.group(1)
        default = match.group(2)
        if named:
            _add_named_imports(named)
        elif default and len(default) >= _MIN_NAME_LENGTH and default.isidentifier():
            candidates.add(default)
        _add_module_name(match.group(3))

    # export { Foo } from './module'
    for match in _JS_REEXPORT_RE.finditer(source_text):
        _add_named_imports(match.group(1))
        _add_module_name(match.group(2))

    return candidates


def extract_type_candidates(source_text: str) -> set[str]:
    """Extract potential type references from source code.

    For all languages: extracts PascalCase identifiers (>= 3 chars) after
    stripping comments, strings, and attributes.

    Additionally extracts imported names from language-specific import syntax:
      - Python: ``import`` / ``from ... import`` statements
      - JS/TS: ESM ``import``, ``export from``, and CommonJS ``require``

    These regexes only match their respective language syntax, so running on
    other languages is a no-op. The caller intersects with known class names
    to filter false positives.
    """
    cleaned = _clean_source(source_text)
    candidates = {m for m in _TYPE_CANDIDATE_RE.findall(cleaned) if len(m) >= _MIN_NAME_LENGTH}

    # Language-specific import extraction (safe to run on all — no cross-language false positives)
    candidates.update(_extract_python_import_names(source_text))
    candidates.update(_extract_js_import_names(source_text))

    return candidates


def build_dep_graph(records: list[DepRecord]) -> dict[str, dict]:
    """Build class dependency graph via known-class intersection.

    Two passes:
    1. Collect all known class names from records
    2. For each class, intersect its type_candidates with known classes (excluding self)

    Returns:
        {"ClassA": {"file": "path/to/ClassA.cs", "module": "...", "namespace": "...", "deps": ["ClassB", "IFoo"]}}
    """
    all_classes = {r[0] for r in records}

    graph: dict[str, dict] = {}
    for class_name, file_path, module, namespace, type_candidates in records:
        deps = sorted(type_candidates & all_classes - {class_name})
        graph[class_name] = {
            "file": file_path,
            "module": module,
            "namespace": namespace,
            "deps": deps,
        }
    return graph


def save_dep_graph(repo_name: str, records: list[DepRecord]) -> None:
    """Build and save dependency graph sidecar to disk."""
    graph = build_dep_graph(records)
    path = DATA_DIR / f"{repo_name}_dep_graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    total_edges = sum(len(v["deps"]) for v in graph.values())
    logger.info("Saved dep graph for %s: %d classes, %d edges",
                repo_name, len(graph), total_edges)


def load_dep_graph(repo_name: str) -> dict[str, dict]:
    """Load dependency graph from disk."""
    path = DATA_DIR / f"{repo_name}_dep_graph.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load dep graph for %s: %s", repo_name, e)
        return {}
