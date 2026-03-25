"""Find exact text references to a symbol across source files."""

import logging
import re
import shutil
import subprocess
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

from src.config import REPOS, DATA_DIR, resolve_repo, LANGUAGE_EXTENSIONS
from src.indexer.file_scanner import _is_excluded
from src.indexer.hierarchy_builder import load_type_hierarchy

logger = logging.getLogger(__name__)

# Languages where symbol reference search is useful (code, not data)
_CODE_LANGUAGES = frozenset({"csharp", "cpp", "python", "javascript", "html", "lua", "typescript", "rust"})

# Check for ripgrep availability once at import time
_RG_PATH = shutil.which("rg")

# Patterns that indicate a line is a definition (not just a reference)
_DEF_PATTERNS_CS = [
    # interface/class/struct/enum declaration
    re.compile(r'\b(?:interface|class|struct|enum|record)\s+{symbol}\b'),
    # method/property declaration (return_type Symbol( or return_type Symbol {)
    re.compile(r'(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|readonly|new|partial)\s+.*\b{symbol}\s*[\(<{{]'),
    # field/property with the symbol as the *name* (type Symbol = / type Symbol;)
    re.compile(r'(?:public|private|protected|internal|static|readonly|const|volatile)\s+\S+\s+{symbol}\s*[;=\{{]'),
]

_DEF_PATTERNS_CPP = [
    # class/struct/enum
    re.compile(r'\b(?:class|struct|enum)\s+{symbol}\b'),
    # function definition: ReturnType ClassName::Symbol( or ReturnType Symbol(
    re.compile(r'\b\w[\w:*&<> ]*\s+(?:\w+::)?{symbol}\s*\('),
]

_DEF_PATTERNS_PY = [
    re.compile(r'\b(?:class|def)\s+{symbol}\b'),
]

_DEF_PATTERNS_JS = [
    re.compile(r'\b(?:class|function)\s+{symbol}\b'),
    re.compile(r'\b(?:const|let|var)\s+{symbol}\b'),
]

_DEF_PATTERNS_LUA = [
    # function Name() / function Table.Name() / function Table:Name()
    re.compile(r'\b(?:local\s+)?function\s+(?:\w+[.:])?{symbol}\s*\('),
    # local Name = (class, table, function, value)
    re.compile(r'\blocal\s+{symbol}\s*='),
]

_DEF_PATTERN_MAP = {
    ".cs": _DEF_PATTERNS_CS,
    ".cpp": _DEF_PATTERNS_CPP,
    ".h": _DEF_PATTERNS_CPP,
    ".hpp": _DEF_PATTERNS_CPP,
    ".py": _DEF_PATTERNS_PY,
    ".js": _DEF_PATTERNS_JS,
    ".ts": _DEF_PATTERNS_JS,
    ".tsx": _DEF_PATTERNS_JS,
    ".jsx": _DEF_PATTERNS_JS,
    ".lua": _DEF_PATTERNS_LUA,
}


def _is_definition_line(line: str, symbol: str, file_path: str) -> bool:
    """Check if a source line is a definition of the symbol (not just a usage)."""
    # Determine file extension
    ext = ""
    dot_idx = file_path.rfind(".")
    if dot_idx >= 0:
        ext = file_path[dot_idx:].lower()

    patterns = _DEF_PATTERN_MAP.get(ext)
    if not patterns:
        return False

    escaped = re.escape(symbol)
    for pat_template in patterns:
        # Build the actual pattern with the symbol substituted in
        pat = re.compile(pat_template.pattern.replace("{symbol}", escaped))
        if pat.search(line):
            return True
    return False


def find_references(
    symbol: str,
    repo: str,
    file_pattern: str | None = None,
    whole_word: bool = True,
    max_results: int = 50,
) -> str:
    """Find all exact references to a symbol across source files.

    Searches file contents (not the index) for exact text matches. Useful for
    finding all usages before refactoring.

    Uses ripgrep (rg) for speed when available, falls back to Python scan.

    Results are tagged [DEF] for definitions (class/method/function declarations)
    and [REF] for usages, making it easy to find the source of truth.

    Args:
        symbol: The symbol to search for (e.g., "IRobotDriver", "OnCalibrationComplete").
        repo: Which repo to search (required). Use "all" to search all repos.
        file_pattern: Optional glob filter (e.g., "*.cs", "*.prefab").
        whole_word: If True (default), match whole words only (\\b boundaries).
        max_results: Max results to return (default 50).

    Returns:
        Formatted list of file:line matches with [DEF]/[REF] tags and context.
    """
    # Cross-repo search
    if repo == "all":
        return _find_references_all(symbol, file_pattern, whole_word, max_results)

    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())} or 'all'"

    return _find_references_single(symbol, resolved, file_pattern, whole_word, max_results)


def _find_references_single(
    symbol: str,
    resolved: str,
    file_pattern: str | None,
    whole_word: bool,
    max_results: int,
) -> str:
    """Search a single resolved repo for references."""
    config = REPOS[resolved]

    # Collect search directories and their file extensions
    search_dirs: list[tuple[Path, list[str], list[str]]] = []
    for sd in config.source_dirs:
        # Skip Unity asset dirs for symbol search (YAML, not code)
        if sd.language not in _CODE_LANGUAGES:
            continue
        extensions = list(sd.extensions)
        search_dirs.append((sd.path, extensions, sd.exclude_patterns))

    if not search_dirs:
        return f"No code source directories configured for '{resolved}'."

    # Try ripgrep first
    if _RG_PATH and not file_pattern:
        result = _search_with_rg(symbol, whole_word, max_results, search_dirs, config.root)
        if result is not None:
            # Append fuzzy suggestions on zero results
            if result.startswith("No references to"):
                suggestion = _suggest_similar_symbols(symbol, resolved)
                if suggestion:
                    result += "\n" + suggestion
            return result

    # Python fallback
    result = _search_with_python(symbol, whole_word, max_results, search_dirs,
                                  config.root, file_pattern)
    # Append fuzzy suggestions on zero results
    if result.startswith("No references to"):
        suggestion = _suggest_similar_symbols(symbol, resolved)
        if suggestion:
            result += "\n" + suggestion
    return result


def _find_references_all(
    symbol: str,
    file_pattern: str | None,
    whole_word: bool,
    max_results: int,
) -> str:
    """Search all repos for references and merge results."""
    all_results: list[str] = []
    repos_searched: list[str] = []

    for repo_name in REPOS:
        result = _find_references_single(symbol, repo_name, file_pattern, whole_word, max_results)
        # Collect results that found matches (skip "No references" / "No code source" messages)
        if not result.startswith("No references to") and not result.startswith("No code source"):
            all_results.append(f"=== {repo_name} ===\n{result}")
        repos_searched.append(repo_name)

    if not all_results:
        mode = "whole-word" if whole_word else "substring"
        return (f"No references to '{symbol}' found ({mode} search "
                f"across {len(repos_searched)} repos: {', '.join(repos_searched)}).")

    return "\n\n".join(all_results)


def _search_with_rg(symbol: str, whole_word: bool, max_results: int,
                     search_dirs: list[tuple[Path, list[str], list[str]]],
                     repo_root: Path) -> str | None:
    """Search using ripgrep. Returns None if rg fails (caller should fallback)."""
    try:
        all_results: list[str] = []

        for search_path, extensions, exclude_patterns in search_dirs:
            if not search_path.exists():
                continue

            cmd = [_RG_PATH, "--no-heading", "--line-number", "--max-count", str(max_results)]
            if whole_word:
                cmd.extend(["--word-regexp"])

            # Add file type globs
            for ext in extensions:
                cmd.extend(["--glob", f"*{ext}"])

            # Add exclude patterns
            for pat in exclude_patterns:
                if pat.endswith("/"):
                    cmd.extend(["--glob", f"!**/{pat.rstrip('/')}/**"])
                elif pat.startswith("*."):
                    cmd.extend(["--glob", f"!{pat}"])

            cmd.append(symbol)
            cmd.append(str(search_path))

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if proc.returncode > 1:
                return None  # rg error, fallback to Python

            for line in proc.stdout.strip().split("\n"):
                if not line:
                    continue
                # rg output: /abs/path:line:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    abs_path = parts[0]
                    line_num = parts[1]
                    content = parts[2].strip()
                    try:
                        rel_path = str(Path(abs_path).relative_to(repo_root)).replace("\\", "/")
                    except ValueError:
                        rel_path = abs_path
                    if len(content) > 200:
                        content = content[:200] + "..."
                    tag = "[DEF]" if _is_definition_line(content, symbol, rel_path) else "[REF]"
                    all_results.append(f"  {tag} {rel_path}:{line_num}  {content}")
                    if len(all_results) >= max_results:
                        break
            if len(all_results) >= max_results:
                break

        if not all_results:
            mode = "whole-word" if whole_word else "substring"
            return f"No references to '{symbol}' found ({mode} search via rg)."

        header = f"Found {len(all_results)} reference(s) to '{symbol}'"
        if len(all_results) >= max_results:
            # Get true total count via rg --count
            total = _count_with_rg(symbol, whole_word, search_dirs)
            if total and total > max_results:
                header += f" (showing {max_results} of {total} total — increase max_results to see more)"
            else:
                header += f" (capped at {max_results})"
        header += ":\n"
        return header + "\n".join(all_results)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("rg search failed, falling back to Python: %s", e)
        return None


def _count_with_rg(symbol: str, whole_word: bool,
                    search_dirs: list[tuple[Path, list[str], list[str]]]) -> int | None:
    """Get total match count via rg --count. Returns None on failure."""
    try:
        total = 0
        for search_path, extensions, exclude_patterns in search_dirs:
            if not search_path.exists():
                continue
            cmd = [_RG_PATH, "--count-matches", "--no-filename"]
            if whole_word:
                cmd.append("--word-regexp")
            for ext in extensions:
                cmd.extend(["--glob", f"*{ext}"])
            for pat in exclude_patterns:
                if pat.endswith("/"):
                    cmd.extend(["--glob", f"!**/{pat.rstrip('/')}/**"])
                elif pat.startswith("*."):
                    cmd.extend(["--glob", f"!{pat}"])
            cmd.extend([symbol, str(search_path)])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in proc.stdout.strip().split("\n"):
                if line.strip().isdigit():
                    total += int(line.strip())
        return total if total > 0 else None
    except Exception:
        return None


def _search_with_python(symbol: str, whole_word: bool, max_results: int,
                         search_dirs: list[tuple[Path, list[str], list[str]]],
                         repo_root: Path, file_pattern: str | None) -> str:
    """Search using pure Python (fallback)."""
    if whole_word:
        pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')
    else:
        pattern = re.compile(re.escape(symbol))

    results: list[str] = []
    total_matches = 0
    files_searched = 0
    capped = False

    for search_path, extensions, exclude_patterns in search_dirs:
        for ext in extensions:
            glob_pat = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            for fp in search_path.rglob(glob_pat):
                if _is_excluded(fp, exclude_patterns):
                    continue
                if file_pattern and not fp.name.endswith(file_pattern.lstrip("*")):
                    continue

                files_searched += 1
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                rel_path = str(fp.relative_to(repo_root)).replace("\\", "/")
                for i, line in enumerate(text.split("\n"), 1):
                    if pattern.search(line):
                        total_matches += 1
                        if not capped:
                            trimmed = line.strip()
                            if len(trimmed) > 200:
                                trimmed = trimmed[:200] + "..."
                            tag = "[DEF]" if _is_definition_line(trimmed, symbol, rel_path) else "[REF]"
                            results.append(f"  {tag} {rel_path}:{i}  {trimmed}")
                            if len(results) >= max_results:
                                capped = True

    if not results:
        mode = "whole-word" if whole_word else "substring"
        return f"No references to '{symbol}' found ({mode} search, {files_searched} files)."

    header = f"Found {len(results)} reference(s) to '{symbol}'"
    if capped and total_matches > max_results:
        header += f" (showing {max_results} of {total_matches} total — increase max_results to see more)"
    elif capped:
        header += f" (capped at {max_results})"
    header += f" across {files_searched} files:\n"
    return header + "\n".join(results)


def _suggest_similar_symbols(symbol: str, repo: str) -> str:
    """Suggest similar symbol names from the type hierarchy when a search finds nothing.

    Uses difflib.get_close_matches against all known class/interface names
    from the type hierarchy (both keys and implementation class names).
    """
    try:
        hierarchy = load_type_hierarchy(repo)
        if not hierarchy:
            return ""

        # Build a set of all known symbol names from the hierarchy
        known_names: set[str] = set()
        for base_type, impls in hierarchy.items():
            known_names.add(base_type)
            for impl in impls:
                if isinstance(impl, dict) and "class" in impl:
                    known_names.add(impl["class"])

        if not known_names:
            return ""

        # Try close matches (edit distance), excluding the exact query symbol
        matches = [m for m in get_close_matches(symbol, sorted(known_names), n=6, cutoff=0.5)
                   if m != symbol][:5]

        # Also try substring matching as a fallback
        if len(matches) < 3:
            lower = symbol.lower()
            substring_matches = [n for n in sorted(known_names)
                                 if lower in n.lower() and n not in matches
                                 and n != symbol][:5 - len(matches)]
            matches.extend(substring_matches)

        if not matches:
            return ""

        return "Did you mean: " + ", ".join(matches) + "?"
    except Exception:
        return ""
