"""Parse Zenject installer files for DI binding information."""

import re
import logging
from pathlib import Path

from src.models.binding import ZenjectBinding

logger = logging.getLogger(__name__)

# Patterns for common Zenject binding calls.
# Applied against whitespace-collapsed statements so multi-line fluent chains match.
BIND_PATTERN = re.compile(
    r"""
    (?:Container|container)\s*\.\s*
    (?P<bind_type>
        Bind<(?P<bind_iface>[^>]+)>\s*\(\s*\)\s*\.\s*To<(?P<bind_concrete>[^>]+)>
        |BindInterfacesTo<(?P<bit_concrete>[^>]+)>
        |BindInterfacesAndSelfTo<(?P<biast_concrete>[^>]+)>
        |BindFactory<(?P<factory_type>[^>]+),\s*(?P<factory_class>[^>]+)>
        |DeclareSignal<(?P<signal_type>[^>]+)>
    )
    """,
    re.VERBOSE,
)

SCOPE_PATTERN = re.compile(r"\.\s*(AsSingle|AsTransient|AsCached|NonLazy)\s*\(\s*\)")

# Matches the start of a Container binding statement (used to find statement boundaries).
# Matches both "Container." on one line and standalone "Container" at end of line
# (the dot comes on the next continuation line).
_STATEMENT_START = re.compile(r"(?:Container|container)\s*(?:\.|$)")


def _collapse_statements(source: str) -> list[tuple[str, int]]:
    """Collapse multi-line C# statements into single lines for regex matching.

    Returns a list of (collapsed_statement, original_line_number) tuples.
    Each statement runs from a 'Container.' call to the next semicolon.
    """
    lines = source.split("\n")
    results = []
    current_stmt = []
    stmt_start_line = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if _STATEMENT_START.search(stripped):
            # Start of a new statement — flush any previous incomplete one
            if current_stmt:
                joined = " ".join(current_stmt)
                results.append((joined, stmt_start_line))
                current_stmt = []
            stmt_start_line = i + 1
            current_stmt.append(stripped)
        elif current_stmt:
            # Continuation of a multi-line statement
            current_stmt.append(stripped)

        # Check if statement ended (semicolon)
        if current_stmt and ";" in stripped:
            joined = " ".join(current_stmt)
            results.append((joined, stmt_start_line))
            current_stmt = []

    # Flush any trailing statement without semicolon
    if current_stmt:
        joined = " ".join(current_stmt)
        results.append((joined, stmt_start_line))

    return results


def parse_installer_file(file_path: Path) -> list[ZenjectBinding]:
    """Parse a single C# installer file for Zenject bindings."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read %s: %s", file_path, e)
        return []

    bindings = []

    for statement, line_num in _collapse_statements(source):
        for match in BIND_PATTERN.finditer(statement):
            binding = _extract_binding(match, str(file_path), line_num)
            if binding:
                scope_match = SCOPE_PATTERN.search(statement)
                if scope_match:
                    binding.scope = scope_match.group(1)
                bindings.append(binding)

    return bindings


def _extract_binding(match: re.Match, file_path: str, line_number: int) -> ZenjectBinding | None:
    """Extract a ZenjectBinding from a regex match."""
    groups = match.groupdict()

    # Bind<I>().To<C>()
    if groups.get("bind_iface"):
        return ZenjectBinding(
            interface_name=groups["bind_iface"].strip(),
            concrete_type=groups["bind_concrete"].strip(),
            scope="",
            binding_type="Bind",
            installer_file=file_path,
            line_number=line_number,
        )

    # BindInterfacesTo<C>() — binds all interfaces the concrete type implements.
    # We store the concrete type as interface_name since the actual interfaces
    # aren't known from the installer file alone.
    if groups.get("bit_concrete"):
        concrete = groups["bit_concrete"].strip()
        return ZenjectBinding(
            interface_name=concrete,
            concrete_type=concrete,
            scope="",
            binding_type="BindInterfacesTo",
            installer_file=file_path,
            line_number=line_number,
        )

    # BindInterfacesAndSelfTo<C>()
    if groups.get("biast_concrete"):
        concrete = groups["biast_concrete"].strip()
        return ZenjectBinding(
            interface_name=concrete,
            concrete_type=concrete,
            scope="",
            binding_type="BindInterfacesAndSelfTo",
            installer_file=file_path,
            line_number=line_number,
        )

    # BindFactory<T, F>()
    if groups.get("factory_type"):
        return ZenjectBinding(
            interface_name=groups["factory_class"].strip(),
            concrete_type=groups["factory_type"].strip(),
            scope="",
            binding_type="BindFactory",
            installer_file=file_path,
            line_number=line_number,
        )

    # DeclareSignal<S>()
    if groups.get("signal_type"):
        signal = groups["signal_type"].strip()
        return ZenjectBinding(
            interface_name=signal,
            concrete_type=signal,
            scope="",
            binding_type="DeclareSignal",
            installer_file=file_path,
            line_number=line_number,
        )

    return None


def find_installer_files(source_dir: Path) -> list[Path]:
    """Find all Zenject installer files in a directory."""
    installers = []
    for cs_file in source_dir.rglob("*.cs"):
        name = cs_file.stem.lower()
        if "installer" in name:
            installers.append(cs_file)
    return sorted(installers)


def parse_all_installers(source_dir: Path) -> list[ZenjectBinding]:
    """Parse all installer files in a directory tree."""
    all_bindings = []
    for installer_file in find_installer_files(source_dir):
        bindings = parse_installer_file(installer_file)
        if bindings:
            logger.info("Found %d bindings in %s", len(bindings), installer_file.name)
            all_bindings.extend(bindings)
    return all_bindings
