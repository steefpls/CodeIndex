"""Unity prefab and scene file chunking.

Parses Unity's custom multi-document YAML format (prefab/scene files) into
CodeChunk objects for embedding and search. Uses regex-based document splitting
(not PyYAML) since Unity YAML uses non-standard !u! tags and scene files can
be 80K+ lines.

Unlike other chunkers, this module needs an external guid_map to resolve
script GUIDs to human-readable class names.
"""

import logging
import re
from dataclasses import dataclass, field

from src.models.chunk import CodeChunk

logger = logging.getLogger(__name__)

# Unity class ID -> human-readable type name (common built-in components)
_UNITY_CLASS_IDS: dict[int, str] = {
    1: "GameObject",
    4: "Transform",
    20: "Camera",
    23: "MeshRenderer",
    25: "Renderer",
    33: "MeshFilter",
    54: "Rigidbody",
    64: "MeshCollider",
    65: "BoxCollider",
    81: "AudioListener",
    82: "AudioSource",
    108: "Light",
    111: "Animation",
    114: "MonoBehaviour",
    120: "LineRenderer",
    124: "Behaviour",
    135: "SphereCollider",
    136: "CapsuleCollider",
    137: "SkinnedMeshRenderer",
    198: "ParticleSystem",
    212: "SpriteRenderer",
    222: "CanvasRenderer",
    223: "Canvas",
    224: "RectTransform",
    225: "CanvasGroup",
}

# Scene-level settings class IDs to skip (not useful for search)
_SCENE_SETTINGS_IDS: set[int] = {
    29,   # OcclusionCullingSettings
    104,  # RenderSettings
    127,  # LevelGameManager
    157,  # LightmapSettings
    196,  # NavMeshSettings
    1001, # PrefabInstance (handled separately)
}

# Fields in MonoBehaviour/component blocks that are Unity internal noise
_INTERNAL_FIELDS: set[str] = {
    "m_ObjectHideFlags", "m_CorrespondingSourceObject", "m_PrefabInstance",
    "m_PrefabAsset", "m_GameObject", "m_Enabled", "m_EditorHideFlags",
    "m_Script", "m_Name", "m_EditorClassIdentifier", "serializedVersion",
}

# Thresholds
_LARGE_FILE_LINE_THRESHOLD = 500
_DEGRADED_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB: switch to degraded parsing
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB: hard skip (corrupt/binary)
_MAX_GO_CHUNKS = 200  # cap per-GO chunks for very large scenes

# Regex patterns
_DOC_SEPARATOR_RE = re.compile(r"^---\s+!u!(\d+)\s+&(\d+)(?:\s+stripped)?\s*$")
_SCRIPT_GUID_RE = re.compile(r"guid:\s*([0-9a-f]{32})")
_FILEID_RE = re.compile(r"\{fileID:\s*(\d+)\}")
_COMPONENT_RE = re.compile(r"-\s*component:\s*\{fileID:\s*(\d+)\}")
_CHILDREN_RE = re.compile(r"-\s*\{fileID:\s*(\d+)\}")


@dataclass
class _UnityDoc:
    """A single YAML document from a Unity file (one component/object)."""
    class_id: int
    file_id: str
    type_name: str
    start_line: int
    end_line: int
    raw_text: str
    is_stripped: bool = False


@dataclass
class _GameObject:
    """Parsed GameObject with its components."""
    file_id: str
    name: str
    component_ids: list[str] = field(default_factory=list)
    start_line: int = 0


@dataclass
class _TransformInfo:
    """Parsed Transform with hierarchy info."""
    file_id: str
    go_file_id: str
    children_ids: list[str] = field(default_factory=list)
    parent_id: str = "0"


@dataclass
class _MonoBehaviourInfo:
    """Parsed MonoBehaviour with script reference and custom properties."""
    file_id: str
    go_file_id: str
    script_guid: str = ""
    script_name: str = ""  # resolved from guid_map
    script_file_path: str = ""  # relative path to .cs file
    properties: dict[str, str] = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0


def chunk_file_unity(source: bytes, rel_path: str, module: str,
                     guid_map: dict[str, tuple[str, str]] | None = None) -> list[CodeChunk]:
    """Parse a Unity prefab/scene file and return code chunks.

    Args:
        source: Raw file bytes.
        rel_path: Relative path from repo root (forward slashes).
        module: Module name (from .asmdef detection or directory).
        guid_map: Optional GUID -> (class_name, file_path) mapping for script resolution.

    Returns:
        List of CodeChunk objects (prefab_summary + optional gameobject chunks).
    """
    if rel_path.endswith(".asset"):
        return _chunk_scriptable_object(source, rel_path, module, guid_map or {})

    if len(source) > _MAX_FILE_SIZE_BYTES:
        logger.warning("Skipping extremely large file (%d bytes): %s", len(source), rel_path)
        return []

    degraded = len(source) > _DEGRADED_FILE_SIZE_BYTES
    if degraded:
        logger.info("Degraded mode for large file (%d bytes): %s", len(source), rel_path)

    guid_map = guid_map or {}

    try:
        text = source.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to decode %s: %s", rel_path, e)
        return []

    lines = text.split("\n")
    total_lines = len(lines)
    is_scene = rel_path.endswith(".unity")

    # Split into documents
    docs = _split_documents(lines)
    if not docs:
        return []

    # Parse documents into structured data (two-pass: build lookups first,
    # then extract MonoBehaviour properties so cross-references can resolve)
    gameobjects: dict[str, _GameObject] = {}
    transforms: dict[str, _TransformInfo] = {}
    monobehaviours: list[_MonoBehaviourInfo] = []
    component_types: dict[str, str] = {}  # file_id -> type name (for built-in components)
    mb_docs: list[_UnityDoc] = []  # deferred for second pass

    # Pass 1: Build all GameObject and Transform lookups
    for doc in docs:
        if doc.class_id in _SCENE_SETTINGS_IDS:
            continue
        if doc.is_stripped:
            if doc.class_id == 1:
                name = _extract_field(doc.raw_text, "m_Name")
                if name:
                    gameobjects[doc.file_id] = _GameObject(
                        file_id=doc.file_id, name=name, start_line=doc.start_line)
            continue

        if doc.class_id == 1:  # GameObject
            name = _extract_field(doc.raw_text, "m_Name") or "Unnamed"
            comp_ids = _COMPONENT_RE.findall(doc.raw_text)
            gameobjects[doc.file_id] = _GameObject(
                file_id=doc.file_id, name=name,
                component_ids=comp_ids, start_line=doc.start_line)

        elif doc.class_id in (4, 224):  # Transform or RectTransform
            go_id = _extract_fileid(doc.raw_text, "m_GameObject")
            children = _CHILDREN_RE.findall(
                _extract_list_section(doc.raw_text, "m_Children"))
            parent = _extract_fileid(doc.raw_text, "m_Father")
            transforms[doc.file_id] = _TransformInfo(
                file_id=doc.file_id, go_file_id=go_id,
                children_ids=children, parent_id=parent)
            component_types[doc.file_id] = doc.type_name

        elif doc.class_id == 114:  # MonoBehaviour — defer property extraction
            mb_docs.append(doc)

        else:
            # Built-in component (Camera, Light, MeshRenderer, etc.)
            component_types[doc.file_id] = doc.type_name

    # Pass 2: Process MonoBehaviours now that all GOs/Transforms are known
    for doc in mb_docs:
        go_id = _extract_fileid(doc.raw_text, "m_GameObject")
        script_line = _extract_field(doc.raw_text, "m_Script") or ""
        script_guid = ""
        script_name = ""
        script_file_path = ""
        guid_match = _SCRIPT_GUID_RE.search(script_line)
        if guid_match:
            script_guid = guid_match.group(1)
            guid_entry = guid_map.get(script_guid)
            if guid_entry is not None:
                script_name, script_file_path = guid_entry
            else:
                script_name = f"[{script_guid[:8]}...]"

        props = _extract_custom_properties(doc.raw_text, guid_map,
                                            gameobjects, transforms)
        monobehaviours.append(_MonoBehaviourInfo(
            file_id=doc.file_id, go_file_id=go_id,
            script_guid=script_guid, script_name=script_name,
            script_file_path=script_file_path,
            properties=props, start_line=doc.start_line, end_line=doc.end_line))
        component_types[doc.file_id] = script_name or "MonoBehaviour"

    # Build GO -> Transform mapping
    go_transform: dict[str, _TransformInfo] = {}
    for t in transforms.values():
        if t.go_file_id != "0":
            go_transform[t.go_file_id] = t

    # Find root GameObjects (Transform with m_Father: {fileID: 0})
    root_go_ids = []
    for go_id, go in gameobjects.items():
        t = go_transform.get(go_id)
        if t and t.parent_id == "0":
            root_go_ids.append(go_id)

    # If no root found (e.g., all stripped), use the first GO
    if not root_go_ids and gameobjects:
        root_go_ids = [next(iter(gameobjects))]

    # Determine file type name and root name
    file_type = "Scene" if is_scene else "Prefab"
    if root_go_ids and root_go_ids[0] in gameobjects:
        root_name = gameobjects[root_go_ids[0]].name
    else:
        # Fallback to filename
        root_name = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    # Build hierarchy tree
    hierarchy = _build_hierarchy(root_go_ids, gameobjects, go_transform, transforms)

    # Group MonoBehaviours by GO
    go_monobehaviours: dict[str, list[_MonoBehaviourInfo]] = {}
    for mb in monobehaviours:
        go_monobehaviours.setdefault(mb.go_file_id, []).append(mb)

    # Build component list per GO (resolved names)
    go_components: dict[str, list[str]] = {}
    for go_id, go in gameobjects.items():
        comps = []
        for cid in go.component_ids:
            ctype = component_types.get(cid)
            if ctype and ctype not in ("Transform", "RectTransform"):
                comps.append(ctype)
        go_components[go_id] = comps

    # Collect all script names for base_types
    all_script_names = [mb.script_name for mb in monobehaviours if mb.script_name and not mb.script_name.startswith("[")]

    # Derive namespace from directory path
    parts = rel_path.replace("\\", "/").split("/")
    # Find the source_dir portion (Prefabs, Scenes, etc.) and take subdirs
    namespace = ""
    for i, p in enumerate(parts):
        if p in ("Prefabs", "Prefab", "Scenes"):
            namespace = "/".join(parts[i:len(parts) - 1])
            break

    # --- Produce chunks ---
    chunks: list[CodeChunk] = []
    is_large = total_lines > _LARGE_FILE_LINE_THRESHOLD
    has_scripts = bool(go_monobehaviours)

    # For large files with scripts: compact summary (hierarchy only) + per-GO detail chunks.
    # For small files or files without scripts: single combined summary.
    if is_large and has_scripts:
        # 1. Compact prefab_summary: hierarchy + component list only (no script configs)
        compact_source = _format_compact_summary(
            file_type, root_name, rel_path, hierarchy, go_components)
        chunks.append(CodeChunk(
            file_path=rel_path,
            class_name=root_name,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=compact_source,
            chunk_type="prefab_summary",
            module=module,
            base_types=all_script_names[:20],
        ))

        # 2. Per-GO chunks for GOs with MonoBehaviours (capped at 200)
        go_items = list(go_monobehaviours.items())[:_MAX_GO_CHUNKS]
        for go_id, mbs in go_items:
            if go_id not in gameobjects:
                continue
            go = gameobjects[go_id]
            go_source = _format_gameobject_chunk(
                go, go_components.get(go_id, []), mbs, root_name, gameobjects)
            chunks.append(CodeChunk(
                file_path=rel_path,
                class_name=root_name,
                method_name=go.name,
                namespace=namespace,
                start_line=go.start_line,
                end_line=mbs[-1].end_line if mbs else go.start_line,
                source=go_source,
                chunk_type="gameobject",
                module=module,
                base_types=[mb.script_name for mb in mbs if mb.script_name and not mb.script_name.startswith("[")],
            ))

        if len(go_monobehaviours) > _MAX_GO_CHUNKS:
            logger.info("Capped GO chunks at %d for %s (%d total GOs with scripts)",
                        _MAX_GO_CHUNKS, rel_path, len(go_monobehaviours))
    else:
        # Single combined summary for small files or files without scripts
        summary_source = _format_summary(
            file_type, root_name, rel_path, hierarchy,
            go_components, go_monobehaviours, gameobjects)
        chunks.append(CodeChunk(
            file_path=rel_path,
            class_name=root_name,
            method_name=None,
            namespace=namespace,
            start_line=1,
            end_line=total_lines,
            source=summary_source,
            chunk_type="prefab_summary",
            module=module,
            base_types=all_script_names[:20],
        ))

    return chunks


# --- Document splitting ---

def _split_documents(lines: list[str]) -> list[_UnityDoc]:
    """Split a Unity YAML file into documents by --- !u! separators."""
    docs: list[_UnityDoc] = []
    current_class_id = -1
    current_file_id = ""
    current_start = 0
    current_lines: list[str] = []
    current_stripped = False

    for i, line in enumerate(lines):
        match = _DOC_SEPARATOR_RE.match(line)
        if match:
            # Save previous document
            if current_class_id >= 0:
                docs.append(_UnityDoc(
                    class_id=current_class_id,
                    file_id=current_file_id,
                    type_name=_UNITY_CLASS_IDS.get(current_class_id, f"Component_{current_class_id}"),
                    start_line=current_start + 1,
                    end_line=i,
                    raw_text="\n".join(current_lines),
                    is_stripped=current_stripped,
                ))
            current_class_id = int(match.group(1))
            current_file_id = match.group(2)
            current_start = i
            current_lines = []
            current_stripped = "stripped" in line
        elif current_class_id >= 0:
            current_lines.append(line)

    # Save last document
    if current_class_id >= 0:
        docs.append(_UnityDoc(
            class_id=current_class_id,
            file_id=current_file_id,
            type_name=_UNITY_CLASS_IDS.get(current_class_id, f"Component_{current_class_id}"),
            start_line=current_start + 1,
            end_line=len(lines),
            raw_text="\n".join(current_lines),
            is_stripped=current_stripped,
        ))

    return docs


# --- Field extraction ---

def _extract_field(text: str, field_name: str) -> str | None:
    """Extract a simple top-level field value from a YAML text block.

    Matches "  field_name: value" at 2-space indent (Unity standard indent).
    Returns None if not found.
    """
    pattern = re.compile(rf"^\s{{2}}{re.escape(field_name)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_fileid(text: str, field_name: str) -> str:
    """Extract a fileID reference from a field like 'm_GameObject: {fileID: 123}'."""
    val = _extract_field(text, field_name)
    if val:
        fid_match = _FILEID_RE.search(val)
        if fid_match:
            return fid_match.group(1)
    return "0"


def _extract_list_section(text: str, field_name: str) -> str:
    """Extract the list section following a field (e.g., m_Children: followed by - items)."""
    pattern = re.compile(
        rf"^\s{{2}}{re.escape(field_name)}:\s*\n((?:\s{{2}}-\s.*\n?)*)",
        re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return ""


def _extract_custom_properties(text: str, guid_map: dict[str, tuple[str, str]],
                                gameobjects: dict[str, _GameObject],
                                transforms: dict[str, _TransformInfo]) -> dict[str, str]:
    """Extract custom serialized properties from a MonoBehaviour document.

    Returns properties that appear after m_EditorClassIdentifier: line,
    excluding Unity internal fields and nested sub-objects.
    """
    props: dict[str, str] = {}

    # Find the m_EditorClassIdentifier line
    marker = "m_EditorClassIdentifier:"
    marker_idx = text.find(marker)
    if marker_idx < 0:
        return props

    # Everything after the marker
    after = text[marker_idx + len(marker):]
    lines = after.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Only take top-level properties (2 or 4 spaces indent in the raw text,
        # which after the document start is typically 2 extra spaces)
        # We want lines that look like "  PropertyName: value" (not deeper nested)
        if line.startswith("    ") and not line.startswith("      "):
            # This is a top-level property line (4-space indent = 2 for YAML + 2 for property)
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()

                if key in _INTERNAL_FIELDS or key.startswith("m_"):
                    continue

                # Resolve fileID references in values
                val = _resolve_value(val, gameobjects, transforms)

                # Skip empty/zero/default values
                if val in ("", "0", "{fileID: 0}", "[]"):
                    continue

                props[key] = val
        elif line.startswith("  ") and not line.startswith("    "):
            # Same check for 2-space indent (some MonoBehaviours)
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()

                if key in _INTERNAL_FIELDS or key.startswith("m_"):
                    continue

                val = _resolve_value(val, gameobjects, transforms)

                if val in ("", "0", "{fileID: 0}", "[]"):
                    continue

                props[key] = val

    return props


def _resolve_value(val: str, gameobjects: dict[str, _GameObject],
                   transforms: dict[str, _TransformInfo]) -> str:
    """Resolve fileID references in property values to GO names."""
    fid_match = _FILEID_RE.search(val)
    if fid_match:
        fid = fid_match.group(1)
        if fid != "0":
            # Try direct GO lookup
            if fid in gameobjects:
                return f"-> {gameobjects[fid].name}"
            # Try finding GO via transform
            if fid in transforms:
                go_id = transforms[fid].go_file_id
                if go_id in gameobjects:
                    return f"-> {gameobjects[go_id].name}"
    return val


# --- Hierarchy building ---

def _build_hierarchy(root_ids: list[str], gameobjects: dict[str, _GameObject],
                     go_transform: dict[str, _TransformInfo],
                     transforms: dict[str, _TransformInfo]) -> list[tuple[str, str, int]]:
    """Build a flattened hierarchy tree: list of (go_id, go_name, depth) tuples.

    Uses stable Unity fileIDs instead of names to avoid duplicate-name collisions.
    """
    result: list[tuple[str, str, int]] = []
    visited: set[str] = set()

    def _walk(go_id: str, depth: int) -> None:
        if go_id in visited or go_id not in gameobjects:
            return
        visited.add(go_id)
        go = gameobjects[go_id]
        result.append((go_id, go.name, depth))

        # Find children via Transform
        t = go_transform.get(go_id)
        if t:
            for child_tid in t.children_ids:
                # child_tid is a Transform fileID -> find its GO
                child_t = transforms.get(child_tid)
                if child_t and child_t.go_file_id in gameobjects:
                    _walk(child_t.go_file_id, depth + 1)

    for root_id in root_ids:
        _walk(root_id, 0)

    return result


# --- Formatting ---

def _format_summary(file_type: str, root_name: str, rel_path: str,
                    hierarchy: list[tuple[str, str, int]],
                    go_components: dict[str, list[str]],
                    go_monobehaviours: dict[str, list[_MonoBehaviourInfo]],
                    gameobjects: dict[str, _GameObject]) -> str:
    """Format the prefab_summary pseudo-code for embedding."""
    lines: list[str] = []
    lines.append(f"// {file_type}: {root_name}")
    lines.append(f"// File: {rel_path}")
    lines.append("")

    safe_name = re.sub(r"[^a-zA-Z0-9]", "", root_name)
    lines.append(f"class {safe_name}_{file_type} {{")

    # Hierarchy with components
    if hierarchy:
        lines.append("  // Hierarchy")
        for go_id, go_name, depth in hierarchy:
            indent = "  " * (depth + 1)
            comps = go_components.get(go_id, [])
            comp_str = f" [{', '.join(comps)}]" if comps else ""
            lines.append(f"{indent}GameObject {go_name}{comp_str}")

    # Script configurations (MonoBehaviour properties)
    mb_sections: list[str] = []
    for go_id, mbs in go_monobehaviours.items():
        go_name = gameobjects[go_id].name if go_id in gameobjects else "Unknown"
        for mb in mbs:
            if not mb.properties or mb.script_name.startswith("["):
                continue
            props_str = "; ".join(f"{k} = {v}" for k, v in list(mb.properties.items())[:10])
            mb_sections.append(f"  {mb.script_name} on {go_name} {{ {props_str} }}")
            if mb.script_file_path:
                mb_sections.append(f"    // -> {mb.script_file_path}")

    if mb_sections:
        lines.append("")
        lines.append("  // Script configurations")
        lines.extend(mb_sections)

    lines.append("}")
    return "\n".join(lines)


def _format_compact_summary(file_type: str, root_name: str, rel_path: str,
                            hierarchy: list[tuple[str, str, int]],
                            go_components: dict[str, list[str]]) -> str:
    """Format a compact prefab_summary for large files (hierarchy + components only, no script configs).

    Script details are in the per-GO gameobject chunks instead.
    """
    lines: list[str] = []
    lines.append(f"// {file_type}: {root_name}")
    lines.append(f"// File: {rel_path}")
    lines.append("")

    safe_name = re.sub(r"[^a-zA-Z0-9]", "", root_name)
    lines.append(f"class {safe_name}_{file_type} {{")

    if hierarchy:
        lines.append("  // Hierarchy")
        for go_id, go_name, depth in hierarchy:
            indent = "  " * (depth + 1)
            comps = go_components.get(go_id, [])
            comp_str = f" [{', '.join(comps)}]" if comps else ""
            lines.append(f"{indent}GameObject {go_name}{comp_str}")

    lines.append("}")
    return "\n".join(lines)


def _format_gameobject_chunk(go: _GameObject, components: list[str],
                              monobehaviours: list[_MonoBehaviourInfo],
                              root_name: str,
                              gameobjects: dict[str, _GameObject]) -> str:
    """Format a per-GameObject chunk."""
    lines: list[str] = []
    lines.append(f"// GameObject: {go.name}")
    lines.append(f"// In: {root_name}")
    lines.append("")

    safe_name = re.sub(r"[^a-zA-Z0-9]", "", go.name)
    lines.append(f"class {safe_name}_GameObject {{")

    # List all component types
    if components:
        lines.append(f"  components: [{', '.join(components)}]")

    # MonoBehaviour details
    for mb in monobehaviours:
        if mb.script_name.startswith("["):
            continue
        lines.append("")
        lines.append(f"  {mb.script_name} {{")
        if mb.script_file_path:
            lines.append(f"    // -> {mb.script_file_path}")
        for key, val in list(mb.properties.items())[:15]:
            lines.append(f"    {key} = {val}")
        lines.append("  }")

    lines.append("}")
    return "\n".join(lines)



def _chunk_scriptable_object(source: bytes, rel_path: str, module: str,
                              guid_map: dict[str, tuple[str, str]]) -> list[CodeChunk]:
    """Parse a Unity .asset file (ScriptableObject) into a single chunk.

    .asset files are multi-document YAML like prefabs. We look for a MonoBehaviour
    document (class_id 114) with a custom script GUID. System .asset files (no
    custom MonoBehaviour) are skipped.
    """
    try:
        text = source.decode("utf-8", errors="replace")
    except Exception:
        return []

    lines = text.split("\n")
    docs = _split_documents(lines)
    if not docs:
        return []

    # Find the MonoBehaviour doc with a custom script
    for doc in docs:
        if doc.class_id != 114:
            continue
        script_line = _extract_field(doc.raw_text, "m_Script") or ""
        guid_match = _SCRIPT_GUID_RE.search(script_line)
        if not guid_match:
            continue
        script_guid = guid_match.group(1)
        guid_entry = guid_map.get(script_guid)
        if guid_entry is None:
            continue  # system script, skip

        class_name, script_file_path = guid_entry
        name = _extract_field(doc.raw_text, "m_Name") or rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        # Extract custom properties
        props = {}
        marker = "m_EditorClassIdentifier:"
        marker_idx = doc.raw_text.find(marker)
        if marker_idx >= 0:
            after = doc.raw_text[marker_idx + len(marker):]
            for line in after.split("\n"):
                stripped = line.strip()
                if not stripped or ":" not in stripped:
                    continue
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()
                if key in _INTERNAL_FIELDS or key.startswith("m_"):
                    continue
                if val in ("", "0", "{fileID: 0}", "[]"):
                    continue
                props[key] = val

        # Format as pseudo-code
        source_lines = [
            f"// ScriptableObject: {name}",
            f"// Type: {class_name}",
            f"// File: {rel_path}",
            f"// Script: {script_file_path}",
            "",
            f"class {class_name}_Asset {{",
        ]
        for k, v in list(props.items())[:30]:
            source_lines.append(f"  {k} = {v}")
        source_lines.append("}")

        return [CodeChunk(
            file_path=rel_path,
            class_name=class_name,
            method_name=None,
            namespace="",
            start_line=doc.start_line,
            end_line=doc.end_line,
            source="\n".join(source_lines),
            chunk_type="scriptable_object",
            module=module,
            base_types=[class_name],
        )]

    return []  # No custom MonoBehaviour found — system .asset file
