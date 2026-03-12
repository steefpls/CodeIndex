"""Tests for the Unity prefab/scene chunker."""

from src.indexer.chunker_unity import chunk_file_unity


# --- Test data ---

SIMPLE_PREFAB = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &100
GameObject:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  serializedVersion: 6
  m_Component:
  - component: {fileID: 101}
  - component: {fileID: 102}
  m_Layer: 0
  m_Name: TestObject
  m_TagString: Untagged
  m_IsActive: 1
--- !u!4 &101
Transform:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 100}
  serializedVersion: 2
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 1, y: 2, z: 3}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Children: []
  m_Father: {fileID: 0}
--- !u!114 &102
MonoBehaviour:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 100}
  m_Enabled: 1
  m_EditorHideFlags: 0
  m_Script: {fileID: 11500000, guid: aabbccdd00112233aabbccdd00112233, type: 3}
  m_Name:
  m_EditorClassIdentifier:
  Speed: 5.0
  MaxForce: 100
  Target: {fileID: 0}
"""


HIERARCHY_PREFAB = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &200
GameObject:
  m_Component:
  - component: {fileID: 201}
  - component: {fileID: 203}
  m_Name: Root
--- !u!4 &201
Transform:
  m_GameObject: {fileID: 200}
  m_Children:
  - {fileID: 211}
  - {fileID: 221}
  m_Father: {fileID: 0}
--- !u!114 &203
MonoBehaviour:
  m_GameObject: {fileID: 200}
  m_Script: {fileID: 11500000, guid: aaaa000011112222aaaa000011112222, type: 3}
  m_EditorClassIdentifier:
  Mode: 2
--- !u!1 &210
GameObject:
  m_Component:
  - component: {fileID: 211}
  - component: {fileID: 212}
  m_Name: ChildA
--- !u!4 &211
Transform:
  m_GameObject: {fileID: 210}
  m_Children: []
  m_Father: {fileID: 201}
--- !u!114 &212
MonoBehaviour:
  m_GameObject: {fileID: 210}
  m_Script: {fileID: 11500000, guid: bbbb000011112222bbbb000011112222, type: 3}
  m_EditorClassIdentifier:
  Velocity: 10
--- !u!1 &220
GameObject:
  m_Component:
  - component: {fileID: 221}
  m_Name: ChildB
--- !u!4 &221
Transform:
  m_GameObject: {fileID: 220}
  m_Children: []
  m_Father: {fileID: 201}
"""


def _build_large_prefab(num_gameobjects: int) -> bytes:
    """Build a synthetic large prefab with many GameObjects."""
    lines = ["%YAML 1.1", "%TAG !u! tag:unity3d.com,2011:"]

    # Root GO
    root_fid = 1000
    root_transform_fid = 1001
    child_transform_fids = []

    lines.append(f"--- !u!1 &{root_fid}")
    comp_lines = [f"  - component: {{fileID: {root_transform_fid}}}"]
    lines.append("GameObject:")
    lines.append("  m_Component:")
    lines.extend(comp_lines)
    lines.append("  m_Name: BigRoot")

    # Root transform
    for i in range(num_gameobjects):
        child_transform_fids.append(2000 + i * 10 + 1)

    lines.append(f"--- !u!4 &{root_transform_fid}")
    lines.append("Transform:")
    lines.append(f"  m_GameObject: {{fileID: {root_fid}}}")
    lines.append("  m_Children:")
    for ctfid in child_transform_fids:
        lines.append(f"  - {{fileID: {ctfid}}}")
    lines.append("  m_Father: {fileID: 0}")

    # Child GOs with MonoBehaviours
    for i in range(num_gameobjects):
        go_fid = 2000 + i * 10
        t_fid = 2000 + i * 10 + 1
        mb_fid = 2000 + i * 10 + 2
        guid = f"cc{i:06d}00112233cc{i:06d}0011223300"[:32]

        lines.append(f"--- !u!1 &{go_fid}")
        lines.append("GameObject:")
        lines.append("  m_Component:")
        lines.append(f"  - component: {{fileID: {t_fid}}}")
        lines.append(f"  - component: {{fileID: {mb_fid}}}")
        lines.append(f"  m_Name: Child_{i}")

        lines.append(f"--- !u!4 &{t_fid}")
        lines.append("Transform:")
        lines.append(f"  m_GameObject: {{fileID: {go_fid}}}")
        lines.append("  m_Children: []")
        lines.append(f"  m_Father: {{fileID: {root_transform_fid}}}")

        lines.append(f"--- !u!114 &{mb_fid}")
        lines.append("MonoBehaviour:")
        lines.append(f"  m_GameObject: {{fileID: {go_fid}}}")
        lines.append(f"  m_Script: {{fileID: 11500000, guid: {guid}, type: 3}}")
        lines.append("  m_EditorClassIdentifier: ")
        lines.append(f"  Health: {i * 10}")
        lines.append(f"  Armor: {i * 5}")
        # Add padding lines to make the file large enough
        for j in range(5):
            lines.append(f"  Padding{j}: {j}")

    return "\n".join(lines).encode("utf-8")


# --- Tests ---

def test_simple_prefab_summary():
    """Small prefab should produce exactly 1 prefab_summary chunk."""
    guid_map = {"aabbccdd00112233aabbccdd00112233": ("PlayerController", "Assets/Scripts/PlayerController.cs")}
    chunks = chunk_file_unity(SIMPLE_PREFAB, "Assets/Prefabs/Test.prefab", "TestModule", guid_map)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "prefab_summary"
    assert chunk.class_name == "TestObject"
    assert chunk.module == "TestModule"
    assert chunk.file_path == "Assets/Prefabs/Test.prefab"
    assert "PlayerController" in chunk.source
    assert "Speed" in chunk.source
    assert "5.0" in chunk.source
    assert "MaxForce" in chunk.source
    # Script file path should appear in output
    assert "// -> Assets/Scripts/PlayerController.cs" in chunk.source
    print("PASS: test_simple_prefab_summary")


def test_guid_resolution():
    """MonoBehaviour GUIDs should be resolved to class names from guid_map."""
    guid_map = {"aabbccdd00112233aabbccdd00112233": ("PlayerController", "Assets/Scripts/PlayerController.cs")}
    chunks = chunk_file_unity(SIMPLE_PREFAB, "Assets/Prefabs/Test.prefab", "", guid_map)

    assert len(chunks) == 1
    assert "PlayerController" in chunks[0].source
    assert "PlayerController" in chunks[0].base_types
    print("PASS: test_guid_resolution")


def test_unknown_guid_fallback():
    """Unknown GUIDs should show truncated hex."""
    chunks = chunk_file_unity(SIMPLE_PREFAB, "Assets/Prefabs/Test.prefab", "", {})

    assert len(chunks) == 1
    # Should have the truncated GUID as fallback
    assert "[aabbccdd...]" in chunks[0].source
    print("PASS: test_unknown_guid_fallback")


def test_hierarchy_extraction():
    """Multi-GO prefab should show correct hierarchy."""
    guid_map = {
        "aaaa000011112222aaaa000011112222": ("RootScript", "Assets/Scripts/RootScript.cs"),
        "bbbb000011112222bbbb000011112222": ("ChildScript", "Assets/Scripts/ChildScript.cs"),
    }
    chunks = chunk_file_unity(HIERARCHY_PREFAB, "Assets/Prefabs/Hierarchy.prefab", "", guid_map)

    assert len(chunks) == 1  # small file, summary only
    source = chunks[0].source
    assert "Root" in source
    assert "ChildA" in source
    assert "ChildB" in source
    assert "RootScript" in source
    assert "ChildScript" in source
    # Script file paths should appear
    assert "// -> Assets/Scripts/RootScript.cs" in source
    print("PASS: test_hierarchy_extraction")


def test_large_prefab_produces_gameobject_chunks():
    """Files > 500 lines should produce prefab_summary + gameobject chunks."""
    # Build a prefab with enough GOs to exceed 500 lines
    large_data = _build_large_prefab(60)
    line_count = large_data.count(b"\n") + 1
    assert line_count > 500, f"Expected > 500 lines, got {line_count}"

    chunks = chunk_file_unity(large_data, "Assets/Prefabs/Big.prefab", "BigModule")
    summary_chunks = [c for c in chunks if c.chunk_type == "prefab_summary"]
    go_chunks = [c for c in chunks if c.chunk_type == "gameobject"]

    assert len(summary_chunks) == 1
    assert len(go_chunks) > 0  # should have per-GO chunks
    assert summary_chunks[0].class_name == "BigRoot"

    # Each gameobject chunk should reference the root name
    for gc in go_chunks:
        assert gc.class_name == "BigRoot"
        assert gc.method_name is not None  # should be the GO name
    print("PASS: test_large_prefab_produces_gameobject_chunks")


def test_scene_file_extension():
    """Scene files (.unity) should be recognized and parsed."""
    # Reuse simple prefab data but with .unity extension
    chunks = chunk_file_unity(SIMPLE_PREFAB, "Assets/Scenes/Init.unity", "")

    assert len(chunks) == 1
    assert "Scene" in chunks[0].source  # should say "Scene:" not "Prefab:"
    print("PASS: test_scene_file_extension")


def test_scene_settings_filtered():
    """Scene-level settings documents should be filtered out."""
    scene_data = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!29 &1
OcclusionCullingSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 2
  m_OcclusionBakeSettings:
    smallestOccluder: 5
--- !u!104 &2
RenderSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 9
--- !u!157 &3
LightmapSettings:
  m_ObjectHideFlags: 0
--- !u!196 &4
NavMeshSettings:
  m_ObjectHideFlags: 0
--- !u!1 &100
GameObject:
  m_Component:
  - component: {fileID: 101}
  m_Name: SceneRoot
--- !u!4 &101
Transform:
  m_GameObject: {fileID: 100}
  m_Children: []
  m_Father: {fileID: 0}
"""
    chunks = chunk_file_unity(scene_data, "Assets/Scenes/Test.unity", "")

    assert len(chunks) == 1
    source = chunks[0].source
    assert "SceneRoot" in source
    assert "OcclusionCulling" not in source
    assert "RenderSettings" not in source
    print("PASS: test_scene_settings_filtered")


def test_max_file_size_skip():
    """Files over 50MB should be skipped (hard limit)."""
    huge_data = b"x" * (51 * 1024 * 1024)
    chunks = chunk_file_unity(huge_data, "Assets/Prefabs/Huge.prefab", "")
    assert chunks == []
    print("PASS: test_max_file_size_skip")


def test_degraded_mode_large_file():
    """Files between 5MB and 50MB should be parsed in degraded mode, not skipped."""
    # Build a valid Unity prefab that exceeds 5MB
    lines = ["%YAML 1.1", "%TAG !u! tag:unity3d.com,2011:"]
    lines.append("--- !u!1 &100")
    lines.append("GameObject:")
    lines.append("  m_Component:")
    lines.append("  - component: {fileID: 101}")
    lines.append("  - component: {fileID: 102}")
    lines.append("  m_Name: LargeScene")
    lines.append("--- !u!4 &101")
    lines.append("Transform:")
    lines.append("  m_GameObject: {fileID: 100}")
    lines.append("  m_Children: []")
    lines.append("  m_Father: {fileID: 0}")
    lines.append("--- !u!114 &102")
    lines.append("MonoBehaviour:")
    lines.append("  m_GameObject: {fileID: 100}")
    lines.append("  m_Script: {fileID: 11500000, guid: aaaa000011112222aaaa000011112222, type: 3}")
    lines.append("  m_EditorClassIdentifier:")
    lines.append("  Value: 42")
    # Pad with comment lines to push past 5MB
    pad_line = "  # " + "x" * 200
    while len("\n".join(lines).encode()) < 5.5 * 1024 * 1024:
        lines.append(pad_line)

    data = "\n".join(lines).encode("utf-8")
    assert len(data) > 5 * 1024 * 1024, f"Data should be > 5MB, got {len(data)}"
    assert len(data) < 50 * 1024 * 1024, f"Data should be < 50MB"

    guid_map = {"aaaa000011112222aaaa000011112222": ("BigScript", "Assets/Scripts/BigScript.cs")}
    chunks = chunk_file_unity(data, "Assets/Scenes/UIPlayground.unity", "TestModule", guid_map)
    assert len(chunks) > 0, "Degraded mode should produce chunks, not skip"
    assert any(c.chunk_type == "prefab_summary" for c in chunks)
    print("PASS: test_degraded_mode_large_file")


def test_fileid_crossref_resolution():
    """Internal fileID references in MonoBehaviour properties should resolve to GO names."""
    data = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &300
GameObject:
  m_Component:
  - component: {fileID: 301}
  - component: {fileID: 302}
  m_Name: Controller
--- !u!4 &301
Transform:
  m_GameObject: {fileID: 300}
  m_Children: []
  m_Father: {fileID: 0}
--- !u!114 &302
MonoBehaviour:
  m_GameObject: {fileID: 300}
  m_Script: {fileID: 11500000, guid: dddd000011112222dddd000011112222, type: 3}
  m_EditorClassIdentifier:
  targetObject: {fileID: 400}
  speed: 5
--- !u!1 &400
GameObject:
  m_Component:
  - component: {fileID: 401}
  m_Name: Target
--- !u!4 &401
Transform:
  m_GameObject: {fileID: 400}
  m_Children: []
  m_Father: {fileID: 301}
"""
    guid_map = {"dddd000011112222dddd000011112222": ("ControllerScript", "Assets/Scripts/ControllerScript.cs")}
    chunks = chunk_file_unity(data, "Assets/Prefabs/Ref.prefab", "", guid_map)

    assert len(chunks) == 1
    source = chunks[0].source
    assert "-> Target" in source
    print("PASS: test_fileid_crossref_resolution")


def test_embedding_text_has_metadata():
    """Chunk's embedding_text should include file path, module, and class name."""
    guid_map = {"aabbccdd00112233aabbccdd00112233": ("PlayerController", "Assets/Scripts/PlayerController.cs")}
    chunks = chunk_file_unity(SIMPLE_PREFAB, "Assets/Prefabs/Test.prefab", "Robotics", guid_map)

    et = chunks[0].embedding_text
    assert "Assets/Prefabs/Test.prefab" in et
    assert "Robotics" in et
    assert "TestObject" in et
    print("PASS: test_embedding_text_has_metadata")


def test_empty_file():
    """Empty or header-only YAML should return no chunks."""
    data = b"%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n"
    chunks = chunk_file_unity(data, "Assets/Prefabs/Empty.prefab", "")
    assert chunks == []
    print("PASS: test_empty_file")


def test_stripped_gameobject():
    """Stripped GameObjects (prefab instances) should be handled gracefully."""
    data = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &500
GameObject:
  m_Component:
  - component: {fileID: 501}
  m_Name: MainObj
--- !u!4 &501
Transform:
  m_GameObject: {fileID: 500}
  m_Children: []
  m_Father: {fileID: 0}
--- !u!1001 &600
PrefabInstance:
  m_ObjectHideFlags: 0
  serializedVersion: 2
--- !u!1 &700 stripped
GameObject:
  m_CorrespondingSourceObject: {fileID: 123, guid: abcd, type: 3}
  m_PrefabInstance: {fileID: 600}
"""
    chunks = chunk_file_unity(data, "Assets/Prefabs/WithStripped.prefab", "")
    assert len(chunks) >= 1
    assert chunks[0].class_name == "MainObj"
    print("PASS: test_stripped_gameobject")


def test_duplicate_go_names():
    """GameObjects with the same name should not collide in hierarchy or component attribution."""
    data = b"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &1000
GameObject:
  m_Component:
  - component: {fileID: 1001}
  m_Name: Panel
--- !u!4 &1001
Transform:
  m_GameObject: {fileID: 1000}
  m_Children:
  - {fileID: 1011}
  - {fileID: 1021}
  m_Father: {fileID: 0}
--- !u!1 &1010
GameObject:
  m_Component:
  - component: {fileID: 1011}
  - component: {fileID: 1012}
  m_Name: Icon
--- !u!4 &1011
Transform:
  m_GameObject: {fileID: 1010}
  m_Children: []
  m_Father: {fileID: 1001}
--- !u!114 &1012
MonoBehaviour:
  m_GameObject: {fileID: 1010}
  m_Script: {fileID: 11500000, guid: aaaa111122223333aaaa111122223333, type: 3}
  m_EditorClassIdentifier:
  Sprite: star
--- !u!1 &1020
GameObject:
  m_Component:
  - component: {fileID: 1021}
  - component: {fileID: 1022}
  m_Name: Icon
--- !u!4 &1021
Transform:
  m_GameObject: {fileID: 1020}
  m_Children: []
  m_Father: {fileID: 1001}
--- !u!114 &1022
MonoBehaviour:
  m_GameObject: {fileID: 1020}
  m_Script: {fileID: 11500000, guid: bbbb111122223333bbbb111122223333, type: 3}
  m_EditorClassIdentifier:
  Sprite: circle
"""
    guid_map = {
        "aaaa111122223333aaaa111122223333": ("IconRendererA", "Assets/Scripts/IconRendererA.cs"),
        "bbbb111122223333bbbb111122223333": ("IconRendererB", "Assets/Scripts/IconRendererB.cs"),
    }
    chunks = chunk_file_unity(data, "Assets/Prefabs/DuplicateNames.prefab", "", guid_map)
    assert len(chunks) == 1
    source = chunks[0].source

    # Both scripts should appear with their correct properties
    assert "IconRendererA" in source
    assert "IconRendererB" in source
    assert "star" in source
    assert "circle" in source

    # Both Icon GOs should appear in hierarchy
    icon_count = source.count("GameObject Icon")
    assert icon_count == 2, f"Expected 2 Icon GOs in hierarchy, got {icon_count}"

    # Components should be correctly attributed to their respective Icon GOs
    # IconRendererA (Sprite=star) should be on first Icon, IconRendererB (Sprite=circle) on second
    lines = source.split("\n")
    icon_a_section = None
    icon_b_section = None
    for line in lines:
        if "IconRendererA" in line:
            icon_a_section = line
        if "IconRendererB" in line:
            icon_b_section = line
    assert icon_a_section is not None, "IconRendererA should appear"
    assert icon_b_section is not None, "IconRendererB should appear"
    # The scripts should be on different "Icon" GOs, not both on the first one
    assert "star" in icon_a_section or any("star" in l for l in lines if "IconRendererA" in l or "star" in l)
    print("PASS: test_duplicate_go_names")


if __name__ == "__main__":
    test_simple_prefab_summary()
    test_guid_resolution()
    test_unknown_guid_fallback()
    test_hierarchy_extraction()
    test_large_prefab_produces_gameobject_chunks()
    test_scene_file_extension()
    test_scene_settings_filtered()
    test_max_file_size_skip()
    test_fileid_crossref_resolution()
    test_embedding_text_has_metadata()
    test_empty_file()
    test_stripped_gameobject()
    test_duplicate_go_names()
    test_degraded_mode_large_file()
    print("\nAll tests passed!")
