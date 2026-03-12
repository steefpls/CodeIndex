"""Integration tests for Unity indexing, sidecar freshness, and disambiguation.

End-to-end tests that exercise the full pipeline flow with synthetic Unity
data: chunking -> sidecar state -> materialization -> tool queries.

Run with: PYTHONPATH=. python tests/test_integration_unity_indexing.py
"""

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ── Sidecar state: full lifecycle ────────────────────────────────────────────

class TestSidecarStateLifecycle(unittest.TestCase):
    """End-to-end sidecar state: add files, remove files, rematerialize."""

    def test_add_remove_rematerialize(self):
        """Adding and removing files keeps sidecars consistent."""
        from src.indexer.sidecar_state import (
            _empty_state, set_hierarchy_contributions,
            set_dep_graph_contributions, set_asset_ref_contributions,
            remove_file_contributions, materialize_hierarchy,
            materialize_dep_graph, materialize_asset_refs,
        )

        state = _empty_state()

        # Add file A with hierarchy + dep_graph data
        set_hierarchy_contributions(state, "Scripts/A.cs", [
            ("class_summary", "ClassA", "Scripts/A.cs", "Mod", "Ns", ["IFoo", "MonoBehaviour"]),
        ])
        set_dep_graph_contributions(state, "Scripts/A.cs", {
            ("ClassA", "Scripts/A.cs"): {
                "module": "Mod", "namespace": "Ns", "refs": {"ClassB", "ClassC"},
            },
        })

        # Add file B
        set_hierarchy_contributions(state, "Scripts/B.cs", [
            ("class_summary", "ClassB", "Scripts/B.cs", "Mod", "Ns", ["IFoo"]),
        ])
        set_dep_graph_contributions(state, "Scripts/B.cs", {
            ("ClassB", "Scripts/B.cs"): {
                "module": "Mod", "namespace": "Ns", "refs": {"ClassA"},
            },
        })

        # Add asset file
        set_asset_ref_contributions(state, "Prefabs/X.prefab", {
            "guid-a": {"class_name": "ClassA", "script_path": "Scripts/ClassA.cs"},
            "guid-b": {"class_name": "ClassB", "script_path": "Scripts/ClassB.cs"},
        })

        # Materialize
        h = materialize_hierarchy(state)
        self.assertIn("IFoo", h)
        self.assertEqual(len(h["IFoo"]), 2)
        self.assertIn("MonoBehaviour", h)
        self.assertEqual(len(h["MonoBehaviour"]), 1)

        dg = materialize_dep_graph(state)
        self.assertIn("Ns.ClassA", dg["nodes"])
        self.assertIn("Ns.ClassB", dg["nodes"]["Ns.ClassA"]["deps"])
        self.assertIn("Ns.ClassA", dg["nodes"]["Ns.ClassB"]["deps"])

        ar = materialize_asset_refs(state)
        self.assertIn("ClassA", ar)
        self.assertIn("Prefabs/X.prefab", ar["ClassA"])

        # Remove file A (simulates file deletion during incremental reindex)
        remove_file_contributions(state, "Scripts/A.cs")

        # Re-materialize
        h2 = materialize_hierarchy(state)
        self.assertIn("IFoo", h2)
        self.assertEqual(len(h2["IFoo"]), 1)  # Only ClassB remains
        self.assertNotIn("MonoBehaviour", h2)

        dg2 = materialize_dep_graph(state)
        self.assertNotIn("Ns.ClassA", dg2["nodes"])
        self.assertIn("Ns.ClassB", dg2["nodes"])
        # ClassB's dep on ClassA should be filtered out (ClassA no longer in known classes)
        self.assertEqual(dg2["nodes"]["Ns.ClassB"]["deps"], [])

    def test_file_update_replaces_contributions(self):
        """Updating a file replaces its contributions without duplicating."""
        from src.indexer.sidecar_state import (
            _empty_state, set_hierarchy_contributions, materialize_hierarchy,
        )

        state = _empty_state()

        # Initial version of file
        set_hierarchy_contributions(state, "Scripts/X.cs", [
            ("class_summary", "OldClass", "Scripts/X.cs", "Mod", "", ["IOld"]),
        ])

        h = materialize_hierarchy(state)
        self.assertIn("IOld", h)

        # Update file (new class replaces old)
        set_hierarchy_contributions(state, "Scripts/X.cs", [
            ("class_summary", "NewClass", "Scripts/X.cs", "Mod", "", ["INew"]),
        ])

        h2 = materialize_hierarchy(state)
        self.assertNotIn("IOld", h2)
        self.assertIn("INew", h2)
        self.assertEqual(h2["INew"][0]["class"], "NewClass")


# ── Chunk payload store roundtrip ────────────────────────────────────────────

class TestChunkPayloadStoreIntegration(unittest.TestCase):
    """Integration test for payload store with realistic chunk data."""

    def test_store_and_retrieve_large_chunk(self):
        """Large chunks are stored and retrievable by chunk_id."""
        from src.indexer.chunk_payload_store import (
            load_payloads, save_payloads, update_payloads_for_file,
            remove_payloads_for_file, get_payload,
        )
        from src.models.chunk import CodeChunk, MAX_EMBED_CHARS

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch DATA_DIR to temp
            with patch("src.indexer.chunk_payload_store.DATA_DIR", Path(tmpdir)):
                # Create a chunk with source exceeding MAX_EMBED_CHARS
                big_source = "// Big chunk\n" + ("x" * (MAX_EMBED_CHARS + 500))
                chunk = CodeChunk(
                    file_path="Assets/Prefabs/Big.prefab",
                    class_name="BigRoot",
                    method_name="SomeGO",
                    namespace="",
                    start_line=1,
                    end_line=100,
                    source=big_source,
                    chunk_type="gameobject",
                )

                # Update payloads
                update_payloads_for_file("testrepo", "Assets/Prefabs/Big.prefab", [chunk])

                # Retrieve
                payload = get_payload("testrepo", chunk.chunk_id)
                self.assertIsNotNone(payload)
                self.assertEqual(payload, big_source)

                # Small chunk should NOT be stored
                small_chunk = CodeChunk(
                    file_path="Assets/Prefabs/Small.prefab",
                    class_name="SmallRoot",
                    method_name=None,
                    namespace="",
                    start_line=1,
                    end_line=10,
                    source="// Short",
                    chunk_type="prefab_summary",
                )
                update_payloads_for_file("testrepo", "Assets/Prefabs/Small.prefab", [small_chunk])
                self.assertIsNone(get_payload("testrepo", small_chunk.chunk_id))

                # Remove file payloads
                remove_payloads_for_file("testrepo", "Assets/Prefabs/Big.prefab")
                self.assertIsNone(get_payload("testrepo", chunk.chunk_id))


# ── Unity context retrieval tool ─────────────────────────────────────────────

class TestUnityContextRetrieval(unittest.TestCase):
    """Test get_unity_entity_context with payload store."""

    def test_retrieves_from_payload_store(self):
        """Tool should return full payload when available in store."""
        from src.tools.unity_context import get_unity_entity_context
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = RepoConfig(
                name="testrepo", root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="unity")],
            )
            with patch("src.tools.unity_context.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.unity_context.resolve_repo", return_value="testrepo"), \
                 patch("src.tools.unity_context.get_payload", return_value="Full expanded payload content"):
                result = get_unity_entity_context("testrepo", chunk_id="some_chunk_id")
                self.assertIn("Full expanded payload content", result)

    def test_retrieves_from_payload_store_json(self):
        """Tool should return structured JSON when requested."""
        from src.tools.unity_context import get_unity_entity_context
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = RepoConfig(
                name="testrepo", root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="unity")],
            )
            with patch("src.tools.unity_context.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.unity_context.resolve_repo", return_value="testrepo"), \
                 patch("src.tools.unity_context.get_payload", return_value="Full expanded payload content"):
                result = get_unity_entity_context("testrepo", chunk_id="some_chunk_id", output_format="json")
                payload = json.loads(result)
                self.assertEqual(payload["status"], "found")
                self.assertEqual(payload["chunk_id"], "some_chunk_id")

    def test_missing_chunk_id(self):
        """Missing chunk_id returns helpful error."""
        from src.tools.unity_context import get_unity_entity_context
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = RepoConfig(
                name="testrepo", root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="unity")],
            )
            with patch("src.tools.unity_context.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.unity_context.resolve_repo", return_value="testrepo"), \
                 patch("src.tools.unity_context.get_payload", return_value=None), \
                 patch("src.tools.unity_context.get_collection") as mock_coll:
                mock_coll.return_value.get.return_value = {"ids": [], "metadatas": []}
                result = get_unity_entity_context("testrepo", chunk_id="nonexistent")
                self.assertIn("not found", result)

    def test_requires_params(self):
        """No params returns usage hint."""
        from src.tools.unity_context import get_unity_entity_context
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = RepoConfig(
                name="testrepo", root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="unity")],
            )
            with patch("src.tools.unity_context.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.unity_context.resolve_repo", return_value="testrepo"):
                result = get_unity_entity_context("testrepo")
                self.assertIn("Provide either", result)


# ── GUID-based asset reference disambiguation ───────────────────────────────

class TestAssetRefDisambiguation(unittest.TestCase):
    """Test that GUID-keyed refs disambiguate same-name scripts."""

    def test_build_guid_refs_separate_guids(self):
        """Two scripts with the same class name but different GUIDs are tracked separately."""
        from src.indexer.asset_ref_builder import build_asset_references_by_guid

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create two prefabs referencing scripts with different GUIDs
            prefab_a = root / "PrefabA.prefab"
            prefab_a.write_text(
                "m_Script: {fileID: 123, guid: aaaa000011112222aaaa000011112222}",
                encoding="utf-8",
            )
            prefab_b = root / "PrefabB.prefab"
            prefab_b.write_text(
                "m_Script: {fileID: 456, guid: bbbb000011112222bbbb000011112222}",
                encoding="utf-8",
            )

            from src.config import RepoConfig, SourceDirConfig
            sd = SourceDirConfig(path=root, language="unity")
            config = RepoConfig(name="test", root=root, collection_name="test",
                                source_dirs=[sd])

            # Both GUIDs resolve to the same class name
            guid_map = {
                "aaaa000011112222aaaa000011112222": ("PlayerController", "Scripts/Gameplay/PlayerController.cs"),
                "bbbb000011112222bbbb000011112222": ("PlayerController", "Scripts/UI/PlayerController.cs"),
            }
            source_files = [(prefab_a, sd), (prefab_b, sd)]
            by_guid = build_asset_references_by_guid(source_files, guid_map, config)

            # Both GUIDs should be tracked separately
            self.assertEqual(len(by_guid), 2)
            self.assertIn("aaaa000011112222aaaa000011112222", by_guid)
            self.assertIn("bbbb000011112222bbbb000011112222", by_guid)

            # Each should have a different script path
            paths = {by_guid[g]["script_path"] for g in by_guid}
            self.assertEqual(len(paths), 2)

    def test_class_keyed_materialization_merges_same_name(self):
        """The class-keyed JSON merges files from different GUIDs with the same class name."""
        from src.indexer.asset_ref_builder import _materialize_class_keyed

        by_guid = {
            "aaaa": {
                "class_name": "Foo",
                "script_path": "Scripts/A/Foo.cs",
                "files": ["Prefabs/A.prefab"],
            },
            "bbbb": {
                "class_name": "Foo",
                "script_path": "Scripts/B/Foo.cs",
                "files": ["Prefabs/B.prefab"],
            },
        }
        class_keyed = _materialize_class_keyed(by_guid)
        self.assertIn("Foo", class_keyed)
        self.assertEqual(len(class_keyed["Foo"]), 2)


# ── UnityScriptRef identity model ───────────────────────────────────────────

class TestUnityScriptRef(unittest.TestCase):
    """Test the rich script identity model."""

    def test_qualified_name_with_namespace(self):
        from src.models.unity_script_ref import UnityScriptRef
        ref = UnityScriptRef(
            guid="abc123", class_name="RobotDriver",
            script_path="Scripts/RobotDriver.cs",
            namespace="Augmentus.Robotics",
        )
        self.assertEqual(ref.qualified_name, "Augmentus.Robotics.RobotDriver")

    def test_qualified_name_without_namespace(self):
        from src.models.unity_script_ref import UnityScriptRef
        ref = UnityScriptRef(guid="abc123", class_name="Foo", script_path="Foo.cs")
        self.assertEqual(ref.qualified_name, "Foo")

    def test_backward_compat_tuple(self):
        from src.models.unity_script_ref import UnityScriptRef
        ref = UnityScriptRef(
            guid="abc123", class_name="Foo",
            script_path="Scripts/Foo.cs",
        )
        cn, sp = ref.to_tuple()
        self.assertEqual(cn, "Foo")
        self.assertEqual(sp, "Scripts/Foo.cs")


# ── Rich GUID map with namespace extraction ──────────────────────────────────

class TestRichGuidMap(unittest.TestCase):
    """Test build_rich_guid_map namespace extraction."""

    def test_extracts_namespace(self):
        from src.indexer.guid_resolver import build_rich_guid_map

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "UnityProject" / "Assets" / "Scripts"
            root.mkdir(parents=True)

            # Create .cs file with namespace
            cs = root / "MyClass.cs"
            cs.write_text("using System;\nnamespace Augmentus.Core {\n  public class MyClass {}\n}", encoding="utf-8")

            # Create .meta file
            meta = root / "MyClass.cs.meta"
            meta.write_text("fileFormatVersion: 2\nguid: aabbccdd00112233aabbccdd00112233\n", encoding="utf-8")

            guid_map = build_rich_guid_map(root, Path(tmpdir))
            self.assertIn("aabbccdd00112233aabbccdd00112233", guid_map)
            ref = guid_map["aabbccdd00112233aabbccdd00112233"]
            self.assertEqual(ref.class_name, "MyClass")
            self.assertEqual(ref.namespace, "Augmentus.Core")
            self.assertEqual(ref.qualified_name, "Augmentus.Core.MyClass")


# ── find_references skips Unity YAML dirs ────────────────────────────────────

class TestFindReferencesSkipsUnity(unittest.TestCase):
    """Verify that find_references only searches code dirs, not Unity YAML."""

    def test_unity_dir_excluded(self):
        """Unity language source dirs are skipped by find_references."""
        from src.tools.references import find_references

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a code dir and a unity dir
            scripts = Path(tmpdir) / "Scripts"
            scripts.mkdir()
            (scripts / "Foo.cs").write_text("public class Foo { IBar x; }", encoding="utf-8")

            assets = Path(tmpdir) / "Assets"
            assets.mkdir()
            (assets / "Test.prefab").write_text("m_Script: IBar reference here", encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo", root=Path(tmpdir),
                collection_name="test",
                source_dirs=[
                    SourceDirConfig(path=scripts, language="csharp"),
                    SourceDirConfig(path=assets, language="unity"),
                ],
            )
            with patch("src.tools.references.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.references.resolve_repo", return_value="testrepo"), \
                 patch("src.tools.references._RG_PATH", None):  # Force Python fallback
                result = find_references("IBar", repo="testrepo")
                self.assertIn("IBar", result)
                self.assertIn("Foo.cs", result)
                # The prefab should NOT appear (unity dir excluded)
                self.assertNotIn("Test.prefab", result)


# ── Unity chunker: large file degradation (end-to-end) ──────────────────────

class TestLargeFileDegradation(unittest.TestCase):
    """End-to-end test: large Unity file produces compact chunks, not empty list."""

    def test_compact_summary_for_large_file(self):
        """A large prefab (> 5MB) should produce at least a compact summary chunk."""
        from src.indexer.chunker_unity import chunk_file_unity

        # Build a valid prefab that exceeds 5MB
        lines = ["%YAML 1.1", "%TAG !u! tag:unity3d.com,2011:"]
        lines.append("--- !u!1 &100")
        lines.append("GameObject:")
        lines.append("  m_Component:")
        lines.append("  - component: {fileID: 101}")
        lines.append("  m_Name: HugeScene")
        lines.append("--- !u!4 &101")
        lines.append("Transform:")
        lines.append("  m_GameObject: {fileID: 100}")
        lines.append("  m_Children: []")
        lines.append("  m_Father: {fileID: 0}")
        # Pad past 5MB
        pad = "  # " + "y" * 200
        while len("\n".join(lines).encode()) < 5.5 * 1024 * 1024:
            lines.append(pad)

        data = "\n".join(lines).encode("utf-8")
        self.assertGreater(len(data), 5 * 1024 * 1024)

        chunks = chunk_file_unity(data, "Assets/Scenes/Huge.unity", "TestModule")
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].chunk_type, "prefab_summary")
        self.assertIn("HugeScene", chunks[0].source)


# ── Sidecar state persistence roundtrip ──────────────────────────────────────

class TestSidecarStatePersistence(unittest.TestCase):
    """Test save/load cycle with DATA_DIR patched to temp."""

    def test_save_load_roundtrip(self):
        from src.indexer.sidecar_state import (
            _empty_state, set_hierarchy_contributions,
            save_state, load_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.indexer.sidecar_state.DATA_DIR", Path(tmpdir)):
                state = _empty_state()
                set_hierarchy_contributions(state, "Scripts/X.cs", [
                    ("class_summary", "ClassX", "Scripts/X.cs", "Mod", "Ns", ["IFoo"]),
                ])
                save_state("test_repo", state)

                loaded = load_state("test_repo")
                self.assertEqual(loaded["version"], 2)
                self.assertIn("Scripts/X.cs", loaded["hierarchy"])
                self.assertEqual(loaded["hierarchy"]["Scripts/X.cs"][0]["class"], "ClassX")


if __name__ == "__main__":
    unittest.main()
