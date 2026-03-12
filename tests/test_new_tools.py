"""Tests for the new tools added in commits b098cfb, 6993574, cb5e3f5.

Covers: find_references, assembly_graph, type_hierarchy, asset_references,
project_info, scriptable_object chunking, serialized field tagging,
base_types in metadata, .asset extension support, and the dependency graph.

Run with: PYTHONPATH=. python tests/test_new_tools.py
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


# ── find_references ──────────────────────────────────────────────────────────

class TestFindReferences(unittest.TestCase):
    """Test the find_references tool."""

    def test_find_in_temp_files(self):
        from src.tools.references import find_references
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake repo structure
            scripts = Path(tmpdir) / "UnityProject" / "Assets" / "Scripts"
            scripts.mkdir(parents=True)
            (scripts / "Foo.cs").write_text("public class Foo : IBar { }", encoding="utf-8")
            (scripts / "Bar.cs").write_text("public class Bar : IBar { Foo x; }", encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=scripts, language="csharp")],
            )
            with patch("src.tools.references.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.references.resolve_repo", return_value="testrepo"):
                result = find_references("IBar", repo="testrepo")
                self.assertIn("IBar", result)
                self.assertIn("Foo.cs", result)
                self.assertIn("Bar.cs", result)

    def test_whole_word_match(self):
        from src.tools.references import find_references
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = Path(tmpdir) / "src"
            scripts.mkdir(parents=True)
            (scripts / "test.cs").write_text("Foobar Foo FooX", encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=scripts, language="csharp")],
            )
            with patch("src.tools.references.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.references.resolve_repo", return_value="testrepo"):
                result = find_references("Foo", repo="testrepo", whole_word=True)
                # Should match "Foo" but report only lines containing whole-word Foo
                self.assertIn("Foo", result)

    def test_unknown_repo(self):
        from src.tools.references import find_references
        result = find_references("Test", repo="nonexistent")
        self.assertIn("Unknown repo", result)


# ── assembly_graph ───────────────────────────────────────────────────────────

class TestAssemblyGraph(unittest.TestCase):
    """Test the assembly_graph tool."""

    def test_no_asmdef_files(self):
        from src.tools.assembly_graph import get_assembly_graph
        with tempfile.TemporaryDirectory() as tmpdir:
            assets = Path(tmpdir) / "UnityProject" / "Assets"
            assets.mkdir(parents=True)

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=assets, language="unity")],
            )
            with patch("src.tools.assembly_graph.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.assembly_graph.resolve_repo", return_value="testrepo"):
                # Clear cache first
                from src.tools.assembly_graph import invalidate_graph_cache
                invalidate_graph_cache("testrepo")
                result = get_assembly_graph(repo="testrepo")
                self.assertIn("No .asmdef files", result)

    def test_with_asmdef(self):
        from src.tools.assembly_graph import get_assembly_graph, invalidate_graph_cache
        with tempfile.TemporaryDirectory() as tmpdir:
            assets = Path(tmpdir) / "UnityProject" / "Assets"
            assets.mkdir(parents=True)
            asmdef = assets / "Main.asmdef"
            asmdef.write_text(json.dumps({"name": "Main", "references": []}), encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=assets, language="unity")],
            )
            with patch("src.tools.assembly_graph.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.assembly_graph.resolve_repo", return_value="testrepo"):
                invalidate_graph_cache("testrepo")
                result = get_assembly_graph(repo="testrepo")
                self.assertIn("Main", result)
                self.assertIn("1 assemblies", result)


# ── type_hierarchy ───────────────────────────────────────────────────────────

class TestTypeHierarchy(unittest.TestCase):
    """Test the type_hierarchy tool."""

    def test_build_and_query(self):
        from src.indexer.hierarchy_builder import build_type_hierarchy
        records = [
            ("class_summary", "FooImpl", "path/Foo.cs", "Module", "Ns", ["IFoo", "BaseClass"]),
            ("class_summary", "BarImpl", "path/Bar.cs", "Module", "Ns", ["IFoo"]),
        ]
        h = build_type_hierarchy(records)
        self.assertIn("IFoo", h)
        self.assertEqual(len(h["IFoo"]), 2)
        self.assertIn("BaseClass", h)
        self.assertEqual(len(h["BaseClass"]), 1)

    def test_find_implementations_not_found(self):
        from src.tools.type_hierarchy import find_implementations, invalidate_hierarchy_cache
        with patch("src.tools.type_hierarchy.REPOS", {"testrepo": object()}), \
             patch("src.tools.type_hierarchy.resolve_repo", return_value="testrepo"), \
             patch("src.tools.type_hierarchy._get_hierarchy", return_value={}):
            invalidate_hierarchy_cache("testrepo")
            result = find_implementations("NonExistent", repo="testrepo")
            self.assertIn("No type hierarchy data", result)


# ── asset_references ─────────────────────────────────────────────────────────

class TestAssetReferences(unittest.TestCase):
    """Test the asset_references tool."""

    def test_build_refs(self):
        from src.indexer.asset_ref_builder import build_asset_references
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prefab = root / "test.prefab"
            prefab.write_text("m_Script: {fileID: 123, guid: aabbccdd00112233aabbccdd00112233}", encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            config = RepoConfig(name="testrepo", root=root, collection_name="test",
                                source_dirs=[SourceDirConfig(path=root, language="unity")])
            guid_map = {"aabbccdd00112233aabbccdd00112233": ("PlayerController", "Scripts/PlayerController.cs")}
            source_files = [(prefab, config.source_dirs[0])]
            refs = build_asset_references(source_files, guid_map, config)
            self.assertIn("PlayerController", refs)
            self.assertIn("test.prefab", refs["PlayerController"][0])

    def test_unknown_repo(self):
        from src.tools.asset_references import find_asset_references
        result = find_asset_references("Test", repo="nonexistent")
        self.assertIn("Unknown repo", result)

    def test_json_output(self):
        from src.tools.asset_references import find_asset_references
        with patch("src.tools.asset_references.REPOS", {"testrepo": object()}), \
             patch("src.tools.asset_references.resolve_repo", return_value="testrepo"), \
             patch("src.tools.asset_references._get_refs", return_value={"Foo": ["A.prefab", "B.unity"]}), \
             patch("src.tools.asset_references._get_guid_refs", return_value={}):
            result = find_asset_references("Foo", repo="testrepo", output_format="json")
            payload = json.loads(result)
            self.assertEqual(payload["status"], "found")
            self.assertEqual(payload["class_name"], "Foo")
            self.assertEqual(payload["total_files"], 2)


# ── project_info ─────────────────────────────────────────────────────────────

class TestProjectInfo(unittest.TestCase):
    """Test the project_info tool."""

    def test_reads_version(self):
        from src.tools.project_info import get_project_info
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Path(tmpdir) / "UnityProject" / "ProjectSettings"
            proj.mkdir(parents=True)
            (proj / "ProjectVersion.txt").write_text("m_EditorVersion: 6000.0.23f1\n", encoding="utf-8")

            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="csharp")],
            )
            with patch("src.tools.project_info.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.project_info.resolve_repo", return_value="testrepo"):
                result = get_project_info(repo="testrepo")
                self.assertIn("6000.0.23f1", result)

    def test_no_unity_project(self):
        from src.tools.project_info import get_project_info
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import RepoConfig, SourceDirConfig
            fake_config = RepoConfig(
                name="testrepo",
                root=Path(tmpdir),
                collection_name="test",
                source_dirs=[SourceDirConfig(path=Path(tmpdir), language="csharp")],
            )
            with patch("src.tools.project_info.REPOS", {"testrepo": fake_config}), \
                 patch("src.tools.project_info.resolve_repo", return_value="testrepo"):
                result = get_project_info(repo="testrepo")
                self.assertIn("No UnityProject", result)


# ── ScriptableObject chunking ────────────────────────────────────────────────

class TestScriptableObjectChunking(unittest.TestCase):
    """Test that .asset files with MonoScript are chunked as scriptable_object."""

    def test_scriptable_object_asset(self):
        from src.indexer.chunker_unity import chunk_file_unity
        yaml_content = textwrap.dedent("""\
            %YAML 1.1
            %TAG !u! tag:unity3d.com,2011:
            --- !u!114 &11400000
            MonoBehaviour:
              m_Script: {fileID: 11500000, guid: abc123def456abc123def456abc123de, type: 3}
              m_Name: MyConfig
              someField: 42
              listField:
              - item1
              - item2
        """).encode("utf-8")

        guid_map = {"abc123def456abc123def456abc123de": ("MyConfigSO", "Scripts/MyConfigSO.cs")}
        chunks = chunk_file_unity(yaml_content, "Assets/Data/MyConfig.asset", "Data", guid_map=guid_map)
        self.assertTrue(len(chunks) >= 1)
        so_chunks = [c for c in chunks if c.chunk_type == "scriptable_object"]
        self.assertTrue(len(so_chunks) >= 1, f"Expected scriptable_object chunks, got: {[c.chunk_type for c in chunks]}")
        self.assertIn("MyConfigSO", so_chunks[0].source)


# ── Serialized field tagging ─────────────────────────────────────────────────

class TestSerializedFieldTagging(unittest.TestCase):
    """Test that class_summary tags [SerializeField] public fields."""

    def test_public_field_tagged(self):
        from src.indexer.chunker import chunk_file
        source = textwrap.dedent("""\
            using UnityEngine;
            namespace Test {
                public class BigClass : MonoBehaviour {
                    [SerializeField] private float _speed;
                    public int health;
                    private string _name;
                    public void Method1() { }
                    public void Method2() { }
                    public void Method3() { }
                    public void Method4() { }
                    public void Method5() { }
                    public void Method6() { }
                    public void Method7() { }
                    public void Method8() { }
                    public void Method9() { }
                    public void Method10() { }
                    public void Method11() { }
                    public void Method12() { }
                    public void Method13() { }
                    public void Method14() { }
                    public void Method15() { }
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                    // filler lines
                }
            }
        """).encode("utf-8")
        chunks = chunk_file(source, "Test/BigClass.cs", "Test")
        summaries = [c for c in chunks if c.chunk_type == "class_summary"]
        self.assertTrue(len(summaries) > 0, "Expected class_summary chunk for large class")
        summary_src = summaries[0].source
        self.assertIn("[serialized]", summary_src)


# ── base_types in metadata ───────────────────────────────────────────────────

class TestBaseTypesInMetadata(unittest.TestCase):
    """Test that CodeChunk metadata includes base_types."""

    def test_base_types_field(self):
        from src.models.chunk import CodeChunk
        c = CodeChunk(
            file_path="Foo.cs",
            class_name="Foo",
            method_name=None,
            namespace="",
            start_line=1,
            end_line=1,
            source="class Foo : IBar, IBaz {}",
            chunk_type="class_summary",
            base_types=["IBar", "IBaz"],
        )
        meta = c.metadata
        self.assertEqual(meta["base_types"], "IBar,IBaz")

    def test_empty_base_types(self):
        from src.models.chunk import CodeChunk
        c = CodeChunk(
            file_path="Foo.cs",
            class_name="Foo",
            method_name="DoStuff",
            namespace="",
            start_line=1,
            end_line=1,
            source="void DoStuff() {}",
            chunk_type="method",
            base_types=[],
        )
        meta = c.metadata
        self.assertEqual(meta["base_types"], "")


# ── .asset extension support ─────────────────────────────────────────────────

class TestAssetExtension(unittest.TestCase):
    """Test that file_scanner includes .asset files for unity language."""

    def test_asset_in_extensions(self):
        from src.indexer.file_scanner import _LANGUAGE_EXTENSIONS
        self.assertIn(".asset", _LANGUAGE_EXTENSIONS.get("unity", []))


# ── Dependency graph ─────────────────────────────────────────────────────────

class TestDepGraph(unittest.TestCase):
    """Test the dependency graph builder and tool."""

    def test_extract_type_candidates(self):
        from src.indexer.dep_graph_builder import extract_type_candidates
        source = "public class Foo : IBar { private Baz _baz; }"
        candidates = extract_type_candidates(source)
        self.assertIn("Foo", candidates)
        self.assertIn("IBar", candidates)
        self.assertIn("Baz", candidates)

    def test_extract_filters_allcaps(self):
        from src.indexer.dep_graph_builder import extract_type_candidates
        source = "MAX_VALUE SOME_CONST FooBar"
        candidates = extract_type_candidates(source)
        self.assertNotIn("MAX_VALUE", candidates)
        self.assertNotIn("SOME_CONST", candidates)
        self.assertIn("FooBar", candidates)

    def test_extract_strips_comments_strings_attributes(self):
        from src.indexer.dep_graph_builder import extract_type_candidates
        source = textwrap.dedent("""\
            // FakeComment class reference
            /* MultiLineComment */
            "StringLiteral"
            [SerializeField]
            public RealClass _field;
        """)
        candidates = extract_type_candidates(source)
        self.assertNotIn("FakeComment", candidates)
        self.assertNotIn("MultiLineComment", candidates)
        self.assertNotIn("StringLiteral", candidates)
        self.assertNotIn("SerializeField", candidates)
        self.assertIn("RealClass", candidates)

    def test_extract_filters_short_names(self):
        from src.indexer.dep_graph_builder import extract_type_candidates
        source = "On If Do RealClass"
        candidates = extract_type_candidates(source)
        self.assertNotIn("On", candidates)
        self.assertNotIn("If", candidates)
        self.assertNotIn("Do", candidates)
        self.assertIn("RealClass", candidates)

    def test_build_dep_graph(self):
        from src.indexer.dep_graph_builder import build_dep_graph
        records = [
            ("ClassA", "a.cs", "Mod", "Ns", {"ClassB", "ClassC", "Unknown"}),
            ("ClassB", "b.cs", "Mod", "Ns", {"ClassA"}),
            ("ClassC", "c.cs", "Mod", "Ns", set()),
        ]
        graph = build_dep_graph(records)
        self.assertEqual(len(graph), 3)
        # ClassA depends on ClassB and ClassC (Unknown filtered by intersection)
        self.assertIn("ClassB", graph["ClassA"]["deps"])
        self.assertIn("ClassC", graph["ClassA"]["deps"])
        self.assertNotIn("Unknown", graph["ClassA"]["deps"])
        # ClassB depends on ClassA
        self.assertIn("ClassA", graph["ClassB"]["deps"])
        # ClassC has no deps
        self.assertEqual(graph["ClassC"]["deps"], [])

    def test_reverse_deps(self):
        from src.tools.class_deps import _get_reverse
        graph = {
            "ClassA": {"deps": ["ClassB", "ClassC"]},
            "ClassB": {"deps": ["ClassC"]},
            "ClassC": {"deps": []},
        }
        # Clear any cached reverse
        from src.tools.class_deps import _reverse_cache
        _reverse_cache.pop("test_repo", None)
        reverse = _get_reverse("test_repo", graph)
        self.assertIn("ClassB", reverse)
        self.assertIn("ClassA", reverse["ClassB"])
        self.assertIn("ClassC", reverse)
        self.assertIn("ClassA", reverse["ClassC"])
        self.assertIn("ClassB", reverse["ClassC"])
        # Clean up
        _reverse_cache.pop("test_repo", None)

    def test_get_class_dependencies_format(self):
        from src.tools.class_deps import get_class_dependencies, invalidate_dep_cache, _dep_cache
        # Inject mock graph data
        _dep_cache["testrepo"] = {
            "ClassA": {"file": "a.cs", "module": "Mod", "namespace": "Ns", "deps": ["ClassB"]},
            "ClassB": {"file": "b.cs", "module": "Mod", "namespace": "Ns", "deps": []},
        }
        with patch("src.tools.class_deps.REPOS", {"testrepo": object()}), \
             patch("src.tools.class_deps.resolve_repo", return_value="testrepo"):
            result = get_class_dependencies("ClassA", repo="testrepo")
            self.assertIn("ClassA", result)
            self.assertIn("Depends on", result)
            self.assertIn("ClassB", result)
            self.assertIn("Depended on by", result)

            # Summary mode
            result = get_class_dependencies(None, repo="testrepo")
            self.assertIn("Dependency Graph", result)
            self.assertIn("2 classes", result)
        # Clean up
        invalidate_dep_cache("testrepo")

    def test_get_class_dependencies_json(self):
        from src.tools.class_deps import get_class_dependencies, invalidate_dep_cache, _dep_cache
        _dep_cache["testrepo"] = {
            "schema_version": 2,
            "nodes": {
                "MyNs.ClassA": {
                    "class_name": "ClassA",
                    "namespace": "MyNs",
                    "module": "Mod",
                    "files": ["a.cs"],
                    "deps": ["MyNs.ClassB"],
                },
                "MyNs.ClassB": {
                    "class_name": "ClassB",
                    "namespace": "MyNs",
                    "module": "Mod",
                    "files": ["b.cs"],
                    "deps": [],
                },
            },
            "name_index": {
                "ClassA": ["MyNs.ClassA"],
                "ClassB": ["MyNs.ClassB"],
            },
        }
        with patch("src.tools.class_deps.REPOS", {"testrepo": object()}), \
             patch("src.tools.class_deps.resolve_repo", return_value="testrepo"):
            result = get_class_dependencies("ClassA", repo="testrepo", output_format="json")
            payload = json.loads(result)
            self.assertEqual(payload["status"], "found")
            self.assertEqual(payload["node_key"], "MyNs.ClassA")
            self.assertEqual(payload["depends_on"][0]["node_key"], "MyNs.ClassB")

            summary = get_class_dependencies(None, repo="testrepo", output_format="json")
            summary_payload = json.loads(summary)
            self.assertEqual(summary_payload["status"], "summary")
            self.assertEqual(summary_payload["nodes"], 2)
        invalidate_dep_cache("testrepo")

    def test_get_class_dependencies_disambiguates_v2_name_collisions(self):
        from src.tools.class_deps import get_class_dependencies, invalidate_dep_cache, _dep_cache
        _dep_cache["testrepo"] = {
            "schema_version": 2,
            "nodes": {
                "Foo@@Core@@A/Foo.cs": {
                    "class_name": "Foo",
                    "namespace": "",
                    "module": "Core",
                    "files": ["A/Foo.cs"],
                    "deps": [],
                },
                "Foo@@Core@@B/Foo.cs": {
                    "class_name": "Foo",
                    "namespace": "",
                    "module": "Core",
                    "files": ["B/Foo.cs"],
                    "deps": [],
                },
            },
            "name_index": {
                "Foo": ["Foo@@Core@@A/Foo.cs", "Foo@@Core@@B/Foo.cs"],
            },
        }
        with patch("src.tools.class_deps.REPOS", {"testrepo": object()}), \
             patch("src.tools.class_deps.resolve_repo", return_value="testrepo"):
            result = get_class_dependencies("Foo", repo="testrepo")
            self.assertIn("Multiple classes match", result)
            self.assertIn("Foo@@Core@@A/Foo.cs", result)
            self.assertIn("Foo@@Core@@B/Foo.cs", result)
        invalidate_dep_cache("testrepo")

    def test_get_class_dependencies_exact_node_key_v2(self):
        from src.tools.class_deps import get_class_dependencies, invalidate_dep_cache, _dep_cache
        _dep_cache["testrepo"] = {
            "schema_version": 2,
            "nodes": {
                "MyNs.Foo": {
                    "class_name": "Foo",
                    "namespace": "MyNs",
                    "module": "Core",
                    "files": ["Foo.cs"],
                    "deps": [],
                },
            },
            "name_index": {"Foo": ["MyNs.Foo"]},
        }
        with patch("src.tools.class_deps.REPOS", {"testrepo": object()}), \
             patch("src.tools.class_deps.resolve_repo", return_value="testrepo"):
            result = get_class_dependencies("MyNs.Foo", repo="testrepo")
            self.assertIn("=== MyNs.Foo", result)
            self.assertIn("Class: Foo", result)
        invalidate_dep_cache("testrepo")


if __name__ == "__main__":
    unittest.main()
