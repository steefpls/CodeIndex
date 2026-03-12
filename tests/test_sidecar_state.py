"""Tests for per-file sidecar contribution state model.

Run with: PYTHONPATH=. python tests/test_sidecar_state.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.indexer.sidecar_state import (
    load_state, save_state, _empty_state,
    set_hierarchy_contributions, set_dep_graph_contributions,
    set_asset_ref_contributions, remove_file_contributions,
    materialize_hierarchy, materialize_dep_graph, materialize_asset_refs,
    materialize_asset_refs_by_guid,
    materialize_and_save_all,
)


class TestSidecarStateRoundtrip(unittest.TestCase):
    """Test load/save roundtrip and version checking."""

    def test_empty_state_structure(self):
        state = _empty_state()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["hierarchy"], {})
        self.assertEqual(state["dep_graph"], {})
        self.assertEqual(state["asset_refs"], {})

    def test_roundtrip_save_load(self):
        state = _empty_state()
        set_hierarchy_contributions(state, "Foo.cs", [
            ("class_summary", "Foo", "Foo.cs", "Core", "MyApp", ["IFoo", "MonoBehaviour"]),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.indexer.sidecar_state.DATA_DIR", Path(tmpdir)):
                save_state("testrepo", state)
                loaded = load_state("testrepo")
        self.assertEqual(loaded["hierarchy"]["Foo.cs"], state["hierarchy"]["Foo.cs"])

    def test_version_mismatch_resets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "testrepo_sidecar_state.json"
            path.write_text(json.dumps({"version": 999, "hierarchy": {"x": []}}))
            with patch("src.indexer.sidecar_state.DATA_DIR", Path(tmpdir)):
                state = load_state("testrepo")
        self.assertEqual(state, _empty_state())

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.indexer.sidecar_state.DATA_DIR", Path(tmpdir)):
                state = load_state("nonexistent")
        self.assertEqual(state, _empty_state())


class TestHierarchyContributions(unittest.TestCase):
    """Test hierarchy contribution add/remove/materialize."""

    def test_set_and_materialize(self):
        state = _empty_state()
        set_hierarchy_contributions(state, "Foo.cs", [
            ("class_summary", "Foo", "Foo.cs", "Core", "MyApp", ["IFoo", "MonoBehaviour"]),
        ])
        set_hierarchy_contributions(state, "Bar.cs", [
            ("class_summary", "Bar", "Bar.cs", "Core", "MyApp", ["IFoo"]),
        ])
        hierarchy = materialize_hierarchy(state)
        self.assertIn("IFoo", hierarchy)
        self.assertEqual(len(hierarchy["IFoo"]), 2)
        self.assertIn("MonoBehaviour", hierarchy)
        self.assertEqual(len(hierarchy["MonoBehaviour"]), 1)

    def test_remove_and_materialize(self):
        state = _empty_state()
        set_hierarchy_contributions(state, "Foo.cs", [
            ("class_summary", "Foo", "Foo.cs", "Core", "MyApp", ["IFoo"]),
        ])
        set_hierarchy_contributions(state, "Bar.cs", [
            ("class_summary", "Bar", "Bar.cs", "Core", "MyApp", ["IFoo"]),
        ])
        remove_file_contributions(state, "Foo.cs")
        hierarchy = materialize_hierarchy(state)
        self.assertEqual(len(hierarchy["IFoo"]), 1)
        self.assertEqual(hierarchy["IFoo"][0]["class"], "Bar")

    def test_empty_records_remove_key(self):
        state = _empty_state()
        set_hierarchy_contributions(state, "Foo.cs", [
            ("class_summary", "Foo", "Foo.cs", "Core", "MyApp", ["IFoo"]),
        ])
        set_hierarchy_contributions(state, "Foo.cs", [])
        self.assertNotIn("Foo.cs", state["hierarchy"])


class TestDepGraphContributions(unittest.TestCase):
    """Test dependency graph contribution add/remove/materialize."""

    def test_set_and_materialize(self):
        state = _empty_state()
        set_dep_graph_contributions(state, "Foo.cs", {
            ("Foo", "Foo.cs"): {"module": "Core", "namespace": "MyApp", "refs": {"Bar", "Baz", "String"}},
        })
        set_dep_graph_contributions(state, "Bar.cs", {
            ("Bar", "Bar.cs"): {"module": "Core", "namespace": "MyApp", "refs": {"Foo", "List"}},
        })
        graph = materialize_dep_graph(state)
        self.assertEqual(graph["schema_version"], 2)
        foo_key = "MyApp.Foo"
        bar_key = "MyApp.Bar"
        # Foo's deps should be intersected with known classes {Foo, Bar}
        self.assertIn(bar_key, graph["nodes"][foo_key]["deps"])
        self.assertNotIn("String", graph["nodes"][foo_key]["deps"])  # not a known class
        self.assertIn(foo_key, graph["nodes"][bar_key]["deps"])

    def test_remove_updates_known_classes(self):
        state = _empty_state()
        set_dep_graph_contributions(state, "Foo.cs", {
            ("Foo", "Foo.cs"): {"module": "Core", "namespace": "MyApp", "refs": {"Bar"}},
        })
        set_dep_graph_contributions(state, "Bar.cs", {
            ("Bar", "Bar.cs"): {"module": "Core", "namespace": "MyApp", "refs": {"Foo"}},
        })
        remove_file_contributions(state, "Bar.cs")
        graph = materialize_dep_graph(state)
        # Bar is no longer a known class, so Foo should have no deps
        self.assertEqual(graph["nodes"]["MyApp.Foo"]["deps"], [])

    def test_duplicate_unqualified_class_name_no_overwrite(self):
        state = _empty_state()
        set_dep_graph_contributions(state, "A/Foo.cs", {
            ("Foo", "A/Foo.cs"): {"module": "Core", "namespace": "", "refs": {"Bar"}},
        })
        set_dep_graph_contributions(state, "B/Foo.cs", {
            ("Foo", "B/Foo.cs"): {"module": "Core", "namespace": "", "refs": {"Bar"}},
        })
        set_dep_graph_contributions(state, "Bar.cs", {
            ("Bar", "Bar.cs"): {"module": "Core", "namespace": "", "refs": set()},
        })

        graph = materialize_dep_graph(state)
        self.assertIn("Foo", graph["name_index"])
        self.assertEqual(len(graph["name_index"]["Foo"]), 2)
        for node_key in graph["name_index"]["Foo"]:
            self.assertIn("Bar@@Core@@Bar.cs", graph["nodes"][node_key]["deps"])


class TestAssetRefContributions(unittest.TestCase):
    """Test asset reference contribution add/remove/materialize."""

    def test_set_and_materialize(self):
        state = _empty_state()
        set_asset_ref_contributions(state, "Prefabs/A.prefab", {
            "guid-a": {"class_name": "ScriptA", "script_path": "Scripts/A.cs"},
            "guid-b": {"class_name": "ScriptB", "script_path": "Scripts/B.cs"},
        })
        set_asset_ref_contributions(state, "Scenes/Main.unity", {
            "guid-a": {"class_name": "ScriptA", "script_path": "Scripts/A.cs"},
        })
        refs = materialize_asset_refs(state)
        self.assertEqual(sorted(refs["ScriptA"]), ["Prefabs/A.prefab", "Scenes/Main.unity"])
        self.assertEqual(refs["ScriptB"], ["Prefabs/A.prefab"])
        by_guid = materialize_asset_refs_by_guid(state)
        self.assertEqual(by_guid["guid-a"]["files"], ["Prefabs/A.prefab", "Scenes/Main.unity"])

    def test_remove_file_clears_contributions(self):
        state = _empty_state()
        set_asset_ref_contributions(state, "Prefabs/A.prefab", {
            "guid-a": {"class_name": "ScriptA", "script_path": "Scripts/A.cs"},
        })
        set_asset_ref_contributions(state, "Scenes/Main.unity", {
            "guid-a": {"class_name": "ScriptA", "script_path": "Scripts/A.cs"},
        })
        remove_file_contributions(state, "Prefabs/A.prefab")
        refs = materialize_asset_refs(state)
        self.assertEqual(refs["ScriptA"], ["Scenes/Main.unity"])

    def test_empty_classes_remove_key(self):
        state = _empty_state()
        set_asset_ref_contributions(state, "Prefabs/A.prefab", {
            "guid-a": {"class_name": "ScriptA", "script_path": "Scripts/A.cs"},
        })
        set_asset_ref_contributions(state, "Prefabs/A.prefab", {})
        self.assertNotIn("Prefabs/A.prefab", state["asset_refs"])


class TestMaterializeAndSaveAll(unittest.TestCase):
    """Test full materialization + file writing."""

    def test_saves_all_sidecar_files(self):
        state = _empty_state()
        set_hierarchy_contributions(state, "Foo.cs", [
            ("class_summary", "Foo", "Foo.cs", "Core", "MyApp", ["IFoo"]),
        ])
        set_dep_graph_contributions(state, "Foo.cs", {
            ("Foo", "Foo.cs"): {"module": "Core", "namespace": "MyApp", "refs": {"Bar"}},
        })
        set_asset_ref_contributions(state, "Prefabs/A.prefab", {
            "guid-foo": {"class_name": "Foo", "script_path": "Scripts/Foo.cs"},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.indexer.sidecar_state.DATA_DIR", Path(tmpdir)):
                materialize_and_save_all("testrepo", state)

            # Check all files were created
            self.assertTrue((Path(tmpdir) / "testrepo_type_hierarchy.json").exists())
            self.assertTrue((Path(tmpdir) / "testrepo_dep_graph.json").exists())
            self.assertTrue((Path(tmpdir) / "testrepo_asset_references.json").exists())
            self.assertTrue((Path(tmpdir) / "testrepo_asset_references_by_guid.json").exists())
            self.assertTrue((Path(tmpdir) / "testrepo_sidecar_state.json").exists())

            # Verify content
            hierarchy = json.loads((Path(tmpdir) / "testrepo_type_hierarchy.json").read_text())
            self.assertIn("IFoo", hierarchy)
            refs = json.loads((Path(tmpdir) / "testrepo_asset_references.json").read_text())
            self.assertIn("Foo", refs)
            dep_graph = json.loads((Path(tmpdir) / "testrepo_dep_graph.json").read_text())
            self.assertEqual(dep_graph["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
