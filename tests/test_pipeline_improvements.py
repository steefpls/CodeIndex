"""Tests for pipeline performance improvements.

Covers:
- file_scanner: os.walk with directory pruning (replaces rglob)
- pipeline: timer split (setup vs indexing), setup progress bar,
  batched payload store with periodic flush, O(1) payload removal
  via reverse index, empty payload store on full reindex
- index_management: result formatter with setup + indexing time split

Run with: PYTHONPATH=. python tests/test_pipeline_improvements.py
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import RepoConfig, SourceDirConfig
from src.models.chunk import CodeChunk, MAX_EMBED_CHARS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(file_path: str, class_name: str, method: str | None,
                source: str, start_line: int = 1, end_line: int = 10,
                chunk_type: str = "method") -> CodeChunk:
    """Create a CodeChunk for testing."""
    return CodeChunk(
        file_path=file_path,
        class_name=class_name,
        method_name=method,
        namespace="TestNs",
        start_line=start_line,
        end_line=end_line,
        source=source,
        chunk_type=chunk_type,
        module="TestModule",
    )


def _create_tree(root: Path, structure: dict) -> None:
    """Create a directory tree from a nested dict.

    Keys ending with / are directories. Leaf strings are file contents.
    Example: {"src/": {"foo.py": "print(1)"}, "README.md": "hello"}
    """
    for name, value in structure.items():
        path = root / name.rstrip("/")
        if isinstance(value, dict):
            path.mkdir(parents=True, exist_ok=True)
            _create_tree(path, value)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")


# ── file_scanner tests ──────────────────────────────────────────────────────

class TestFileScannerDirectoryPruning(unittest.TestCase):
    """Test that os.walk-based scanner prunes excluded directories."""

    def test_excludes_directory_patterns(self):
        """Files inside excluded directories (e.g. node_modules/) are not returned."""
        from src.indexer.file_scanner import _scan_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {"app.py": "# app", "utils.py": "# utils"},
                "node_modules/": {"pkg/": {"index.py": "# should be excluded"}},
                ".venv/": {"lib/": {"site.py": "# should be excluded"}},
            })
            files = _scan_dir(root, [".py"], ["node_modules/", ".venv/"])
            names = {f.name for f in files}
            self.assertIn("app.py", names)
            self.assertIn("utils.py", names)
            self.assertNotIn("index.py", names)
            self.assertNotIn("site.py", names)

    def test_excludes_extension_patterns(self):
        """Files matching extension exclusion (e.g. *.meta) are not returned."""
        from src.indexer.file_scanner import _scan_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "Script.cs": "class Foo {}",
                "Script.cs.meta": "guid: abc123",
                "Other.cs": "class Bar {}",
            })
            files = _scan_dir(root, [".cs"], ["*.meta"])
            names = {f.name for f in files}
            self.assertIn("Script.cs", names)
            self.assertIn("Other.cs", names)
            # .meta is a different extension, so it wouldn't match .cs anyway,
            # but verify extension filtering works
            self.assertNotIn("Script.cs.meta", names)

    def test_excludes_substring_patterns(self):
        """Files matching substring exclusion patterns are not returned."""
        from src.indexer.file_scanner import _scan_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {"main.json": '{}'},
                "package-lock.json": '{}',
            })
            files = _scan_dir(root, [".json"], ["package-lock.json"])
            names = {f.name for f in files}
            self.assertIn("main.json", names)
            self.assertNotIn("package-lock.json", names)

    def test_multiple_extensions_single_walk(self):
        """Multiple extensions are collected in one walk, not separate rglob calls."""
        from src.indexer.file_scanner import _scan_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {
                    "main.cpp": "int main() {}",
                    "utils.h": "void foo();",
                    "helper.hpp": "class X {};",
                    "readme.txt": "ignore me",
                },
            })
            files = _scan_dir(root, [".cpp", ".h", ".hpp"], [])
            names = {f.name for f in files}
            self.assertEqual(names, {"main.cpp", "utils.h", "helper.hpp"})

    def test_nested_excluded_dirs_not_walked(self):
        """Deeply nested excluded dirs are pruned at the first level they appear."""
        from src.indexer.file_scanner import _scan_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {
                    "good.py": "# keep",
                    "obj/": {
                        "deep/": {
                            "deeper/": {"bad.py": "# exclude"},
                        },
                    },
                },
            })
            files = _scan_dir(root, [".py"], ["obj/"])
            names = {f.name for f in files}
            self.assertIn("good.py", names)
            self.assertNotIn("bad.py", names)

    def test_missing_source_dir_raises(self):
        """FileNotFoundError is raised when source_dir doesn't exist."""
        from src.indexer.file_scanner import _scan_dir
        with self.assertRaises(FileNotFoundError):
            _scan_dir(Path("/nonexistent/path"), [".py"], [])

    def test_scan_repo_files_combines_source_dirs(self):
        """scan_repo_files combines files from multiple source directories."""
        from src.indexer.file_scanner import scan_repo_files

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {"app.py": "# app"},
                "tests/": {"test_app.py": "# test"},
            })
            config = RepoConfig(
                name="test",
                root=root,
                collection_name="test_code",
                source_dirs=[
                    SourceDirConfig(path=root / "src", language="python"),
                    SourceDirConfig(path=root / "tests", language="python"),
                ],
            )
            files = scan_repo_files(config)
            names = {f.name for f, _ in files}
            self.assertEqual(names, {"app.py", "test_app.py"})

    def test_results_are_sorted(self):
        """scan_repo_files returns results sorted by path."""
        from src.indexer.file_scanner import scan_repo_files

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_tree(root, {
                "src/": {"z.py": "#", "a.py": "#", "m.py": "#"},
            })
            config = RepoConfig(
                name="test", root=root, collection_name="test_code",
                source_dirs=[SourceDirConfig(path=root / "src", language="python")],
            )
            files = scan_repo_files(config)
            names = [f.name for f, _ in files]
            self.assertEqual(names, sorted(names))


# ── Payload store in-memory helpers ─────────────────────────────────────────

class TestPayloadInMemoryHelpers(unittest.TestCase):
    """Test _update_payloads_in_memory and _remove_payloads_in_memory with reverse index."""

    def test_update_adds_large_chunks(self):
        """Chunks exceeding MAX_EMBED_CHARS are added to the payload store."""
        from src.indexer.pipeline import _update_payloads_in_memory

        store = {}
        idx = {}
        large_source = "x" * (MAX_EMBED_CHARS + 100)
        small_source = "y" * 100
        chunks = [
            _make_chunk("src/Foo.cs", "Foo", "BigMethod", large_source),
            _make_chunk("src/Foo.cs", "Foo", "SmallMethod", small_source),
        ]
        _update_payloads_in_memory(store, idx, "src/Foo.cs", chunks)

        # Only the large chunk should be in the store
        self.assertEqual(len(store), 1)
        self.assertIn(chunks[0].chunk_id, store)
        self.assertNotIn(chunks[1].chunk_id, store)
        # Reverse index should have the file
        self.assertIn("src/Foo.cs", idx)
        self.assertEqual(idx["src/Foo.cs"], [chunks[0].chunk_id])

    def test_update_removes_old_entries_via_index(self):
        """Re-indexing a file removes old payload entries using the reverse index."""
        from src.indexer.pipeline import _update_payloads_in_memory

        store = {"old_chunk_id": "old source"}
        idx = {"src/Foo.cs": ["old_chunk_id"]}
        large_source = "x" * (MAX_EMBED_CHARS + 100)
        chunks = [_make_chunk("src/Foo.cs", "Foo", "NewMethod", large_source)]

        _update_payloads_in_memory(store, idx, "src/Foo.cs", chunks)

        # Old chunk should be gone, new chunk should be present
        self.assertNotIn("old_chunk_id", store)
        self.assertEqual(len(store), 1)
        self.assertIn(chunks[0].chunk_id, store)

    def test_update_with_no_large_chunks_clears_file(self):
        """If a file's new chunks are all small, its entries are removed."""
        from src.indexer.pipeline import _update_payloads_in_memory

        store = {"old_id": "old source"}
        idx = {"src/Foo.cs": ["old_id"]}
        small_chunks = [_make_chunk("src/Foo.cs", "Foo", "Small", "tiny")]

        _update_payloads_in_memory(store, idx, "src/Foo.cs", small_chunks)

        self.assertEqual(len(store), 0)
        self.assertNotIn("src/Foo.cs", idx)

    def test_remove_clears_file_entries(self):
        """_remove_payloads_in_memory removes all entries for a file."""
        from src.indexer.pipeline import _remove_payloads_in_memory

        store = {"chunk_a": "source_a", "chunk_b": "source_b", "other": "keep"}
        idx = {"file_a": ["chunk_a", "chunk_b"]}

        _remove_payloads_in_memory(store, idx, "file_a")

        self.assertNotIn("chunk_a", store)
        self.assertNotIn("chunk_b", store)
        self.assertIn("other", store)
        self.assertNotIn("file_a", idx)

    def test_remove_nonexistent_file_is_noop(self):
        """Removing a file not in the index doesn't raise."""
        from src.indexer.pipeline import _remove_payloads_in_memory

        store = {"chunk_a": "source_a"}
        idx = {}
        _remove_payloads_in_memory(store, idx, "nonexistent.cs")
        self.assertEqual(len(store), 1)

    def test_update_multiple_files_independent(self):
        """Updating one file doesn't affect another file's payloads."""
        from src.indexer.pipeline import _update_payloads_in_memory

        store = {}
        idx = {}
        large = "x" * (MAX_EMBED_CHARS + 1)
        chunks_a = [_make_chunk("src/A.cs", "A", "Method", large)]
        chunks_b = [_make_chunk("src/B.cs", "B", "Method", large)]

        _update_payloads_in_memory(store, idx, "src/A.cs", chunks_a)
        _update_payloads_in_memory(store, idx, "src/B.cs", chunks_b)

        self.assertEqual(len(store), 2)
        self.assertIn("src/A.cs", idx)
        self.assertIn("src/B.cs", idx)

        # Re-index A — B should be unaffected
        new_chunks_a = [_make_chunk("src/A.cs", "A", "NewMethod", large, start_line=20, end_line=30)]
        _update_payloads_in_memory(store, idx, "src/A.cs", new_chunks_a)

        self.assertEqual(len(store), 2)
        self.assertNotIn(chunks_a[0].chunk_id, store)
        self.assertIn(new_chunks_a[0].chunk_id, store)
        self.assertIn(chunks_b[0].chunk_id, store)


# ── Pipeline timer and progress ─────────────────────────────────────────────

class TestPipelineTimerSplit(unittest.TestCase):
    """Test that index_repo returns separate setup and indexing times."""

    def _make_fake_repo(self, tmpdir: str) -> tuple[RepoConfig, Path]:
        root = Path(tmpdir)
        src = root / "src"
        src.mkdir()
        (src / "hello.py").write_text("print('hello')", encoding="utf-8")
        config = RepoConfig(
            name="timer_test",
            root=root,
            collection_name="timer_test_code",
            source_dirs=[SourceDirConfig(path=src, language="python")],
        )
        return config, root

    @patch("src.indexer.pipeline.get_collection")
    @patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100)
    def test_result_contains_setup_and_elapsed(self, mock_batch, mock_coll):
        """index_repo result dict has both setup_seconds and elapsed_seconds."""
        from src.indexer.pipeline import index_repo

        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        with tempfile.TemporaryDirectory() as tmpdir:
            config, root = self._make_fake_repo(tmpdir)
            with patch("src.indexer.pipeline.REPOS", {"timer_test": config}), \
                 patch("src.indexer.pipeline.DATA_DIR", Path(tmpdir)):
                result = index_repo("timer_test", incremental=False)

        self.assertIn("setup_seconds", result)
        self.assertIn("elapsed_seconds", result)
        self.assertIsInstance(result["setup_seconds"], float)
        self.assertIsInstance(result["elapsed_seconds"], float)
        self.assertGreaterEqual(result["setup_seconds"], 0)
        self.assertGreaterEqual(result["elapsed_seconds"], 0)

    @patch("src.indexer.pipeline.get_collection")
    @patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100)
    def test_setup_progress_callback_fires(self, mock_batch, mock_coll):
        """Progress callback receives setup phase messages."""
        from src.indexer.pipeline import index_repo

        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        messages = []
        def capture_progress(current, total, msg):
            messages.append(msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            config, root = self._make_fake_repo(tmpdir)
            with patch("src.indexer.pipeline.REPOS", {"timer_test": config}), \
                 patch("src.indexer.pipeline.DATA_DIR", Path(tmpdir)):
                index_repo("timer_test", incremental=False,
                           progress_callback=capture_progress)

        # Should have setup messages
        setup_msgs = [m for m in messages if "Setup" in m]
        self.assertTrue(len(setup_msgs) >= 2, f"Expected setup messages, got: {messages}")
        # Should contain scanning and loading steps
        self.assertTrue(any("Scanning" in m for m in setup_msgs))
        self.assertTrue(any("Done" in m for m in setup_msgs))


# ── Payload periodic flush ──────────────────────────────────────────────────

class TestPayloadPeriodicFlush(unittest.TestCase):
    """Test that payload store is flushed periodically, not just at end."""

    @patch("src.indexer.pipeline.get_collection")
    @patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100)
    def test_periodic_flush_occurs(self, mock_batch, mock_coll):
        """save_payloads is called during indexing for repos with enough files."""
        from src.indexer.pipeline import index_repo

        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            # Create 25 files — flush interval is max(1, 25//20) = 1,
            # so it should flush frequently
            for i in range(25):
                (src / f"mod{i}.py").write_text(f"# module {i}", encoding="utf-8")

            config = RepoConfig(
                name="flush_test", root=root, collection_name="flush_test_code",
                source_dirs=[SourceDirConfig(path=src, language="python")],
            )
            with patch("src.indexer.pipeline.REPOS", {"flush_test": config}), \
                 patch("src.indexer.pipeline.DATA_DIR", Path(tmpdir)), \
                 patch("src.indexer.pipeline.save_payloads") as mock_save:
                index_repo("flush_test", incremental=False)

            # save_payloads should have been called at least once
            # (periodic flushes + possible final flush)
            self.assertTrue(mock_save.called,
                            "save_payloads should be called for periodic flush")


# ── Full reindex uses empty payload store ───────────────────────────────────

class TestFullReindexEmptyPayloadStore(unittest.TestCase):
    """Test that full reindex starts with empty payload store."""

    @patch("src.indexer.pipeline.get_collection")
    @patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100)
    def test_full_reindex_does_not_load_old_payloads(self, mock_batch, mock_coll):
        """On incremental=False, load_payloads should not be called."""
        from src.indexer.pipeline import index_repo

        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "app.py").write_text("print('hi')", encoding="utf-8")

            # Pre-seed a payload file that should NOT be loaded
            payload_path = Path(tmpdir) / "empty_test_chunk_payloads.json"
            payload_path.write_text('{"stale_chunk": "stale data"}', encoding="utf-8")

            config = RepoConfig(
                name="empty_test", root=root, collection_name="empty_test_code",
                source_dirs=[SourceDirConfig(path=src, language="python")],
            )
            with patch("src.indexer.pipeline.REPOS", {"empty_test": config}), \
                 patch("src.indexer.pipeline.DATA_DIR", Path(tmpdir)), \
                 patch("src.indexer.pipeline.load_payloads") as mock_load:
                mock_load.return_value = {"stale_chunk": "stale data"}
                index_repo("empty_test", incremental=False)

            # load_payloads should NOT be called for full reindex
            mock_load.assert_not_called()

    @patch("src.indexer.pipeline.get_collection")
    @patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100)
    def test_incremental_reindex_loads_payloads(self, mock_batch, mock_coll):
        """On incremental=True, load_payloads IS called."""
        from src.indexer.pipeline import index_repo

        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "src"
            src.mkdir()
            (src / "app.py").write_text("print('hi')", encoding="utf-8")

            config = RepoConfig(
                name="inc_test", root=root, collection_name="inc_test_code",
                source_dirs=[SourceDirConfig(path=src, language="python")],
            )
            with patch("src.indexer.pipeline.REPOS", {"inc_test": config}), \
                 patch("src.indexer.pipeline.DATA_DIR", Path(tmpdir)), \
                 patch("src.indexer.pipeline.load_payloads", return_value={}) as mock_load:
                index_repo("inc_test", incremental=True)

            mock_load.assert_called_once()


# ── index_management result formatter ───────────────────────────────────────

class TestResultFormatter(unittest.TestCase):
    """Test _format_index_result shows setup + indexing time split."""

    def test_format_includes_time_split(self):
        from src.tools.index_management import _format_index_result

        result = {
            "repo": "testrepo",
            "files_scanned": 100,
            "files_indexed": 95,
            "files_skipped": 5,
            "chunks_added": 300,
            "chunks_deleted": 0,
            "total_chunks": 300,
            "errors": [],
            "setup_seconds": 0.12,
            "elapsed_seconds": 45.3,
        }
        formatted = _format_index_result(result)
        self.assertIn("setup 0.12s", formatted)
        self.assertIn("indexing 45.3s", formatted)
        # Total should be sum
        self.assertIn("45.4s", formatted)

    def test_format_without_setup_seconds(self):
        """Backward compat: if setup_seconds is missing, default to 0."""
        from src.tools.index_management import _format_index_result

        result = {
            "repo": "testrepo",
            "files_scanned": 10,
            "files_indexed": 10,
            "files_skipped": 0,
            "chunks_added": 30,
            "chunks_deleted": 0,
            "total_chunks": 30,
            "errors": [],
            "elapsed_seconds": 5.0,
        }
        formatted = _format_index_result(result)
        self.assertIn("setup 0s", formatted)
        self.assertIn("indexing 5.0s", formatted)

    def test_format_error_result(self):
        from src.tools.index_management import _format_index_result

        result = {"repo": "bad", "error": "Source directory not found"}
        formatted = _format_index_result(result)
        self.assertIn("Error", formatted)
        self.assertIn("Source directory not found", formatted)


# ── Reverse index correctness at scale ──────────────────────────────────────

class TestPayloadReverseIndexScale(unittest.TestCase):
    """Test that the reverse index approach scales correctly (no O(n²) scan)."""

    def test_many_files_no_cross_contamination(self):
        """With many files, updating one doesn't scan or corrupt others."""
        from src.indexer.pipeline import _update_payloads_in_memory, _remove_payloads_in_memory

        store = {}
        idx = {}
        large = "x" * (MAX_EMBED_CHARS + 1)

        # Add 100 files
        for i in range(100):
            fp = f"src/File{i}.cs"
            chunks = [_make_chunk(fp, f"Class{i}", "Method", large, start_line=i)]
            _update_payloads_in_memory(store, idx, fp, chunks)

        self.assertEqual(len(store), 100)
        self.assertEqual(len(idx), 100)

        # Remove file 50 — only its entry should be gone
        _remove_payloads_in_memory(store, idx, "src/File50.cs")
        self.assertEqual(len(store), 99)
        self.assertNotIn("src/File50.cs", idx)

        # File 49 and 51 should still be present
        self.assertIn("src/File49.cs", idx)
        self.assertIn("src/File51.cs", idx)

        # Re-index file 0 with different content
        new_chunks = [_make_chunk("src/File0.cs", "Class0", "NewMethod", large, start_line=999)]
        _update_payloads_in_memory(store, idx, "src/File0.cs", new_chunks)
        self.assertEqual(len(store), 99)  # replaced, not added
        self.assertEqual(len(idx["src/File0.cs"]), 1)
        self.assertEqual(idx["src/File0.cs"][0], new_chunks[0].chunk_id)

    def test_reverse_index_performance_linear(self):
        """Verify that operations complete in reasonable time for large stores.

        With the old O(n²) prefix scan, 5000 files would take seconds.
        With the reverse index, it should be near-instant.
        """
        from src.indexer.pipeline import _update_payloads_in_memory

        store = {}
        idx = {}
        large = "x" * (MAX_EMBED_CHARS + 1)

        t0 = time.perf_counter()
        for i in range(5000):
            fp = f"src/deep/path/File{i}.cs"
            chunks = [_make_chunk(fp, f"C{i}", "M", large, start_line=i)]
            _update_payloads_in_memory(store, idx, fp, chunks)
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(store), 5000)
        # Should complete in well under 5 seconds (old approach would take 30s+)
        self.assertLess(elapsed, 5.0,
                        f"5000 payload updates took {elapsed:.1f}s — likely O(n²) regression")


# ── Sequence length bucketing (VRAM leak fix) ──────────────────────────────

class TestSequenceLengthBucketing(unittest.TestCase):
    """Test that ONNX inputs are padded to fixed bucket sizes."""

    def test_bucket_pad_rounds_up(self):
        """Inputs are padded to the next bucket boundary."""
        import numpy as np
        from src.indexer.embedder import CodeRankEmbedder

        cases = [
            (7, 64),
            (64, 64),
            (65, 128),
            (100, 128),
            (128, 128),
            (129, 256),
            (500, 512),
            (513, 1024),
            (1000, 1024),
            (2000, 2048),
            (4000, 4096),
            (7000, 8192),
        ]
        for seq_len, expected_bucket in cases:
            ids = np.ones((1, seq_len), dtype=np.int64)
            mask = np.ones((1, seq_len), dtype=np.int64)
            ids_p, mask_p = CodeRankEmbedder._bucket_pad(ids, mask)
            self.assertEqual(ids_p.shape[1], expected_bucket,
                             f"seq_len={seq_len} should bucket to {expected_bucket}, got {ids_p.shape[1]}")

    def test_bucket_pad_preserves_content(self):
        """Original token values are preserved, padding positions are 0."""
        import numpy as np
        from src.indexer.embedder import CodeRankEmbedder

        ids = np.array([[101, 2054, 2003, 102]], dtype=np.int64)  # 4 tokens
        mask = np.array([[1, 1, 1, 1]], dtype=np.int64)
        ids_p, mask_p = CodeRankEmbedder._bucket_pad(ids, mask)

        # Original content preserved
        self.assertEqual(ids_p[0, 0], 101)
        self.assertEqual(ids_p[0, 3], 102)
        self.assertEqual(mask_p[0, 0], 1)
        self.assertEqual(mask_p[0, 3], 1)
        # Padding is zeros
        self.assertEqual(ids_p[0, 4], 0)
        self.assertEqual(mask_p[0, 4], 0)
        # Padded to 64
        self.assertEqual(ids_p.shape[1], 64)

    def test_bucket_pad_exact_match_no_copy(self):
        """When seq_len exactly matches a bucket, no padding is needed."""
        import numpy as np
        from src.indexer.embedder import CodeRankEmbedder

        ids = np.ones((2, 256), dtype=np.int64)
        mask = np.ones((2, 256), dtype=np.int64)
        ids_p, mask_p = CodeRankEmbedder._bucket_pad(ids, mask)
        self.assertEqual(ids_p.shape[1], 256)
        # Should be the same object (no copy)
        self.assertIs(ids_p, ids)
        self.assertIs(mask_p, mask)

    def test_bucket_pad_batch_dimension_preserved(self):
        """Batch dimension is not affected by padding."""
        import numpy as np
        from src.indexer.embedder import CodeRankEmbedder

        for batch_size in [1, 4, 16]:
            ids = np.ones((batch_size, 100), dtype=np.int64)
            mask = np.ones((batch_size, 100), dtype=np.int64)
            ids_p, mask_p = CodeRankEmbedder._bucket_pad(ids, mask)
            self.assertEqual(ids_p.shape[0], batch_size)
            self.assertEqual(mask_p.shape[0], batch_size)
            self.assertEqual(ids_p.shape[1], 128)

    def test_bucket_pad_beyond_max_bucket(self):
        """Sequences longer than max bucket (8192) are not padded further."""
        import numpy as np
        from src.indexer.embedder import CodeRankEmbedder

        ids = np.ones((1, 8192), dtype=np.int64)
        mask = np.ones((1, 8192), dtype=np.int64)
        ids_p, mask_p = CodeRankEmbedder._bucket_pad(ids, mask)
        self.assertEqual(ids_p.shape[1], 8192)
        self.assertIs(ids_p, ids)


if __name__ == "__main__":
    unittest.main()
