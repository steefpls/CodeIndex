"""Tests for embedding backend routing by role (index vs search)."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestBackendRoleRouting(unittest.TestCase):
    def test_get_embedding_function_defaults_search_cpu_index_gpu(self):
        from src.indexer import embedder

        class _FakeEmbedder:
            def __init__(self, backend_mode="auto", role="index"):
                self.backend_mode = backend_mode
                self.role = role
                self.backend = f"fake-{backend_mode}"

            def close(self):
                return None

        embedder._embedding_fns.clear()
        embedder._active_backends.clear()

        with patch("src.indexer.embedder.CodeRankEmbedder", _FakeEmbedder):
            search_ef = embedder.get_embedding_function(role="search")
            index_ef = embedder.get_embedding_function(role="index")

            self.assertEqual(search_ef.backend_mode, "cpu")
            self.assertEqual(index_ef.backend_mode, "gpu")
            self.assertEqual(search_ef.role, "search")
            self.assertEqual(index_ef.role, "index")
            self.assertEqual(embedder.get_active_backend(role="search"), "fake-cpu")
            self.assertEqual(embedder.get_active_backend(role="index"), "fake-gpu")

            embedder.release_embedding_function(role="search")
            embedder.release_embedding_function(role="index")

    def test_search_code_uses_search_role_embedder(self):
        from src.tools.search import search_code
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            src_dir.mkdir()

            fake_repo = RepoConfig(
                name="search_test",
                root=root,
                collection_name="search_test_code",
                source_dirs=[SourceDirConfig(path=src_dir, language="python")],
            )

            fake_collection = MagicMock()
            fake_collection.query.return_value = {
                "ids": [["chunk_1"]],
                "metadatas": [[{
                    "file_path": "src/app.py",
                    "class_name": "App",
                    "method_name": "run",
                    "chunk_type": "method",
                    "module": "Core",
                    "start_line": 1,
                    "end_line": 10,
                    "source": "def run():\n    pass",
                }]],
                "documents": [["def run():\n    pass"]],
                "distances": [[1.0]],
            }

            fake_embedder = MagicMock()
            fake_embedder.embed_queries.return_value = [[0.1, 0.2, 0.3]]

            with patch("src.tools.search.REPOS", {"search_test": fake_repo}), \
                 patch("src.tools.search.resolve_repo", return_value="search_test"), \
                 patch("src.tools.search.get_collection", return_value=fake_collection), \
                 patch("src.tools.search.get_active_backend", return_value="ONNX + CPU (forced)"), \
                 patch("src.tools.search.get_embedding_function", return_value=fake_embedder) as mock_get_ef, \
                 patch("src.tools.search._get_thresholds_cached", return_value={
                     "HIGH": 2.0,
                     "MEDIUM": 4.0,
                     "LOW": 6.0,
                 }):
                result = search_code("find run", repo="search_test", n_results=1, output_format="json")

            payload = json.loads(result)
            self.assertEqual(payload["results"][0]["chunk_id"], "chunk_1")
            mock_get_ef.assert_called_once_with(role="search")
            fake_embedder.embed_queries.assert_called_once()

    def test_search_code_returns_init_message_on_guard_timeout(self):
        from src.tools.search import search_code
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            src_dir.mkdir()

            fake_repo = RepoConfig(
                name="search_test",
                root=root,
                collection_name="search_test_code",
                source_dirs=[SourceDirConfig(path=src_dir, language="python")],
            )

            with patch("src.tools.search.REPOS", {"search_test": fake_repo}), \
                 patch("src.tools.search.resolve_repo", return_value="search_test"), \
                 patch("src.tools.search.get_collection"), \
                 patch("src.tools.search._SEARCH_INIT_MODE", "blocking"), \
                 patch("src.tools.search._get_query_embeddings_with_guard",
                       side_effect=TimeoutError("guard timeout")):
                result = search_code("find run", repo="search_test", n_results=1, output_format="text")

            self.assertIn("initializing", result)

    def test_search_code_retries_once_after_timeout_and_can_succeed(self):
        from src.tools.search import search_code
        from src.config import RepoConfig, SourceDirConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            src_dir.mkdir()

            fake_repo = RepoConfig(
                name="search_test",
                root=root,
                collection_name="search_test_code",
                source_dirs=[SourceDirConfig(path=src_dir, language="python")],
            )

            fake_collection = MagicMock()
            fake_collection.query.return_value = {
                "ids": [["chunk_1"]],
                "metadatas": [[{
                    "file_path": "src/app.py",
                    "class_name": "App",
                    "method_name": "run",
                    "chunk_type": "method",
                    "module": "Core",
                    "start_line": 1,
                    "end_line": 10,
                    "source": "def run():\n    pass",
                }]],
                "documents": [["def run():\n    pass"]],
                "distances": [[1.0]],
            }

            with patch("src.tools.search.REPOS", {"search_test": fake_repo}), \
                 patch("src.tools.search.resolve_repo", return_value="search_test"), \
                 patch("src.tools.search.get_collection", return_value=fake_collection), \
                 patch("src.tools.search._SEARCH_INIT_MODE", "blocking"), \
                 patch("src.tools.search._get_thresholds_cached", return_value={
                     "HIGH": 2.0,
                     "MEDIUM": 4.0,
                     "LOW": 6.0,
                 }), \
                 patch("src.tools.search._get_query_embeddings_with_guard",
                       side_effect=[TimeoutError("guard timeout"), [[0.1, 0.2, 0.3]]]) as mock_emb:
                result = search_code("find run", repo="search_test", n_results=1, output_format="json")

            payload = json.loads(result)
            self.assertEqual(payload["results"][0]["chunk_id"], "chunk_1")
            self.assertEqual(mock_emb.call_count, 2)

    def test_search_init_guard_fast_fails_on_retry_when_already_over_timeout(self):
        import src.tools.search as search_mod

        event = search_mod.threading.Event()  # intentionally unset
        old_started = search_mod.time.monotonic() - 10.0

        with patch("src.tools.search.get_active_backend", return_value="not initialized"), \
             patch("src.tools.search._ensure_search_backend_init_started", return_value=event):
            original_guard = search_mod._SEARCH_INIT_GUARD_SECONDS
            original_started = search_mod._search_init_started_at
            try:
                search_mod._SEARCH_INIT_GUARD_SECONDS = 1.0
                search_mod._search_init_started_at = old_started
                t0 = time.perf_counter()
                with self.assertRaises(TimeoutError):
                    search_mod._get_query_embeddings_with_guard("x")
                elapsed = time.perf_counter() - t0
            finally:
                search_mod._SEARCH_INIT_GUARD_SECONDS = original_guard
                search_mod._search_init_started_at = original_started

        self.assertLess(elapsed, 0.2, "Retry path should fail fast without waiting guard again")

    def test_index_repo_uses_index_role_embedder_and_explicit_embeddings(self):
        from src.config import RepoConfig, SourceDirConfig
        from src.indexer.pipeline import index_repo

        class _FakeEmbedder:
            def __init__(self):
                self.calls = 0

            def __call__(self, docs):
                self.calls += 1
                return [[0.01, 0.02, 0.03] for _ in docs]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_dir = root / "src"
            src_dir.mkdir()
            (src_dir / "app.py").write_text("def hi():\n    return 1\n", encoding="utf-8")

            fake_repo = RepoConfig(
                name="index_test",
                root=root,
                collection_name="index_test_code",
                source_dirs=[SourceDirConfig(path=src_dir, language="python")],
            )

            fake_collection = MagicMock()
            fake_collection.count.return_value = 1
            fake_embedder = _FakeEmbedder()

            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            # Skip calibration path (incremental + file exists)
            (data_dir / "index_test_calibration.json").write_text("{}", encoding="utf-8")

            with patch("src.indexer.pipeline.REPOS", {"index_test": fake_repo}), \
                 patch("src.indexer.pipeline.DATA_DIR", data_dir), \
                 patch("src.indexer.pipeline.get_collection", return_value=fake_collection), \
                 patch("src.indexer.pipeline.get_embedding_function", return_value=fake_embedder) as mock_get_ef, \
                 patch("src.indexer.pipeline.get_embedding_batch_size", return_value=100):
                result = index_repo("index_test", incremental=True)

            self.assertNotIn("error", result)
            mock_get_ef.assert_called_once_with(role="index")
            self.assertGreater(fake_embedder.calls, 0)

            add_calls = fake_collection.add.call_args_list
            self.assertGreater(len(add_calls), 0)
            first_kwargs = add_calls[0].kwargs
            self.assertIn("embeddings", first_kwargs)
            self.assertIn("documents", first_kwargs)

    def test_session_init_strategy_search_cpu_prefers_fast_cold_start(self):
        from src.indexer.embedder import _session_init_strategy, _use_lightweight_profile

        self.assertEqual(_session_init_strategy("search", "cpu"), ("extended", False))
        self.assertEqual(_session_init_strategy("search", "gpu"), ("all", True))
        self.assertEqual(_session_init_strategy("index", "cpu"), ("all", True))
        self.assertEqual(_session_init_strategy("index", "gpu"), ("all", True))
        self.assertTrue(_use_lightweight_profile("search", "cpu"))
        self.assertFalse(_use_lightweight_profile("search", "gpu"))
        self.assertFalse(_use_lightweight_profile("index", "cpu"))


if __name__ == "__main__":
    unittest.main()
