"""Tests for the get_file_chunks tool."""

import json
import chromadb
from unittest.mock import patch

from src.tools.search import get_file_chunks
from src.tools.search import _format_results, _format_results_json


def _setup_test_collection():
    """Create an in-memory ChromaDB collection with known test data."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("test_code")

    # Add chunks simulating a C# file with class_summary + methods
    collection.add(
        ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4"],
        documents=[
            "class RobotDriver : IRobotDriver, IDisposable { ... }",
            "public async Task<bool> Connect(string ip, int port) { ... }",
            "public void MoveToPosition(Vector3 pos) { ... }",
            "public RobotDriver(ILogger logger) { ... }",
        ],
        metadatas=[
            {
                "file_path": "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
                "class_name": "RobotDriver",
                "method_name": "",
                "chunk_type": "class_summary",
                "module": "Robotics",
                "start_line": 1,
                "end_line": 245,
                "source": "namespace Augmentus.Robotics\nclass RobotDriver : IRobotDriver, IDisposable {\n  // Manages robot communication\n}",
            },
            {
                "file_path": "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
                "class_name": "RobotDriver",
                "method_name": "Connect",
                "chunk_type": "method",
                "module": "Robotics",
                "start_line": 45,
                "end_line": 78,
                "source": "public async Task<bool> Connect(string ip, int port) {\n  // connect logic\n}",
            },
            {
                "file_path": "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
                "class_name": "RobotDriver",
                "method_name": "MoveToPosition",
                "chunk_type": "method",
                "module": "Robotics",
                "start_line": 80,
                "end_line": 120,
                "source": "public void MoveToPosition(Vector3 pos) {\n  // movement\n}",
            },
            {
                "file_path": "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
                "class_name": "RobotDriver",
                "method_name": "RobotDriver",
                "chunk_type": "constructor",
                "module": "Robotics",
                "start_line": 20,
                "end_line": 40,
                "source": "public RobotDriver(ILogger logger) {\n  _logger = logger;\n}",
            },
        ],
    )

    # Add a chunk from a different file to verify filtering
    collection.add(
        ids=["chunk_other"],
        documents=["class OtherClass { ... }"],
        metadatas=[{
            "file_path": "UnityProject/Assets/Scripts/Other/OtherClass.cs",
            "class_name": "OtherClass",
            "method_name": "",
            "chunk_type": "class_summary",
            "module": "Other",
            "start_line": 1,
            "end_line": 50,
            "source": "class OtherClass { }",
        }],
    )

    return client, collection


class _FakeConfig:
    def __init__(self, collection_name):
        self.collection_name = collection_name


def test_get_file_chunks_returns_all_chunks():
    """Should return all chunks for the requested file, not others."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs", repo="test"
        )

    assert "4 total" in result
    assert "RobotDriver" in result
    assert "[class_summary]" in result
    assert "[method]" in result
    assert "[constructor]" in result
    assert "Chunk ID: chunk_1" in result
    assert "Chunk ID: chunk_2" in result
    assert "OtherClass" not in result
    print("PASS: test_get_file_chunks_returns_all_chunks")


def test_get_file_chunks_ordering():
    """class_summary should come first, then constructor, then methods by line."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs", repo="test"
        )

    # class_summary should appear before constructor, which appears before methods
    summary_pos = result.index("[class_summary]")
    constructor_pos = result.index("[constructor]")
    method_pos = result.index("[method]")
    assert summary_pos < constructor_pos < method_pos
    print("PASS: test_get_file_chunks_ordering")


def test_get_file_chunks_not_found():
    """Non-existent file should return a helpful message."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks("NonExistent/File.cs", repo="test")

    assert "No chunks found" in result
    print("PASS: test_get_file_chunks_not_found")


def test_get_file_chunks_unknown_repo():
    """Unknown repo should return error."""
    result = get_file_chunks("some/file.cs", repo="nonexistent_repo_xyz")
    assert "Error" in result or "Unknown repo" in result
    print("PASS: test_get_file_chunks_unknown_repo")


def test_search_formatter_includes_chunk_id():
    """search_code formatter should print chunk IDs for copy/paste chaining."""
    results = {
        "ids": [["chunk_search_1"]],
        "metadatas": [[{
            "file_path": "UnityProject/Assets/Scripts/Foo.cs",
            "class_name": "Foo",
            "method_name": "Bar",
            "chunk_type": "method",
            "module": "Core",
            "start_line": 10,
            "end_line": 20,
            "source": "public void Bar() {}",
        }]],
        "documents": [["public void Bar() {}"]],
        "distances": [[1.0]],
    }

    with patch("src.tools.search._get_thresholds_cached", return_value={
        "HIGH": 2.0,
        "MEDIUM": 4.0,
        "LOW": 6.0,
    }):
        output = _format_results(results, repo="test")

    assert "Chunk ID: chunk_search_1" in output
    print("PASS: test_search_formatter_includes_chunk_id")


def test_search_formatter_json_includes_chunk_id():
    """JSON formatter should include chunk IDs and structured fields."""
    results = {
        "ids": [["chunk_search_1"]],
        "metadatas": [[{
            "file_path": "UnityProject/Assets/Scripts/Foo.cs",
            "class_name": "Foo",
            "method_name": "Bar",
            "chunk_type": "method",
            "module": "Core",
            "start_line": 10,
            "end_line": 20,
            "source": "public void Bar() {}",
        }]],
        "documents": [["public void Bar() {}"]],
        "distances": [[1.0]],
    }

    with patch("src.tools.search._get_thresholds_cached", return_value={
        "HIGH": 2.0,
        "MEDIUM": 4.0,
        "LOW": 6.0,
    }):
        output = _format_results_json(results, repo="test")

    payload = json.loads(output)
    assert payload["results"][0]["chunk_id"] == "chunk_search_1"
    assert payload["results"][0]["file_path"].endswith("Foo.cs")
    print("PASS: test_search_formatter_json_includes_chunk_id")


def test_get_file_chunks_json_mode():
    """get_file_chunks should support structured JSON output."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
            repo="test",
            output_format="json",
        )

    payload = json.loads(result)
    assert payload["total_chunks"] == 4
    assert payload["offset"] == 0
    assert payload["chunks"][0]["chunk_id"] == "chunk_1"
    print("PASS: test_get_file_chunks_json_mode")


def test_get_file_chunks_pagination_limit():
    """limit should restrict how many chunks are returned."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
            repo="test", limit=2,
        )

    assert "4 total" in result
    assert "showing 1-2" in result
    assert "2 more chunks" in result
    # class_summary and constructor should be in the first page (sorted order)
    assert "[class_summary]" in result
    assert "[constructor]" in result
    print("PASS: test_get_file_chunks_pagination_limit")


def test_get_file_chunks_pagination_offset():
    """offset should skip chunks."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
            repo="test", offset=2, limit=2,
        )

    assert "4 total" in result
    assert "showing 3-4" in result
    # Methods should be on page 2 (after class_summary and constructor)
    assert "[method]" in result
    assert "[class_summary]" not in result
    print("PASS: test_get_file_chunks_pagination_offset")


def test_get_file_chunks_pagination_beyond_end():
    """offset past the end should return a helpful message."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
            repo="test", offset=100,
        )

    assert "No more chunks" in result
    assert "total: 4" in result
    print("PASS: test_get_file_chunks_pagination_beyond_end")


def test_get_file_chunks_pagination_json():
    """JSON mode should include pagination metadata."""
    client, collection = _setup_test_collection()

    with patch("src.tools.search.REPOS", {"test": _FakeConfig("test_code")}), \
         patch("src.tools.search.resolve_repo", lambda r: r), \
         patch("src.tools.search.get_collection", lambda name: collection):

        result = get_file_chunks(
            "UnityProject/Assets/Scripts/Robotics/RobotDriver.cs",
            repo="test", offset=1, limit=2, output_format="json",
        )

    payload = json.loads(result)
    assert payload["total_chunks"] == 4
    assert payload["offset"] == 1
    assert payload["limit"] == 2
    assert payload["returned"] == 2
    assert len(payload["chunks"]) == 2
    print("PASS: test_get_file_chunks_pagination_json")


if __name__ == "__main__":
    test_get_file_chunks_returns_all_chunks()
    test_get_file_chunks_ordering()
    test_get_file_chunks_not_found()
    test_get_file_chunks_unknown_repo()
    test_search_formatter_includes_chunk_id()
    test_search_formatter_json_includes_chunk_id()
    test_get_file_chunks_json_mode()
    test_get_file_chunks_pagination_limit()
    test_get_file_chunks_pagination_offset()
    test_get_file_chunks_pagination_beyond_end()
    test_get_file_chunks_pagination_json()
    print("\nAll tests passed!")
