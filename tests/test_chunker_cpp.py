"""Tests for the tree-sitter C++ chunker."""

from src.indexer.chunker_cpp import chunk_file_cpp


def test_small_file_whole_class():
    source = b"""
namespace Pipeline {

struct Point3D {
    float x;
    float y;
    float z;
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/Point3D.h", "Pipeline")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "Point3D"
    assert chunks[0].module == "Pipeline"


def test_enum_always_whole():
    source = b"""
namespace Pipeline {

enum class FilterType {
    Gaussian,
    Bilateral,
    Statistical,
    Voxel
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/FilterType.h", "Pipeline")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "FilterType"


def test_large_class_method_chunks():
    """Classes in files >= 150 lines should produce class_summary + method chunks."""
    lines = ["#include <vector>", "", "namespace Pipeline {", ""]
    lines.append("class MeshProcessor {")
    lines.append("public:")
    for i in range(20):
        lines.append(f"    void Process{i}(int param) {{")
        for j in range(5):
            lines.append(f"        auto val{j} = compute({j});")
        lines.append("    }")
        lines.append("")
    lines.append("};")
    lines.append("")
    lines.append("}  // namespace Pipeline")

    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_cpp(source, "Source/MeshProcessor.cpp", "Pipeline")

    # Should have 1 class_summary + 20 methods
    assert len(chunks) == 21, f"Expected 21 chunks, got {len(chunks)}"
    types = [c.chunk_type for c in chunks]
    assert types[0] == "class_summary"
    assert types.count("method") == 20
    assert all(c.module == "Pipeline" for c in chunks)


def test_free_functions():
    """Free functions outside classes should be chunked as methods."""
    lines = ["#include <iostream>", "", "namespace Pipeline {", ""]
    # Add enough lines to exceed small file threshold
    for i in range(20):
        lines.append(f"void FreeFunc{i}(int x) {{")
        for j in range(5):
            lines.append(f"    auto result{j} = process({j});")
        lines.append("}")
        lines.append("")
    lines.append("}  // namespace Pipeline")

    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_cpp(source, "Source/Utils.cpp", "Pipeline")

    assert len(chunks) > 0
    assert all(c.chunk_type == "method" for c in chunks)
    # Free functions should use filename stem as class_name
    assert all(c.class_name == "Utils" for c in chunks)


def test_qualified_function_name():
    """ClassName::Method definitions should extract just the method name."""
    lines = ["#include <string>", "", "namespace Pipeline {", ""]
    for i in range(20):
        lines.append(f"void MeshFilter::Apply{i}(int param) {{")
        for j in range(5):
            lines.append(f"    auto val{j} = filter({j});")
        lines.append("}")
        lines.append("")
    lines.append("}  // namespace Pipeline")

    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_cpp(source, "Source/MeshFilter.cpp", "Pipeline")

    assert len(chunks) > 0
    for c in chunks:
        assert c.class_name == "MeshFilter"
        assert "::" not in c.method_name


def test_base_types_extracted():
    source = b"""
namespace Pipeline {

class VoxelFilter : public FilterBase {
    void Apply() {
        // implementation
    }
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/VoxelFilter.h", "Pipeline")
    assert len(chunks) >= 1
    assert "FilterBase" in chunks[0].base_types


def test_doc_comment_extracted():
    source = b"""
namespace Pipeline {

/// Processes point cloud data for registration.
class PointCloudProcessor {
    int data;
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/PointCloudProcessor.h", "Pipeline")
    assert len(chunks) == 1
    assert "registration" in chunks[0].doc_comment.lower()


def test_namespace_extraction():
    source = b"""
namespace Pipeline {

struct Config {
    int maxIterations;
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/Config.h", "Pipeline")
    assert len(chunks) == 1
    assert chunks[0].namespace == "Pipeline"


def test_empty_file_fallback():
    """Files with no types or functions should still produce a chunk."""
    source = b"""
#include <vector>

// Just some includes and defines
#define MAX_SIZE 100

typedef int PointIndex;
"""
    chunks = chunk_file_cpp(source, "Source/Types.h", "Pipeline")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"


def test_chunk_id_uniqueness():
    source = b"""
namespace Pipeline {

class Foo {
    void Bar() { }
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/Foo.h", "Pipeline")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_forward_declarations_skipped():
    """Forward declarations (no body) should not produce chunks."""
    source = b"""
namespace Pipeline {

class ForwardDeclared;
struct AnotherForward;

struct Actual {
    int value;
};

}  // namespace Pipeline
"""
    chunks = chunk_file_cpp(source, "Source/Decls.h", "Pipeline")
    assert len(chunks) == 1
    assert chunks[0].class_name == "Actual"


if __name__ == "__main__":
    test_small_file_whole_class()
    test_enum_always_whole()
    test_large_class_method_chunks()
    test_free_functions()
    test_qualified_function_name()
    test_base_types_extracted()
    test_doc_comment_extracted()
    test_namespace_extraction()
    test_empty_file_fallback()
    test_chunk_id_uniqueness()
    test_forward_declarations_skipped()
    print("\nAll C++ chunker tests passed!")
