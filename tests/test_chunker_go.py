"""Tests for the tree-sitter Go chunker."""

from src.indexer.chunker_go import chunk_file_go


def test_small_struct_whole_class():
    """Small file with a struct -> whole_class chunk."""
    source = b"""package geom

type Point3D struct {
    X float64
    Y float64
    Z float64
}
"""
    chunks = chunk_file_go(source, "geom/point.go", "geom")
    type_chunks = [c for c in chunks if c.class_name == "Point3D"]
    assert len(type_chunks) == 1
    assert type_chunks[0].chunk_type == "whole_class"
    assert type_chunks[0].namespace == "geom"


def test_interface_whole_class():
    """Interface types should be captured."""
    source = b"""package io

type Greeter interface {
    Greet() string
    Farewell(name string) string
}
"""
    chunks = chunk_file_go(source, "io/greeter.go", "io")
    assert any(c.class_name == "Greeter" for c in chunks)


def test_method_attached_to_receiver():
    """Methods should be attached to their receiver type as class_name."""
    source = b"""package foo

type Bar struct {
    X int
}

// Do returns the stored X value.
func (b *Bar) Do() int {
    result := b.X
    return result
}

func (b *Bar) Set(v int) {
    b.X = v
    return
}
"""
    chunks = chunk_file_go(source, "foo/bar.go", "foo")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert len(methods) == 2
    assert all(m.class_name == "Bar" for m in methods)
    assert {m.method_name for m in methods} == {"Do", "Set"}


def test_top_level_functions():
    """Top-level functions use file stem as class_name and chunk_type=function."""
    source = b"""package utils

func CalculateDistance(a, b float64) float64 {
    diff := a - b
    return diff
}

func CalculateArea(w, h float64) float64 {
    area := w * h
    return area
}
"""
    chunks = chunk_file_go(source, "utils/math.go", "utils")
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 2
    assert all(c.class_name == "math" for c in func_chunks)
    assert {c.method_name for c in func_chunks} == {"CalculateDistance", "CalculateArea"}


def test_doc_comment_extracted():
    """// comments directly preceding a declaration should be captured."""
    source = b"""package foo

// Widget represents a display element.
// Used for UI composition.
type Widget struct {
    Name string
}
"""
    chunks = chunk_file_go(source, "foo/widget.go", "foo")
    widget_chunks = [c for c in chunks if c.class_name == "Widget"]
    assert len(widget_chunks) == 1
    assert "display element" in widget_chunks[0].doc_comment


def test_large_file_class_summary():
    """Large files produce class_summary + method chunks with signatures."""
    lines = ["package big", ""]
    lines.append("type Processor struct {")
    lines.append("    data []byte")
    lines.append("}")
    lines.append("")
    for i in range(20):
        lines.append(f"func (p *Processor) Process{i}(v int) int {{")
        for j in range(5):
            lines.append(f"    tmp{j} := v * {j}")
        lines.append("    return v")
        lines.append("}")
        lines.append("")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_go(source, "big/processor.go", "big")

    summaries = [c for c in chunks if c.chunk_type == "class_summary" and c.class_name == "Processor"]
    methods = [c for c in chunks if c.chunk_type == "method" and c.class_name == "Processor"]
    assert len(summaries) == 1
    assert len(methods) == 20


def test_pointer_and_value_receivers():
    """Both (b *Bar) and (b Bar) receivers should resolve to Bar."""
    source = b"""package foo

type Bar struct{ X int }

func (b *Bar) Pointer() int {
    result := b.X
    return result
}

func (b Bar) Value() int {
    result := b.X
    return result
}
"""
    chunks = chunk_file_go(source, "foo/bar.go", "foo")
    methods = [c for c in chunks if c.chunk_type == "method"]
    assert len(methods) == 2
    assert all(m.class_name == "Bar" for m in methods)


def test_empty_file_fallback():
    """Files with only package and imports still get a fallback chunk."""
    source = b"""package prelude

import (
    "fmt"
    "os"
)
"""
    chunks = chunk_file_go(source, "prelude/prelude.go", "prelude")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "prelude"


def test_namespace_from_package():
    """Namespace should be the package name, not the file path."""
    source = b"""package whatsapp

type Client struct {
    Name string
}
"""
    chunks = chunk_file_go(source, "deeply/nested/path/client.go", "whatsapp")
    client_chunks = [c for c in chunks if c.class_name == "Client"]
    assert client_chunks[0].namespace == "whatsapp"


if __name__ == "__main__":
    test_small_struct_whole_class()
    test_interface_whole_class()
    test_method_attached_to_receiver()
    test_top_level_functions()
    test_doc_comment_extracted()
    test_large_file_class_summary()
    test_pointer_and_value_receivers()
    test_empty_file_fallback()
    test_namespace_from_package()
    print("\nAll Go chunker tests passed!")
