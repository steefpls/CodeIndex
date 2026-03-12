"""Tests for the tree-sitter Rust chunker."""

from src.indexer.chunker_rust import chunk_file_rust


def test_small_struct_whole_class():
    """Small file with a struct -> whole_class chunk."""
    source = b"""
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
"""
    chunks = chunk_file_rust(source, "src/point.rs", "geometry")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "Point3D"
    assert chunks[0].module == "geometry"


def test_enum_always_whole():
    """Enums should always be whole_class regardless of file size."""
    lines = ["// padding"] * 160
    lines.append("pub enum Color {")
    lines.append("    Red,")
    lines.append("    Green,")
    lines.append("    Blue,")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_rust(source, "src/color.rs", "")
    # Should have enum as whole_class (plus possibly a fallback for padding)
    enum_chunks = [c for c in chunks if c.class_name == "Color"]
    assert len(enum_chunks) == 1
    assert enum_chunks[0].chunk_type == "whole_class"


def test_large_impl_with_methods():
    """Large file with impl block -> class_summary + method chunks."""
    lines = []
    lines.append("pub struct Processor {")
    lines.append("    data: Vec<u8>,")
    lines.append("}")
    lines.append("")
    lines.append("impl Processor {")
    for i in range(20):
        lines.append(f"    pub fn process_{i}(&self, input: i32) -> i32 {{")
        for j in range(5):
            lines.append(f"        let val_{j} = input * {j};")
        lines.append("        input")
        lines.append("    }")
        lines.append("")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_rust(source, "src/processor.rs", "")

    # Should have struct_summary + impl_summary + 20 methods
    summary_chunks = [c for c in chunks if c.chunk_type == "class_summary"]
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(summary_chunks) >= 1
    assert len(method_chunks) == 20
    assert all(c.class_name == "Processor" for c in method_chunks)


def test_trait_declaration():
    """Traits should capture method signatures and supertraits."""
    source = b"""
pub trait Drawable: Clone + Debug {
    fn draw(&self);
    fn resize(&mut self, width: u32, height: u32);
}
"""
    chunks = chunk_file_rust(source, "src/traits.rs", "")
    assert len(chunks) >= 1
    trait_chunk = chunks[0]
    assert trait_chunk.class_name == "Drawable"
    assert "Clone" in trait_chunk.base_types
    assert "Debug" in trait_chunk.base_types


def test_trait_impl_base_types():
    """impl Trait for Struct should set base_types to the trait."""
    source = b"""
pub struct Circle {
    radius: f64,
}

impl Drawable for Circle {
    fn draw(&self) {
        println!("Drawing circle");
    }
}
"""
    chunks = chunk_file_rust(source, "src/circle.rs", "")
    impl_chunks = [c for c in chunks if c.class_name == "Circle" and c.base_types]
    assert len(impl_chunks) >= 1
    assert "Drawable" in impl_chunks[0].base_types


def test_top_level_functions():
    """Top-level functions should use file stem as class_name."""
    source = b"""
pub fn calculate_distance(a: f64, b: f64) -> f64 {
    let diff = a - b;
    diff.abs()
}

pub fn calculate_area(width: f64, height: f64) -> f64 {
    let result = width * height;
    result
}
"""
    chunks = chunk_file_rust(source, "src/math/utils.rs", "")
    assert len(chunks) >= 1
    assert all(c.chunk_type == "function" for c in chunks)
    assert all(c.class_name == "utils" for c in chunks)


def test_macro_definition():
    """Macro definitions should be indexed as function chunks."""
    source = b"""
macro_rules! my_macro {
    ($x:expr) => {
        println!("{}", $x);
        $x + 1
    };
}
"""
    chunks = chunk_file_rust(source, "src/macros.rs", "")
    assert len(chunks) >= 1
    macro_chunks = [c for c in chunks if c.method_name == "my_macro"]
    assert len(macro_chunks) == 1
    assert macro_chunks[0].chunk_type == "function"


def test_doc_comment_extracted():
    """/// doc comments should be extracted and attached to chunks."""
    source = b"""
/// Represents a 3D point in space.
/// Used for mesh processing.
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
"""
    chunks = chunk_file_rust(source, "src/point.rs", "")
    assert len(chunks) == 1
    assert "3D point" in chunks[0].doc_comment


def test_namespace_from_path():
    """Namespace should be derived from file path using :: separator."""
    source = b"""
pub struct TcpStream {
    fd: i32,
}
"""
    chunks = chunk_file_rust(source, "src/network/tcp.rs", "")
    assert len(chunks) == 1
    assert chunks[0].namespace == "network"


def test_multiple_impls():
    """Multiple impl blocks for the same type should each produce chunks."""
    source = b"""
pub struct Widget {
    name: String,
}

impl Widget {
    pub fn new(name: String) -> Self {
        let w = Widget { name };
        w
    }
}

impl Display for Widget {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.name)?;
        Ok(())
    }
}
"""
    chunks = chunk_file_rust(source, "src/widget.rs", "")
    widget_chunks = [c for c in chunks if c.class_name == "Widget"]
    assert len(widget_chunks) >= 2  # struct + at least 2 impl blocks


def test_empty_file_fallback():
    """Files with no types or functions should produce a fallback chunk."""
    source = b"""
// Just some comments and use statements
use std::io;
use std::collections::HashMap;
"""
    chunks = chunk_file_rust(source, "src/prelude.rs", "")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "prelude"


def test_generic_impl():
    """Generic impl blocks should extract the base type name."""
    source = b"""
pub struct Container<T> {
    items: Vec<T>,
}

impl<T> Container<T> {
    pub fn new() -> Self {
        let items = Vec::new();
        Container { items }
    }

    pub fn add(&mut self, item: T) {
        self.items.push(item);
        println!("added");
    }
}
"""
    chunks = chunk_file_rust(source, "src/container.rs", "")
    container_chunks = [c for c in chunks if c.class_name == "Container"]
    assert len(container_chunks) >= 1


if __name__ == "__main__":
    test_small_struct_whole_class()
    test_enum_always_whole()
    test_large_impl_with_methods()
    test_trait_declaration()
    test_trait_impl_base_types()
    test_top_level_functions()
    test_macro_definition()
    test_doc_comment_extracted()
    test_namespace_from_path()
    test_multiple_impls()
    test_empty_file_fallback()
    test_generic_impl()
    print("\nAll Rust chunker tests passed!")
