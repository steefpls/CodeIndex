"""Tests for the tree-sitter CSS chunker."""

from src.indexer.chunker_css import chunk_file_css


def test_small_file_whole_class():
    """Small CSS files (< 150 lines) should produce a single whole_class chunk."""
    source = b"""
body {
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}
"""
    chunks = chunk_file_css(source, "styles/main.css", "")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "main"


def test_large_file_rule_chunks():
    """Large CSS files should produce individual css_rule chunks."""
    lines = []
    for i in range(50):
        lines.append(f".class-{i} {{")
        lines.append(f"    color: #{i:06x};")
        lines.append(f"    font-size: {i + 10}px;")
        lines.append("}")
        lines.append("")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/large.css", "")

    rule_chunks = [c for c in chunks if c.chunk_type == "css_rule"]
    assert len(rule_chunks) == 50
    assert all(c.class_name == "large" for c in rule_chunks)


def _filler_rules(count: int) -> list[str]:
    """Generate multi-line CSS rules to push file past 150-line threshold."""
    lines = []
    for i in range(count):
        lines.append(f".filler-{i} {{")
        lines.append(f"    color: #{i:06x};")
        lines.append(f"    font-size: {i + 10}px;")
        lines.append("}")
        lines.append("")
    return lines


def test_media_query():
    """@media statements should be chunked with the condition as method_name."""
    lines = _filler_rules(40)
    lines.append("@media (max-width: 768px) {")
    lines.append("    .container { width: 100%; }")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/responsive.css", "")

    media_chunks = [c for c in chunks if c.method_name and c.method_name.startswith("@media")]
    assert len(media_chunks) == 1
    assert "768px" in media_chunks[0].method_name


def test_keyframes():
    """@keyframes statements should be chunked with animation name."""
    lines = _filler_rules(40)
    lines.append("@keyframes fadeIn {")
    lines.append("    from { opacity: 0; }")
    lines.append("    to { opacity: 1; }")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/animations.css", "")

    kf_chunks = [c for c in chunks if c.method_name and "fadeIn" in c.method_name]
    assert len(kf_chunks) == 1
    assert kf_chunks[0].chunk_type == "css_rule"


def test_css_comment_extracted():
    """Block comments preceding rules should be extracted as doc_comment."""
    lines = _filler_rules(40)
    lines.append("/* Main navigation styles */")
    lines.append(".nav {")
    lines.append("    display: flex;")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/nav.css", "")

    nav_chunks = [c for c in chunks if c.method_name and "nav" in c.method_name.lower()]
    assert len(nav_chunks) >= 1
    assert "navigation" in nav_chunks[0].doc_comment.lower()


def test_selector_as_method_name():
    """Rule selectors should be used as method_name."""
    lines = _filler_rules(40)
    lines.append("header .logo img {")
    lines.append("    width: 200px;")
    lines.append("    height: auto;")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/header.css", "")

    logo_chunks = [c for c in chunks if c.method_name and "logo" in c.method_name]
    assert len(logo_chunks) == 1


def test_empty_file_fallback():
    """Large empty-ish files should still produce a fallback chunk."""
    lines = ["/* empty stylesheet */"] + [""] * 160
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/empty.css", "")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "empty"


def test_long_selector_truncation():
    """Selectors longer than 100 chars should be truncated."""
    # Build a very long selector
    long_selector = ", ".join([f".very-long-class-name-{i}" for i in range(20)])
    lines = _filler_rules(40)
    lines.append(f"{long_selector} {{")
    lines.append("    color: blue;")
    lines.append("}")
    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_css(source, "styles/long.css", "")

    long_chunks = [c for c in chunks if c.method_name and "..." in c.method_name]
    assert len(long_chunks) == 1
    assert len(long_chunks[0].method_name) <= 100


if __name__ == "__main__":
    test_small_file_whole_class()
    test_large_file_rule_chunks()
    test_media_query()
    test_keyframes()
    test_css_comment_extracted()
    test_selector_as_method_name()
    test_empty_file_fallback()
    test_long_selector_truncation()
    print("\nAll CSS chunker tests passed!")
