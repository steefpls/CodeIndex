"""CodeChunk dataclass representing a parsed code unit."""

from dataclasses import dataclass, field


@dataclass
class CodeChunk:
    file_path: str
    class_name: str
    method_name: str | None  # None for whole_class / class_summary chunks
    namespace: str
    start_line: int
    end_line: int
    source: str  # raw source code
    chunk_type: str  # "whole_class", "class_summary", "method", "constructor", "property", "function", "component", "template", "prefab_summary", "gameobject", "scriptable_object", "css_rule"
    module: str = ""
    doc_comment: str = ""
    base_types: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        """Unique ID for ChromaDB. Uses file path + class + method + lines."""
        safe_path = self.file_path.replace("\\", "/").replace("/", "_").replace(":", "")
        parts = [safe_path, self.class_name]
        if self.method_name:
            parts.append(self.method_name)
        parts.append(f"L{self.start_line}-{self.end_line}")
        return "__".join(parts)

    @property
    def embedding_text(self) -> str:
        """Front-loaded structural context for better embedding quality."""
        header_parts = []
        if self.namespace:
            header_parts.append(self.namespace)
        header_parts.append(self.class_name)
        if self.base_types:
            header_parts.append(": " + ", ".join(self.base_types))

        header = " ".join(header_parts)
        file_line = f"File: {self.file_path} | Module: {self.module}" if self.module else f"File: {self.file_path}"

        lines = [header, file_line]
        if self.doc_comment:
            lines.append("")
            lines.append(self.doc_comment)
        lines.append("")
        source = self.source[:MAX_EMBED_CHARS]
        lines.append(source)
        text = "\n".join(lines)
        # Hard cap at MAX_EMBED_CHARS to stay within token limits
        return text[:MAX_EMBED_CHARS]

    @property
    def metadata(self) -> dict:
        """Metadata dict for ChromaDB storage."""
        # Truncate source for metadata (ChromaDB metadata values must be
        # scalar types; large strings are fine but we cap for sanity).
        source_for_meta = self.source[:MAX_EMBED_CHARS]
        return {
            "file_path": self.file_path,
            "class_name": self.class_name,
            "method_name": self.method_name or "",
            "namespace": self.namespace,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "module": self.module,
            "doc_comment": self.doc_comment,
            "base_types": ",".join(self.base_types),
            "source": source_for_meta,
        }


# Keep embedding text under token limit
MAX_EMBED_CHARS = 5000
