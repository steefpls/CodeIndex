"""Tests for the tree-sitter TypeScript chunker."""

from src.indexer.chunker_ts import chunk_file_ts


def test_small_class_whole():
    """Small file with a class -> whole_class chunk."""
    source = b"""
class UserService {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    getName(): string {
        return this.name;
    }
}
"""
    chunks = chunk_file_ts(source, "src/services/user.ts", "services")
    assert len(chunks) >= 1
    class_chunk = [c for c in chunks if c.class_name == "UserService"][0]
    assert class_chunk.chunk_type == "whole_class"
    assert class_chunk.module == "services"


def test_interface_always_whole():
    """Interfaces should always be whole_class chunks."""
    source = b"""
export interface IRepository<T> {
    findById(id: string): Promise<T>;
    save(entity: T): Promise<void>;
    delete(id: string): Promise<boolean>;
}
"""
    chunks = chunk_file_ts(source, "src/interfaces.ts", "")
    assert len(chunks) >= 1
    iface_chunks = [c for c in chunks if c.class_name == "IRepository"]
    assert len(iface_chunks) == 1
    assert iface_chunks[0].chunk_type == "whole_class"


def test_type_alias_large():
    """Type aliases >= 3 lines should produce function chunks."""
    source = b"""
export type ComplexConfig = {
    host: string;
    port: number;
    ssl: boolean;
    timeout: number;
};
"""
    chunks = chunk_file_ts(source, "src/types.ts", "")
    type_chunks = [c for c in chunks if c.method_name == "ComplexConfig"]
    assert len(type_chunks) == 1
    assert type_chunks[0].chunk_type == "function"


def test_enum_always_whole():
    """Enums should always be whole_class chunks."""
    source = b"""
export enum Direction {
    Up = "UP",
    Down = "DOWN",
    Left = "LEFT",
    Right = "RIGHT",
}
"""
    chunks = chunk_file_ts(source, "src/enums.ts", "")
    enum_chunks = [c for c in chunks if c.class_name == "Direction"]
    assert len(enum_chunks) == 1
    assert enum_chunks[0].chunk_type == "whole_class"


def test_abstract_class():
    """Abstract class declarations should be detected."""
    source = b"""
export abstract class BaseController {
    abstract handle(req: Request): Response;

    protected log(msg: string): void {
        console.log(msg);
    }
}
"""
    chunks = chunk_file_ts(source, "src/base.ts", "")
    assert len(chunks) >= 1
    class_chunks = [c for c in chunks if c.class_name == "BaseController"]
    assert len(class_chunks) >= 1


def test_tsx_component():
    """PascalCase functions with JSX should be detected as components."""
    source = b"""
export function UserCard({ name, email }: UserCardProps) {
    return (
        <div className="card">
            <h2>{name}</h2>
            <p>{email}</p>
        </div>
    );
}
"""
    chunks = chunk_file_ts(source, "src/components/UserCard.tsx", "")
    assert len(chunks) >= 1
    comp_chunks = [c for c in chunks if c.method_name == "UserCard"]
    assert len(comp_chunks) == 1
    assert comp_chunks[0].chunk_type == "component"


def test_arrow_function_export():
    """Exported arrow functions should be indexed."""
    source = b"""
export const formatDate = (date: Date): string => {
    const year = date.getFullYear();
    const month = date.getMonth();
    return `${year}-${month}`;
};
"""
    chunks = chunk_file_ts(source, "src/utils.ts", "")
    func_chunks = [c for c in chunks if c.method_name == "formatDate"]
    assert len(func_chunks) == 1
    assert func_chunks[0].chunk_type == "function"


def test_extends_and_implements():
    """Classes with extends and implements should capture base_types."""
    source = b"""
class AdminService extends BaseService implements IAdmin, ILoggable {
    doAdmin(): void {
        console.log("admin");
    }
}
"""
    chunks = chunk_file_ts(source, "src/admin.ts", "")
    assert len(chunks) >= 1
    class_chunk = [c for c in chunks if c.class_name == "AdminService"][0]
    assert "BaseService" in class_chunk.base_types


def test_large_file_method_chunks():
    """Classes in large files should produce class_summary + method chunks."""
    lines = ["import { Injectable } from '@nestjs/common';", ""]
    lines.append("class BigService {")
    for i in range(20):
        lines.append(f"    process{i}(input: number): number {{")
        for j in range(5):
            lines.append(f"        const val{j} = input * {j};")
        lines.append("        return input;")
        lines.append("    }")
        lines.append("")
    lines.append("}")

    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file_ts(source, "src/big.ts", "")

    summary_chunks = [c for c in chunks if c.chunk_type == "class_summary"]
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(summary_chunks) == 1
    assert len(method_chunks) == 20


def test_minified_file_skip():
    """Files with avg line length > 200 should be skipped."""
    # Create a single very long line
    source = ("const x = " + "a".join(["1"] * 300) + ";").encode("utf-8")
    chunks = chunk_file_ts(source, "dist/bundle.ts", "")
    assert len(chunks) == 0


def test_export_unwrapping():
    """Export-wrapped declarations should be properly unwrapped."""
    source = b"""
export class ExportedClass {
    value: number;

    constructor() {
        this.value = 0;
    }
}

export function exportedFunc(x: number): number {
    const result = x * 2;
    return result;
}
"""
    chunks = chunk_file_ts(source, "src/exported.ts", "")
    class_chunks = [c for c in chunks if c.class_name == "ExportedClass"]
    func_chunks = [c for c in chunks if c.method_name == "exportedFunc"]
    assert len(class_chunks) >= 1
    assert len(func_chunks) == 1


def test_tsdoc_extracted():
    """TSDoc comments (/** ... */) should be extracted."""
    source = b"""
/**
 * Represents a user in the system.
 * Contains authentication data.
 */
export class User {
    name: string;
    email: string;
}
"""
    chunks = chunk_file_ts(source, "src/user.ts", "")
    assert len(chunks) >= 1
    user_chunk = [c for c in chunks if c.class_name == "User"][0]
    assert "user" in user_chunk.doc_comment.lower()


def test_ts_vs_tsx_parser():
    """TS and TSX files should use different parsers but both work."""
    ts_source = b"""
export function add(a: number, b: number): number {
    const sum = a + b;
    return sum;
}
"""
    tsx_source = b"""
export function Button({ label }: { label: string }) {
    return <button>{label}</button>;
}
"""
    ts_chunks = chunk_file_ts(ts_source, "src/math.ts", "")
    tsx_chunks = chunk_file_ts(tsx_source, "src/Button.tsx", "")
    assert len(ts_chunks) >= 1
    assert len(tsx_chunks) >= 1


def test_empty_file_fallback():
    """Files with no declarations should produce a fallback chunk."""
    source = b"""
// Just imports
import { something } from 'somewhere';
"""
    chunks = chunk_file_ts(source, "src/index.ts", "")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "index"


if __name__ == "__main__":
    test_small_class_whole()
    test_interface_always_whole()
    test_type_alias_large()
    test_enum_always_whole()
    test_abstract_class()
    test_tsx_component()
    test_arrow_function_export()
    test_extends_and_implements()
    test_large_file_method_chunks()
    test_minified_file_skip()
    test_export_unwrapping()
    test_tsdoc_extracted()
    test_ts_vs_tsx_parser()
    test_empty_file_fallback()
    print("\nAll TypeScript chunker tests passed!")
