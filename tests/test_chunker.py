"""Tests for the tree-sitter C# chunker."""

from src.indexer.chunker import chunk_file


def test_small_file_whole_class():
    source = b"""
namespace Augmentus.Models
{
    public class Point3D
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
}
"""
    chunks = chunk_file(source, "Source/Models/Point3D.cs", "Models")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "Point3D"
    assert chunks[0].namespace == "Augmentus.Models"
    assert chunks[0].module == "Models"
    print("PASS: test_small_file_whole_class")


def test_enum_always_whole():
    source = b"""
namespace Augmentus
{
    public enum RobotState
    {
        Idle,
        Moving,
        Error,
        Homing
    }
}
"""
    chunks = chunk_file(source, "Source/RobotState.cs", "")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "whole_class"
    assert chunks[0].class_name == "RobotState"
    print("PASS: test_enum_always_whole")


def test_large_file_method_chunks():
    """Files >= 150 lines should produce class_summary + method chunks."""
    lines = ["using System;", "", "namespace Augmentus.Robotics", "{"]
    lines.append("    public class BigDriver : IRobotDriver")
    lines.append("    {")
    lines.append("        private bool _connected;")
    for i in range(20):
        lines.append(f"        public void Action{i}()")
        lines.append("        {")
        for j in range(5):
            lines.append(f"            var x{j} = Process({j});")
        lines.append("        }")
        lines.append("")
    lines.append("    }")
    lines.append("}")

    source = "\n".join(lines).encode("utf-8")
    chunks = chunk_file(source, "Source/Robotics/BigDriver.cs", "Robotics")

    # Should have 1 class_summary + 20 methods
    assert len(chunks) == 21, f"Expected 21 chunks, got {len(chunks)}"
    types = [c.chunk_type for c in chunks]
    assert types[0] == "class_summary"
    assert types.count("method") == 20
    assert all(c.namespace == "Augmentus.Robotics" for c in chunks)
    assert all(c.module == "Robotics" for c in chunks)
    print("PASS: test_large_file_method_chunks")


def test_file_scoped_namespace():
    source = b"""
namespace Augmentus.Commands;

public class MoveRobot
{
    public string TargetPosition { get; set; }
    public void Execute() { }
}
"""
    chunks = chunk_file(source, "Source/Commands/MoveRobot.cs", "Commands")
    assert len(chunks) == 1
    assert chunks[0].namespace == "Augmentus.Commands"
    print("PASS: test_file_scoped_namespace")


def test_interface_chunk():
    source = b"""
namespace Augmentus.Network
{
    public interface INetworkHandler
    {
        void Connect(string host, int port);
        void Disconnect();
        bool IsConnected { get; }
    }
}
"""
    chunks = chunk_file(source, "Source/Network/INetworkHandler.cs", "Network")
    assert len(chunks) == 1
    assert chunks[0].class_name == "INetworkHandler"
    print("PASS: test_interface_chunk")


def test_base_types_extracted():
    source = b"""
namespace Augmentus
{
    public class AbbDriver : RobotDriverBase, IRobotDriver, IDisposable
    {
        public void Connect() { }
    }
}
"""
    chunks = chunk_file(source, "Source/AbbDriver.cs", "")
    assert len(chunks) == 1
    assert "RobotDriverBase" in chunks[0].base_types[0]
    print("PASS: test_base_types_extracted")


def test_chunk_id_uniqueness():
    source = b"""
namespace Augmentus
{
    public class Foo
    {
        public void Bar() { }
    }
}
"""
    chunks = chunk_file(source, "Source/Foo.cs", "")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
    print("PASS: test_chunk_id_uniqueness")


def test_embedding_text_format():
    source = b"""
namespace Augmentus.Network
{
    /// <summary>Handles TCP connections.</summary>
    public class TcpHandler : IDisposable
    {
        public void Connect() { }
    }
}
"""
    chunks = chunk_file(source, "Source/Network/TcpHandler.cs", "Network")
    text = chunks[0].embedding_text
    assert "Augmentus.Network" in text
    assert "TcpHandler" in text
    assert "File:" in text
    assert "Module: Network" in text
    print("PASS: test_embedding_text_format")


if __name__ == "__main__":
    test_small_file_whole_class()
    test_enum_always_whole()
    test_large_file_method_chunks()
    test_file_scoped_namespace()
    test_interface_chunk()
    test_base_types_extracted()
    test_chunk_id_uniqueness()
    test_embedding_text_format()
    print("\nAll chunker tests passed!")
