"""Tests for the Zenject binding parser."""

import os
import tempfile
from pathlib import Path

from src.indexer.zenject_parser import parse_installer_file


def _parse_source(source: str):
    """Helper: write source to a temp file, parse it, clean up."""
    fd, path = tempfile.mkstemp(suffix=".cs", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        return parse_installer_file(Path(path))
    finally:
        os.unlink(path)


def test_bind_to():
    source = """
using Zenject;

public class GameInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.Bind<IRobotDriver>().To<AbbRobotDriver>().AsSingle();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.interface_name == "IRobotDriver"
    assert b.concrete_type == "AbbRobotDriver"
    assert b.binding_type == "Bind"
    assert b.scope == "AsSingle"


def test_bind_interfaces_to():
    source = """
public class ServiceInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.BindInterfacesTo<NetworkService>().AsSingle().NonLazy();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.concrete_type == "NetworkService"
    assert b.binding_type == "BindInterfacesTo"
    # BindInterfacesTo stores concrete type as interface_name
    # (actual interfaces aren't known from installer alone)
    assert b.interface_name == "NetworkService"


def test_bind_factory():
    source = """
public class FactoryInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.BindFactory<Robot, Robot.Factory>();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.binding_type == "BindFactory"
    assert b.concrete_type == "Robot"
    assert b.interface_name == "Robot.Factory"


def test_declare_signal():
    source = """
public class SignalInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.DeclareSignal<RobotMovedSignal>();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.binding_type == "DeclareSignal"
    assert b.interface_name == "RobotMovedSignal"


def test_multiple_bindings():
    source = """
public class MultiInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.Bind<ILogger>().To<FileLogger>().AsSingle();
        Container.Bind<ICache>().To<MemoryCache>().AsTransient();
        Container.BindInterfacesAndSelfTo<GameManager>().AsSingle();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 3
    assert bindings[0].interface_name == "ILogger"
    assert bindings[1].interface_name == "ICache"
    assert bindings[2].binding_type == "BindInterfacesAndSelfTo"


def test_multiline_bind():
    """Multi-line fluent bindings should be parsed correctly."""
    source = """
public class MultiLineInstaller : MonoInstaller
{
    public override void InstallBindings()
    {
        Container.Bind<IRobotDriver>()
            .To<AbbRobotDriver>()
            .AsSingle()
            .NonLazy();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.interface_name == "IRobotDriver"
    assert b.concrete_type == "AbbRobotDriver"
    assert b.scope == "AsSingle"


def test_multiline_bind_interfaces_to():
    """Multi-line BindInterfacesTo should work."""
    source = """
public class Installer : MonoInstaller
{
    public override void InstallBindings()
    {
        Container
            .BindInterfacesTo<ScanService>()
            .AsSingle();
    }
}
"""
    bindings = _parse_source(source)
    assert len(bindings) == 1
    assert bindings[0].concrete_type == "ScanService"


if __name__ == "__main__":
    test_bind_to()
    test_bind_interfaces_to()
    test_bind_factory()
    test_declare_signal()
    test_multiple_bindings()
    test_multiline_bind()
    test_multiline_bind_interfaces_to()
    print("\nAll Zenject parser tests passed!")
