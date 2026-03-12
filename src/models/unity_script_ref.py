"""Rich identity model for Unity script references (GUID-based)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UnityScriptRef:
    """Resolved identity for a Unity script referenced by GUID.

    Carries enough information to disambiguate scripts with the same
    filename-derived class name in different namespaces or assemblies.
    """
    guid: str
    class_name: str       # filename-derived class name (e.g. "RobotDriver")
    script_path: str      # relative path to .cs file
    namespace: str = ""   # declared namespace (best-effort parse, may be empty)
    assembly: str = ""    # .asmdef assembly name (may be empty)

    @property
    def qualified_name(self) -> str:
        """Namespace-qualified class name, or just class_name if no namespace."""
        if self.namespace:
            return f"{self.namespace}.{self.class_name}"
        return self.class_name

    def to_tuple(self) -> tuple[str, str]:
        """Backward-compatible (class_name, script_path) tuple."""
        return (self.class_name, self.script_path)
