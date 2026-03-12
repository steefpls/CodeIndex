"""ZenjectBinding dataclass for DI binding information."""

from dataclasses import dataclass


@dataclass
class ZenjectBinding:
    interface_name: str
    concrete_type: str
    scope: str  # "Singleton", "Transient", "Cached", etc.
    binding_type: str  # "Bind", "BindInterfacesTo", "BindInterfacesAndSelfTo", "BindFactory", "DeclareSignal"
    installer_file: str
    line_number: int

    def to_dict(self) -> dict:
        return {
            "interface_name": self.interface_name,
            "concrete_type": self.concrete_type,
            "scope": self.scope,
            "binding_type": self.binding_type,
            "installer_file": self.installer_file,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ZenjectBinding":
        return cls(**d)
