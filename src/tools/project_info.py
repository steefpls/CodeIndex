"""Unity project metadata: version, packages, build scenes, scripting defines."""

import json
import logging
import re
from pathlib import Path

from src.config import REPOS, resolve_repo

logger = logging.getLogger(__name__)


def get_project_info(repo: str = "mainapp") -> str:
    """Get Unity project metadata.

    Reads ProjectVersion.txt, manifest.json, ProjectSettings.asset,
    and EditorBuildSettings.asset.

    Args:
        repo: Which repo to query.

    Returns:
        Formatted project information.
    """
    resolved = resolve_repo(repo)
    if resolved not in REPOS:
        return f"Unknown repo: '{repo}'. Available: {list(REPOS.keys())}"

    config = REPOS[resolved]
    project_root = config.root / "UnityProject"

    if not project_root.exists():
        return f"No UnityProject directory found in '{resolved}'."

    lines = [f"=== Project Info: {resolved} ==="]

    # Unity version
    version_file = project_root / "ProjectSettings" / "ProjectVersion.txt"
    if version_file.exists():
        try:
            text = version_file.read_text(encoding="utf-8")
            for line in text.split("\n"):
                if line.startswith("m_EditorVersion:"):
                    version = line.split(":", 1)[1].strip()
                    lines.append(f"\nUnity Version: {version}")
                    break
        except OSError:
            lines.append("\nUnity Version: N/A")
    else:
        lines.append("\nUnity Version: N/A")

    # Packages
    manifest_file = project_root / "Packages" / "manifest.json"
    if manifest_file.exists():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            deps = data.get("dependencies", {})
            lines.append(f"\nPackages ({len(deps)}):")
            for pkg, ver in sorted(deps.items()):
                lines.append(f"  {pkg}: {ver}")
        except (json.JSONDecodeError, OSError):
            lines.append("\nPackages: N/A")
    else:
        lines.append("\nPackages: N/A")

    # Scripting defines
    settings_file = project_root / "ProjectSettings" / "ProjectSettings.asset"
    if settings_file.exists():
        try:
            text = settings_file.read_text(encoding="utf-8", errors="replace")
            defines_match = re.search(r"scriptingDefineSymbols:.*?\n((?:\s+\d+:.*\n)*)", text)
            if defines_match:
                defines_text = defines_match.group(1).strip()
                if defines_text:
                    defines = []
                    for line in defines_text.split("\n"):
                        parts = line.strip().split(":", 1)
                        if len(parts) == 2:
                            defines.append(parts[1].strip())
                    if defines:
                        lines.append(f"\nScripting Defines: {'; '.join(defines)}")
                    else:
                        lines.append("\nScripting Defines: (none)")
                else:
                    lines.append("\nScripting Defines: (none)")
            else:
                lines.append("\nScripting Defines: (none)")
        except OSError:
            lines.append("\nScripting Defines: N/A")
    else:
        lines.append("\nScripting Defines: N/A")

    # Build scenes
    build_settings = project_root / "ProjectSettings" / "EditorBuildSettings.asset"
    if build_settings.exists():
        try:
            text = build_settings.read_text(encoding="utf-8", errors="replace")
            scene_matches = re.findall(r"path:\s*(Assets/Scenes/[^\n]+)", text)
            if scene_matches:
                lines.append(f"\nBuild Scenes ({len(scene_matches)}):")
                for i, scene in enumerate(scene_matches, 1):
                    lines.append(f"  {i}. {scene.strip()}")
            else:
                lines.append("\nBuild Scenes: (none)")
        except OSError:
            lines.append("\nBuild Scenes: N/A")
    else:
        lines.append("\nBuild Scenes: N/A")

    return "\n".join(lines)
