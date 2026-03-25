"""Configuration for repo paths, collection names, and constants.

Each repo can have multiple source directories (e.g. C# + C++ in the same repo, or Python + JavaScript).
Repo paths can be overridden or new repos added via config.local.json in the project root.
Example config.local.json (new format):
{
    "repos": {
        "perception": {
            "root": "~/Documents/Augmentus-Perception",
            "source_dirs": [
                {"path": "~/Documents/Augmentus-Perception/Source", "language": "csharp"}
            ]
        },
        "my-project": {
            "root": "~/Documents/MyProject",
            "collection_name": "myproject_code",
            "source_dirs": [
                {"path": "~/Documents/MyProject/src", "language": "csharp", "exclude_patterns": ["bin/", "obj/"]}
            ]
        }
    },
    "aliases": {
        "pipeline": "perception"
    }
}

Augmentus repos are auto-discovered at startup. Set AUGMENTUS_REPO_ROOT env var to
override the search locations, or use add_repo() for any custom project.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_FILE = DATA_DIR / "server.log"

# CodeRankEmbed (code-specific embedding model, 137M params, 768-dim)
CODERANK_MODEL = "nomic-ai/CodeRankEmbed"
CODERANK_QUERY_PREFIX = "Represent this query for searching relevant code: "
CODERANK_ONNX_DIR = DATA_DIR / "coderank_onnx"

# Chunking thresholds
SMALL_FILE_LINE_THRESHOLD = 150
_EMBEDDING_BATCH_SIZE_DEFAULT = 100


def get_embedding_batch_size() -> int:
    """Get optimal ChromaDB embedding batch size based on detected hardware.

    Lazy import to avoid circular dependency (hardware -> config -> hardware).
    Falls back to default if hardware detection hasn't run yet.
    """
    try:
        from src.hardware import get_hardware_profile
        return get_hardware_profile().embedding_batch_size
    except Exception:
        return _EMBEDDING_BATCH_SIZE_DEFAULT


# Keep module-level name for backwards compat with any direct imports,
# but pipeline.py should use get_embedding_batch_size() for dynamic values.
EMBEDDING_BATCH_SIZE = _EMBEDDING_BATCH_SIZE_DEFAULT


def _expand_path(raw: str) -> Path:
    """Expand environment variables and ~ in a path string, then resolve."""
    return Path(os.path.expandvars(raw)).expanduser().resolve()


# Language -> file extensions mapping (canonical source of truth)
LANGUAGE_EXTENSIONS: dict[str, frozenset[str]] = {
    "csharp": frozenset({".cs"}),
    "cpp": frozenset({".cpp", ".h", ".hpp"}),
    "python": frozenset({".py"}),
    "javascript": frozenset({".js", ".jsx", ".mjs"}),
    "html": frozenset({".html", ".htm", ".vue"}),
    "unity": frozenset({".prefab", ".unity", ".asset"}),
    "json": frozenset({".json"}),
    "yaml": frozenset({".yaml", ".yml"}),
    "markdown": frozenset({".md"}),
    "rust": frozenset({".rs"}),
    "typescript": frozenset({".ts", ".tsx"}),
    "css": frozenset({".css"}),
    "lua": frozenset({".lua"}),
}


@dataclass
class SourceDirConfig:
    """A single source directory within a repo, with its own language and exclude patterns."""
    path: Path
    language: str = "csharp"  # "csharp", "cpp", "python", "javascript", "html", "unity", "json", "yaml", "markdown", "rust", "typescript", or "css"
    exclude_patterns: list[str] = field(default_factory=list)
    extensions: frozenset[str] = field(default=None)  # auto-populated from language

    def __post_init__(self):
        if self.extensions is None:
            self.extensions = LANGUAGE_EXTENSIONS.get(self.language, frozenset({".cs"}))


@dataclass
class RepoConfig:
    name: str
    root: Path
    collection_name: str
    source_dirs: list[SourceDirConfig] = field(default_factory=list)
    strip_prefixes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Auto-discovery: Augmentus repo templates (path-independent)
# ---------------------------------------------------------------------------

_AUGMENTUS_REPO_TEMPLATES: dict[str, dict] = {
    "perception": {
        "dir_name": "Augmentus-Perception",
        "collection_name": "perception_code",
        "strip_prefixes": ["Augmentus.", "Augmentus.MainApp."],
        "source_dirs": [
            {
                "relative_path": "Augmentus-Perception/Source",
                "language": "csharp",
                "exclude_patterns": ["bin/", "obj/", "Properties/"],
            },
            {
                "relative_path": "Augmentus-Perception/Forms",
                "language": "csharp",
                "exclude_patterns": ["bin/", "obj/"],
            },
            {
                "relative_path": "Augmentus-Perception.Test",
                "language": "csharp",
                "exclude_patterns": ["bin/", "obj/"],
            },
            {
                "relative_path": "Augmentus-Perception.Extern/3DDataProcessingPipeline",
                "language": "cpp",
                "exclude_patterns": [
                    "boost/", "vcpkg/", "ThirdParty/", "packages/", "out/",
                    "bin/", "obj/", ".vs/", "x64/", ".tests", "open3d/",
                    "CSharp/", "DebugDlls/", "ReleaseDlls/", "FilteringDll/",
                ],
            },
        ],
    },
    "mainapp": {
        "dir_name": "Augmentus-MainApp-U6",
        "collection_name": "mainapp_code",
        "strip_prefixes": ["Augmentus.", "Augmentus.MainApp."],
        "source_dirs": [
            {
                "relative_path": "UnityProject/Assets",
                "language": "csharp",
                "exclude_patterns": [
                    "*.meta", "bin/", "obj/",
                    "Plugins/", "Libraries/", "OPS/", "PlayerPrefsEditor/",
                    "ExternalDependencyManager/", "NuGet/", "TriLib/", "Doozy/",
                    "TextMesh Pro/", "NodyGraphs/", "AddressableAssetsData/",
                    "AsimpL/", "Beebyte/", "Dreamteck/", "Linefy/",
                    "ModernUIPack/", "NVIDIA/", "ProceduralUIImage/",
                    "Shapes/", "TriLibCore/", "UIGradient/", "Vectrosity/",
                    "Packages/", "PlayServicesResolver/", "BuildReports/",
                ],
            },
            {
                "relative_path": "UnityProject/Assets",
                "language": "unity",
                "exclude_patterns": [
                    "*.meta",
                    "Plugins/", "Libraries/", "OPS/", "PlayerPrefsEditor/",
                    "ExternalDependencyManager/", "NuGet/", "TriLib/", "Doozy/",
                    "TextMesh Pro/", "NodyGraphs/", "AddressableAssetsData/",
                    "AsimpL/", "Beebyte/", "Dreamteck/", "Linefy/",
                    "ModernUIPack/", "NVIDIA/", "ProceduralUIImage/",
                    "Shapes/", "TriLibCore/", "UIGradient/", "Vectrosity/",
                    "Packages/", "PlayServicesResolver/", "BuildReports/",
                    "Editor/", "Samples/", "PackageCache/",
                    "StreamingAssets/", "Resources/Fonts/", "Editor Default Resources/",
                ],
            },
        ],
    },
}

# Common parent directories to probe for Augmentus repos.
# AUGMENTUS_REPO_ROOT env var takes highest priority.
_SEARCH_LOCATIONS = [
    "~/Documents",
    "~/source/repos",
    "~/repos",
    "~/dev",
    "~/projects",
]

# ---------------------------------------------------------------------------
# Auto-detect source dirs for add_repo() (extension → language mapping)
# ---------------------------------------------------------------------------

# Reverse mapping for auto-detecting language from file extension
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".cs": "csharp",
    ".cpp": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".html": "html", ".htm": "html", ".vue": "html",
    ".prefab": "unity", ".unity": "unity", ".asset": "unity",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown",
    ".rs": "rust",
    ".ts": "typescript", ".tsx": "typescript",
    ".css": "css",
}

_DEFAULT_EXCLUDE_PATTERNS: dict[str, list[str]] = {
    "csharp": ["bin/", "obj/", "Properties/"],
    "cpp": ["boost/", "vcpkg/", "ThirdParty/", "packages/", "out/", "bin/", "obj/"],
    "python": ["__pycache__/", ".venv/", "venv/", "dist/", "build/", ".egg-info/"],
    "javascript": ["node_modules/", "dist/", "build/", ".next/"],
    "html": ["node_modules/", "dist/"],
    "unity": ["Plugins/", "TextMesh Pro/", "PackageCache/"],
    "json": ["node_modules/", "package-lock.json", ".venv/", "venv/"],
    "yaml": ["node_modules/", ".github/"],
    "markdown": ["node_modules/", ".venv/", "CHANGELOG.md"],
    "rust": ["target/", ".cargo/"],
    "typescript": ["node_modules/", "dist/", "build/", ".next/", "*.d.ts"],
    "css": ["node_modules/", "dist/", "*.min.css"],
}

# Directories to always skip when walking for auto-detection
_ALWAYS_SKIP_DIRS = frozenset({
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    "bin", "obj", ".vs", ".idea", ".vscode", "dist", "build", "out",
    "packages", "PackageCache", ".next", ".egg-info",
})

# Safety limit for auto-detection walk
_AUTO_DETECT_MAX_FILES = 100_000


def _discover_augmentus_repos() -> dict[str, RepoConfig]:
    """Auto-discover Augmentus repos by probing common parent directories.

    For each template in _AUGMENTUS_REPO_TEMPLATES, checks search locations
    (env var first, then _SEARCH_LOCATIONS) for <location>/<dir_name>.
    First match wins. Only repos that exist on disk are returned.
    """
    repos: dict[str, RepoConfig] = {}

    # Build ordered list of parent dirs to probe
    search_dirs: list[Path] = []
    env_root = os.environ.get("AUGMENTUS_REPO_ROOT")
    if env_root:
        search_dirs.append(_expand_path(env_root))
    for loc in _SEARCH_LOCATIONS:
        search_dirs.append(_expand_path(loc))

    for repo_name, template in _AUGMENTUS_REPO_TEMPLATES.items():
        dir_name = template["dir_name"]
        found_root: Path | None = None

        for parent in search_dirs:
            candidate = parent / dir_name
            if candidate.is_dir():
                found_root = candidate.resolve()
                break

        if found_root is None:
            continue

        # Build source dirs with absolute paths
        sd_configs: list[SourceDirConfig] = []
        for sd_template in template["source_dirs"]:
            sd_path = (found_root / sd_template["relative_path"]).resolve()
            sd_configs.append(SourceDirConfig(
                path=sd_path,
                language=sd_template.get("language", "csharp"),
                exclude_patterns=list(sd_template.get("exclude_patterns", [])),
            ))

        repos[repo_name] = RepoConfig(
            name=repo_name,
            root=found_root,
            collection_name=template["collection_name"],
            source_dirs=sd_configs,
            strip_prefixes=list(template.get("strip_prefixes", [])),
        )
        logger.info("Auto-discovered '%s' at %s", repo_name, found_root)

    return repos


# Set of repo names that came from auto-discovery (tracked for _save_config_local)
_DISCOVERED_REPO_NAMES: set[str] = set()


# Accept "pipeline" as an alias for "perception" (they share the same repo and collection).
REPO_ALIASES: dict[str, str] = {
    "pipeline": "perception",
}


def _repo_config_from_dict(name: str, d: dict) -> RepoConfig:
    """Create a RepoConfig from a JSON dict (used by config.local.json loading and add_repo)."""
    root = _expand_path(d["root"])
    collection_name = d.get("collection_name", f"{name}_code")
    strip_prefixes = d.get("strip_prefixes", [])
    source_dirs = []
    for sd in d.get("source_dirs", []):
        source_dirs.append(SourceDirConfig(
            path=_expand_path(sd["path"]),
            language=sd.get("language", "csharp"),
            exclude_patterns=sd.get("exclude_patterns", []),
        ))
    return RepoConfig(
        name=name, root=root, collection_name=collection_name,
        source_dirs=source_dirs, strip_prefixes=strip_prefixes,
    )


def _is_new_format(data: dict) -> bool:
    """Detect whether config.local.json uses the new {repos: ..., aliases: ...} format."""
    return "repos" in data


def _load_config_overrides() -> dict[str, RepoConfig]:
    """Load repo configs: auto-discover Augmentus repos, then apply config.local.json overrides.

    Supports two config.local.json formats:
    - New format: {"repos": {...}, "aliases": {...}} -- full repo definitions + aliases
    - Old format: {"perception": {...}, "mainapp": {...}} -- per-repo overrides only
    """
    global _DISCOVERED_REPO_NAMES

    # Start with auto-discovered repos instead of hardcoded defaults
    repos = _discover_augmentus_repos()
    _DISCOVERED_REPO_NAMES = set(repos.keys())

    local_config = PROJECT_ROOT / "config.local.json"
    if not local_config.exists():
        return repos

    try:
        data = json.loads(local_config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load config.local.json: %s", e)
        return repos

    if _is_new_format(data):
        _apply_new_format(repos, data)
    else:
        _apply_old_format(repos, data)

    logger.info("Applied config overrides from config.local.json")
    return repos


def _apply_new_format(repos: dict[str, RepoConfig], data: dict) -> None:
    """Apply new-format config.local.json: full repo definitions + aliases."""
    for repo_name, repo_dict in data.get("repos", {}).items():
        if repo_dict is None:
            # Explicitly removed repo
            repos.pop(repo_name, None)
            continue
        if repo_name in repos:
            # Override existing default repo
            cfg = repos[repo_name]
            if "root" in repo_dict:
                cfg.root = _expand_path(repo_dict["root"])
            if "strip_prefixes" in repo_dict:
                cfg.strip_prefixes = repo_dict["strip_prefixes"]
            if "source_dirs" in repo_dict:
                cfg.source_dirs = [
                    SourceDirConfig(
                        path=_expand_path(sd["path"]),
                        language=sd.get("language", "csharp"),
                        exclude_patterns=sd.get("exclude_patterns", []),
                    ) for sd in repo_dict["source_dirs"]
                ]
        else:
            # New repo defined in config.local.json
            repos[repo_name] = _repo_config_from_dict(repo_name, repo_dict)

    # Apply aliases
    for alias, target in data.get("aliases", {}).items():
        if target is None:
            REPO_ALIASES.pop(alias, None)
        else:
            REPO_ALIASES[alias] = target


def _apply_old_format(repos: dict[str, RepoConfig], data: dict) -> None:
    """Apply old-format config.local.json: per-repo path overrides only."""
    for repo_name, overrides_dict in data.items():
        if repo_name not in repos:
            logger.warning("config.local.json (old format): unknown repo '%s', skipping", repo_name)
            continue
        cfg = repos[repo_name]
        if "root" in overrides_dict:
            cfg.root = _expand_path(overrides_dict["root"])
        if "source_dirs" in overrides_dict:
            for i, sd_override in enumerate(overrides_dict["source_dirs"]):
                if i < len(cfg.source_dirs):
                    if "path" in sd_override:
                        cfg.source_dirs[i].path = _expand_path(sd_override["path"])
                    if "exclude_patterns" in sd_override:
                        cfg.source_dirs[i].exclude_patterns = sd_override["exclude_patterns"]


def _save_config_local(repos_override: dict[str, RepoConfig], aliases_override: dict[str, str]) -> None:
    """Persist current repos/aliases to config.local.json (new format).

    Saves ALL repos (including defaults). Discovered repos not present in
    repos_override are written as null so they stay removed on next load.
    """
    local_config = PROJECT_ROOT / "config.local.json"

    # Serialize every current repo
    repos_data = {}
    for name, cfg in repos_override.items():
        repos_data[name] = _repo_config_to_dict(cfg)

    # Mark any discovered repos that were removed as null
    for name in _DISCOVERED_REPO_NAMES:
        if name not in repos_data:
            repos_data[name] = None

    output = {"repos": repos_data}
    if aliases_override:
        output["aliases"] = dict(aliases_override)

    local_config.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved config.local.json")


def _repo_config_to_dict(cfg: RepoConfig) -> dict:
    """Serialize a RepoConfig to a JSON-compatible dict."""
    d = {
        "root": str(cfg.root),
        "collection_name": cfg.collection_name,
        "source_dirs": [
            {
                "path": str(sd.path),
                "language": sd.language,
                "exclude_patterns": sd.exclude_patterns,
            }
            for sd in cfg.source_dirs
        ],
    }
    if cfg.strip_prefixes:
        d["strip_prefixes"] = cfg.strip_prefixes
    return d


# ---------------------------------------------------------------------------
# Auto-detect source directories for simplified add_repo()
# ---------------------------------------------------------------------------

def auto_detect_source_dirs(root: Path) -> tuple[list[dict], str]:
    """Walk a repo root and auto-detect source directories by file extension.

    Classifies files by extension, groups them by language, finds the deepest
    common ancestor directory for each language, and applies default excludes.

    Args:
        root: Absolute path to the repo root.

    Returns:
        (source_dirs_list, human_readable_summary) where source_dirs_list is
        ready to pass to add_repo_config().
    """
    # Collect files per language
    lang_files: dict[str, list[Path]] = {}
    files_visited = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune irrelevant directories in-place
        dirnames[:] = [d for d in dirnames if d not in _ALWAYS_SKIP_DIRS]

        for fname in filenames:
            files_visited += 1
            if files_visited > _AUTO_DETECT_MAX_FILES:
                break

            ext = os.path.splitext(fname)[1].lower()
            lang = _EXTENSION_TO_LANGUAGE.get(ext)
            if lang:
                lang_files.setdefault(lang, []).append(Path(dirpath) / fname)

        if files_visited > _AUTO_DETECT_MAX_FILES:
            break

    # Filter out languages with <2 files (noise)
    lang_files = {lang: files for lang, files in lang_files.items() if len(files) >= 2}

    if not lang_files:
        return [], "No supported source files detected."

    # For each language, find deepest common ancestor
    source_dirs: list[dict] = []
    summary_parts: list[str] = []

    for lang, files in sorted(lang_files.items()):
        # Find common ancestor
        parents = [f.parent for f in files]
        if len(parents) == 1:
            common = parents[0]
        else:
            # os.path.commonpath works on absolute paths
            common = Path(os.path.commonpath(parents))

        # Ensure common ancestor is at or below root
        try:
            common.relative_to(root)
        except ValueError:
            common = root

        excludes = list(_DEFAULT_EXCLUDE_PATTERNS.get(lang, []))

        source_dirs.append({
            "path": str(common),
            "language": lang,
            "exclude_patterns": excludes,
        })
        summary_parts.append(f"  {lang}: {len(files)} files in {common.relative_to(root) if common != root else '.'}")

    summary = f"Auto-detected {len(source_dirs)} source dir(s) ({files_visited} files scanned):\n" + "\n".join(summary_parts)
    return source_dirs, summary


def add_repo_config(name: str, root: str, source_dirs: list[dict] | None = None,
                     aliases: list[str] | None = None,
                     strip_prefixes: list[str] | None = None) -> str:
    """Add a new repo at runtime and persist to config.local.json.

    Args:
        name: Repo name (e.g. "weld-detect").
        root: Root path (supports ~ and env vars).
        source_dirs: List of dicts with "path", "language", "exclude_patterns".
                     If None, auto-detects source directories by scanning the repo.
        aliases: Optional list of alias names that resolve to this repo.
        strip_prefixes: Optional list of .asmdef name prefixes to strip.

    Returns:
        Success/error message.
    """
    if name in REPOS:
        return f"Error: Repo '{name}' already exists. Remove it first or choose a different name."

    root_path = _expand_path(root)
    if not root_path.exists():
        return f"Error: Root path does not exist: {root_path}"

    detect_summary = ""

    if source_dirs is None:
        # Auto-detect source directories
        source_dirs, detect_summary = auto_detect_source_dirs(root_path)
        if not source_dirs:
            return (f"Error: Could not auto-detect any source directories in {root_path}. "
                    "Please provide source_dirs explicitly.")

    if not source_dirs:
        return "Error: At least one source_dir is required."

    sd_configs = []
    for sd in source_dirs:
        if "path" not in sd:
            return "Error: Each source_dir must have a 'path' field."
        sd_path = _expand_path(sd["path"])
        if not sd_path.exists():
            return f"Error: Source dir does not exist: {sd_path}"
        sd_configs.append(SourceDirConfig(
            path=sd_path,
            language=sd.get("language", "csharp"),
            exclude_patterns=sd.get("exclude_patterns", []),
        ))

    collection_name = f"{name.replace('-', '_')}_code"
    cfg = RepoConfig(
        name=name, root=root_path, collection_name=collection_name,
        source_dirs=sd_configs, strip_prefixes=strip_prefixes or [],
    )

    REPOS[name] = cfg

    if aliases:
        for alias in aliases:
            REPO_ALIASES[alias] = name

    _save_config_local(REPOS, REPO_ALIASES)

    msg = f"Added repo '{name}' with {len(sd_configs)} source dir(s). Run reindex('{name}') to index it."
    if detect_summary:
        msg += f"\n\n{detect_summary}"
    return msg


def remove_repo_config(name: str) -> str:
    """Remove a repo at runtime, clean up data files, and update config.local.json.

    Args:
        name: Repo name to remove.

    Returns:
        Success/error message.
    """
    if name not in REPOS:
        return f"Error: Repo '{name}' not found. Available: {list(REPOS.keys())}"

    config = REPOS[name]

    # Delete ChromaDB collection
    try:
        from src.indexer.embedder import get_chroma_client
        client = get_chroma_client()
        try:
            client.delete_collection(config.collection_name)
            logger.info("Deleted ChromaDB collection: %s", config.collection_name)
        except Exception:
            pass  # Collection may not exist yet
    except Exception as e:
        logger.warning("Could not delete collection for %s: %s", name, e)

    # Delete manifest and calibration files
    for suffix in ("_manifest.json", "_calibration.json"):
        path = DATA_DIR / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            logger.info("Deleted %s", path)

    # Remove from in-memory dicts
    del REPOS[name]

    # Remove any aliases pointing to this repo
    to_remove = [alias for alias, target in REPO_ALIASES.items() if target == name]
    for alias in to_remove:
        del REPO_ALIASES[alias]

    _save_config_local(REPOS, REPO_ALIASES)
    return f"Removed repo '{name}' and cleaned up its data."


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_repos() -> None:
    """Log warnings about missing or misconfigured repos at startup.

    Called from server.py before the embedding model is initialized so
    problems are visible early in the log.
    """
    if not REPOS:
        logger.warning(
            "No repos configured. Auto-discovery found nothing. "
            "Use add_repo(name, root) to add a repo, or set AUGMENTUS_REPO_ROOT "
            "env var to the parent directory containing Augmentus repos."
        )
        return

    all_roots_missing = True
    for name, cfg in REPOS.items():
        if not cfg.root.exists():
            logger.warning("Repo '%s': root path does not exist: %s", name, cfg.root)
        else:
            all_roots_missing = False
            for sd in cfg.source_dirs:
                if not sd.path.exists():
                    logger.warning("Repo '%s': source dir does not exist: %s [%s]",
                                   name, sd.path, sd.language)

    if all_roots_missing:
        logger.warning(
            "All configured repo roots are missing. Repos may have moved. "
            "Set AUGMENTUS_REPO_ROOT env var, update config.local.json, "
            "or use add_repo() / remove_repo() to fix."
        )


def resolve_repo(repo: str) -> str:
    """Resolve a repo name, following aliases (e.g. 'pipeline' -> 'perception')."""
    return REPO_ALIASES.get(repo, repo)


REPOS: dict[str, RepoConfig] = _load_config_overrides()
