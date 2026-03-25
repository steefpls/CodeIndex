# code-index

I built this to stop Claude Code from wasting tokens exploring my codebase file by file. Instead of reading through thousands of files, it calls `search_code("hand-eye calibration")` and gets the relevant chunks instantly.

It's a local MCP server that parses source files into semantic chunks using tree-sitter, embeds them with a code-specific model, and exposes search tools that Claude Code calls directly.

## How it works

1. **Parses** source files into meaningful chunks (classes, methods, functions, prefab hierarchies) via tree-sitter AST analysis and Unity YAML parsing
2. **Embeds** each chunk with CodeRankEmbed (137M params, 768-dim) running locally through ONNX Runtime
3. **Stores** embeddings in ChromaDB on disk for instant retrieval
4. **Calibrates** confidence thresholds per collection using distance distribution analysis
5. **Exposes** MCP tools that Claude Code calls:

| Tool | What it does |
|------|-------------|
| `search_code` | Semantic search with optional path filter |
| `get_file_chunks` | All indexed chunks for a file |
| `lookup_binding` | Zenject DI binding lookup |
| `find_references` | Exact text search for symbol references |
| `find_implementations` | Classes implementing an interface or extending a base |
| `get_class_dependencies` | Class dependency graph |
| `get_assembly_graph` | Unity .asmdef dependency graph |
| `find_asset_references` | Prefabs/scenes referencing a script |
| `get_project_info` | Unity project metadata |
| `get_unity_entity_context` | Full un-truncated context for Unity entities |
| `rebuild_sidecars` | Rebuild hierarchy/deps/asset-ref data without re-embedding |
| `reindex` / `start_reindex` | Re-index a repo (incremental by default) |
| `reindex_file` / `remove_file` | Single-file index operations |
| `index_status` | Health check — backends, chunk counts, calibration |
| `add_repo` / `remove_repo` / `list_repos` | Manage indexed repos at runtime |

## What I index

| Repo | Files | Chunks | What |
|------|-------|--------|------|
| `mainapp` | ~3,650 | ~28,500 | Augmentus-MainApp-U6 — Unity 6 C# + prefabs/scenes, 32 DDD modules, Zenject DI |
| `perception` | ~194 | ~1,100 | Augmentus-Perception — C#/C++ 3D data, scanning, calibration |
| `code-index` | ~24 | ~160 | This project |

Repos can be added/removed at runtime via MCP tools — no code edits needed. Custom repos persist in `config.local.json`.

## Language support

| Language | Chunk types | Notes |
|----------|------------|-------|
| C# | `whole_class` / `class_summary` + `method` | XML doc comments, nested types |
| C++ | `whole_class` / `class_summary` + `method` | Qualified names, forward decl skipping |
| Python | `whole_class` / `class_summary` + `method`, top-level `function` | Decorators, docstrings, module paths |
| JavaScript | `whole_class` / `class_summary` + `method`, top-level `function` | Arrow functions, React components, JSDoc |
| TypeScript | Same as JavaScript | `.ts`/`.tsx` |
| Lua | `whole_class`, `class_summary` + `method`/`constructor`, top-level `function` | xLua hotfix extraction (single + batch), module tables, `class("Name", Base)` OOP, table field functions, EmmyLua `---@class` annotations, `CS.X.Y = function` overrides |
| HTML | `template` chunks | `<script>` sub-chunked as JS, Vue SFC support |
| Unity prefab/scene | `prefab_summary` + `gameobject` | GUID-resolved scripts, hierarchy, degraded mode for 5–50 MB files |
| Rust, JSON, YAML, Markdown, CSS | Basic chunking | Structural extraction |

## Quick start

### Prerequisites

- Python 3.11+

### Automated

```bash
cd augmentus-code-index
setup.bat
```

Handles venv, deps, model download, ONNX export, and Claude Code registration.

### Manual

```bash
python -m venv .venv
.venv\Scripts\activate

# GPU runtime — pick ONE (they conflict):
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y
pip install onnxruntime-gpu          # NVIDIA
# pip install onnxruntime-directml   # AMD/Intel
# pip install onnxruntime            # CPU only

# NVIDIA — CUDA runtime DLLs (skip if CUDA Toolkit is system-installed):
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# Everything else:
pip install -e .

# Register with Claude Code:
claude mcp add --scope user augmentus-code-index -- "%CD%\.venv\Scripts\python.exe" "%CD%\src\server.py"

# Then ask Claude to run: reindex('perception') and reindex('mainapp')
```

## Architecture

```
[tree-sitter parsers]  ->  [ChromaDB]  <-  [FastMCP/stdio]  <-  Claude Code
  C#, C++, Python,                              |
  JS/TS, HTML, Lua,                    search_code()
  Unity YAML                           find_references()
      |                                get_class_dependencies()
  AST chunking                         ...
  CodeRankEmbed
  ONNX + GPU
```

### Embedding pipeline

1. **Chunking** — files parsed into semantic units (classes, methods, prefab hierarchies)
2. **Embedding text** — structural header (namespace, class, path) prepended to source, capped at 5,000 chars
3. **Query prefix** — `"Represent this query for searching relevant code: "` per CodeRankEmbed's asymmetric design
4. **Confidence scoring** — L2 distances mapped to HIGH/MEDIUM/LOW/NO MATCH via per-collection calibration

### Incremental indexing

A manifest tracks each file's mtime and chunk IDs. On reindex, only changed files are re-processed — old chunks are deleted and new ones added atomically.

### Sidecar data

Type hierarchy, dependency graph, and asset reference data are maintained incrementally. When files change, only that file's contributions are updated and the global sidecars rematerialized. `rebuild_sidecars(repo)` does a one-time repair without re-embedding.

### Chunk payloads

Unity chunks can exceed the 5,000-char embedding cap. A payload store keeps full source for oversized chunks, retrieved on demand via `get_unity_entity_context`.

## GPU acceleration

| Priority | Backend | GPU | Speed |
|----------|---------|-----|-------|
| 1 | CUDA | NVIDIA | Fastest |
| 2 | DirectML | AMD/Intel | ~10x over CPU |
| 3 | CPU | None | Baseline |

### Role-based backend routing

I split embedding into two roles to control VRAM usage:

- **index** role (reindex, calibration) — defaults to GPU
- **search** role (search_code) — defaults to CPU

```bash
CODERANK_INDEX_BACKEND=gpu    # gpu, cpu, or auto
CODERANK_SEARCH_BACKEND=cpu
CODERANK_SEARCH_INIT_TIMEOUT_SECONDS=30   # avoid MCP timeout on cold start
CODERANK_STARTUP_PREWARM=0                # 1 to eagerly load models at startup
```

After indexing, the index-role embedder is released to free VRAM. The search+CPU path skips GPU probing and uses lighter ONNX optimization for faster cold starts.

### NVIDIA CUDA setup

`setup.bat` handles this automatically — installs `onnxruntime-gpu`, verifies CUDA loads, and installs runtime DLLs if the CUDA Toolkit isn't system-installed. The server registers DLL directories on `PATH` at import time, so no system-wide CUDA install needed.

If `index_status()` shows CPU on an NVIDIA machine:

```bash
.venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
.venv\Scripts\python scripts\detect_gpu.py
```

## How chunking works

- **Small files (< 150 lines):** Single `whole_class` chunk — avoids fragmenting small DTOs/enums
- **Large files (>= 150 lines):** Individual `method` chunks + one `class_summary`
- **Enums:** Always `whole_class` regardless of size
- **Unity prefabs (>= 500 lines):** `prefab_summary` + individual `gameobject` chunks
- **Lua:** Small files as single chunk, large files split per function/method. Extracts xLua hotfix calls (single + batch), `class("Name", Base)` OOP patterns, table field functions, `M.f = function()` assignments, EmmyLua `---@class` annotations for base types. Constructors (`ctor`, `new`, `__init`) auto-detected
- Each chunk carries metadata: file path, class name, method name, namespace, line numbers, module, doc comments, base types
- Embedding text capped at 5,000 chars with a structural header prepended

## Project structure

```
augmentus-code-index/
├── setup.bat
├── pyproject.toml
├── src/
│   ├── server.py                # FastMCP entry point
│   ├── config.py                # Repo config, aliases, config.local.json
│   ├── indexer/
│   │   ├── chunker.py           # C# tree-sitter AST chunking
│   │   ├── chunker_cpp.py       # C++
│   │   ├── chunker_python.py    # Python
│   │   ├── chunker_js.py        # JavaScript
│   │   ├── chunker_html.py      # HTML + Vue SFC
│   │   ├── chunker_lua.py       # Lua (module tables, hotfix scripts)
│   │   ├── chunker_unity.py     # Unity prefab/scene/ScriptableObject YAML
│   │   ├── chunker_rust.py      # Rust
│   │   ├── chunker_ts.py        # TypeScript
│   │   ├── chunker_css.py       # CSS
│   │   ├── chunker_json.py      # JSON
│   │   ├── chunker_yaml.py      # YAML
│   │   ├── chunker_markdown.py  # Markdown
│   │   ├── guid_resolver.py     # GUID -> script ref from .meta files
│   │   ├── embedder.py          # ChromaDB + CodeRankEmbed ONNX
│   │   ├── calibration.py       # Per-collection confidence thresholds
│   │   ├── file_scanner.py      # Source file discovery
│   │   ├── metadata.py          # Module detection, .asmdef parsing
│   │   ├── zenject_parser.py    # Zenject binding extraction
│   │   ├── hierarchy_builder.py # Type hierarchy sidecar
│   │   ├── asset_ref_builder.py # Asset reference sidecar
│   │   ├── dep_graph_builder.py # Class dependency graph sidecar
│   │   ├── sidecar_state.py     # Incremental sidecar maintenance
│   │   ├── chunk_payload_store.py # Full payload store for oversized chunks
│   │   └── pipeline.py          # Orchestrates scan -> chunk -> embed -> store
│   ├── tools/
│   │   ├── search.py            # search_code, get_file_chunks
│   │   ├── zenject.py           # lookup_binding
│   │   ├── references.py        # find_references
│   │   ├── type_hierarchy.py    # find_implementations
│   │   ├── class_deps.py        # get_class_dependencies
│   │   ├── assembly_graph.py    # get_assembly_graph
│   │   ├── asset_references.py  # find_asset_references
│   │   ├── project_info.py      # get_project_info
│   │   ├── unity_context.py     # get_unity_entity_context
│   │   └── index_management.py  # reindex, index_status, repo management
│   └── models/
│       ├── chunk.py             # CodeChunk dataclass
│       ├── unity_script_ref.py  # UnityScriptRef model
│       └── binding.py           # ZenjectBinding dataclass
├── data/                        # ChromaDB, ONNX model, manifests, sidecars (gitignored)
└── tests/
```

## Configuration

Edit `src/config.py` for chunking thresholds, batch sizes, exclusion patterns — or use `add_repo()` at runtime to register new repos without touching code.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `index_status()` shows "Unity Coverage: DISABLED" | Add a `unity` source dir via `add_repo()`, then `reindex(repo)` |
| Stale sidecar data | `rebuild_sidecars(repo)` for quick repair, or `reindex(repo, incremental=False)` for full rebuild |
| `get_unity_entity_context` returns truncated text | `reindex(repo, incremental=False)` to populate the chunk payload store |
| `find_references` slow (> 2s) | Check `index_status()` for YAML dir pollution; rg backend auto-activates if ripgrep is installed |
| Large scene/prefab shows only summary | Expected — use `get_unity_entity_context` for per-GO detail |

## Tech stack

- **Python** + **FastMCP** (stdio)
- **tree-sitter** — C#, C++, Python, JavaScript, TypeScript, HTML, Lua, Rust, CSS
- **ChromaDB** — vector storage
- **CodeRankEmbed** (nomic-ai/CodeRankEmbed, 768-dim, 8K context)
- **ONNX Runtime** + DirectML/CUDA
