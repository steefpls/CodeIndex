# augmentus-code-index

Local MCP server that gives AI coding agents (Claude Code) semantic search over codebases. Supports C#, C++, Python, JavaScript, HTML, and Unity prefabs/scenes. Instead of wasting tokens exploring thousands of files, agents call `search_code("hand-eye calibration")` and get the most relevant code chunks instantly.

## What it does

- **Parses** source files into meaningful chunks (classes, methods, functions, components, prefab hierarchies) using tree-sitter AST analysis and Unity YAML parsing
- **Embeds** each chunk with CodeRankEmbed (code-specific model, 137M params, 768-dim) running locally
- **Accelerates** inference via ONNX Runtime with GPU support (CUDA for NVIDIA, DirectML for AMD/Intel)
- **Stores** embeddings in ChromaDB on disk for instant retrieval
- **Calibrates** confidence thresholds per-collection using distance distribution analysis
- **Exposes** MCP tools that Claude Code can call directly:

| Tool | Description |
|------|-------------|
| `search_code(query, repo, file_path?, output_format?)` | Semantic code search with optional path prefix filter (includes `Chunk ID` per result; supports `output_format="json"`) |
| `get_file_chunks(file_path, repo, output_format?)` | List all indexed chunks for a specific file (includes `Chunk ID` per chunk; supports `output_format="json"`) |
| `lookup_binding(interface)` | Zenject DI binding lookup (Bind/To, BindInterfacesTo, BindFactory, DeclareSignal) |
| `find_references(symbol, repo)` | Exact text search for symbol references across all source files |
| `find_implementations(type_name, repo)` | Find all classes implementing an interface or extending a base class |
| `get_class_dependencies(class_name?, repo, output_format?)` | Class dependency graph — what a class depends on and what depends on it (supports node-key disambiguation and `output_format="json"`) |
| `get_assembly_graph(repo, assembly?)` | Unity .asmdef assembly definition dependency graph |
| `find_asset_references(class_name, repo, output_format?)` | Find prefabs/scenes/assets that reference a script class (`output_format="json"` available) |
| `get_project_info(repo)` | Unity project metadata (version, packages, build scenes, scripting defines) |
| `get_unity_entity_context(repo, chunk_id?, file_path?, entity_name?, output_format?)` | Get full un-truncated context for a Unity entity (chunk or GO), with optional JSON output |
| `rebuild_sidecars(repo)` | Rebuild sidecar data (hierarchy, deps, asset refs) without re-embedding |
| `reindex(repo)` | Re-index a repo (incremental by default, only re-embeds changed files) |
| `start_reindex(repo)` | Reindex with live progress updates (blocks until complete) |
| `get_reindex_status(repo)` | Check status of a reindex job |
| `reindex_file(repo, file_path)` | Re-index a single file |
| `remove_file(repo, file_path)` | Remove a single file from the index |
| `index_status()` | Health check — embedding backend, chunk counts, calibration thresholds, Unity coverage |
| `add_repo(name, root, source_dirs_json)` | Register a new repo for indexing |
| `remove_repo(name)` | Remove a repo and its index |
| `list_repos()` | Show all configured repos with paths and aliases |

## Indexed repos

| Repo | Files | Chunks | Description |
|------|-------|--------|-------------|
| `perception` | ~194 | ~1,100 | Augmentus-Perception (C#/C++ 3D data, scanning, calibration) |
| `mainapp` | ~3,650 | ~28,500 | Augmentus-MainApp-U6 (Unity 6 C# + prefabs/scenes, 32 DDD modules, Zenject DI) |
| `code-index` | ~24 | ~160 | This project (Python) |

Repos can be added/removed at runtime via MCP tools — no code edits needed. Custom repos are persisted in `config.local.json`.

## Quick start

### Prerequisites

- Python 3.11+

### Automated setup

```bash
cd augmentus-code-index
setup.bat
```

This handles everything: venv, dependencies, model download, ONNX export for GPU acceleration, and Claude Code registration.

### Manual setup

```bash
# 1. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\activate

# 2. Install the correct GPU-accelerated onnxruntime FIRST (before other deps).
#    The CPU/DirectML/CUDA variants share the same Python namespace and conflict.
#    Pick ONE: onnxruntime-gpu (NVIDIA), onnxruntime-directml (AMD/Intel), onnxruntime (CPU)
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y
pip install onnxruntime-gpu  # or onnxruntime-directml for AMD/Intel

# 2b. For NVIDIA: install CUDA runtime libraries (skip if CUDA Toolkit is system-installed)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# 3. Install remaining packages (they'll see onnxruntime as already satisfied)
pip install "mcp[cli]" tree-sitter tree-sitter-c-sharp tree-sitter-cpp tree-sitter-python tree-sitter-javascript tree-sitter-html chromadb
pip install sentence-transformers transformers einops onnxscript
pip install -e .

# 4. Register with Claude Code
claude mcp add --scope user augmentus-code-index -- "%CD%\.venv\Scripts\python.exe" "%CD%\src\server.py"

# 5. Restart Claude Code — the tools will appear automatically
# 6. Ask Claude to run: reindex('perception') and reindex('mainapp')
```

## Architecture

```
[tree-sitter parsers]  ->  [ChromaDB on disk]  <-  [MCP Server (FastMCP/stdio)]  <-  Claude Code
  (C#, C++, Python,                                            |
   JavaScript, HTML,                                  search_code()
   Unity YAML)                                        get_file_chunks()
        |                                             lookup_binding()
   AST chunking                                       reindex()
   CodeRankEmbed                                      index_status()
   ONNX + GPU
```

### Embedding pipeline

1. **Chunking** — Source files are parsed into semantic units (classes, methods, prefab hierarchies)
2. **Embedding text** — Each chunk gets a front-loaded header (namespace, class, file path) prepended to its source, truncated to 5,000 characters
3. **Query prefix** — At search time, queries are prefixed with `"Represent this query for searching relevant code: "` per CodeRankEmbed's asymmetric search design
4. **Confidence scoring** — L2 distances are mapped to HIGH/MEDIUM/LOW/NO MATCH using per-collection calibration thresholds derived from distance distribution percentiles

### Incremental indexing

A manifest file (`data/<repo>_manifest.json`) tracks each indexed file's mtime and chunk IDs. On incremental reindex, only files with changed mtimes are re-processed. Chunks for modified files are deleted and re-added atomically.

### Sidecar maintenance

Type hierarchy, dependency graph, and asset reference data are maintained incrementally via a per-file contribution state model (`data/<repo>_sidecar_state.json`). When files are added, modified, or removed — whether during a full reindex, incremental reindex, or single-file operation — only that file's contributions are updated and the global sidecar JSONs are rematerialized. This replaces the previous full-rebuild-only approach and keeps sidecar data fresh without performance penalty. Use `rebuild_sidecars(repo)` for one-time repair after schema changes.

### Chunk payload store

Unity chunks can exceed the 5,000-character embedding text cap. The chunk payload store (`data/<repo>_chunk_payloads.json`) persists the full un-truncated source for oversized chunks, keyed by chunk_id. The `get_unity_entity_context` tool retrieves these payloads on demand, avoiding token blowup in search results while keeping full detail accessible.

## GPU acceleration

The embedding model runs through ONNX Runtime with automatic GPU detection:

| Priority | Backend | GPU | Speed |
|----------|---------|-----|-------|
| 1 | CUDA | NVIDIA | Fastest |
| 2 | DirectML | AMD / Intel | ~10x over CPU |
| 3 | CPU | None | Baseline |

The active backend is reported on startup and via `index_status()`. The ONNX model is exported during `setup.bat` with fp16 weights and cached graph optimizations — if it's missing, the server falls back to PyTorch CPU (slower).

### Role-based backend routing (VRAM control)

Embedding now uses separate runtime roles:

- `index` role (used by `reindex`, `start_reindex`, `reindex_file`, calibration) defaults to GPU
- `search` role (used by `search_code`) defaults to CPU

This is controlled by environment variables:

```bash
CODERANK_INDEX_BACKEND=gpu
CODERANK_SEARCH_BACKEND=cpu
```

Accepted values: `gpu`, `cpu`, `auto`.

First-search guard (to avoid MCP 120s call timeouts on slow cold starts):

```bash
CODERANK_SEARCH_INIT_TIMEOUT_SECONDS=30
```

If first-time search initialization exceeds this guard, `search_code` returns a
retry message immediately instead of hanging until the client-side tool deadline.

Startup no longer prewarms embedding sessions by default, so MCP startup does not eagerly consume GPU VRAM. To opt into prewarm behavior:

```bash
CODERANK_STARTUP_PREWARM=1
```

After indexing completes, the index role embedder is explicitly released. This reduces long-lived VRAM usage when multiple agent instances are running.

For `search + cpu`, startup now uses a fast path:

- skips heavy hardware probing (no GPU/PowerShell VRAM detection)
- uses lighter ONNX graph optimization (`EXTENDED`)
- skips optimized-model serialization on first load

This reduces first-query cold-start latency in fresh terminal/MCP processes.

Verification:

```bash
# Reports both roles explicitly
.venv\Scripts\python -c "from src.tools.index_management import index_status; print(index_status())"

# Direct provider verification
.venv\Scripts\python -c "from src.indexer.embedder import get_embedding_function; s=get_embedding_function(role='search'); i=get_embedding_function(role='index'); print('search', s._ort_session.get_providers()); print('index', i._ort_session.get_providers())"
```

Expected on NVIDIA:

- Search role: `['CPUExecutionProvider']`
- Index role: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

### NVIDIA CUDA setup

`setup.bat` handles CUDA setup automatically. It installs `onnxruntime-gpu` and then verifies that CUDA actually loads. If the CUDA Toolkit isn't system-installed, it installs the required runtime DLLs as pip packages:

```
nvidia-cublas-cu12    nvidia-cudnn-cu12     nvidia-cufft-cu12
nvidia-curand-cu12    nvidia-cusolver-cu12  nvidia-cusparse-cu12
nvidia-cuda-runtime-cu12  nvidia-cuda-nvrtc-cu12
```

The server registers these DLL directories on `PATH` at import time (`embedder.py`), so no system-wide CUDA Toolkit install is needed.

**Manual CUDA fix** (if `index_status()` shows "ONNX + CPU" on an NVIDIA machine):

```bash
# Install CUDA 12 runtime libraries into the venv
.venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

# Verify CUDA loads
.venv\Scripts\python scripts\detect_gpu.py
# Should show: [OK] NVIDIA CUDA verified - GPU acceleration is working
```

**CPU fallback warning:** When reindexing on CPU, `start_reindex` and `reindex` return a warning with the detected GPU vendor and the exact install command to enable GPU acceleration.

**DirectML resilience:** DirectML can fail on first inference at unusual sequence lengths due to lazy kernel compilation. The embedder handles this with a fallback chain: batch inference -> individual retry (3 attempts) -> truncate to 512 tokens. This ensures indexing never fails even on large prefab files.

## Project structure

```
augmentus-code-index/
├── setup.bat                    # One-click setup
├── pyproject.toml
├── src/
│   ├── server.py                # FastMCP entry point, tool definitions, GPU warmup
│   ├── config.py                # Repo config, aliases, runtime add/remove, config.local.json
│   ├── indexer/
│   │   ├── chunker.py           # C# tree-sitter AST chunking (serialized field tagging)
│   │   ├── chunker_cpp.py       # C++ tree-sitter AST chunking
│   │   ├── chunker_python.py    # Python tree-sitter AST chunking
│   │   ├── chunker_js.py        # JavaScript tree-sitter AST chunking
│   │   ├── chunker_html.py      # HTML tree-sitter AST chunking (+ Vue SFC)
│   │   ├── chunker_unity.py     # Unity prefab/scene/ScriptableObject YAML parsing
│   │   ├── guid_resolver.py     # GUID -> UnityScriptRef from .meta files (namespace extraction)
│   │   ├── embedder.py          # ChromaDB + CodeRankEmbed (ONNX/DirectML/CUDA), fallback chain
│   │   ├── calibration.py       # Per-collection distance calibration for confidence thresholds
│   │   ├── file_scanner.py      # Source file discovery with exclusion patterns
│   │   ├── metadata.py          # Module detection, .asmdef parsing
│   │   ├── zenject_parser.py    # Zenject installer binding extraction (5 binding types)
│   │   ├── hierarchy_builder.py # Type hierarchy sidecar: base_type -> implementing classes
│   │   ├── asset_ref_builder.py # Asset reference sidecar: GUID-keyed canonical + class-name-keyed compat
│   │   ├── dep_graph_builder.py # Class dependency graph sidecar: class -> [type references]
│   │   ├── sidecar_state.py     # Per-file sidecar contribution state for incremental maintenance
│   │   ├── chunk_payload_store.py # Full-payload store for oversized Unity chunks
│   │   └── pipeline.py          # Orchestrates scan -> chunk -> embed -> store, sidecar builders
│   ├── tools/
│   │   ├── search.py            # search_code + get_file_chunks implementations
│   │   ├── zenject.py           # lookup_binding implementation (exact + fuzzy match)
│   │   ├── references.py        # find_references: rg backend + Python fallback, code-dirs only
│   │   ├── type_hierarchy.py    # find_implementations: interface/base class implementations
│   │   ├── class_deps.py        # get_class_dependencies: class-to-class dependency graph
│   │   ├── assembly_graph.py    # get_assembly_graph: Unity .asmdef dependency graph
│   │   ├── asset_references.py  # find_asset_references: prefabs/scenes referencing a script (disambiguation)
│   │   ├── project_info.py      # get_project_info: Unity project metadata
│   │   ├── unity_context.py     # get_unity_entity_context: full payload retrieval for Unity entities
│   │   └── index_management.py  # reindex, reindex_file, remove_file, index_status, repo management
│   └── models/
│       ├── chunk.py             # CodeChunk dataclass (base_types in metadata)
│       ├── unity_script_ref.py  # UnityScriptRef: GUID + namespace + assembly identity model
│       └── binding.py           # ZenjectBinding dataclass
├── data/                        # ChromaDB storage, ONNX model, manifests, sidecars, payloads (gitignored)
└── tests/
    ├── test_chunker.py          # C# chunker tests
    ├── test_chunker_cpp.py      # C++ chunker tests
    ├── test_chunker_unity.py    # Unity prefab/scene chunker tests (duplicate names, degraded mode)
    ├── test_sidecar_state.py    # Sidecar state CRUD and materialization tests
    ├── test_zenject_parser.py   # Zenject binding parser tests
    ├── test_get_file_chunks.py  # get_file_chunks tool tests
    ├── test_new_tools.py        # Tool + sidecar builder tests
    ├── test_integration_unity_indexing.py  # End-to-end Unity indexing + sidecar integration tests
    └── bench_find_references.py # find_references latency benchmark
```

## How chunking works

- **Small files (< 150 lines):** Indexed as a single `whole_class` chunk — avoids fragmenting small DTOs/enums
- **Large files (>= 150 lines):** Split into individual `method` chunks + one `class_summary` chunk
- **Enums:** Always indexed as `whole_class` regardless of size
- **Unity prefabs (>= 500 lines):** Split into one `prefab_summary` + individual `gameobject` chunks per hierarchy node
- Each chunk carries metadata: file path, class name, method name, namespace, line numbers, module, doc comments, base types
- Embedding text is capped at 5,000 characters (`MAX_EMBED_CHARS`) with a structural header prepended

### Language-specific behavior

| Language | Classes | Functions | Special |
|----------|---------|-----------|---------|
| C# | `whole_class` / `class_summary` + `method` | — | XML doc comments, nested types |
| C++ | `whole_class` / `class_summary` + `method` | Free functions as `method` | Qualified names, forward decl skipping |
| Python | `whole_class` / `class_summary` + `method` | Top-level as `function` | Decorators, docstrings, module paths |
| JavaScript | `whole_class` / `class_summary` + `method` | Top-level as `function` | Arrow functions, React `component`, JSDoc, minified file skipping |
| HTML | — | — | `template` chunks, `<script>` sub-chunked as JS, Vue SFC support |
| Unity prefab/scene | `prefab_summary` + `gameobject` | — | GUID-resolved script names with file paths, fileID-based attribution, hierarchy, component configs, scene settings filtering, degraded mode for large files (5–50 MB) |

## Configuration

Edit `src/config.py` to change:
- Repo paths and source directories
- Exclusion patterns
- Chunking thresholds (`SMALL_FILE_LINE_THRESHOLD = 150`)
- Embedding batch sizes (`EMBEDDING_BATCH_SIZE = 100` for ChromaDB, `BATCH_SIZE = 8` for ONNX inference)
- Assembly definition prefix stripping

Or add repos at runtime via `add_repo()` — persisted to `config.local.json` with alias support.

## Tech stack

- **Python** + **FastMCP** for the MCP server (stdio transport)
- **tree-sitter** + language grammars (C#, C++, Python, JavaScript, HTML) for AST-based code chunking
- **ChromaDB** (PersistentClient) for vector storage
- **CodeRankEmbed** (nomic-ai/CodeRankEmbed, 768-dim, 8K context) for code-specific embeddings
- **ONNX Runtime** + **DirectML/CUDA** for GPU-accelerated inference

## Migration notes

### After updating to this version

This release introduces several new data structures (sidecar state, chunk payload store, GUID-keyed asset references) that require a **one-time full reindex** to populate:

```
# From Claude Code, run:
reindex('mainapp', incremental=False)
```

This rebuilds all chunks, materializes the new sidecar contribution state, populates the chunk payload store for oversized Unity chunks, and generates GUID-keyed asset reference data. After this initial full reindex, all incremental operations (including `reindex_file` and `remove_file`) will maintain the new data structures automatically.

**What happens if you skip the full reindex:**
- `find_implementations`, `get_class_dependencies`, and `find_asset_references` will still work but may show stale data from the previous sidecar format
- `get_unity_entity_context` will fall back to truncated ChromaDB metadata instead of full payloads
- `rebuild_sidecars(repo)` can partially repair sidecar data without re-embedding, but the chunk payload store requires a full reindex to populate

### Operator runbook

| Symptom | Action |
|---------|--------|
| `index_status()` shows "Unity Coverage: DISABLED" | Add a `unity` source dir via `add_repo()` or edit `config.local.json`, then `reindex(repo)` |
| Sidecar data seems stale (wrong implementations/deps) | Run `rebuild_sidecars(repo)` for quick repair, or `reindex(repo, incremental=False)` for full rebuild |
| `get_unity_entity_context` returns truncated text | Run `reindex(repo, incremental=False)` to populate the chunk payload store |
| `find_references` is slow (> 2s) | Check `index_status()` for Unity YAML dir pollution; the rg backend should auto-activate if ripgrep is installed |
| Asset reference query shows disambiguation warning | Expected when multiple scripts share a class name; use the script path in the output to identify the correct one |
| Large scene/prefab (> 5 MB) shows only summary | Expected degraded mode; use `get_unity_entity_context` to retrieve per-GO detail chunks |
