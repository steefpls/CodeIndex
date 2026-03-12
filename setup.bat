@echo off
setlocal

REM Always run from this script's directory so relative paths are portable.
pushd "%~dp0" >nul
set "EXIT_CODE=0"

echo ============================================
echo   Code Index - Setup
echo ============================================
echo.

REM Resolve Python launcher: prefer python, fall back to py -3.
set "PYTHON_CMD="
python --version >nul 2>&1 && set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
    py -3 --version >nul 2>&1 && set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python not found. Install Python 3.11+ first.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)
echo [OK] Python launcher: %PYTHON_CMD%

REM --- Venv and dependencies ---
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        set "EXIT_CODE=1"
        goto :cleanup_with_pause
    )
)
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found at "%VENV_PYTHON%".
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)
echo [INFO] Installing Python dependencies...

REM Keep pip current to improve wheel compatibility on new machines.
"%VENV_PYTHON%" -m pip install -q --upgrade pip
if errorlevel 1 (
    echo [WARN] Could not upgrade pip - continuing anyway.
)

REM --- GPU-specific onnxruntime (install FIRST, before other deps) ---
REM IMPORTANT: The CPU-only "onnxruntime" and GPU variants (directml/cuda) share the
REM same Python namespace. If both are installed, CPU shadows GPU and acceleration
REM silently breaks. We install the correct variant FIRST, then install everything
REM else, so transitive deps see it as already satisfied and don't pull in CPU.
echo [INFO] Detecting GPU vendor...
set "ORT_PACKAGE="
for /f "delims=" %%i in ('"%VENV_PYTHON%" scripts\detect_gpu.py --install-hint') do set "ORT_PACKAGE=%%i"
if not defined ORT_PACKAGE set "ORT_PACKAGE=onnxruntime"
if defined ORT_PACKAGE_OVERRIDE (
    echo [INFO] Using ORT_PACKAGE_OVERRIDE=%ORT_PACKAGE_OVERRIDE%
    set "ORT_PACKAGE=%ORT_PACKAGE_OVERRIDE%"
)

set "ORT_INSTALL_OK=0"
echo [INFO] Installing %ORT_PACKAGE%...
"%VENV_PYTHON%" -m pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y >nul 2>&1
"%VENV_PYTHON%" -m pip install -q %ORT_PACKAGE%
if not errorlevel 1 set "ORT_INSTALL_OK=1"

if "%ORT_INSTALL_OK%"=="0" if /I "%ORT_PACKAGE%"=="onnxruntime-gpu" (
    echo [WARN] CUDA package install failed - trying DirectML fallback...
    "%VENV_PYTHON%" -m pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y >nul 2>&1
    "%VENV_PYTHON%" -m pip install -q onnxruntime-directml
    if not errorlevel 1 (
        set "ORT_PACKAGE=onnxruntime-directml"
        set "ORT_INSTALL_OK=1"
    )
)

if "%ORT_INSTALL_OK%"=="0" (
    echo [WARN] GPU package install failed - falling back to CPU onnxruntime...
    "%VENV_PYTHON%" -m pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y >nul 2>&1
    "%VENV_PYTHON%" -m pip install -q onnxruntime
    if errorlevel 1 (
        echo [ERROR] Failed to install onnxruntime.
        set "EXIT_CODE=1"
        goto :cleanup_with_pause
    )
    set "ORT_PACKAGE=onnxruntime"
)

echo [INFO] Installing core packages...
"%VENV_PYTHON%" -m pip install -q "mcp[cli]" tree-sitter tree-sitter-c-sharp tree-sitter-cpp tree-sitter-python tree-sitter-javascript tree-sitter-html tree-sitter-rust tree-sitter-typescript tree-sitter-css chromadb
if errorlevel 1 (
    echo [ERROR] Failed to install core packages.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

echo [INFO] Installing embedding model packages...
"%VENV_PYTHON%" -m pip install -q sentence-transformers transformers einops onnxscript
if errorlevel 1 (
    echo [ERROR] Failed to install embedding model packages.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

REM Install project in editable mode (registers entry points, resolves remaining deps)
"%VENV_PYTHON%" -m pip install -q -e .
if errorlevel 1 (
    echo [ERROR] Failed to install project in editable mode.
    set "EXIT_CODE=1"
    goto :cleanup_with_pause
)

REM Final safety check: ensure CPU-only onnxruntime didn't sneak back in as a
REM transitive dependency during the installs above. If it did, uninstalling it
REM can leave a ghost onnxruntime/ directory that shadows the GPU variant, so we
REM must also clean that up and force-reinstall the GPU package.
set "ORT_CONFLICT=0"
"%VENV_PYTHON%" -m pip show onnxruntime >nul 2>&1 && "%VENV_PYTHON%" -m pip show onnxruntime-directml >nul 2>&1 && set "ORT_CONFLICT=1"
"%VENV_PYTHON%" -m pip show onnxruntime >nul 2>&1 && "%VENV_PYTHON%" -m pip show onnxruntime-gpu >nul 2>&1 && set "ORT_CONFLICT=1"
if "%ORT_CONFLICT%"=="1" (
    echo [WARN] CPU onnxruntime detected alongside GPU variant - fixing...
    "%VENV_PYTHON%" -m pip uninstall onnxruntime -y >nul 2>&1
    REM Remove ghost directory left behind by pip uninstall (shared DLLs stay behind)
    if exist ".venv\Lib\site-packages\onnxruntime" (
        rmdir /s /q ".venv\Lib\site-packages\onnxruntime" >nul 2>&1
    )
    REM Force-reinstall the chosen variant to restore its files
    "%VENV_PYTHON%" -m pip install --force-reinstall -q %ORT_PACKAGE% >nul 2>&1
    echo [OK] onnxruntime package restored ^(%ORT_PACKAGE%^)
)

REM --- Install CUDA pip libraries if NVIDIA GPU detected but CUDA doesn't load ---
REM onnxruntime-gpu lists CUDAExecutionProvider but needs CUDA 12 runtime DLLs
REM (cuBLAS, cuDNN, cuFFT, etc.). If CUDA Toolkit isn't system-installed, the
REM nvidia-*-cu12 pip packages provide these DLLs inside the venv.
if /I "%ORT_PACKAGE%"=="onnxruntime-gpu" (
    echo [INFO] Checking if CUDA libraries are available...
    "%VENV_PYTHON%" scripts\detect_gpu.py --check-cuda
    if errorlevel 1 (
        echo [INFO] CUDA libraries missing - installing CUDA pip packages...
        "%VENV_PYTHON%" -m pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
        if errorlevel 1 (
            echo [WARN] Failed to install some CUDA packages - GPU acceleration may not work.
        ) else (
            echo [OK] CUDA pip packages installed
            REM Verify CUDA now loads
            "%VENV_PYTHON%" scripts\detect_gpu.py --check-cuda
            if errorlevel 1 (
                echo [WARN] CUDA still not loading after installing pip packages.
                echo        You may need to install CUDA Toolkit 12.x from NVIDIA.
            ) else (
                echo [OK] CUDA verified - GPU acceleration is working
            )
        )
    ) else (
        echo [OK] CUDA libraries already available
    )
)

echo [OK] Dependencies installed

REM Repo paths can differ across machines when config.local.json is copied over.
if exist "config.local.json" (
    echo [WARN] Found config.local.json ^(machine-specific repo paths^).
    echo        If repo locations changed, edit or remove this file and re-add repos.
    echo        Use list_repos^(^) and add_repo^(^) after setup to verify paths.
)

REM Validate repo discovery / config paths
echo.
echo [INFO] Checking repo paths...
"%VENV_PYTHON%" -c "from src.config import REPOS; found = {n: c.root.exists() for n, c in REPOS.items()}; [print(f'  [OK] {n}: {c.root}') if c.root.exists() else print(f'  [MISS] {n}: {c.root}') for n, c in REPOS.items()]; print() if found else None; print('  (no repos found)') if not found else None; missing = [n for n, ok in found.items() if not ok]; print(f'  TIP: Set AUGMENTUS_REPO_ROOT env var or use add_repo() to configure paths.') if missing or not found else None"

REM --- Download model and export to ONNX ---
echo.
echo [INFO] Downloading CodeRankEmbed model and exporting to ONNX...
echo        (first run downloads ~274MB model, then exports for acceleration)
echo.

set PYTHONIOENCODING=utf-8
"%VENV_PYTHON%" scripts\export_onnx.py
if errorlevel 1 (
    echo [ERROR] ONNX export failed. The server will still work using PyTorch CPU ^(slower^).
    echo         Check data\server.log for details.
)

REM --- Detect GPU backend ---
echo.
echo [INFO] Detecting GPU acceleration...
"%VENV_PYTHON%" scripts\detect_gpu.py

REM --- Register MCP server with available AI CLIs ---
echo.
set "REGISTERED_ANY=0"

REM Check for Claude Code
where claude >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Registering MCP server with Claude Code...
    call claude mcp add --scope user code-index -- "%VENV_PYTHON%" "%CD%\src\server.py" 2>nul
    echo [OK] Registered with Claude Code
    set "REGISTERED_ANY=1"
) else (
    echo [SKIP] Claude Code not found - skipping registration
)

REM Check for Codex CLI
where codex >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Registering MCP server with Codex CLI...
    call codex mcp add code-index -- "%VENV_PYTHON%" "%CD%\src\server.py" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] codex mcp add not available, writing config directly...
        if not exist "%USERPROFILE%\.codex" mkdir "%USERPROFILE%\.codex"
        if not exist "%USERPROFILE%\.codex\config.toml" type nul > "%USERPROFILE%\.codex\config.toml"

        findstr /C:"[mcp_servers.code-index]" "%USERPROFILE%\.codex\config.toml" >nul 2>&1
        if not errorlevel 1 (
            echo [INFO] Existing [mcp_servers.code-index] entry found - skipping direct write.
        ) else (
            REM Build paths with doubled backslashes for TOML
            set "PYTHON_PATH=%VENV_PYTHON%"
            set "SERVER_PATH=%CD%\src\server.py"
            setlocal enabledelayedexpansion
            set "PYTHON_TOML=!PYTHON_PATH:\=\\!"
            set "SERVER_TOML=!SERVER_PATH:\=\\!"
            >> "%USERPROFILE%\.codex\config.toml" (
                echo.
                echo [mcp_servers.code-index]
                echo command = "!PYTHON_TOML!"
                echo args = ["!SERVER_TOML!"]
            )
            endlocal
        )
    )
    echo [OK] Registered with Codex CLI
    set "REGISTERED_ANY=1"
) else (
    echo [SKIP] Codex CLI not found - skipping registration
)

if "%REGISTERED_ANY%"=="0" (
    echo.
    echo [WARN] Neither Claude Code nor Codex CLI found.
    echo        You can register the MCP server manually later:
    echo.
    echo        Claude Code:
    echo          claude mcp add --scope user code-index -- "%VENV_PYTHON%" "%CD%\src\server.py"
    echo.
    echo        Codex CLI:
    echo          codex mcp add code-index -- "%VENV_PYTHON%" "%CD%\src\server.py"
    pause
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Restart Claude Code / Codex CLI to load the new MCP server
echo   2. Ask the AI to run: list_repos()
echo   3. If needed, run add_repo(...) for machine-specific paths
echo   4. Run reindex('perception') / reindex('mainapp') when repos are configured
echo.
goto :cleanup_with_pause

:cleanup_with_pause
pause
goto :cleanup

:cleanup
popd >nul
exit /b %EXIT_CODE%
