@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "COMFY_PYTHON="

if exist "%~dp0..\..\..\python_embeded\python.exe" (
    set "COMFY_PYTHON=%~dp0..\..\..\python_embeded\python.exe"
)

if not defined COMFY_PYTHON if exist "%~dp0..\..\.venv\Scripts\python.exe" (
    set "COMFY_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
)

if not defined COMFY_PYTHON if exist "%~dp0..\..\venv\Scripts\python.exe" (
    set "COMFY_PYTHON=%~dp0..\..\venv\Scripts\python.exe"
)

if not defined COMFY_PYTHON (
    where python.exe >nul 2>nul
    if not errorlevel 1 set "COMFY_PYTHON=python.exe"
)

if not defined COMFY_PYTHON (
    echo ERROR: Could not find ComfyUI's Python.
    echo Put this folder inside ComfyUI\custom_nodes, then run this file again.
    pause
    exit /b 1
)

echo Using Python: %COMFY_PYTHON%
"%COMFY_PYTHON%" "%~dp0install.py"
if errorlevel 1 (
    echo.
    echo Installation failed. Read the error above.
    pause
    exit /b 1
)

echo.
echo Installation completed. Restart ComfyUI.
pause
exit /b 0
