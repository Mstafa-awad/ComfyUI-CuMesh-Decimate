from __future__ import annotations

import importlib
from pathlib import Path
import platform
import re
import subprocess
import sys


NODE_DIR = Path(__file__).resolve().parent


def _run(command):
    print("[CuMesh Installer]", " ".join(map(str, command)), flush=True)
    subprocess.check_call([str(item) for item in command])


def _cumesh_ready():
    try:
        module = importlib.import_module("cumesh")
        return callable(getattr(module, "CuMesh", None))
    except Exception:
        return False


def _torch_folder_name(version: str):
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return None
    major, minor = match.groups()
    return f"Torch{major}{minor}0"


def _matching_sibling_wheel():
    """Find the compatible CuMesh wheel shipped by a sibling Trellis2 node."""
    import torch

    custom_nodes_dir = NODE_DIR.parent
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_folder = _torch_folder_name(torch.__version__)
    system = platform.system().lower()
    platform_token = "win_amd64" if system == "windows" else "linux"

    candidates = []
    for sibling in custom_nodes_dir.iterdir():
        if not sibling.is_dir() or sibling.resolve() == NODE_DIR:
            continue
        wheels_dir = sibling / "wheels"
        if not wheels_dir.is_dir():
            continue
        for wheel in wheels_dir.rglob("cumesh-*.whl"):
            path_text = str(wheel).lower()
            if python_tag not in wheel.name.lower():
                continue
            if platform_token not in wheel.name.lower():
                continue
            if torch_folder and torch_folder.lower() not in path_text:
                continue
            candidates.append(wheel)
    return sorted(candidates)[-1] if candidates else None


def main():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is unavailable. Run this installer with ComfyUI's Python."
        ) from exc

    print(f"[CuMesh Installer] Python: {sys.version.split()[0]}")
    print(f"[CuMesh Installer] PyTorch: {torch.__version__}")
    print(f"[CuMesh Installer] PyTorch CUDA: {torch.version.cuda}")

    if _cumesh_ready():
        print("[CuMesh Installer] CuMesh is already installed and ready.")
        return 0

    wheel = _matching_sibling_wheel()
    if wheel is not None:
        print(f"[CuMesh Installer] Installing matching sibling wheel: {wheel}")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                wheel,
            ]
        )
        importlib.invalidate_caches()
        if _cumesh_ready():
            print("[CuMesh Installer] CuMesh installed successfully.")
            return 0

    raise RuntimeError(
        "No compatible CuMesh installation or sibling wheel was found. Install/update "
        "ComfyUI-Trellis2 so it contains a CuMesh wheel matching this Python and PyTorch, "
        "or build the official visualbruno/CuMesh project for this environment."
    )


if __name__ == "__main__":
    raise SystemExit(main())
