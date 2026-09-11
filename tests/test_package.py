from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_nodes():
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(OutOfMemoryError=RuntimeError)
    fake_torch.float32 = "float32"
    fake_torch.int32 = "int32"
    fake_torch.int64 = "int64"
    sys.modules["torch"] = fake_torch

    latest = types.ModuleType("comfy_api.latest")
    latest.Types = types.SimpleNamespace(MESH=object)
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    spec = importlib.util.spec_from_file_location("cumesh_nodes_test", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PackageTests(unittest.TestCase):
    def test_source_compiles(self):
        for name in ("__init__.py", "nodes.py", "install.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            compile(source, str(ROOT / name), "exec")

    def test_registration_and_real_controls(self):
        module = _load_nodes()
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS["CuMeshGeometryDecimate"],
            "CuMesh - Geometry Decimate (GPU)",
        )
        inputs = module.CuMeshGeometryDecimate.INPUT_TYPES()["required"]
        self.assertEqual(
            list(inputs),
            [
                "mesh",
                "target_faces",
                "initial_threshold",
                "edge_length_weight",
                "skinny_triangle_weight",
                "unload_models",
                "output_smooth_normals",
                "verbose",
            ],
        )
        self.assertEqual(inputs["target_faces"][1]["step"], 1)
        self.assertEqual(inputs["initial_threshold"][1]["default"], 1e-8)
        self.assertEqual(inputs["edge_length_weight"][1]["default"], 1e-2)
        self.assertEqual(inputs["skinny_triangle_weight"][1]["default"], 1e-3)
        self.assertTrue(inputs["unload_models"][1]["default"])
        self.assertTrue(inputs["output_smooth_normals"][1]["default"])

    def test_installer_torch_folder_mapping(self):
        spec = importlib.util.spec_from_file_location(
            "cumesh_installer_test", ROOT / "install.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module._torch_folder_name("2.8.0+cu128"), "Torch280")
        self.assertEqual(module._torch_folder_name("2.10.0"), "Torch2100")

    def test_archive_does_not_bundle_binary_or_wheel(self):
        forbidden = {".whl", ".pyd", ".dll", ".so", ".dylib"}
        found = [path for path in ROOT.rglob("*") if path.suffix.lower() in forbidden]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
