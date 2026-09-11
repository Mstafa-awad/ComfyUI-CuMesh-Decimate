from __future__ import annotations

import gc
import inspect

import torch
from comfy_api.latest import Types


def _release_vram(unload_models: bool) -> None:
    """Release ComfyUI models and cached tensors before CuMesh allocates VRAM."""
    if not torch.cuda.is_available():
        raise RuntimeError("CuMesh requires an NVIDIA CUDA GPU.")

    if unload_models:
        try:
            import comfy.model_management as model_management

            print("[CuMesh Decimate] Unloading ComfyUI models to free VRAM...", flush=True)
            model_management.unload_all_models()
            model_management.soft_empty_cache()
        except Exception as exc:
            print(
                f"[CuMesh Decimate] Warning: could not unload ComfyUI models: {exc}",
                flush=True,
            )

    gc.collect()
    torch.cuda.empty_cache()


def _import_cumesh():
    try:
        import cumesh
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "CuMesh is not installed for this ComfyUI Python/Torch build. Run "
            "Install-CuMesh-Node.cmd, then restart ComfyUI."
        ) from exc

    if not hasattr(cumesh, "CuMesh"):
        raise RuntimeError("The installed cumesh package does not expose cumesh.CuMesh.")
    return cumesh


def _mesh_batch_items(mesh):
    """Extract the real (unpadded) geometry from ComfyUI's native MESH batch."""
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(
            "Expected a native ComfyUI MESH with vertices and faces; "
            f"received {type(mesh).__name__}."
        )

    vertices = mesh.vertices
    faces = mesh.faces
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
    if faces.ndim == 2:
        faces = faces.unsqueeze(0)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"MESH vertices must be [B,N,3], got {tuple(vertices.shape)}.")
    if faces.ndim != 3 or faces.shape[-1] != 3:
        raise ValueError(f"MESH faces must be [B,F,3], got {tuple(faces.shape)}.")
    if vertices.shape[0] != faces.shape[0]:
        raise ValueError("MESH vertices and faces have different batch sizes.")

    vertex_counts = getattr(mesh, "vertex_counts", None)
    face_counts = getattr(mesh, "face_counts", None)
    items = []
    for index in range(vertices.shape[0]):
        vertex_count = (
            int(vertex_counts[index].item())
            if vertex_counts is not None
            else int(vertices.shape[1])
        )
        face_count = (
            int(face_counts[index].item())
            if face_counts is not None
            else int(faces.shape[1])
        )
        if vertex_count <= 0 or face_count <= 0:
            raise ValueError(f"MESH batch item {index} is empty.")

        item_vertices = vertices[index, :vertex_count].detach()
        item_faces = faces[index, :face_count].detach()
        minimum = int(item_faces.min().item())
        maximum = int(item_faces.max().item())
        if minimum < 0 or maximum >= vertex_count:
            raise ValueError(
                f"MESH batch item {index} has a face index outside 0..{vertex_count - 1}."
            )
        items.append((item_vertices, item_faces))
    return items


def _construct_mesh(**values):
    """Remain compatible with ComfyUI MESH revisions that lack newer fields."""
    try:
        accepted = inspect.signature(Types.MESH.__init__).parameters
        values = {key: value for key, value in values.items() if key in accepted}
    except (TypeError, ValueError):
        pass
    return Types.MESH(**values)


def _pack_mesh_batch(items):
    """Pack geometry and optional smooth normals back into a native MESH."""
    vertices = [item[0].to("cpu", torch.float32).contiguous() for item in items]
    faces = [item[1].to("cpu", torch.int64).contiguous() for item in items]
    normals = [
        item[2].to("cpu", torch.float32).contiguous() if item[2] is not None else None
        for item in items
    ]

    if not vertices or any(v.numel() == 0 or f.numel() == 0 for v, f in zip(vertices, faces)):
        raise RuntimeError("CuMesh produced an empty mesh.")

    all_normals = all(normal is not None for normal in normals)
    if len(vertices) == 1:
        values = {
            "vertices": vertices[0].unsqueeze(0),
            "faces": faces[0].unsqueeze(0),
        }
        if all_normals:
            values["normals"] = normals[0].unsqueeze(0)
        return _construct_mesh(**values)

    vertex_counts = torch.tensor([len(v) for v in vertices], dtype=torch.int64)
    face_counts = torch.tensor([len(f) for f in faces], dtype=torch.int64)
    packed_vertices = torch.zeros(
        (len(vertices), int(vertex_counts.max().item()), 3), dtype=torch.float32
    )
    packed_faces = torch.zeros(
        (len(faces), int(face_counts.max().item()), 3), dtype=torch.int64
    )
    packed_normals = torch.zeros_like(packed_vertices) if all_normals else None
    for index, (item_vertices, item_faces) in enumerate(zip(vertices, faces)):
        packed_vertices[index, : len(item_vertices)] = item_vertices
        packed_faces[index, : len(item_faces)] = item_faces
        if packed_normals is not None:
            packed_normals[index, : len(item_vertices)] = normals[index]

    values = {
        "vertices": packed_vertices,
        "faces": packed_faces,
        "vertex_counts": vertex_counts,
        "face_counts": face_counts,
    }
    if packed_normals is not None:
        values["normals"] = packed_normals
    return _construct_mesh(**values)


def _decimate_item(
    cumesh_module,
    vertices,
    faces,
    target_faces: int,
    initial_threshold: float,
    edge_length_weight: float,
    skinny_triangle_weight: float,
    output_smooth_normals: bool,
    verbose: bool,
):
    input_faces = int(faces.shape[0])
    target_faces = max(1, int(target_faces))

    vertices_cuda = vertices.to(
        device="cuda", dtype=torch.float32, non_blocking=False
    ).contiguous()
    faces_cuda = faces.to(
        device="cuda", dtype=torch.int32, non_blocking=False
    ).contiguous()
    cuda_mesh = cumesh_module.CuMesh()

    try:
        cuda_mesh.init(vertices_cuda, faces_cuda)
        # CuMesh owns its geometry after init; release the bridge tensors early.
        del vertices_cuda, faces_cuda

        if input_faces > target_faces:
            options = {
                "thresh": float(initial_threshold),
                "lambda_edge_length": float(edge_length_weight),
                "lambda_skinny": float(skinny_triangle_weight),
            }
            cuda_mesh.simplify(target_faces, verbose=bool(verbose), options=options)

        output_vertices, output_faces = cuda_mesh.read()
        output_normals = None
        if output_smooth_normals:
            cuda_mesh.compute_vertex_normals()
            output_normals = cuda_mesh.read_vertex_normals()

        # Copy the result out before deleting CuMesh's CUDA object.
        output_vertices = output_vertices.detach().to("cpu", torch.float32).contiguous()
        output_faces = output_faces.detach().to("cpu", torch.int64).contiguous()
        if output_normals is not None:
            output_normals = output_normals.detach().to("cpu", torch.float32).contiguous()

        print(
            f"[CuMesh Decimate] Complete: {input_faces:,} -> "
            f"{len(output_faces):,} faces.",
            flush=True,
        )
        return output_vertices, output_faces, output_normals
    finally:
        del cuda_mesh
        gc.collect()
        torch.cuda.empty_cache()


class CuMeshGeometryDecimate:
    """GPU QEM simplification through VisualBruno's MIT-licensed CuMesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "target_faces": (
                    "INT",
                    {"default": 100_000, "min": 1, "max": 2_147_483_647, "step": 1},
                ),
                "initial_threshold": (
                    "FLOAT",
                    {
                        "default": 0.00000001,
                        "min": 0.000000000001,
                        "max": 0.01,
                        "step": 0.000000000001,
                    },
                ),
                "edge_length_weight": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "skinny_triangle_weight": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0001},
                ),
                "unload_models": ("BOOLEAN", {"default": True}),
                "output_smooth_normals": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Mesh/Decimation"
    DESCRIPTION = (
        "Geometry-only CuMesh GPU decimation. Removes UVs, textures, colors, materials, "
        "and prior normals; optionally computes fresh smooth normals after simplification."
    )

    def process(
        self,
        mesh,
        target_faces,
        initial_threshold,
        edge_length_weight,
        skinny_triangle_weight,
        unload_models,
        output_smooth_normals,
        verbose,
    ):
        _release_vram(bool(unload_models))
        cumesh_module = _import_cumesh()
        output_items = []

        try:
            for vertices, faces in _mesh_batch_items(mesh):
                output_items.append(
                    _decimate_item(
                        cumesh_module,
                        vertices,
                        faces,
                        target_faces=target_faces,
                        initial_threshold=initial_threshold,
                        edge_length_weight=edge_length_weight,
                        skinny_triangle_weight=skinny_triangle_weight,
                        output_smooth_normals=output_smooth_normals,
                        verbose=verbose,
                    )
                )
            return (_pack_mesh_batch(output_items),)
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CuMesh ran out of GPU memory. It is GPU-only and must first hold the source "
                "mesh plus its edge/adjacency data in VRAM. Keep unload_models enabled. If it "
                "still fails, pre-reduce the source mesh or use a GPU with more VRAM."
            ) from exc
        finally:
            gc.collect()
            torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "CuMeshGeometryDecimate": CuMeshGeometryDecimate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CuMeshGeometryDecimate": "CuMesh - Geometry Decimate (GPU)",
}
