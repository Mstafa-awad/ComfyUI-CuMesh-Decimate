# ComfyUI CuMesh Decimate

Standalone native ComfyUI `MESH -> MESH` geometry decimation using the
MIT-licensed [VisualBruno CuMesh](https://github.com/visualbruno/CuMesh) CUDA backend.

## Node

`CuMesh - Geometry Decimate (GPU)`

The output contains only vertices, triangle faces, and optional newly computed smooth
normals. UVs, textures, colors, materials, tangents, and maps are deliberately discarded.

## Installation on ComfyUI Easy Install / Portable

1. Extract `ComfyUI-CuMesh-Decimate` into `ComfyUI/custom_nodes/`.
2. Run `Install-CuMesh-Node.cmd`.
3. Restart ComfyUI.

If CuMesh is already importable, the installer changes nothing. Otherwise it searches
sibling custom nodes (including ComfyUI-Trellis2) for a wheel matching the active Python,
PyTorch, and platform, then installs it without changing PyTorch.

## Controls

- `target_faces`: requested maximum triangle count; accepts every integer from 1 upward.
- `initial_threshold`: CuMesh's starting collapse threshold. The upstream default is `1e-8`.
- `edge_length_weight`: discourages uneven edge lengths. Upstream default: `0.01`.
- `skinny_triangle_weight`: penalizes skinny triangles. Upstream default: `0.001`.
- `unload_models`: unload ComfyUI models before CuMesh allocates VRAM. Keep this enabled
  on an 8 GB GPU. ComfyUI reloads a model automatically if a later node needs it.
- `output_smooth_normals`: computes fresh smooth vertex normals after decimation.
- `verbose`: prints CuMesh simplification progress.

CuMesh performs parallel edge collapses, so the final count can be slightly below the
requested target. A very aggressive reduction still removes real geometric detail; no
simplifier can preserve a 50-million-face surface identically at 100,000 faces.

## Suggested starting settings

Use the upstream defaults first:

```text
initial_threshold       0.00000001
edge_length_weight      0.01
skinny_triangle_weight  0.001
unload_models           true
output_smooth_normals   true
verbose                 true
```

If thin triangles remain, increase `skinny_triangle_weight` gradually (for example,
`0.002`, then `0.005`). Large changes can alter the result, so compare visually.

## License

This wrapper is MIT licensed. CuMesh is a separate MIT-licensed dependency; see
`THIRD_PARTY_NOTICES.md` and `licenses/CUMESH-MIT.txt`.
