"""Export editable shell meshes, interior Gaussians and linear HDR illumination."""

import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.int64)

# Every cube uses its lower-to-upper body diagonal. Shared cube faces therefore
# receive identical diagonals, preventing cracks between independently cut cells.
_TETRAHEDRA = (
    (0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
    (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7),
)


@torch.no_grad()
def _sample_sdf(field, bound, resolution, chunk_size=8192):
    """Sample values[ix, iy, iz]; C-order flattening makes z the fast axis."""
    parameter = next(field.parameters())
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    values = np.empty(resolution ** 3, dtype=np.float32)
    for start in range(0, len(values), chunk_size):
        flat = np.arange(start, min(start + chunk_size, len(values)))
        indices = np.column_stack((flat // (resolution * resolution),
                                   (flat // resolution) % resolution,
                                   flat % resolution))
        points = torch.as_tensor(axis[indices], device=parameter.device, dtype=parameter.dtype)
        values[start:start + len(flat)] = field(points).detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Cannot extract a mesh from nonfinite SDF samples")
    return axis, values.reshape(resolution, resolution, resolution)


def _marching_tetrahedra(axis, values):
    """Extract the piecewise-linear zero set, sharing vertices by grid-edge ID."""
    resolution = len(axis)
    low = values[:-1, :-1, :-1].copy()
    high = low.copy()
    for corner in _CORNERS[1:]:
        slices = tuple(slice(int(offset), resolution - 1 + int(offset)) for offset in corner)
        np.minimum(low, values[slices], out=low)
        np.maximum(high, values[slices], out=high)
    active_cells = np.argwhere((low < 0.0) & (high >= 0.0))
    flat_values = values.reshape(-1)
    vertices, faces = [], []
    vertex_cache, face_cache = {}, set()

    def edge_vertex(a, b, pa, pb, va, vb):
        # An exact zero at a grid vertex is shared by all incident grid edges.
        if va == 0.0:
            key = (int(a), int(a))
            position = pa
        elif vb == 0.0:
            key = (int(b), int(b))
            position = pb
        else:
            key = (int(min(a, b)), int(max(a, b)))
            fraction = float(va) / (float(va) - float(vb))
            position = pa + fraction * (pb - pa)
        if key not in vertex_cache:
            vertex_cache[key] = len(vertices)
            vertices.append(np.asarray(position, dtype=np.float64))
        return vertex_cache[key]

    def add_face(triangle, outward):
        if len(set(triangle)) < 3:
            return
        a, b, c = [vertices[index] for index in triangle]
        normal = np.cross(b - a, c - a)
        if not np.dot(normal, normal) > 0.0:
            return
        if np.dot(normal, outward) < 0.0:
            triangle = (triangle[0], triangle[2], triangle[1])
        key = tuple(sorted(triangle))
        if key not in face_cache:
            face_cache.add(key)
            faces.append(triangle)

    for cell in active_cells:
        coordinates = cell[None, :] + _CORNERS
        ids = ((coordinates[:, 0] * resolution + coordinates[:, 1]) * resolution + coordinates[:, 2])
        positions = axis[coordinates].astype(np.float64)
        samples = flat_values[ids]
        for tetrahedron in _TETRAHEDRA:
            inside = [index for index in tetrahedron if samples[index] < 0.0]
            outside = [index for index in tetrahedron if samples[index] >= 0.0]
            if not inside or not outside:
                continue
            # The dot product with the linear SDF gradient is positive for this
            # vector, so it fixes the triangle winding toward f > 0.
            outward = positions[outside].mean(axis=0) - positions[inside].mean(axis=0)

            def cut(a, b):
                return edge_vertex(ids[a], ids[b], positions[a], positions[b], samples[a], samples[b])

            if len(inside) == 1:
                triangle = tuple(cut(inside[0], other) for other in outside)
                add_face(triangle, outward)
            elif len(inside) == 3:
                triangle = tuple(cut(other, outside[0]) for other in inside)
                add_face(triangle, outward)
            else:
                a = cut(inside[0], outside[0])
                b = cut(inside[0], outside[1])
                c = cut(inside[1], outside[0])
                d = cut(inside[1], outside[1])
                # Convex perimeter is a-b-d-c; both triangles use diagonal a-d.
                add_face((a, b, d), outward)
                add_face((a, d, c), outward)

    if not faces:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    faces = np.asarray(faces, dtype=np.int64)
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices = np.asarray(vertices, dtype=np.float32)[used]
    return vertices, remap[faces].astype(np.int32)


def _write_mesh(path, vertices, faces, surface_name):
    normals = np.zeros_like(vertices)
    if len(faces):
        triangles = vertices[faces]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-20)
    vertex_data = np.empty(len(vertices), dtype=[(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz")])
    for index, name in enumerate(("x", "y", "z")):
        vertex_data[name] = vertices[:, index]
        vertex_data["n" + name] = normals[:, index]
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    comments = ["GShell " + surface_name + ": neural SDF zero level set",
                "coordinates: normalized scene units; winding and normals toward f>0",
                "extraction: shared-edge marching tetrahedra; see scene.json for grid and units"]
    PlyData([PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")],
            text=False, comments=comments).write(str(path))


@torch.no_grad()
def export_scene(model, output_dir, resolution=96):
    """Export both zero-set meshes, Gaussian PLY, environment NPY and metadata.

    Meshes remain in normalized coordinates, matching the Gaussian PLY. The
    cube grid covers the complete spherical tracing domain without introducing
    artificial clipping caps; counts expose any SDF vertices beyond that sphere.
    Inner normals point out of the cavity, following the inner SDF sign convention.
    """
    if resolution < 3:
        raise ValueError("Mesh grid resolution must be at least 3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    files = {"outer_shell": "outer_shell.ply", "inner_shell": "inner_shell.ply",
             "interior": "interior.ply", "environment": "environment.npy", "metadata": "scene.json"}
    mesh_info = {}
    for name, field in (("outer_shell", model.outer), ("inner_shell", model.inner)):
        axis, values = _sample_sdf(field, cfg.bound, resolution)
        vertices, faces = _marching_tetrahedra(axis, values)
        _write_mesh(output_dir / files[name], vertices, faces, name)
        mesh_info[name] = {
            "vertices": int(len(vertices)), "triangles": int(len(faces)),
            "vertices_outside_trace_sphere": int((np.linalg.norm(vertices, axis=-1) > cfg.bound).sum()),
            "empty_surface": len(faces) == 0,
            "normal_orientation": "toward positive SDF; outward from enclosed volume",
        }
        del values
    model.gaussians.save_ply(str(output_dir / files["interior"]))
    environment = model.environment.radiance().detach().cpu().numpy()
    np.save(str(output_dir / files["environment"]), environment)
    absorption = model.material.absorption.detach().cpu().numpy()
    metadata = {
        "format": "gshell-export-v1",
        "stage": int(model.stage),
        "files": files,
        "coordinates": {
            "space": "normalized scene coordinates",
            "normalization": "x_normalized = (x_world - scene_center) / scene_scale",
            "inverse": "x_world = scene_scale * x_normalized + scene_center",
            "scene_center": list(cfg.scene_center), "scene_scale": float(cfg.scene_scale),
            "world_scale_conversion": "Gaussian scales multiply by scene_scale; covariance by scene_scale squared",
        },
        "shell": {
            "ior": float(model.material.ior.item()), "surrounding_ior": 1.0,
            "absorption_rgb": absorption.tolist(), "absorption_units": "inverse normalized scene length",
            "absorption_per_world_length_rgb": (absorption / cfg.scene_scale).tolist(),
            "negative_sdf": {"outer_shell": "inside outer envelope", "inner_shell": "inside cavity"},
            "wall_region": "outer_sdf <= 0 and inner_sdf >= 0",
            "tracing_domain": {"shape": "sphere", "center": [0.0, 0.0, 0.0], "radius": float(cfg.bound)},
        },
        "extraction": {
            "method": "marching tetrahedra with shared edge vertices",
            "resolution_per_axis": int(resolution),
            "grid_bounds_xyz": [[-float(cfg.bound), float(cfg.bound)]] * 3,
            "grid_spacing": float(2.0 * cfg.bound / (resolution - 1)),
            "array_axis_order": "values[x_index, y_index, z_index]",
            "flatten_order": "C order: x slowest, z fastest",
            "level": 0.0, "sphere_clipping": False, "surfaces": mesh_info,
        },
        "interior": {
            "gaussian_count": int(model.gaussians.xyz.shape[0]),
            "active_sh_degree": int(model.gaussians.active_sh_degree),
            "density_units": "inverse normalized scene length; divide by scene_scale for world units",
            "opacity": "PLY opacity is a reference-ray viewer approximation; density is authoritative",
        },
        "environment": {
            "shape": list(environment.shape), "layout": "HWC RGB", "color_space": "linear HDR radiance",
            "projection": "equirectangular", "north": "+Z", "central_meridian": "+X",
            "longitude": "atan2(y,x); u = longitude / (2*pi) + 0.5",
            "latitude": "v = acos(z) / pi; row 0 is north",
        },
    }
    (output_dir / files["metadata"]).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {name: str(output_dir / filename) for name, filename in files.items()}
