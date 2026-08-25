"""fTetWild remeshing helpers matching Unified Fig.10.

The legacy script keeps its PolyFEM solver, but its remesh command and output
validation intentionally follow ``Unified_GIPC_new_diff``'s
``2_fig10_hanger_rest_shape/fig10_remesh.py``: absolute target edge length
``--la 0.05``, ASCII Gmsh output, and acceptance of any finite, non-degenerate
tetrahedral result instead of rejecting a valid output solely because its
minimum quality is below the optimization trigger threshold.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REMESH_UTILS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "Diff-GIPC",
        "StiffGIPC",
        "python",
    )
)
_ALT = os.path.abspath(
    os.path.join(os.path.expanduser("~"), "Diff-GIPC", "StiffGIPC", "python")
)
for _p in (_ALT, _REMESH_UTILS_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from remesh_utils import (  # noqa: E402
    boundary_triangle_quality_stats,
    extract_boundary_faces as extract_surface_faces,
    tet_mesh_quality_min,
    tet_quality_stats_from_mesh,
)

DEFAULT_FTETWILD_LA = 5.0e-2

__all__ = [
    "boundary_triangle_quality_stats",
    "extract_surface_faces",
    "run_remesh",
    "tet_mesh_quality_min",
    "tet_quality_stats_from_mesh",
]


def _write_surface_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for point in np.asarray(vertices, dtype=np.float64).reshape(-1, 3):
            out.write(f"v {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")
        for face in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
            out.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _outward_boundary_faces(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return consistently outward-oriented boundary triangles."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tetrahedra = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    faces: dict[tuple[int, int, int], list[int]] = {}
    counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for tet in tetrahedra:
        for local in local_faces:
            face = [int(tet[i]) for i in local]
            opposite_local = next(index for index in range(4) if index not in local)
            opposite = int(tet[opposite_local])
            p0, p1, p2 = points[face]
            normal = np.cross(p1 - p0, p2 - p0)
            if float(np.dot(normal, points[opposite] - p0)) > 0.0:
                face[1], face[2] = face[2], face[1]
            key = tuple(sorted(face))
            counts[key] += 1
            faces.setdefault(key, face)
    return np.asarray(
        [face for key, face in faces.items() if counts[key] == 1],
        dtype=np.int64,
    ).reshape(-1, 3)


def _resolve_binary() -> str:
    candidates = [
        os.environ.get("FTETWILD_BIN"),
        shutil.which("fTetWild"),
        shutil.which("ftetwild"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "fTetWild executable not found; pass FTETWILD_BIN or add fTetWild to PATH"
    )


def _read_tet_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import meshio

    mesh = meshio.read(str(path))
    points = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
    blocks = [
        np.asarray(cell.data, dtype=np.int64).reshape(-1, 4)
        for cell in mesh.cells
        if cell.type in ("tetra", "tetrahedron")
    ]
    if not blocks:
        raise ValueError(f"fTetWild output has no tetrahedra: {path}")
    tetrahedra = np.concatenate(blocks, axis=0)
    if not np.isfinite(points).all():
        raise FloatingPointError(f"fTetWild output vertices are not finite: {path}")
    if tetrahedra.min() < 0 or tetrahedra.max() >= points.shape[0]:
        raise ValueError(f"fTetWild output has invalid tetrahedron indices: {path}")
    quality = tet_quality_stats_from_mesh(str(path))
    if quality is None or not np.isfinite(list(quality.values())).all() or quality["min"] <= 0.0:
        raise ValueError(f"fTetWild output has degenerate tetrahedra: {path}")
    return points, tetrahedra


def run_remesh(
    root: str,
    remesh_iter: int,
    attempt_idx: int,
    vertices: np.ndarray,
    tets: np.ndarray,
    do_boundary_remesh: bool = False,
    surface_remesh_method: Optional[str] = None,
    surface_target_scale: float = 1.0,
    surface_remesh_iters: int = 0,
    *,
    volume_backend: str = "ftetwild",
    ftetwild_opts: Optional[str] = None,
    ftetwild_epsilon: Optional[float] = None,
) -> Tuple[bool, str, str, str, dict]:
    """Run one Unified-style fTetWild attempt.

    The legacy surface-remesh arguments remain in the signature so old callers
    do not break; Unified Fig.10 uses only the volume fTetWild path.
    ``attempt_idx`` controls the same 0.8 epsilon backoff used by Unified.
    """
    del (
        do_boundary_remesh,
        surface_remesh_method,
        surface_target_scale,
        surface_remesh_iters,
    )
    if (volume_backend or "ftetwild").lower() != "ftetwild":
        return False, "", "", "", {"error": f"unsupported volume_backend={volume_backend!r}"}

    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tetrahedra = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    if points.size == 0 or tetrahedra.size == 0:
        return False, "", "", "", {"error": "empty remesh input"}
    if not np.isfinite(points).all():
        return False, "", "", "", {"error": "non-finite remesh input"}

    try:
        executable = _resolve_binary()
    except FileNotFoundError as error:
        return False, "", "", "", {"error": str(error)}

    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"r{int(remesh_iter):03d}_try{int(attempt_idx)}"
    before_obj = out_dir / f"ftetwild_before_{tag}.obj"
    after_msh = out_dir / f"ftetwild_after_{tag}.msh"
    log_path = out_dir / f"ftetwild_{tag}.log"
    faces = _outward_boundary_faces(points, tetrahedra)
    if faces.size == 0:
        return False, str(before_obj), str(after_msh), str(log_path), {
            "backend": "ftetwild",
            "error": "no_boundary_faces",
        }

    _write_surface_obj(before_obj, points, faces)
    after_msh.unlink(missing_ok=True)
    options = shlex.split(ftetwild_opts) if ftetwild_opts else []
    has_length = any(
        token in {"-l", "--lr", "-a", "--la"}
        or token.startswith("--lr=")
        or token.startswith("--la=")
        for token in options
    )
    if not has_length:
        options.extend(["--la", f"{DEFAULT_FTETWILD_LA:.8e}"])
    if "--no-binary" not in options:
        options.append("--no-binary")

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    epsilon = None
    if ftetwild_epsilon is not None:
        epsilon = float(ftetwild_epsilon) * (0.8 ** max(int(attempt_idx) - 1, 0))
    relative_epsilon = None if epsilon is None else epsilon / max(bbox_diag, 1.0e-12)
    command = [executable, "-i", str(before_obj), "-o", str(after_msh)]
    if relative_epsilon is not None:
        command.extend(["--epsr", f"{relative_epsilon:.8e}"])
    command.extend(options)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)

    info = {
        "backend": "ftetwild",
        "ftetwild_return_code": int(process.returncode),
        "epsilon": epsilon,
        "command": command,
        "input_vertices": int(len(points)),
        "input_tets": int(len(tetrahedra)),
        "log": str(log_path),
    }
    if process.returncode != 0 or not after_msh.is_file():
        info["error"] = "ftetwild_output_missing"
        return False, str(before_obj), str(after_msh), str(log_path), info
    try:
        remeshed_points, remeshed_tets = _read_tet_mesh(after_msh)
    except (OSError, ValueError, FloatingPointError) as error:
        info["error"] = str(error)
        return False, str(before_obj), str(after_msh), str(log_path), info
    info.update(
        {
            "output_vertices": int(len(remeshed_points)),
            "output_tets": int(len(remeshed_tets)),
            "output_quality": tet_quality_stats_from_mesh(str(after_msh)),
        }
    )
    return True, str(before_obj), str(after_msh), str(log_path), info
