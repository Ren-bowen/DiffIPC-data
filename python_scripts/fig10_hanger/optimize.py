#!/usr/bin/env python3
"""
PolyFEM implementation of Diff-GIPC Case13 / Fig.10 hanger:
rest-shape optimisation of a 3D Stable NeoHookean solid under spatially varying gravity.

Case13 / fig10_hanger setup (matching Diff-GIPC ``19_fig10_3d_hanger``):
  - Soft mesh:  middle_cleaned_res5x.msh (design variables on the soft body)
  - Geometry:   no wood body (matches ``main.py`` ``ENABLE_WOOD=False``)
  - Material:   StableNeoHookean, E = 1e8, nu = 0.49, rho = 1000
  - Body force: b_y = -300 on the outer side of the two arm-normal cut
    planes (same count as the old ``|z| > 1`` split). Optimisation zeros the
    shape gradient on that same loaded set.
  - Loss:       Stable Neo-Hookean stress norm, matching GIPC's fig10 loss,
                for ``3 < y < 5`` plus an optional boundary
                Laplacian on the same soft Y-band (``--laplacian-weight``).
                The forward material remains StableNeoHookean.
  - Variables:  rest surface vertex positions on soft middle with ``3 <= y <= 5``
                inward of the two arm-normal cuts (same count as old ``|z| <= 1``)

Usage:
    conda activate diffipc
    cd /home/bowen/DiffIPC-data/python_scripts/fig10_hanger
    python optimize.py                  # headless
    python optimize.py --polyscope      # with Polyscope visualisation
    python optimize.py --verify-grad [--verify-eps 1e-5]   # grad FD check vs autograd, then exit

Outputs under ``opt/``:
  - iteration VTU, loss curve, etc.
  - each ``opt/remesh_*_iter_* / surface_rest.obj`` — boundary triangle mesh (rest / design
    geometry) after that optimisation step.
  - ``opt/remesh/`` — remesh-only files (before/after surface OBJ, ``after_remesh_*.msh``, logs, …).

Volume remeshing always uses **fTetWild** (see ``tet_remesh.run_remesh``); requires ``fTetWild``/``ftetwild`` on PATH.
"""

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Optional

import igl
import numpy as np
import torch

from feasibility_backtracking import feasible_backtracking, needs_remesh

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from polyfem_backend import load_polyfem

_backend_parser = argparse.ArgumentParser(add_help=False)
_backend_parser.add_argument('--polyfem-backend', choices=('ren-bowen', 'pinned'), default='ren-bowen')
_backend_args, _ = _backend_parser.parse_known_args()
pf = load_polyfem(_backend_args.polyfem_backend)
from tet_remesh import (
    boundary_triangle_quality_stats,
    extract_surface_faces,
    run_remesh,
    tet_mesh_quality_min,
    tet_quality_stats_from_mesh,
)

torch.set_default_dtype(torch.float64)

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOFT_MESH_NAME = "middle_cleaned_res5x.msh"
# Match Diff-GIPC's scene-12 objective normalization. This keeps the
# Laplacian weight on the same numerical scale as main.py/loss_utils.py.
_STRESS_SCALE = 1e-4
# PolyFEM's differentiable contact path requires a constant barrier stiffness.
# GIPC updates this coefficient internally, so this is the fixed PolyFEM
# counterpart used by the other aligned differentiable-contact cases.
_BARRIER_STIFFNESS = 1.0  # User-selected default contact stiffness.
_RELATIVE_DHAT = 3e-4  # Unified uses relative_dhat * current rest bbox diagonal.
_dynamic_dirichlet_paths: set[str] = set()
_legacy_polysolve_schema = False
_newton_stopping = "unified-linf"
_single_pass_profile = False


def _count_mesh_vertices(mesh_path: str) -> int:
    """Count Vertices block size for Medit ``.mesh`` / Gmsh-like ASCII meshes."""
    path = os.path.abspath(mesh_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.strip().split()
        if not token:
            continue
        if token[0] in ("Vertices", "$Nodes"):
            # Medit: "Vertices" then count on next line.
            # Gmsh ASCII v2: "$Nodes" then "numEntityBlocks numNodes ..."
            if token[0] == "Vertices":
                return int(lines[i + 1].strip().split()[0])
            parts = lines[i + 1].strip().split()
            if len(parts) >= 2:
                return int(parts[1])
            return int(parts[0])
    raise RuntimeError(f"could not count vertices in mesh: {path}")


def _read_medit_vertices(mesh_path: str) -> np.ndarray:
    """Read vertices from the Medit input or a Gmsh remesh output.

    The initial hanger mesh is Medit ``.mesh``.  fTetWild writes Gmsh
    ``.msh`` files, which are also accepted by PolyFEM but do not contain a
    Medit ``Vertices`` section.  Keeping this small fallback here lets the
    post-remesh Newton setup use the same vertex-based Dirichlet mask.
    """
    path = os.path.abspath(mesh_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().split()[:1] != ["Vertices"]:
            continue
        n = int(lines[i + 1].strip().split()[0])
        rows = []
        for row in lines[i + 2 : i + 2 + n]:
            values = row.strip().split()
            if len(values) < 3:
                raise RuntimeError(f"invalid Medit vertex row in {path}: {row!r}")
            rows.append([float(values[0]), float(values[1]), float(values[2])])
        vertices = np.asarray(rows, dtype=np.float64)
        if vertices.shape != (n, 3):
            raise RuntimeError(f"could not read {n} vertices from {path}")
        return vertices

    try:
        import meshio

        mesh = meshio.read(path)
        vertices = np.asarray(mesh.points, dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[1] >= 3 and vertices.shape[0] > 0:
            return vertices[:, :3].copy()
    except Exception as exc:
        raise RuntimeError(f"could not read vertices from mesh: {path}") from exc

    raise RuntimeError(f"could not find Vertices block in mesh: {path}")


def _boundary_facets(tets: np.ndarray) -> np.ndarray:
    """Return boundary triangles across libigl Python API versions."""
    facets = igl.boundary_facets(np.asarray(tets, dtype=np.int64))
    # Newer pyigl returns (facets, indices, signs); older versions return facets.
    if isinstance(facets, tuple):
        facets = facets[0]
    return np.asarray(facets, dtype=np.int32)


def soft_vertex_count_from_config(config: dict, root: str) -> int:
    """Number of vertices belonging to the deformable middle mesh (geometry[0])."""
    soft_rel = config["geometry"][0]["mesh"]
    soft_path = soft_rel if os.path.isabs(soft_rel) else os.path.join(root, soft_rel)
    return _count_mesh_vertices(soft_path)


def gipc_case13_fixed_mask(verts: np.ndarray, soft_n: int) -> np.ndarray:
    """GIPC's fixed set for the soft-only hanger: vertices outside the design strip."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    soft_n = int(soft_n)
    if soft_n < 0 or soft_n > len(verts):
        raise ValueError(f"soft_n={soft_n} incompatible with n_verts={len(verts)}")
    mask = np.zeros(len(verts), dtype=bool)
    y = verts[:soft_n, 1]
    mask[:soft_n] = (y < 3.0) | (y > 5.0)
    return mask


def hanger_arm_axis_planes(
    points: np.ndarray,
    *,
    require_fitted: bool = False,
) -> list[tuple[np.ndarray, float]]:
    """Fit one outward plane per arm, perpendicular to that arm's centerline."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    y = points[:, 1]
    z = points[:, 2]
    band = (y >= 3.0) & (y <= 5.0)
    planes: list[tuple[np.ndarray, float]] = []
    for sign in (1.0, -1.0):
        signed = sign * z
        arm = points[(signed > 0.25) & (signed < 0.90) & band]
        fallback = np.array([0.0, 0.0, sign], dtype=np.float64)
        if len(arm) < 4:
            assert not require_fitted, (
                f"Fig.10 {sign:+.0f}z arm-normal cut fell back to |z|>1: "
                f"only {len(arm)} axis samples"
            )
            planes.append((fallback, 1.0))
            continue
        zy = arm[:, (2, 1)]
        centered = zy - np.mean(zy, axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        _values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(_values))]
        if sign * axis[0] < 0.0:
            axis = -axis
        normal = np.array([0.0, axis[1], axis[0]], dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= 0.0:
            assert not require_fitted, (
                f"Fig.10 {sign:+.0f}z arm-normal cut fell back to |z|>1: "
                "invalid centerline"
            )
            planes.append((fallback, 1.0))
            continue
        normal = normal / length
        projection = points @ normal
        target = int((signed > 1.0).sum())
        candidates = projection[signed > 0.0]
        if candidates.size == 0:
            assert not require_fitted, (
                f"Fig.10 {sign:+.0f}z arm-normal cut fell back to |z|>1: "
                "no projected candidates"
            )
            planes.append((fallback, 1.0))
            continue
        best_error = None
        best_offset = float(np.median(candidates))
        for offset in np.quantile(candidates, np.linspace(0.0, 1.0, 4001)):
            count = int(((signed > 0.0) & (projection > offset)).sum())
            error = abs(count - target)
            if best_error is None or error < best_error:
                best_error = error
                best_offset = float(offset)
                if error == 0:
                    break
        if require_fitted:
            assert not (
                np.allclose(normal, fallback) and abs(best_offset - 1.0) <= 1e-12
            ), f"Fig.10 {sign:+.0f}z arm-normal cut degenerated to |z|>1"
            assert abs(float(normal[2])) < 0.95, (
                f"Fig.10 arm-normal cut is too close to a z-plane: n={normal}"
            )
        planes.append((normal, best_offset))
    return planes


def hanger_arm_loaded_mask(
    points: np.ndarray,
    *,
    require_fitted: bool = False,
) -> np.ndarray:
    """True on the outer/loaded side of the two arm-normal cut planes."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    z = points[:, 2]
    loaded = np.zeros(len(points), dtype=bool)
    for sign, (normal, offset) in zip(
        (1.0, -1.0),
        hanger_arm_axis_planes(points, require_fitted=require_fitted),
    ):
        loaded |= (sign * z > 0.0) & ((points @ normal) > offset)
    return loaded


def hanger_arm_rhs_expression(points: np.ndarray, magnitude: float = -300.0) -> str:
    """PolyFEM ``if`` expression for the two arm-normal body-force planes."""
    planes = hanger_arm_axis_planes(points, require_fitted=True)
    if len(planes) != 2:
        raise RuntimeError("Fig.10 arm-normal cut must not fall back to |z|>1")
    print(
        "fig10 arm-normal cuts require_fitted=True "
        + "; ".join(
            f"sign={sign:+.0f} n=[{normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}] "
            f"offset={offset:.6f}"
            for sign, (normal, offset) in zip((1.0, -1.0), planes)
        )
        + " (no |z|>1 fallback)",
        flush=True,
    )
    (n_pos, t_pos), (n_neg, t_neg) = planes
    pos = f"(({n_pos[1]:.17g})*y + ({n_pos[2]:.17g})*z - ({t_pos:.17g}))"
    neg = f"(({n_neg[1]:.17g})*y + ({n_neg[2]:.17g})*z - ({t_neg:.17g}))"
    return f"if(z, if({pos}, {magnitude:g}, 0), if(-z, if({neg}, {magnitude:g}, 0), 0))"


def gipc_case13_body_force_rhs_mask(verts: np.ndarray) -> np.ndarray:
    """Nonzero body-force vertices: outer side of the two arm-normal cuts."""
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim == 1:
        return np.abs(verts) > 1.0
    return hanger_arm_loaded_mask(verts, require_fitted=True)


def gipc_case13_design_band_mask(y: np.ndarray) -> np.ndarray:
    """Match Diff-GIPC ``19_fig10_3d_hanger`` rest_diff band: ``3 <= y <= 5``."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return (y >= 3.0) & (y <= 5.0)


def gipc_case13_grad_shield_z_mask(z: np.ndarray) -> np.ndarray:
    """Legacy ``|z| > 1`` shield; used only when arm-axis fitting is unavailable."""
    return np.abs(np.asarray(z, dtype=np.float64).reshape(-1)) > 1.0


def gipc_case13_design_mask(
    verts: np.ndarray,
    fixed_mask: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Per-topology surface design DOFs: soft ∩ 3<=y<=5 ∩ inward of arm cuts."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    surface_mask = np.zeros(len(verts), dtype=bool)
    surface_mask[np.unique(np.asarray(faces, dtype=np.int64))] = True
    return (
        surface_mask
        & (~fixed_mask)
        & gipc_case13_design_band_mask(verts[:, 1])
        & (~hanger_arm_loaded_mask(verts, require_fitted=True))
    )


# ---------------------------------------------------------------------------
#  Differentiable dynamic simulator (shape parametrisation)
# ---------------------------------------------------------------------------

def completed_rollout_cache(solver, n_dofs: int, expected_frames: int) -> np.ndarray:
    """Require all u_0,...,u_T states before exposing the terminal state."""
    if expected_frames < 1:
        raise ValueError("expected_frames must be positive")
    cached = np.asarray(solver.get_solutions(), dtype=np.float64)
    expected_shape = (n_dofs, expected_frames + 1)
    if cached.shape != expected_shape:
        raise RuntimeError(
            f"Incomplete or invalid PolyFEM rollout: cache shape {cached.shape}, "
            f"expected {expected_shape} (initial state + {expected_frames} frames)"
        )
    if not np.isfinite(cached).all():
        raise RuntimeError("Non-finite PolyFEM rollout")
    return cached


class AcceptedForward:
    """One-use ownership of a successful native derivative cache."""

    def __init__(self, solver, vertices, expected_frames):
        self.solver = solver
        self.vertices = np.asarray(vertices, dtype=np.float64).copy()
        self.frames = expected_frames
        self.tets = np.asarray(solver.mesh().elements()).copy()
        completed_rollout_cache(solver, self.vertices.size, expected_frames)
        self.consumed = False
        self.reused = False

    def take(self, solver, vertices, expected_frames):
        if self.consumed:
            return None
        self.consumed = True
        if (solver is not self.solver or expected_frames != self.frames
                or not np.array_equal(vertices, self.vertices)
                or not np.array_equal(solver.mesh().vertices(), self.vertices)
                or not np.array_equal(solver.mesh().elements(), self.tets)):
            return None
        cached = completed_rollout_cache(solver, self.vertices.size, expected_frames)
        self.reused = True
        return cached


class Simulate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, solver, vertices, state_seed_mask, expected_frames, accepted_forward=None):
        seed_mask = np.asarray(
            state_seed_mask.detach().cpu(), dtype=bool
        ).reshape(-1)
        if seed_mask.size != vertices.shape[0] or vertices.dim() != 2 or vertices.shape[1] != 3:
            raise ValueError(
                "state_seed_mask must have one entry per (N, 3) vertex input"
            )
        points = vertices.detach().cpu().numpy()
        cached = (accepted_forward.take(solver, points, expected_frames)
                  if accepted_forward is not None else None)
        if cached is None:
            solver.mesh().set_vertices(points)
            solver.set_cache_level(pf.CacheLevel.Derivatives)
            # Propagate failures; never substitute an incomplete trajectory.
            solver.solve(0)
            cached = completed_rollout_cache(solver, vertices.numel(), expected_frames)
        # On reuse, do not call mesh/cache setters: they may invalidate native derivatives.
        # PolyFEM caches u_0, ..., u_T for a transient solve.  The Unified
        # Fig.10 module returns only the final state of its 20-frame rollout,
        # so expose the final displacement here as the differentiable output.
        sol = torch.from_numpy(cached[:, -1].copy())
        if sol.numel() == 0:
            raise RuntimeError("PolyFEM produced an empty cached solution")
        ctx.solver = solver
        ctx.time_steps = int(cached.shape[1] - 1)
        ctx.input_shape = tuple(vertices.shape)
        # The historical state_seed_mask argument is retained for callers,
        # but selects design variables only. Non-design state DOFs still
        # depend on the design and must contribute their full loss seed.
        # The optimization/FD caller restricts the final rest gradient.
        return sol

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        # Never mutate grad_output: a CPU NumPy view can alias another
        # PyTorch branch (in particular deformed = rest + displacement).
        grad = np.array(grad_output.detach().cpu(), dtype=np.float64, copy=True).reshape(-1)
        # solve_adjoint expects one RHS column per cached transient state,
        # including u_0.  The loss is evaluated on u_T only, hence all earlier
        # columns are zero and the final column receives the PyTorch seed.
        adjoint_rhs = np.zeros((grad.size, ctx.time_steps + 1), dtype=np.float64)
        adjoint_rhs[:, -1] = grad
        adjoint_started = time.perf_counter()
        ctx.solver.solve_adjoint(torch.from_numpy(adjoint_rhs))
        adjoint_seconds = time.perf_counter() - adjoint_started
        shape_started = time.perf_counter()
        shape_grad = np.asarray(pf.shape_derivative(ctx.solver), dtype=np.float64)
        shape_seconds = time.perf_counter() - shape_started
        if _single_pass_profile:
            print(
                "BACKWARD_DETAIL "
                + json.dumps({
                    "solve_adjoint_seconds": adjoint_seconds,
                    "shape_derivative_seconds": shape_seconds,
                }),
                flush=True,
            )
        result = (None, torch.as_tensor(
            shape_grad,
            dtype=grad_output.dtype,
            device=grad_output.device,
        ).reshape(ctx.input_shape), None, None)
        return result + (None,) if len(ctx.needs_input_grad) == 5 else result


def init_solver(config, log_level):
    if _legacy_polysolve_schema:
        config = copy.deepcopy(config)
        nonlinear = config["solver"]["nonlinear"]
        nonlinear["grad_norm"] = nonlinear.pop("grad_norm_tol", 1e-3)
        nonlinear["x_delta"] = 0
        for key in ("norm_type", "x_delta_tol", "rel_grad_norm_tol"):
            nonlinear.pop(key, None)
    solver = pf.Solver()
    solver.set_settings(json.dumps(config))
    solver.set_max_threads(int(config["solver"].get("max_threads", 32)))
    solver.set_log_level(log_level)
    solver.load_mesh_from_settings()
    return solver


def init_solver_with_gipc_newton(
    config: dict,
    log_level: int,
    newton_threshold: float,
    newton_dt: float,
) -> pf.Solver:
    """Configure GIPC's tolerance and load one solver instance.

    The old probe-and-reload sequence briefly created two native PolyFEM solver
    objects. Besides increasing peak RSS, that sequence can segfault in some
    polyfempy builds when contact is enabled. This case has no mesh transform,
    so the Medit vertices provide the exact bbox and nodal-BC indexing up front.
    """
    mesh_rel = config["geometry"][0]["mesh"]
    mesh_path = mesh_rel if os.path.isabs(mesh_rel) else os.path.join(CASE_DIR, mesh_rel)
    vertices = _read_medit_vertices(mesh_path)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    config.setdefault("contact", {})["dhat"] = _RELATIVE_DHAT * bbox_diag
    install_gipc_case13_dirichlet(config, vertices)
    config.setdefault("boundary_conditions", {})["rhs"] = [
        0,
        hanger_arm_rhs_expression(vertices),
        0,
    ]
    configure_newton_stopping(config, newton_threshold, newton_dt, bbox_diag)
    return init_solver(config, log_level)


def install_gipc_case13_dirichlet(config: dict, vertices: np.ndarray) -> str:
    """Install nodal BCs matching GIPC's fixed soft strip.

    PolyFEM's JSON boundary IDs only cover boundary facets. GIPC fixes every
    soft node outside ``3 <= y <= 5``, so use PolyFEM's nodal-BC matrix format
    (``node_id ux uy uz``) for the exact same set.
    """
    soft_n = soft_vertex_count_from_config(config, CASE_DIR)
    fixed = gipc_case13_fixed_mask(vertices, soft_n)
    node_ids = np.flatnonzero(fixed).astype(np.int64)
    if node_ids.size == 0:
        raise RuntimeError("no nodal Dirichlet vertices found for Case13")

    fd, path = tempfile.mkstemp(prefix="fig10_hanger_dirichlet_", suffix=".txt")
    os.close(fd)
    values = np.zeros((node_ids.size, 4), dtype=np.float64)
    values[:, 0] = node_ids
    np.savetxt(path, values, fmt=["%d", "%.17g", "%.17g", "%.17g"])

    boundary = config.setdefault("boundary_conditions", {}).setdefault(
        "dirichlet_boundary", []
    )
    if not isinstance(boundary, list):
        boundary = [boundary]
    boundary = [
        entry
        for entry in boundary
        if not (isinstance(entry, str) and entry in _dynamic_dirichlet_paths)
    ]
    boundary.append(path)
    config["boundary_conditions"]["dirichlet_boundary"] = boundary
    _dynamic_dirichlet_paths.add(path)
    return path


def configure_material_and_solver(config: dict, max_threads: int = 32) -> None:
    """Apply the common Stable-NH/Newton settings used by the aligned cases."""
    materials = config.get("materials")
    if isinstance(materials, list):
        for material in materials:
            material["type"] = "StableNeoHookean"
            material["E"] = 1e8
            material["nu"] = 0.49
            material["rho"] = 1000.0
    elif isinstance(materials, dict):
        materials["type"] = "StableNeoHookean"
        materials["E"] = 1e8
        materials["nu"] = 0.49
        materials["rho"] = 1000.0
    else:
        config["materials"] = {
            "type": "StableNeoHookean",
            "E": 1e8,
            "nu": 0.49,
            "rho": 1000.0,
        }

    config.setdefault("space", {}).setdefault("advanced", {})["quadrature_order"] = 1

    solver_cfg = config.setdefault("solver", {})
    solver_cfg["max_threads"] = max(1, int(max_threads))
    # Recent polyfempy validates the separate adjoint_linear block when
    # settings are supplied through the Python API.  The command-line
    # PolyFEM binary fills this default implicitly, but polyfempy does not.
    linear = solver_cfg.setdefault("linear", {})
    if isinstance(linear, dict):
        linear.pop("adjoint_solver", None)
        # Preserve Fig10's measured algorithm even when MKL is now available.
        linear["solver"] = "Eigen::SimplicialLDLT"
    solver_cfg.setdefault(
        "adjoint_linear",
        {
            "solver": "Eigen::SimplicialLDLT",
            "precond": "Eigen::DiagonalPreconditioner",
        },
    )
    solver_cfg.setdefault("advanced", {})[
        "scale_grad_norm_by_characteristic_length"
    ] = True
    nonlinear = solver_cfg.setdefault("nonlinear", {})
    nonlinear["solver"] = "Newton"
    for key in ("x_delta", "grad_norm", "norm_type", "x_delta_tol"):
        nonlinear.pop(key, None)
    nonlinear["grad_norm_tol"] = 1e-3
    nonlinear.setdefault("line_search", {})["method"] = "Backtracking"
    nonlinear["line_search"]["use_grad_norm_tol"] = 0.0

    # These are top-level contact settings in PolyFEM. The nested solver
    # object accepts solver/contact options such as barrier_stiffness only.
    contact = solver_cfg.setdefault("contact", {})
    for key in ("enabled", "dhat", "use_convergent_formulation"):
        contact.pop(key, None)
    contact["barrier_stiffness"] = _BARRIER_STIFFNESS
    contact.setdefault("CCD", {})["broad_phase"] = "bvh"
    config_contact = config.setdefault("contact", {})
    config_contact["enabled"] = True
    # init_solver_with_gipc_newton computes the physical gap from each mesh,
    # including meshes generated by remeshing.
    config_contact["use_convergent_formulation"] = True


def configure_newton_stopping(
    config: dict,
    newton_threshold: float,
    newton_dt: float,
    bbox_diag: float,
) -> float:
    """Select legacy gradient stopping or Unified's absolute direction test."""
    if bbox_diag <= 0.0 or not np.isfinite(bbox_diag):
        raise ValueError(f"bbox_diag must be positive and finite, got {bbox_diag}")
    nonlinear = config["solver"]["nonlinear"]
    for key in ("norm_type", "x_delta_tol", "x_delta", "grad_norm"):
        nonlinear.pop(key, None)
    nonlinear.pop("x_delta_linf_absolute_tol", None)
    if _newton_stopping == "unified-linf":
        tolerance = newton_threshold * newton_dt * bbox_diag
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Unified movement tolerance must be positive and finite")
        nonlinear.update(grad_norm_tol=0.0, first_grad_norm_tol=0.0,
                         max_iterations=10000, allow_out_of_iterations=False)
        advanced = nonlinear.setdefault("advanced", {})
        if _legacy_polysolve_schema:
            nonlinear["x_delta_linf_absolute_tol"] = tolerance
            advanced.update(f_delta=0.0, derivative_along_delta_x_tol=0.0)
        else:
            # Modern NLProblem rescales Linf x_delta_tol by its bbox diagonal.
            # dt is included here exactly once; bbox is included in native code.
            nonlinear.update(norm_type="Linf", x_delta_tol=newton_threshold * newton_dt,
                             rel_grad_norm_tol=0.0, rel_x_delta_tol=0.0,
                             newton_decrement_tol=0.0)
            advanced.pop("f_delta", None)
            advanced.update(f_delta_tol=0.0, derivative_along_delta_x_tol=0.0)
        return tolerance
    nonlinear["grad_norm_tol"] = 1e-3
    return 1e-3


def print_newton_stopping(solver, newton_threshold, newton_dt, prefix="  "):
    vertices = np.asarray(solver.mesh().vertices(), dtype=np.float64)
    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if _newton_stopping == "unified-linf":
        print(f"{prefix}Newton stopping: max(abs(direction)) < "
              f"{newton_threshold * newton_dt * bbox_diag:.17g}; "
              f"threshold={newton_threshold:.17g} dt={newton_dt:.17g} "
              f"bboxDiag={bbox_diag:.17g}; gradient stopping disabled", flush=True)
        return
    print(
        f"{prefix}Newton stopping: grad_norm_tol=1e-3  "
        f"threshold={newton_threshold:.6e}  GIPC_dt={newton_dt:.6e}  "
        f"bboxDiag={bbox_diag:.6e}"
    )


def save_tetrahedral_volume_mesh(path: str, verts: np.ndarray, tets: np.ndarray) -> None:
    """Write volume tet mesh. Tries meshio, else VTK legacy ASCII."""
    verts = np.asarray(verts, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must be (n_tet, 4)")
    try:
        import meshio
        meshio.write(
            path,
            meshio.Mesh(points=verts, cells=[("tetra", tets)]),
        )
        return
    except Exception:
        pass
    base, ext = os.path.splitext(path)
    if ext.lower() not in (".vtk",):
        path = base + ".vtk"
    n_v, n_t = verts.shape[0], tets.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("tet mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_v} double\n")
        for v in verts:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        f.write(f"CELLS {n_t} {n_t * 5}\n")
        for t in tets:
            f.write(f"4 {int(t[0])} {int(t[1])} {int(t[2])} {int(t[3])}\n")
        f.write(f"CELL_TYPES {n_t}\n")
        for _ in range(n_t):
            f.write("10\n")


# ---------------------------------------------------------------------------
#  Boundary Laplacian smoothing (same functional as StiffGIPC ``scene12_torch.laplacian_loss_torch``)
# ---------------------------------------------------------------------------


def _undirected_edges_from_triangles(faces: torch.Tensor) -> torch.Tensor:
    """Unique undirected edges as (E, 2) with faces[:,0] < faces[:,1]."""
    f = faces.long()
    if f.numel() == 0:
        return f.new_empty((0, 2))
    e = torch.cat(
        [
            torch.stack([f[:, 0], f[:, 1]], dim=1),
            torch.stack([f[:, 1], f[:, 2]], dim=1),
            torch.stack([f[:, 2], f[:, 0]], dim=1),
        ],
        dim=0,
    )
    a = torch.minimum(e[:, 0], e[:, 1])
    b = torch.maximum(e[:, 0], e[:, 1])
    return torch.unique(torch.stack([a, b], dim=1), dim=0)


def laplacian_loss_torch(
    x: torch.Tensor,
    faces: torch.Tensor,
    boundary_vertex_ids: Optional[torch.Tensor] = None,
    p: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale-invariant boundary smoothing; ``x`` is rest vertices (V,3), ``faces`` boundary triangles."""
    if x.dim() != 2 or x.shape[1] != 3:
        raise ValueError("x must be (n_vert, 3)")
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (n_face, 3)")
    if p < 1.0:
        raise ValueError("p must be >= 1")

    device, dt = x.device, x.dtype
    faces = faces.to(device=device)
    edges = _undirected_edges_from_triangles(faces)
    if edges.numel() == 0:
        return x.new_zeros(())

    n_v = int(x.shape[0])
    if boundary_vertex_ids is None:
        b_idx = torch.unique(faces.long().ravel())
    else:
        b_idx = boundary_vertex_ids.long().ravel().unique()
    bmask = torch.zeros(n_v, dtype=torch.bool, device=device)
    bmask[b_idx] = True

    i, j = edges[:, 0], edges[:, 1]
    keep = bmask[i] & bmask[j]
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return x.new_zeros(())

    d = x[i] - x[j]
    ell = torch.linalg.norm(d, dim=1)
    num = torch.zeros((n_v, 3), dtype=dt, device=device)
    den = torch.zeros(n_v, dtype=dt, device=device)
    num.index_add_(0, i, d)
    num.index_add_(0, j, -d)
    den.index_add_(0, i, ell)
    den.index_add_(0, j, ell)

    safe = den > eps
    s = torch.where(
        safe.unsqueeze(-1),
        num / den.unsqueeze(-1).clamp(min=eps),
        torch.zeros((n_v, 3), dtype=dt, device=device),
    )
    norms = torch.linalg.norm(s[bmask], dim=1)
    return norms.pow(p).sum()


def gipc_case13_loss_tet_mask(
    rest_np: np.ndarray,
    tets_np: np.ndarray,
    soft_n: int,
) -> np.ndarray:
    """Select soft tets whose rest centroid is strictly inside ``3 < y < 5``."""
    rest_np = np.asarray(rest_np, dtype=np.float64).reshape(-1, 3)
    tets_np = np.asarray(tets_np, dtype=np.int64).reshape(-1, 4)
    soft_tet = np.all(tets_np < int(soft_n), axis=1)
    centroids = rest_np[tets_np].mean(axis=1)
    return soft_tet & (centroids[:, 1] > 3.0) & (centroids[:, 1] < 5.0)


def gipc_case13_laplacian_surface(
    rest_np: np.ndarray,
    faces_np: np.ndarray,
    soft_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return soft surface faces and vertex ids in the strict loss Y-band."""
    rest_np = np.asarray(rest_np, dtype=np.float64).reshape(-1, 3)
    faces_np = np.asarray(faces_np, dtype=np.int64)
    if faces_np.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
    faces_np = faces_np.reshape(-1, 3)
    band_v = np.zeros(rest_np.shape[0], dtype=bool)
    band_v[: int(soft_n)] = (
        (rest_np[: int(soft_n), 1] > 3.0)
        & (rest_np[: int(soft_n), 1] < 5.0)
    )
    # Keep the full surface stencil; the loss selects edges with both ends in band.
    soft_faces = faces_np[np.all(faces_np < int(soft_n), axis=1)]
    surface_ids = np.unique(soft_faces)
    return soft_faces, surface_ids[band_v[surface_ids]].astype(np.int64)


def _stable_nhk_rates(lam: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """StableNeoHookean rates matching PolyFEM's pinned implementation."""
    return 4.0 * mu / 3.0, lam + 5.0 * mu / 6.0 + 1e-4


def _stable_nhk_cofactor(F: torch.Tensor) -> torch.Tensor:
    """Return the batch cofactor matrix for 3x3 deformation gradients."""
    c = torch.empty_like(F)
    c[..., 0, 0] = F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1]
    c[..., 0, 1] = F[..., 1, 2] * F[..., 2, 0] - F[..., 1, 0] * F[..., 2, 2]
    c[..., 0, 2] = F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0]
    c[..., 1, 0] = F[..., 2, 1] * F[..., 0, 2] - F[..., 2, 2] * F[..., 0, 1]
    c[..., 1, 1] = F[..., 2, 2] * F[..., 0, 0] - F[..., 2, 0] * F[..., 0, 2]
    c[..., 1, 2] = F[..., 2, 0] * F[..., 0, 1] - F[..., 2, 1] * F[..., 0, 0]
    c[..., 2, 0] = F[..., 0, 1] * F[..., 1, 2] - F[..., 0, 2] * F[..., 1, 1]
    c[..., 2, 1] = F[..., 0, 2] * F[..., 1, 0] - F[..., 0, 0] * F[..., 1, 2]
    c[..., 2, 2] = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
    return c


def _first_piola_stable_nhk(F: torch.Tensor, lam: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """StableNeoHookean first Piola stress matching PolyFEM/GIPC."""
    length_rate, volume_rate = _stable_nhk_rates(lam, mu)
    J = torch.linalg.det(F)
    q = volume_rate * (J - 1.0) - length_rate
    return length_rate * F + q[..., None, None] * _stable_nhk_cofactor(F)


def stable_nh_stress_norm_loss(
    deformed: torch.Tensor,
    rest: torch.Tensor,
    tets: torch.Tensor,
    stress_power: int,
    tet_mask: Optional[torch.Tensor] = None,
    stress_weight: float = 1.0,
) -> torch.Tensor:
    """StableNeoHookean ``stress_norm`` loss over selected tets."""
    if deformed.dim() != 2 or deformed.shape[1] != 3:
        raise ValueError("deformed must be (n_vert, 3)")
    if rest.dim() != 2 or rest.shape[1] != 3:
        raise ValueError("rest must be (n_vert, 3)")
    tets = tets.to(device=rest.device, dtype=torch.long)
    if tets.dim() != 2 or tets.shape[1] != 4:
        raise ValueError("tets must be (n_tet, 4)")
    if stress_power < 1:
        raise ValueError("stress_power must be >= 1")

    dtype = torch.float64
    deformed = deformed.to(dtype=dtype)
    rest = rest.to(dtype=dtype, device=deformed.device)
    tets = tets.to(device=deformed.device)
    rest_tet = rest[tets]
    def_tet = deformed[tets]
    dm = torch.stack(
        [rest_tet[:, 1] - rest_tet[:, 0], rest_tet[:, 2] - rest_tet[:, 0], rest_tet[:, 3] - rest_tet[:, 0]],
        dim=-1,
    )
    ds = torch.stack(
        [def_tet[:, 1] - def_tet[:, 0], def_tet[:, 2] - def_tet[:, 0], def_tet[:, 3] - def_tet[:, 0]],
        dim=-1,
    )
    eye = torch.eye(3, dtype=dtype, device=deformed.device).expand(tets.shape[0], -1, -1)
    F = torch.bmm(ds, torch.linalg.solve(dm, eye))
    if not bool(torch.all(torch.linalg.det(F) > 0.0)):
        raise FloatingPointError("Fig.10 deformation contains an inverted tet")

    E = torch.as_tensor(1e8, dtype=dtype, device=deformed.device)
    nu = torch.as_tensor(0.49, dtype=dtype, device=deformed.device)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    P = _first_piola_stable_nhk(F, lam, mu) * _STRESS_SCALE

    volume = torch.linalg.det(dm).abs() / 6.0
    p = float(stress_power)
    stress_sq = (P * P).sum(dim=(-2, -1)).clamp(min=0.0)
    per_tet = stress_sq.pow(p / 2.0) * volume
    if tet_mask is not None:
        per_tet = per_tet * tet_mask.to(device=per_tet.device, dtype=per_tet.dtype)
    return stress_weight * per_tet.sum()


_eval_stress_norm_mesh_seq = [0]


def eval_stress_norm_loss(
    solver,
    verts_np: np.ndarray,
    stress_power: int,
    soft_n: int,
    *,
    dump_mesh: bool = False,
    strict_solve: bool = False,
    tet_mask: Optional[np.ndarray] = None,
    expected_frames: int = 20,
    stress_weight: float = 1.0,
) -> float:
    """Evaluate the selected StableNeoHookean stress norm after a dynamic rollout."""
    verts_np = np.asarray(verts_np, dtype=np.float64)
    n = verts_np.shape[0]
    solver.mesh().set_vertices(verts_np)
    solver.set_cache_level(pf.CacheLevel.Derivatives)
    # All evaluations are strict, including FD and optional legacy line search.
    # Retain strict_solve only for compatibility with existing callers.
    solver.solve()
    solution_cache = completed_rollout_cache(solver, n * 3, expected_frames)
    sol = solution_cache[:, -1].copy()
    if dump_mesh and sol.size == n * 3:
        u = sol.reshape(n, 3)
        deformed = verts_np + u
        _eval_stress_norm_mesh_seq[0] += 1
        _seq = _eval_stress_norm_mesh_seq[0]
        _mesh_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "eval_stress_norm_mesh_dump"
        )
        os.makedirs(_mesh_dir, exist_ok=True)
        tets = np.asarray(solver.mesh().elements(), dtype=np.int64)
        _p_def = os.path.join(_mesh_dir, f"deformed_{_seq:04d}.vtk")
        _p_rest = os.path.join(_mesh_dir, f"rest_{_seq:04d}.vtk")
        save_tetrahedral_volume_mesh(_p_def, deformed, tets)
        save_tetrahedral_volume_mesh(_p_rest, verts_np, tets)
        print(f"[eval_stress_norm_loss] saved tet mesh: {_p_def}  (rest: {_p_rest})")
    tets = np.asarray(solver.mesh().elements(), dtype=np.int64).reshape(-1, 4)
    mask = torch.as_tensor(
        (gipc_case13_loss_tet_mask(verts_np, tets, soft_n)
         if tet_mask is None else tet_mask), dtype=torch.bool
    )
    rest_t = torch.tensor(verts_np, dtype=torch.float64)
    def_t = torch.tensor(verts_np + sol.reshape(n, 3), dtype=torch.float64)
    tet_t = torch.as_tensor(tets, dtype=torch.long)
    return float(stable_nh_stress_norm_loss(def_t, rest_t, tet_t, stress_power, mask, stress_weight=stress_weight))


def eval_laplacian_loss_np(
    verts_np: np.ndarray,
    surf_faces: np.ndarray,
    boundary_vertex_ids: Optional[np.ndarray] = None,
) -> float:
    """Scalar Laplacian regulariser at ``verts_np`` (no autograd)."""
    verts_np = np.asarray(verts_np, dtype=np.float64)
    if surf_faces.size == 0:
        return 0.0
    vt = torch.tensor(verts_np, dtype=torch.float64)
    ft = torch.as_tensor(np.asarray(surf_faces, dtype=np.int64), dtype=torch.long)
    b_ids = None
    if boundary_vertex_ids is not None:
        b_ids = torch.as_tensor(boundary_vertex_ids, dtype=torch.long)
    return float(laplacian_loss_torch(vt, ft, boundary_vertex_ids=b_ids).detach())


def verify_grad(
    solver,
    current_verts: np.ndarray,
    fixed_mask: np.ndarray,
    soft_n: int,
    stress_power: int,
    eps: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
    rtol: float = 1e-2,
    atol: float = 1e-8,
    laplacian_weight: float = 0.0,
    surf_faces: Optional[np.ndarray] = None,
    expected_frames: int = 20,
    stress_weight: float = 1.0,
) -> dict:
    """Compare autograd directional derivative :math:`g\\cdot\\theta` with central FD for the
    optimisation objective (stress-norm, optionally plus Laplacian on the boundary surface).

    ``fixed_mask`` True = excluded from the design update. Design DOFs match
    Diff-GIPC hanger: soft ∩ ``3<=y<=5`` ∩ ``|z|<=1``. θ is L2-normalised
    on remaining design vertices.

    Use ``eps`` ~ 1e-5 … 1e-3 for stiff problems; 1e-7 often gives fd≈0 by float cancellation.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    base = np.asarray(current_verts, dtype=np.float64).copy()
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    n_verts = base.shape[0]
    design_mask = gipc_case13_design_mask(base, fixed_mask, extract_surface_faces(solver.mesh().elements()))

    tets_np = np.asarray(solver.mesh().elements(), dtype=np.int64).reshape(-1, 4)
    tet_mask_t = torch.as_tensor(
        gipc_case13_loss_tet_mask(base, tets_np, soft_n), dtype=torch.bool
    )
    lap_faces_np, lap_boundary_ids_np = gipc_case13_laplacian_surface(
        base,
        extract_surface_faces(tets_np) if surf_faces is None else surf_faces,
        soft_n,
    )
    if laplacian_weight > 0.0 and surf_faces is None:
        surf_faces = np.asarray(
            extract_surface_faces(np.asarray(solver.mesh().elements(), dtype=np.int64)),
            dtype=np.int64,
        )
    faces_tt: Optional[torch.Tensor] = None
    lap_boundary_ids_t: Optional[torch.Tensor] = None
    if laplacian_weight > 0.0 and lap_faces_np.size > 0:
        faces_tt = torch.as_tensor(lap_faces_np, dtype=torch.long)
        lap_boundary_ids_t = torch.as_tensor(lap_boundary_ids_np, dtype=torch.long)

    theta = np.zeros((n_verts, 3), dtype=np.float64)
    flex = design_mask
    theta[flex] = rng.standard_normal((int(flex.sum()), 3))
    # theta[flex] = np.ones((int(flex.sum()), 3))
    tn = float(np.linalg.norm(theta[flex].ravel()))
    if tn == 0.0:
        raise ValueError("theta on flexible vertices is zero")
    theta[flex] /= tn

    # Autograd: g·θ — same graph as the optimisation loop.
    verts_t = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    sol = Simulate.apply(solver, verts_t, torch.as_tensor(design_mask, dtype=torch.bool), expected_frames)
    deformed_t = verts_t + sol.reshape_as(verts_t)
    loss_stress = stable_nh_stress_norm_loss(
        deformed_t,
        verts_t,
        torch.as_tensor(tets_np, dtype=torch.long),
        stress_power,
        tet_mask_t,
        stress_weight=stress_weight,
    )
    if laplacian_weight > 0.0 and faces_tt is not None:
        loss = loss_stress + laplacian_weight * laplacian_loss_torch(
            verts_t, faces_tt, boundary_vertex_ids=lap_boundary_ids_t
        )
    else:
        loss = loss_stress
    loss.backward()
    grad = verts_t.grad.detach().numpy().copy()
    grad[~design_mask] = 0.0
    g_norm = float(np.linalg.norm(grad[flex]))
    analytic = float(np.sum(grad * theta))

    print("eps * theta: ", eps * theta)
    print("theta: ", theta)
    f_plus = eval_stress_norm_loss(solver, base + eps * theta, stress_power, soft_n, tet_mask=tet_mask_t.numpy(), expected_frames=expected_frames, stress_weight=stress_weight)
    f_minus = eval_stress_norm_loss(solver, base - eps * theta, stress_power, soft_n, tet_mask=tet_mask_t.numpy(), expected_frames=expected_frames, stress_weight=stress_weight)
    if laplacian_weight > 0.0 and lap_faces_np.size > 0:
        f_plus += laplacian_weight * eval_laplacian_loss_np(
            base + eps * theta, lap_faces_np, lap_boundary_ids_np
        )
        f_minus += laplacian_weight * eval_laplacian_loss_np(
            base - eps * theta, lap_faces_np, lap_boundary_ids_np
        )
    fd = (f_plus - f_minus) / (2.0 * eps)
    delta_L = f_plus - f_minus

    denom = max(abs(analytic), 1e-30)
    rel_err = abs(analytic - fd) / denom
    mixed_denom = max(abs(analytic), abs(fd), 1e-30)
    rel_err_mixed = abs(analytic - fd) / mixed_denom

    out = {
        "analytic": analytic,
        "fd": fd,
        "f_plus": f_plus,
        "f_minus": f_minus,
        "relative_err": rel_err,
        "relative_err_mixed": rel_err_mixed,
        "eps": eps,
        "g_norm": g_norm,
        "delta_L": float(delta_L),
    }
    print(f"[verify_grad] ||g||_flex={g_norm:.6e}  analytic={analytic:.12e}  fd={fd:.12e}  "
          f"rel_err={rel_err:.3e} (mixed {rel_err_mixed:.3e})  eps={eps:.2e}")
    print(f"[verify_grad] L(x+εθ)={f_plus:.12e}  L(x−εθ)={f_minus:.12e}  ΔL={delta_L:.12e}")
    if abs(fd) < 1e-30 * max(abs(analytic), 1.0) and abs(analytic) > atol:
        print(
            "[verify_grad] Hint: fd ~ 0 while analytic is not — try **larger** eps (e.g. 1e-4 … 1e-3)."
        )
    if rel_err_mixed > rtol and abs(analytic - fd) > atol:
        print(
            f"[verify_grad] WARNING: mixed rel_err {rel_err_mixed:.3e} > rtol={rtol} "
            f"(atol={atol:.2e})."
        )
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polyscope", action="store_true")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="number of implicit-Euler frames in each dynamic rollout (Unified Fig.10 default: 20)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=2e-2,
        help="dynamic time step (Unified Fig.10 default: 0.02)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="opt",
        help="Output directory for optimization histories and snapshots.",
    )
    parser.add_argument("--remesh-step-scale-threshold", type=float, default=0.25,
                        help="Remesh when the accepted step scale is <= this threshold.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="step size (2x the previous Unified/PolyFEM default 5e-3)")
    parser.add_argument(
        "--line-search",
        action="store_true",
        help="Use legacy loss-decrease search instead of default feasibility Backtracking.",
    )
    parser.add_argument(
        "--trust-region",
        action="store_true",
        help="Backtrack the normalized rest-shape step to preserve the current tet-quality threshold.",
    )
    parser.add_argument(
        "--skip-vtu",
        action="store_true",
        help="Skip per-iteration VTU export to reduce runtime and disk I/O.",
    )
    parser.add_argument(
        "--eval-init-loss",
        action="store_true",
        help="Evaluate the initial stress+Laplacian loss and exit.",
    )
    parser.add_argument("--stress-power", type=int, default=2)
    parser.add_argument("--eval-init-gradient", action="store_true",
                        help="One complete forward and backward, then exit before any parameter update.")
    parser.add_argument("--legacy-polysolve-schema", action="store_true",
                        help="Use grad_norm/x_delta fields for the installed legacy Python binding.")
    parser.add_argument("--newton-stopping", choices=("gradient", "unified-linf"),
                        default="unified-linf",
                        help="match Unified's pre-line-search Linf movement test")
    parser.add_argument("--polyfem-backend", choices=("ren-bowen", "pinned"), default="ren-bowen",
                        help="Fig10 core backend; pinned is retained for historical reproduction")
    parser.add_argument("--quality-threshold", type=float, default=3e-3,
                        help="remesh if min tet quality < threshold (Unified default 3e-3)")
    parser.add_argument("--boundary-quality-threshold", type=float, default=-1.0,
                        help="boundary remesh threshold; <0 disables (GIPC default -1)")
    parser.add_argument("--max-remesh", type=int, default=100,
                        help="maximum remesh rounds")
    parser.add_argument("--remesh-retries", type=int, default=3,
                        help="maximum remesh attempts when quality is still below threshold")
    parser.add_argument(
        "--surface-remesh-method",
        type=str,
        default="pymeshlab",
        choices=["pymeshlab"],
        help="boundary remesh method before tetrahedralization",
    )
    parser.add_argument(
        "--surface-target-scale",
        type=float,
        default=1.0,
        help="target edge length scale for pymeshlab remesh",
    )
    parser.add_argument(
        "--surface-remesh-iters",
        type=int,
        default=10,
        help="iterations for pymeshlab isotropic remesh",
    )
    parser.add_argument(
        "--ftetwild-opts",
        type=str,
        default=None,
        help='extra CLI flags for fTetWild (e.g. \'-c 1000000\'), split with shlex',
    )
    parser.add_argument(
        "--ftetwild-eps",
        type=float,
        default=1e-3,
        help="absolute fTetWild envelope tolerance (Unified Fig.10 default: 1e-3)",
    )
    parser.add_argument(
        "--verify-grad",
        action="store_true",
        help="Run verify_grad (autograd vs central FD) on initial mesh then exit",
    )
    parser.add_argument(
        "--verify-eps",
        type=float,
        default=5e-7,
        help="FD step along unit θ for --verify-grad (try 1e-5…1e-3)",
    )
    parser.add_argument(
        "--laplacian-weight",
        type=float,
        default=1.0,
        help="Weight for boundary Laplacian smoothing on rest shape "
             "(Unified Fig.10 effective default: 1.0)",
    )
    parser.add_argument(
        "--newton-threshold",
        type=float,
        default=1e-3,
        help="GIPC Newton_solver_threshold used for the Linf stopping criterion",
    )
    parser.add_argument(
        "--newton-dt",
        type=float,
        default=0.02,
        help="time step used to map the GIPC Newton threshold to PolyFEM movement tolerance",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=32,
        help="maximum PolyFEM worker threads for the dynamic solve and adjoint",
    )
    parser.add_argument("--stress-weight", type=float, default=1.0,
                        help="Weight of the volume-integrated Frobenius stress power.")
    args = parser.parse_args()
    if not np.isfinite(args.stress_weight) or args.stress_weight < 0:
        parser.error("--stress-weight must be finite and >= 0")
    if not 0 <= args.remesh_step_scale_threshold <= 1:
        parser.error("--remesh-step-scale-threshold must be between 0 and 1")
    global _legacy_polysolve_schema
    _legacy_polysolve_schema = args.legacy_polysolve_schema
    global _newton_stopping
    _newton_stopping = args.newton_stopping
    if args.polyfem_backend == "ren-bowen" and _legacy_polysolve_schema:
        parser.error("Ren-bowen uses the modern schema; omit --legacy-polysolve-schema")
    if args.polyfem_backend == "pinned" and not _legacy_polysolve_schema:
        parser.error("Pinned requires --legacy-polysolve-schema")
    print(f"PolyFEM Python package: {pf.__file__}", flush=True)
    print("PolyFEM build info: " + json.dumps(getattr(pf, "build_info", {})), flush=True)
    global _single_pass_profile
    _single_pass_profile = args.eval_init_gradient
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if args.dt <= 0:
        parser.error("--dt must be > 0")
    if args.newton_threshold <= 0:
        parser.error("--newton-threshold must be > 0")
    if args.newton_dt <= 0:
        parser.error("--newton-dt must be > 0")
    if args.max_threads < 1:
        parser.error("--max-threads must be >= 1")
    ftetwild_eps = None if args.ftetwild_eps is not None and args.ftetwild_eps < 0 else args.ftetwild_eps

    # ── PolyFEM solver ──
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "run.json")) as f:
        config = json.load(f)
    config["root_path"] = os.path.join(root, "run.json")
    if args.eval_init_loss or args.eval_init_gradient:
        config.setdefault("output", {})["directory"] = os.path.abspath(args.output_dir)
    # Match Unified Fig.10: every objective evaluation is a 20-frame dynamic
    # rollout with implicit Euler, not a static equilibrium solve.
    config["time"] = {
        "t0": 0.0,
        "dt": float(args.dt),
        "time_steps": int(args.frames),
        "integrator": "ImplicitEuler",
        "quasistatic": False,
    }
    config.setdefault("output", {}).setdefault("advanced", {})[
        "save_time_sequence"
    ] = False
    # The native State::init() recreates the logger from the JSON settings
    # during set_settings().  Set this before constructing the solver so the
    # trace-level forward/backward timers are not reset to the old "info"
    # level from run.json.
    config.setdefault("output", {}).setdefault("log", {})["level"] = "trace"
    configure_material_and_solver(config, args.max_threads)

    # Enable trace timers so the forward and adjoint phase timings are
    # available in the comparison log.  The numeric API uses 0 for trace.
    log_level = 0
    solver = init_solver_with_gipc_newton(
        config, log_level, args.newton_threshold, args.newton_dt
    )
    print_newton_stopping(solver, args.newton_threshold, args.newton_dt)

    mesh = solver.mesh()
    v0 = mesh.vertices().copy()
    soft_n = soft_vertex_count_from_config(config, root)
    fixed_mask = gipc_case13_fixed_mask(v0, soft_n)
    print(
        f"Soft middle verts: {soft_n}  excluded/fixed verts: {int(fixed_mask.sum())}"
    )

    if args.verify_grad:
        verify_grad(
            solver,
            v0,
            fixed_mask,
            soft_n,
            args.stress_power,
            eps=args.verify_eps,
            expected_frames=args.frames,
            laplacian_weight=args.laplacian_weight,
            stress_weight=args.stress_weight,
        )
        print("verify_grad finished.")
        return

    if args.eval_init_loss or args.eval_init_gradient:
        tets = mesh.elements()
        surf_faces_np = extract_surface_faces(tets).astype(np.int64)
        verts_tensor = torch.tensor(v0, requires_grad=args.eval_init_gradient)
        print("SINGLE_PASS_FORWARD_BEGIN", flush=True)
        forward_started = time.perf_counter()
        solutions = Simulate.apply(
            solver,
            verts_tensor,
            torch.as_tensor(gipc_case13_design_mask(v0, fixed_mask, extract_surface_faces(tets)), dtype=torch.bool),
            args.frames,
        )
        forward_seconds = time.perf_counter() - forward_started
        print("SINGLE_PASS_FORWARD_END", flush=True)
        cached_frames = np.asarray(solver.get_solutions()).shape[1] - 1
        if cached_frames != args.frames:
            raise RuntimeError(f"Incomplete rollout: {cached_frames}/{args.frames} frames")
        tet_mask = torch.as_tensor(
            gipc_case13_loss_tet_mask(v0, tets, soft_n),
            dtype=torch.bool,
        )
        deformed_tensor = verts_tensor + solutions.reshape_as(verts_tensor)
        loss_stress = stable_nh_stress_norm_loss(
            deformed_tensor,
            verts_tensor,
            torch.as_tensor(tets, dtype=torch.long),
            args.stress_power,
            tet_mask,
            stress_weight=args.stress_weight,
        )
        lap_faces_np, lap_boundary_ids_np = gipc_case13_laplacian_surface(
            v0, surf_faces_np, soft_n
        )
        lap_raw = verts_tensor.new_zeros(())
        if args.laplacian_weight > 0.0 and lap_faces_np.size > 0:
            lap_raw = laplacian_loss_torch(
                verts_tensor,
                torch.as_tensor(lap_faces_np, dtype=torch.long),
                boundary_vertex_ids=torch.as_tensor(
                    lap_boundary_ids_np, dtype=torch.long
                ),
            )
        loss = loss_stress + args.laplacian_weight * lap_raw
        if not torch.isfinite(loss):
            raise RuntimeError("Initial loss contains NaN or Inf")
        print(
            f"INIT_LOSS {float(loss):.17g} "
            f"stress={float(loss_stress):.17g} "
            f"laplacian={float(args.laplacian_weight * lap_raw):.17g}"
        )
        if args.eval_init_gradient:
            print("SINGLE_PASS_BACKWARD_BEGIN", flush=True)
            backward_started = time.perf_counter()
            loss.backward()
            backward_seconds = time.perf_counter() - backward_started
            if not torch.isfinite(verts_tensor.grad).all():
                raise RuntimeError("Initial gradient contains NaN or Inf")
            print("SINGLE_PASS_BACKWARD_END", flush=True)
            print("SINGLE_PASS_RESULT " + json.dumps({
                "loss": float(loss.detach()),
                "gradient_norm": float(verts_tensor.grad.norm()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "completed_frames": int(cached_frames),
                "forward_calls": 1, "backward_calls": 1,
                "parameter_updates": 0,
            }), flush=True)
        return

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    # All remesh intermediates (OBJ/MSH/logs from tet_remesh) live here, not in opt/ root.
    remesh_out_dir = os.path.join(out_dir, "remesh")
    os.makedirs(remesh_out_dir, exist_ok=True)

    # ── Polyscope (optional) ──
    ps_mesh = None
    if args.polyscope:
        import polyscope as ps
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("shadow_only")

    # ── Optimisation loop ──
    loss_history = []
    stress_loss_history = []
    laplacian_loss_history = []
    grad_norm_history = []
    t_start = time.time()
    lr = args.lr
    stress_power = args.stress_power
    laplacian_weight = args.laplacian_weight
    remesh_iter = 0
    accepted_forward = None
    total_iterations = 0

    print(f"\n{'='*60}")
    print("  Starting shape optimisation (stress-norm minimisation)")
    print(f"  lr={lr:.2e}  stress_power={stress_power}  stress_weight={args.stress_weight:.6g}  laplacian_weight={laplacian_weight:.3e}")
    print(
        f"  remesh threshold={args.quality_threshold:.3e}  "
        f"boundary_threshold={args.boundary_quality_threshold:.3e}  "
        f"max_remesh={args.max_remesh}  remesh_retries={args.remesh_retries}"
    )
    print(f"  dynamic rollout: frames={args.frames}  dt={args.dt:.6e}  integrator=ImplicitEuler")
    print(f"  PolyFEM threads: {args.max_threads}")
    print(f"  volume remesh: fTetWild only  ftetwild_eps={ftetwild_eps}")
    print(f"{'='*60}\n")
    print(f"  remesh artifacts directory: {remesh_out_dir}\n")

    while True:
        # ``--n-iters`` is the total optimization budget.  A remesh starts a
        # new local topology loop, but must not silently grant another full
        # budget of iterations.
        if total_iterations >= args.n_iters:
            break
        mesh = solver.mesh()
        v0 = mesh.vertices().copy()
        tets = mesh.elements()
        n_verts = mesh.n_vertices()
        n_elems = mesh.n_elements()
        base_boundary_faces = _boundary_facets(tets).shape[0]
        print(f"\n[remesh {remesh_iter}] Mesh: {n_verts} vertices, {n_elems} tets")

        soft_n = soft_vertex_count_from_config(config, root)
        fixed_mask = gipc_case13_fixed_mask(v0, soft_n)
        bf0 = gipc_case13_body_force_rhs_mask(v0)
        design0 = gipc_case13_design_mask(v0, fixed_mask, extract_surface_faces(tets))
        print(f"Excluded/fixed design verts: {fixed_mask.sum()} (soft_n={soft_n})")
        print(f"Body-force arm-normal cut (physics rhs): {bf0.sum()}")
        print(f"Surface design 3<=y<=5 inward of arm cuts (per topology): {design0.sum()}")
        print(f"Boundary faces: {base_boundary_faces}")

        surf_faces_np = extract_surface_faces(tets).astype(np.int64)

        # Freeze selectors for this topology, including all line-search trials.
        loss_tet_mask = gipc_case13_loss_tet_mask(v0, tets, soft_n)
        lap_faces_np, lap_boundary_ids_np = gipc_case13_laplacian_surface(
            v0, surf_faces_np, soft_n
        )
        current_verts = v0.copy()
        remesh_requested = False
        remesh_failed_this_round = False

        if args.polyscope:
            import polyscope as ps
            surf_faces = extract_surface_faces(tets)
            ps_mesh = ps.register_surface_mesh(f"case13_r{remesh_iter}", v0, surf_faces)
            ps_mesh.set_smooth_shade(True)
            region_q = np.zeros(n_verts)
            region_q[fixed_mask] = 1.0
            ps_mesh.add_scalar_quantity("fixed", region_q, cmap="blues", enabled=False)
        else:
            ps_mesh = None

        for it in range(args.n_iters - total_iterations):
            verts_before = current_verts.copy()
            # Unified fixes the selected parameter IDs when a topology is
            # built. Reuse this per-topology mask for the final rest update;
            # every state DOF contributes to the dynamic-adjoint RHS.
            design_mask = design0
            t0 = time.time()

            # Forward solve + loss + backward
            verts_tensor = torch.tensor(current_verts, requires_grad=True)
            solutions = Simulate.apply(
                solver,
                verts_tensor,
                torch.as_tensor(design_mask, dtype=torch.bool),
                args.frames,
                accepted_forward,
            )
            print(f"        reference_forward_reused={int(accepted_forward is not None and accepted_forward.reused)}")
            accepted_forward = None
            tet_mask = torch.as_tensor(
                loss_tet_mask,
                dtype=torch.bool,
            )
            deformed_tensor = verts_tensor + solutions.reshape_as(verts_tensor)
            loss_stress = stable_nh_stress_norm_loss(
                deformed_tensor,
                verts_tensor,
                torch.as_tensor(tets, dtype=torch.long),
                stress_power,
                tet_mask,
                stress_weight=args.stress_weight,
            )
            if laplacian_weight > 0.0 and lap_faces_np.size > 0:
                lap_faces_t = torch.as_tensor(lap_faces_np, dtype=torch.long)
                lap_boundary_ids_t = torch.as_tensor(
                    lap_boundary_ids_np, dtype=torch.long
                )
                lap_raw = laplacian_loss_torch(
                    verts_tensor,
                    lap_faces_t,
                    boundary_vertex_ids=lap_boundary_ids_t,
                )
                laplacian_loss_term_val = float(laplacian_weight * lap_raw.detach().cpu())
                loss = loss_stress + laplacian_weight * lap_raw
            else:
                laplacian_loss_term_val = 0.0
                loss = loss_stress
            loss.backward()

            loss_val = float(loss)
            grad = verts_tensor.grad.numpy().copy()

            # Zero out gradient outside GIPC rest-diff set: soft ∩ 3<=y<=5 ∩ |z|<=1
            grad[~design_mask] = 0.0
            grad_norm = np.linalg.norm(grad)

            # Match scene12_torch.gradient_descent_rest_update: scale so max|g|<=1 before the step
            grad_step = grad.copy()
            g_abs_max = float(np.max(np.abs(grad_step))) if grad_step.size else 0.0
            if g_abs_max > 1.0:
                grad_step /= g_abs_max

            dt = time.time() - t0
            loss_history.append(loss_val)
            stress_loss_history.append(float(loss_stress.detach().cpu()))
            laplacian_loss_history.append(laplacian_loss_term_val)
            grad_norm_history.append(float(grad_norm))
            total_iterations += 1
            # Persist after every iteration so an interrupted long run keeps
            # the completed loss/gradient history.
            np.savetxt(os.path.join(out_dir, "loss_history.txt"), loss_history)
            np.savetxt(os.path.join(out_dir, "stress_loss_history.txt"), stress_loss_history)
            np.savetxt(os.path.join(out_dir, "laplacian_loss_history.txt"), laplacian_loss_history)
            np.savetxt(os.path.join(out_dir, "grad_norm_history.txt"), grad_norm_history)

            scale_note = f"  |g|_max={g_abs_max:.4e}" if g_abs_max > 1.0 else ""
            lap_note = (
                f"  lap={laplacian_loss_term_val:.6e} (w={laplacian_weight:.3e})"
                if laplacian_weight > 0.0
                else ""
            )
            print(
                f"Iter {total_iterations - 1:3d} | loss {loss_val:.6e}{lap_note} | |grad| {grad_norm:.4e}{scale_note} "
                f"| {dt:.2f}s | total {time.time()-t_start:.1f}s"
            )

            # Export VTU unless a long headless comparison run disables it.
            iter_dir = os.path.join(out_dir, f"remesh_{remesh_iter}_iter_{it}")
            os.makedirs(iter_dir, exist_ok=True)
            if not args.skip_vtu:
                cached_solution = np.asarray(solver.get_solutions(), dtype=np.float64)
                final_solution = (
                    cached_solution[:, -1:]
                    if cached_solution.ndim == 2
                    else cached_solution.reshape(-1, 1)
                )
                solver.export_vtu(
                    os.path.join(iter_dir, "solution.vtu"),
                    final_solution,
                    time=float(args.frames * args.dt),
                    dt=float(args.dt),
                )

            step_scale = 1.0
            if args.line_search:
                # Optional legacy loss-decrease mode.
                step_ok = False
                t = lr
                for _ in range(20):
                    candidate = current_verts - t * grad_step
                    try:
                        solver.mesh().set_vertices(candidate)
                        solver.set_cache_level(pf.CacheLevel.Derivatives)
                        new_stress = eval_stress_norm_loss(
                            solver, candidate, stress_power, soft_n, tet_mask=loss_tet_mask,
                            expected_frames=args.frames,
                            stress_weight=args.stress_weight,
                        )
                        new_lap_faces, new_lap_boundary_ids = lap_faces_np, lap_boundary_ids_np
                        if laplacian_weight > 0.0 and new_lap_faces.size > 0:
                            new_lap = laplacian_weight * eval_laplacian_loss_np(
                                candidate, new_lap_faces, new_lap_boundary_ids
                            )
                        else:
                            new_lap = 0.0
                        new_loss = new_stress + new_lap
                        if np.isfinite(new_loss) and new_loss < loss_val:
                            current_verts = candidate
                            step_ok = True
                            print(
                                f"        step t={t:.2e}  new_loss={new_loss:.6e} "
                                f"(stress={new_stress:.6e}  lap_term={new_lap:.6e})"
                            )
                            break
                    except Exception:
                        pass
                    t *= 0.5
                step_scale = t / lr if step_ok else 0.0
                if not step_ok:
                    solver.mesh().set_vertices(current_verts)
                    print("        line search failed, keeping current vertices")
            elif args.trust_region:
                # Match Unified Fig.10's trust-region guard around the
                # normalized GIPC step: backtrack the rest-shape update until
                # the current topology remains above the remesh threshold.
                # This avoids spending the optimization budget on a topology
                # change caused solely by an unnecessarily large step.
                step_scale = 1.0
                step_accepted = False
                tets_step = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
                soft_tets_step = tets_step[
                    np.all(tets_step < int(soft_n), axis=1)
                ]
                for _ in range(9):
                    candidate = current_verts - (step_scale * lr) * grad_step
                    candidate_quality = tet_mesh_quality_min(
                        candidate[:soft_n], soft_tets_step
                    )
                    if np.isfinite(candidate_quality) and (
                        candidate_quality >= args.quality_threshold
                    ):
                        current_verts = candidate
                        step_accepted = True
                        break
                    step_scale *= 0.5
                if not step_accepted:
                    step_scale = 0.0
                    current_verts = current_verts.copy()
                print(
                    f"        trust-region step_scale={step_scale:.6e}"
                )
                solver.mesh().set_vertices(current_verts)
            else:
                def evaluate_candidate(candidate):
                    stress = eval_stress_norm_loss(
                        solver, candidate, stress_power, soft_n, strict_solve=True,
                        tet_mask=loss_tet_mask, expected_frames=args.frames,
                        stress_weight=args.stress_weight,
                    )
                    faces, boundary_ids = lap_faces_np, lap_boundary_ids_np
                    lap = (laplacian_weight * eval_laplacian_loss_np(candidate, faces, boundary_ids)
                           if laplacian_weight > 0 and faces.size > 0 else 0.0)
                    return stress + lap

                current_verts, step_scale, trial_loss = feasible_backtracking(
                    current_verts, -lr * grad_step,
                    np.asarray(tets, dtype=np.int64).reshape(-1, 4), evaluate_candidate,
                )
                accepted_forward = AcceptedForward(solver, current_verts, args.frames)
                print(f"        feasibility Backtracking step_scale={step_scale:.6e} loss={trial_loss:.6e}")

            surf_faces_step = extract_surface_faces(tets).astype(np.int32)
            surf_rest_path = os.path.join(iter_dir, "surface_rest.obj")
            igl.write_triangle_mesh(surf_rest_path, current_verts, surf_faces_step)

            delta = current_verts - verts_before
            flex = design_mask
            if np.any(flex):
                disp_norms = np.linalg.norm(delta[flex], axis=1)
                rest_mean = float(np.mean(disp_norms))
                rest_max = float(np.max(disp_norms))
            else:
                rest_mean = rest_max = 0.0
            print(
                f"        rest shape update |Δx|: mean={rest_mean:.6e}  max={rest_max:.6e} "
                f"(flexible vertices)"
            )

            soft_n = soft_vertex_count_from_config(config, root)
            soft_verts = np.asarray(current_verts[:soft_n], dtype=np.float64)
            tets_np = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
            soft_tets = tets_np[np.all(tets_np < soft_n, axis=1)]
            if soft_tets.size == 0:
                print("        no soft-body tets found; skip remesh check")
                continue

            quality_min = tet_mesh_quality_min(soft_verts, soft_tets)
            print(f"        min tet quality (soft) = {quality_min:.6e}")
            if needs_remesh(quality_min, args.quality_threshold, step_scale,
                            args.remesh_step_scale_threshold) and not remesh_failed_this_round:
                if remesh_iter >= args.max_remesh:
                    print("        remesh requested, but max remesh reached")
                    break
                bq = boundary_triangle_quality_stats(soft_verts, soft_tets)
                # Unified Fig.10 uses the volume-only fTetWild path.  The
                # legacy surface-remesh switches remain accepted for CLI
                # compatibility, but are deliberately not used here.
                do_boundary_remesh = False
                print("        Mesh quality or small step triggered remesh of soft middle...")
                print(
                    "        boundary quality: "
                    f"min={bq['min']:.6e}, p01={bq['p01']:.6e}, "
                    f"p05={bq['p05']:.6e}, median={bq['median']:.6e}"
                )
                print(
                    "        remesh mode: "
                    + ("boundary+volume" if do_boundary_remesh else "volume_only")
                )

                surf_before_path = os.path.join(
                    remesh_out_dir, f"surface_before_remesh_{remesh_iter}.obj"
                )
                surf_faces_before = _boundary_facets(soft_tets)
                igl.write_triangle_mesh(surf_before_path, soft_verts, surf_faces_before)
                print(f"        saved surface before remesh: {surf_before_path}")

                old_mesh_file = config["geometry"][0]["mesh"]
                accepted = False
                last_info = {}

                phases = [("initial", False)]

                for phase_name, phase_do_boundary in phases:
                    if accepted:
                        break

                    for attempt in range(1, args.remesh_retries + 1):
                        scale = args.surface_target_scale * (0.9 ** (attempt - 1))
                        attempt_tag = (
                            f"{attempt}/{args.remesh_retries}"
                            if phase_name == "initial"
                            else f"{attempt}/{args.remesh_retries} [boundary+volume fallback]"
                        )
                        print(
                            f"        remesh attempt {attempt_tag} "
                            f"(surface_target_scale={scale:.4f})"
                        )
                        ok, before_obj, after_msh, remesh_log, remesh_info = run_remesh(
                            remesh_out_dir,
                            remesh_iter,
                            attempt,
                            soft_verts,
                            soft_tets,
                            phase_do_boundary,
                            args.surface_remesh_method,
                            scale,
                            args.surface_remesh_iters,
                            volume_backend="ftetwild",
                            ftetwild_opts=args.ftetwild_opts,
                            ftetwild_epsilon=ftetwild_eps,
                        )
                        last_info = remesh_info
                        if not ok:
                            print("        Remesh attempt failed.")
                            if before_obj:
                                print(f"        obj={before_obj}")
                            print(f"        log={remesh_log}")
                            print(f"        info={remesh_info}")
                            continue

                        post_q = tet_quality_stats_from_mesh(after_msh)
                        if post_q is None:
                            print("        Remesh output has no tetra, retrying...")
                            continue
                        remesh_info.update({
                            "post_min_quality": post_q["min"],
                            "post_p01_quality": post_q["p01"],
                            "post_p05_quality": post_q["p05"],
                            "post_median_quality": post_q["median"],
                        })
                        print(f"        Remesh stats: {remesh_info}")

                        print(f"        Remesh finished: {after_msh}")
                        config["geometry"][0]["mesh"] = os.path.relpath(
                            after_msh, root
                        ).replace("\\", "/")
                        try:
                            new_solver = init_solver_with_gipc_newton(
                                config,
                                log_level,
                                args.newton_threshold,
                                args.newton_dt,
                            )
                            print_newton_stopping(
                                new_solver,
                                args.newton_threshold,
                                args.newton_dt,
                                prefix="        ",
                            )
                            new_mesh = new_solver.mesh()
                            new_tets = new_mesh.elements()
                            new_soft_n = soft_vertex_count_from_config(config, root)
                            new_fixed = int(
                                gipc_case13_fixed_mask(
                                    new_mesh.vertices(), new_soft_n
                                ).sum()
                            )
                            new_boundary_faces = _boundary_facets(new_tets).shape[0]
                            print(
                                "        Remesh compare: "
                                f"fixed {fixed_mask.sum()} -> {new_fixed}, "
                                f"boundary_faces {base_boundary_faces} -> {new_boundary_faces}, "
                                f"tets {n_elems} -> {new_mesh.n_elements()}"
                            )

                            new_solver.set_cache_level(pf.CacheLevel.Derivatives)
                            new_solver.solve()
                            # Only the new topology's validated cache may be reused.
                            new_accepted_forward = AcceptedForward(new_solver, new_mesh.vertices(), args.frames)
                            solver = new_solver
                            accepted_forward = new_accepted_forward

                            surf_after_path = os.path.join(
                                remesh_out_dir, f"surface_after_remesh_{remesh_iter}.obj"
                            )
                            new_surf_faces = _boundary_facets(new_tets)
                            igl.write_triangle_mesh(
                                surf_after_path,
                                new_mesh.vertices(),
                                new_surf_faces,
                            )
                            print(f"        saved surface after remesh: {surf_after_path}")

                            remesh_iter += 1
                            accepted = True
                        except Exception as e:
                            config["geometry"][0]["mesh"] = old_mesh_file
                            print(f"        Remesh rejected (solver check failed): {e}")
                        if accepted:
                            break

                if not accepted:
                    raise RuntimeError(
                        "Remesh failed to meet quality threshold after "
                        f"all Unified-style fTetWild attempts. "
                        f"last_info={last_info}"
                    )

                remesh_requested = True
                break

            # Update Polyscope
            if ps_mesh is not None:
                ps_mesh.update_vertex_positions(current_verts)

        if not remesh_requested:
            break

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  Optimisation finished")
    print(f"{'='*60}")
    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")
    if loss_history:
        print(f"  Initial loss: {loss_history[0]:.6e}")
        print(f"  Final   loss: {loss_history[-1]:.6e}")

    np.savetxt(os.path.join(out_dir, "loss_history.txt"), loss_history)
    np.savetxt(os.path.join(out_dir, "stress_loss_history.txt"), stress_loss_history)
    np.savetxt(os.path.join(out_dir, "laplacian_loss_history.txt"), laplacian_loss_history)
    np.savetxt(os.path.join(out_dir, "grad_norm_history.txt"), grad_norm_history)
    np.save(os.path.join(out_dir, "final_rest_vertices.npy"), current_verts)
    np.save(os.path.join(out_dir, "final_tets.npy"), np.asarray(solver.mesh().elements(), dtype=np.int64))

    # ── Plot loss curve ──
    if len(loss_history) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(loss_history, "o-", markersize=3, label="total")
        axes[0, 0].plot(stress_loss_history, "o-", markersize=3, label="stress")
        axes[0, 0].plot(laplacian_loss_history, "o-", markersize=3, label="laplacian")
        axes[0, 0].set_xlabel("Iteration")
        ylbl = "Total loss (stress + Laplacian)" if args.laplacian_weight > 0.0 else "Stress-norm loss"
        axes[0, 0].set_ylabel(ylbl)
        axes[0, 0].set_title("Loss curve")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].semilogy(loss_history, "o-", markersize=3, label="total")
        axes[0, 1].semilogy(np.maximum(stress_loss_history, 1e-30), "o-", markersize=3, label="stress")
        axes[0, 1].semilogy(np.maximum(laplacian_loss_history, 1e-30), "o-", markersize=3, label="laplacian")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel(f"{ylbl} (log)")
        axes[0, 1].set_title("Loss curve (log scale)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(grad_norm_history, "o-", markersize=3, color="tab:orange")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("||grad||")
        axes[1, 0].set_title("Gradient norm")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].semilogy(np.maximum(grad_norm_history, 1e-30), "o-", markersize=3, color="tab:orange")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("||grad|| (log)")
        axes[1, 1].set_title("Gradient norm (log scale)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "loss_and_grad_curve.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Loss curve saved to: {fig_path}")

    if args.polyscope:
        import polyscope as ps
        ps.show()


if __name__ == "__main__":
    main()
