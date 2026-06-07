#!/usr/bin/env python3
"""
Reimplement the old polyfem `adjust_weight_period` / `adjustment_coeff` logic
from `ShapeProblem.cpp` (commit 0805b4dc5), plus the legacy `remesh_period` /
`remesh_tolerance` hook backed by `wmtk_interior_tetopt_bin`:

    every `adjust_weight_period` iterations:
        smoothing_w = smoothing_w * adjustment_coeff
                      * (barrier_val + target_val + volume_val) / n_boundary_nodes

    every `remesh_period` cumulative iters, or whenever the worst sphere
    scaled-Jacobian drops below `remesh_tolerance`, call the wmtk binary on
    the sphere-only rest mesh to retet the interior (boundary held fixed).

Modern polyfem dropped the `ShapeProblem` post-step hook, so we drive both
behaviours from Python: run polyfem in chunks of `PERIOD` iters, parse the
`print_energy` values from the log, patch `opt-shape.json` between chunks,
and replace `run-shape.json`'s deformable-body mesh with a sphere-only
(remeshed if needed) extract of the latest `opt_state_0_iter_*.msh`.

Notes vs the original C++ logic:
  - L-BFGS history is reset at each chunk boundary (the old hook kept it).
  - We don't have a separate "barrier" functional in the new opt setup, so the
    `barrier_val` term is omitted. Add it here if you wire one up.
  - polyfem saves the *combined* plank+sphere state mesh with all body tags
    collapsed to 1; we recover the sphere by picking the connected component
    with the smaller bbox and feed only that back into geometry[1].
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np

# --- configuration ---------------------------------------------------------
HERE       = Path(__file__).resolve().parent
OPT_JSON   = HERE / "opt-shape.json"
RUN_JSON   = HERE / "run-shape.json"
POLYFEM    = "/home/zizhouhuang/workspaces/repro-diff-ipc/polyfem/build/PolyFEM_bin"
WMTK_BIN   = "/home/zizhouhuang/workspaces/repro-diff-ipc/wildmeshing-toolkit/build/app/interior_tet_opt/wmtk_interior_tetopt_bin"

PERIOD              = 20     # L-BFGS iters between adapt/remesh/probe events
ADJUSTMENT_COEFF    = 0.5    # adjustment_coeff (unused while REBALANCE=False)
TOTAL_CHUNKS        = 10     # outer iterations (PERIOD each) → 200 total L-BFGS iters
REBALANCE_SMOOTHING = False  # legacy boundary_smoothing weight rebalance disabled

REMESH_TOLERANCE    = 1e-1   # min scaled-jacobian below this triggers remesh
REMESH_PERIOD_ITERS = 10     # force remesh every N cumulative L-BFGS iters
WMTK_MAX_ITERS      = 30     # -i passed to wmtk

# Map opt-shape.json functional "type" -> human label used in probe output
FUNCTIONAL_PROBE_TYPES = {
    "power":            "stress",
    "soft_constraint":  "volume",
    "boundary_smoothing": "boundary_smoothing",
}

ENERGY_KEYS_BASE    = ("stress", "volume")  # functionals that contribute to the base term
DEFORMABLE_GEOM_IX  = 1                     # geometry[] index of the body being shape-opt'd

REMESH_DIR          = HERE / "remesh_work"  # sphere-only / remeshed meshes saved here
# ---------------------------------------------------------------------------


def load(p): return json.loads(p.read_text())
def save(p, o): p.write_text(json.dumps(o, indent=2))


def _fmt_cell(v) -> str:
    """Compact, fixed-width number formatting for the metrics table."""
    if v is None:
        return "."
    if isinstance(v, bool):
        return "Y" if v else "."
    if isinstance(v, (int, np.integer)):
        return f"{int(v):d}"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(x):
        return "nan"
    if x == 0:
        return "0"
    return f"{x:.3e}"


def parse_energies(log: str) -> dict:
    """Return the last value seen for each [<keyword>] <number> log line."""
    out = {}
    for m in re.finditer(r"\[(\w[\w\-]*)\]\s+(-?\d[\d\.eE+-]*)", log):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def count_msh_nodes(mesh_path: Path) -> int:
    """Count vertices in any meshio-readable .msh file."""
    return len(meshio.read(mesh_path).points)


def latest_iter_mesh(out_dir: Path, newer_than: float = 0.0):
    """
    Return the opt_state_0_iter_*.msh with the largest iter number that was
    written after `newer_than` (a unix mtime). Prevents picking up stale meshes
    from previous driver runs that still sit in out_dir with iter numbers
    higher than what the current run produced.
    """
    cands = [p for p in out_dir.glob("opt_state_0_iter_*.msh")
             if p.stat().st_mtime > newer_than]
    cands.sort(key=lambda p: int(re.search(r"iter_(\d+)", p.name).group(1)))
    return cands[-1] if cands else None


def set_max_iters(opt: dict, n: int):
    opt.setdefault("solver", {}).setdefault("nonlinear", {})["max_iterations"] = n


def set_smoothing_weight(opt: dict, new_weight: float) -> float | None:
    for f in opt.get("functionals", []):
        if f.get("type") == "boundary_smoothing":
            old = f["weight"]
            f["weight"] = new_weight
            return old
    return None


# --- mesh extraction / quality / remesh ------------------------------------

def _connected_components(n_pts: int, tets: np.ndarray) -> np.ndarray:
    """Union-find over tet connectivity. Returns root id per vertex."""
    parent = np.arange(n_pts)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in tets:
        a = find(t[0])
        for k in range(1, 4):
            b = find(t[k])
            if a != b:
                parent[b] = a
                a = find(a)
    return np.array([find(i) for i in range(n_pts)])


def extract_sphere(combined_msh: Path, out_path: Path):
    """
    polyfem saves the combined plank+sphere rest mesh with all body tags = 1.
    Pick the connected component with the smaller bbox (the sphere) and write
    it as Gmsh 2.2 (required by wmtk and accepted by polyfem).
    """
    m = meshio.read(combined_msh)
    pts = np.asarray(m.points)
    tets = np.concatenate([c.data for c in m.cells if c.type == "tetra"])
    roots = _connected_components(len(pts), tets)
    labels = np.unique(roots)
    if len(labels) < 2:
        raise RuntimeError(f"{combined_msh} has only {len(labels)} component(s); expected ≥2")
    # smallest-bbox component = sphere
    diags = sorted(
        ((np.linalg.norm(np.ptp(pts[roots == r], axis=0)), r) for r in labels),
        key=lambda x: x[0],
    )
    sphere_root = diags[0][1]
    mask = roots == sphere_root
    old_to_new = -np.ones(len(pts), dtype=int)
    old_to_new[mask] = np.arange(mask.sum())
    keep = np.array([mask[t].all() for t in tets])
    new_tets = old_to_new[tets[keep]]
    out = meshio.Mesh(points=pts[mask], cells=[("tetra", new_tets)])
    meshio.write(out_path, out, file_format="gmsh22")
    return out


def min_scaled_jacobian(mesh: meshio.Mesh) -> float:
    """
    Worst (min) unsigned scaled Jacobian over all tets:
      |sj(tet)| = min over 4 corners of  sqrt(2) * |det([e1,e2,e3])| / (|e1||e2||e3|)
    Equilateral tet → 1; degenerate (sliver/flat) → 0. We take abs because polyfem
    may save tets with the opposite orientation convention; actual inversions are
    caught separately by the `min_jacobian` stopping condition in opt-shape.json.
    """
    pts = np.asarray(mesh.points)
    tets = np.concatenate([c.data for c in mesh.cells if c.type == "tetra"])
    V = pts[tets]  # (N, 4, 3)
    corners = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2)]
    sj = np.full(len(tets), np.inf)
    for c0, a, b, c in corners:
        e1 = V[:, a] - V[:, c0]
        e2 = V[:, b] - V[:, c0]
        e3 = V[:, c] - V[:, c0]
        jac = np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3))
        denom = (np.linalg.norm(e1, axis=1)
                 * np.linalg.norm(e2, axis=1)
                 * np.linalg.norm(e3, axis=1))
        sj = np.minimum(sj, jac / np.maximum(denom, 1e-30))
    return float((sj * np.sqrt(2)).min())


def min_surface_scaled_jacobian(mesh: meshio.Mesh) -> float:
    """
    Worst (min) scaled Jacobian over boundary triangles of a tet mesh.
    Triangle scaled-Jac at a corner v (with adjacent edges u, w):
        sj(v) = (2/sqrt(3)) * (2*area) / (|u| * |w|)
    Equilateral → 1, degenerate → 0. We take the min over all 3 corners
    of every boundary triangle.
    """
    pts = np.asarray(mesh.points)
    tets = np.concatenate([c.data for c in mesh.cells if c.type == "tetra"])
    # Boundary triangles = appear in exactly one tet.
    faces = np.vstack([
        tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
    ])
    sorted_faces = np.sort(faces, axis=1)
    _, inv, counts = np.unique(sorted_faces, axis=0,
                                return_inverse=True, return_counts=True)
    tris = faces[counts[inv] == 1]
    if len(tris) == 0:
        return float("nan")
    a = pts[tris[:, 0]]
    b = pts[tris[:, 1]]
    c = pts[tris[:, 2]]
    e1 = b - a
    e2 = c - a
    e3 = c - b
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)  # = 2 * triangle area
    n1 = np.linalg.norm(e1, axis=1)
    n2 = np.linalg.norm(e2, axis=1)
    n3 = np.linalg.norm(e3, axis=1)
    SQ32 = 2.0 / np.sqrt(3.0)
    sj_a = SQ32 * area2 / np.maximum(n1 * n2, 1e-30)
    sj_b = SQ32 * area2 / np.maximum(n1 * n3, 1e-30)
    sj_c = SQ32 * area2 / np.maximum(n2 * n3, 1e-30)
    return float(np.minimum(np.minimum(sj_a, sj_b), sj_c).min())


def stress_diagnostics(vtu_path: Path, sphere_body_id: int = 1) -> dict:
    """
    Read polyfem's volume VTU and compute max- and L^8-norm of the Cauchy
    stress over sphere nodes (filter by body_ids == sphere_body_id).

    L^8 is a nodal (unweighted) sample-mean of ‖σ‖^8 raised to 1/8 — a
    smooth proxy for max-stress that's cheap to compute without quadrature.
    """
    m = meshio.read(vtu_path)
    pd = m.point_data
    s1 = pd["cauchy_stess_1"]
    s2 = pd["cauchy_stess_2"]
    s3 = pd["cauchy_stess_3"]
    # Frobenius norm of σ per node
    sn = np.sqrt(np.sum(s1**2 + s2**2 + s3**2, axis=1))
    body = pd.get("body_ids")
    if body is not None:
        mask = np.asarray(body, dtype=int).ravel() == sphere_body_id
        sn = sn[mask]
    if len(sn) == 0:
        return {"max_stress": float("nan"), "L8_stress": float("nan")}
    return {
        "max_stress": float(np.max(sn)),
        "L8_stress": float(np.mean(sn ** 8) ** (1.0 / 8.0)),
    }


def parse_lbfgs(log: str) -> dict:
    """
    Pull outer L-BFGS line-search summary out of a polyfem chunk log.

    polysolve's Backtracking shares one `cur_iter` counter between failed
    attempts (element-flip etc., logged as "Failed to take step") and finite-
    energy attempts (logged as "ls it: {cur_iter}"). The step trial at
    cur_iter is `starting_step * step_ratio^cur_iter`. We read the *largest*
    cur_iter seen (`max(ls_it, n_failed)`) and combine with the actual
    starting step + ratio from opt-shape.json so the reported step matches
    what polyfem accepted.

    Returns:
      n_pre:        number of L-BFGS iterations started in this chunk
      n_failed:     total "Failed to take step" warnings
      ls_it:        last `ls it: N` value (None if absent)
      step_size:    accepted step ≈ starting * ratio ** last_cur_iter
      first_grad:   ‖∇f‖ at the start of the first LS in this chunk
    """
    pre = re.findall(
        r"\[L-BFGS\]\[Backtracking\] pre LS iter=(\d+) f=([0-9eE.+-]+)\s*‖∇f‖=([0-9eE.+-]+)",
        log,
    )
    failed = re.findall(r"Failed to take step", log)
    ls_it_vals = [int(x) for x in re.findall(r"ls it:\s*(\d+)", log)]
    # Read live line-search starting step & ratio from opt-shape.json.
    try:
        ls = load(OPT_JSON)["solver"]["nonlinear"]["line_search"]
        starting = float(ls.get("default_init_step_size", 1.0))
        ratio = float(ls.get("step_ratio", 0.5))
    except Exception:
        starting, ratio = 1.0, 0.5
    cur_iter = max([0, *ls_it_vals, len(failed) - 1]) if (ls_it_vals or failed) else 0
    step = starting * (ratio ** cur_iter)
    return {
        "n_pre":     len(pre),
        "n_failed":  len(failed),
        "ls_it":     ls_it_vals[-1] if ls_it_vals else None,
        "step_size": step,
        "first_grad": float(pre[0][2]) if pre else None,
    }


def call_wmtk(in_msh: Path, out_msh: Path):
    """Run wmtk_interior_tetopt_bin in/out. Boundary is held fixed."""
    subprocess.run(
        [WMTK_BIN, "-i", str(WMTK_MAX_ITERS), str(in_msh), str(out_msh)],
        check=True, cwd=HERE,
    )


def probe_term_gradients(probe_log_dir: Path) -> dict:
    """
    Report ‖∇f_i‖ for each top-level functional at the current rest mesh.

    Uses polyfem's `POLYFEM_REPORT_PER_TERM_GRADS` env-var entry point:
    ONE forward solve, then one adjoint solve per child functional. The
    polyfem log prints lines of the form

        [per-term-grad] <label> ||g||=<value>

    which we parse into {label: ‖∇f_i‖}. Strictly faster than the previous
    3-forward implementation (1 forward + N adjoints vs N forward+adjoints),
    and leaves opt-shape.json untouched.
    """
    log_file = probe_log_dir / "probe_all_terms.log"
    env = {**__import__("os").environ, "POLYFEM_REPORT_PER_TERM_GRADS": "1"}
    with log_file.open("w") as fp:
        subprocess.run(
            [POLYFEM, "-j", str(OPT_JSON), "--ns"],
            stdout=fp, stderr=subprocess.STDOUT, cwd=HERE, env=env,
        )
    grads = {}
    for m in re.finditer(
        r"\[per-term-grad\]\s+(\S+)\s+\|\|g\|\|=([0-9eE.+-]+)",
        log_file.read_text(),
    ):
        grads[m.group(1)] = float(m.group(2))
    return grads


# --- main loop -------------------------------------------------------------

def main():
    opt = load(OPT_JSON)
    set_max_iters(opt, PERIOD)
    save(OPT_JSON, opt)

    REMESH_DIR.mkdir(exist_ok=True)
    PROBE_DIR = HERE / "probe_logs"
    PROBE_DIR.mkdir(exist_ok=True)
    out_dir = HERE / load(RUN_JSON)["output"]["directory"].lstrip("./")
    log_path = HERE / "adaptive_smoothing.log"
    metrics_path = HERE / "metrics.csv"

    # Per-chunk tabular metrics: max & L8 stress at final frame, each loss
    # value + ‖∇‖, sphere mesh quality, L-BFGS line-search counters.
    metric_cols = [
        "chunk", "cum_iters",
        "max_stress", "L8_stress",
        "stress_val", "volume_val", "smooth_val",
        "grad_stress", "grad_volume", "grad_smooth",
        "sj_tet_pre", "sj_surf_pre",
        "sj_tet_post", "sj_surf_post",
        "tries", "step_size", "remeshed",
    ]
    with metrics_path.open("w") as f:
        f.write(",".join(metric_cols) + "\n")

    def _emit_row(row: dict):
        with metrics_path.open("a") as f:
            f.write(",".join(_fmt_cell(row.get(k)) for k in metric_cols) + "\n")
        # Human-readable summary to stdout
        header_freq = 5
        if row["chunk"] == 1 or row["chunk"] % header_freq == 1:
            hdr = (f"  {'chunk':>5} {'maxσ':>10} {'L8σ':>10}  "
                   f"{'stress':>10} {'volume':>10} {'smooth':>10}  "
                   f"{'‖∇str‖':>10} {'‖∇vol‖':>10} {'‖∇smt‖':>10}  "
                   f"{'sjT':>6} {'sjS':>6}  "
                   f"{'sjT*':>6} {'sjS*':>6}  "
                   f"{'try':>3} {'step':>8} {'rem':>3}")
            print(hdr, flush=True)
        print(
            f"  {row['chunk']:>5d} "
            f"{_fmt_cell(row['max_stress']):>10} {_fmt_cell(row['L8_stress']):>10}  "
            f"{_fmt_cell(row['stress_val']):>10} {_fmt_cell(row['volume_val']):>10} {_fmt_cell(row['smooth_val']):>10}  "
            f"{_fmt_cell(row['grad_stress']):>10} {_fmt_cell(row['grad_volume']):>10} {_fmt_cell(row['grad_smooth']):>10}  "
            f"{_fmt_cell(row['sj_tet_pre']):>6} {_fmt_cell(row['sj_surf_pre']):>6}  "
            f"{_fmt_cell(row['sj_tet_post']):>6} {_fmt_cell(row['sj_surf_post']):>6}  "
            f"{row['tries']:>3} {_fmt_cell(row['step_size']):>8} {('Y' if row['remeshed'] else '.'): >3}",
            flush=True,
        )
    # Cutoff so latest_iter_mesh() ignores stale opt_state_0_iter_*.msh files
    # left behind by prior driver runs (which may have larger iter numbers
    # than what the current run will produce per chunk).
    import time
    driver_start_mtime = time.time()

    for chunk in range(TOTAL_CHUNKS):
        cum_iters = (chunk + 1) * PERIOD
        print(f"\n=== chunk {chunk+1}/{TOTAL_CHUNKS}: run {PERIOD} iter(s) (cumulative={cum_iters}) ===", flush=True)

        # Probe per-term gradient contributions at the current rest mesh BEFORE
        # the optimization step (= the gradient the optimizer is about to act on).
        try:
            probe_dir = PROBE_DIR / f"chunk_{chunk+1:03d}"
            probe_dir.mkdir(exist_ok=True)
            grads = probe_term_gradients(probe_dir)
            grads_str = "  ".join(f"{k}={v:.4g}" for k, v in grads.items())
            print(f"   pre-iter ‖∇f‖ per term: {grads_str}", flush=True)
        except Exception as e:
            print(f"   probe failed: {e}", flush=True)

        with log_path.open("w") as f:
            ret = subprocess.run(
                [POLYFEM, "-j", str(OPT_JSON), "--ns"],
                stdout=f, stderr=subprocess.STDOUT, cwd=HERE,
            )
        log = log_path.read_text()

        energies = parse_energies(log)
        run = load(RUN_JSON)
        deformable_mesh = HERE / run["geometry"][DEFORMABLE_GEOM_IX]["mesh"]
        nb = count_msh_nodes(deformable_mesh)
        base = sum(energies.get(k, 0.0) for k in ENERGY_KEYS_BASE)

        # Weighted per-loss values from the polyfem [print_energy] markers.
        stress_val = energies.get("stress", float("nan"))
        volume_val = energies.get("volume", float("nan"))
        smooth_val = energies.get("boundary_smoothing", float("nan"))

        # Outer L-BFGS line-search summary for this chunk.
        ls = parse_lbfgs(log)

        # Max- and L8-norm of Cauchy stress over sphere nodes at the FINAL
        # forward timestep. Pulled from the most recent step_*.vtu (skip
        # surface-only files).
        t_args = run["time"]
        if "time_steps" in t_args:
            time_steps = int(t_args["time_steps"])
        else:
            time_steps = int(round((float(t_args["tend"]) - float(t_args["t0"])) / float(t_args["dt"])))
        final_vtu = out_dir / f"step_{time_steps}.vtu"
        if final_vtu.exists():
            sd = stress_diagnostics(final_vtu)
        else:
            sd = {"max_stress": float("nan"), "L8_stress": float("nan")}

        if REBALANCE_SMOOTHING:
            # legacy rebalance — disabled in this run, kept here for reference
            if base > 0 and all(k in energies for k in ENERGY_KEYS_BASE):
                opt = load(OPT_JSON)
                old = set_smoothing_weight(
                    opt,
                    new_weight=_recompute_weight(opt, base, nb),
                )
                save(OPT_JSON, opt)
                print(f"   smoothing weight: {old}  ->  {get_smoothing_weight(opt)}", flush=True)
            else:
                print(f"   skipping reweight (energies incomplete)", flush=True)

        # Extract sphere-only from the latest combined opt_state mesh, optionally
        # remesh its interior, then hand it back as the next chunk's input.
        latest = latest_iter_mesh(out_dir, newer_than=driver_start_mtime)
        if latest is None:
            print("   no opt_state_0_iter_*.msh found, stopping.", flush=True)
            break

        sphere_path = REMESH_DIR / f"chunk_{chunk+1:03d}_sphere.msh"
        sphere_mesh = extract_sphere(latest, sphere_path)
        sj_tet_pre = min_scaled_jacobian(sphere_mesh)
        sj_surf_pre = min_surface_scaled_jacobian(sphere_mesh)
        force_remesh = (cum_iters % REMESH_PERIOD_ITERS) == 0
        need_remesh = sj_tet_pre < REMESH_TOLERANCE or force_remesh
        reason = "scaled_jac" if sj_tet_pre < REMESH_TOLERANCE else ("period" if force_remesh else "")

        next_mesh = sphere_path
        sj_tet_post = None
        sj_surf_post = None
        did_remesh = False
        if need_remesh:
            remeshed_path = REMESH_DIR / f"chunk_{chunk+1:03d}_sphere_remeshed.msh"
            try:
                call_wmtk(sphere_path, remeshed_path)
                rm = meshio.read(remeshed_path)
                sj_tet_post = min_scaled_jacobian(rm)
                sj_surf_post = min_surface_scaled_jacobian(rm)
                next_mesh = remeshed_path
                did_remesh = True
            except subprocess.CalledProcessError as e:
                print(f"   wmtk failed ({e.returncode}); using non-remeshed sphere", flush=True)

        _emit_row({
            "chunk":        chunk + 1,
            "cum_iters":    cum_iters,
            "max_stress":   sd["max_stress"],
            "L8_stress":    sd["L8_stress"],
            "stress_val":   stress_val,
            "volume_val":   volume_val,
            "smooth_val":   smooth_val,
            "grad_stress":  grads.get("stress"),
            "grad_volume":  grads.get("volume"),
            "grad_smooth":  grads.get("boundary_smoothing"),
            "sj_tet_pre":   sj_tet_pre,
            "sj_surf_pre":  sj_surf_pre,
            "sj_tet_post":  sj_tet_post,
            "sj_surf_post": sj_surf_post,
            "tries":        ls["n_failed"] + 1,
            "step_size":    ls["step_size"],
            "remeshed":     did_remesh,
        })

        run["geometry"][DEFORMABLE_GEOM_IX]["mesh"] = str(next_mesh.relative_to(HERE))
        # the deformed mesh is already in world coords; drop any pending transformation
        run["geometry"][DEFORMABLE_GEOM_IX].pop("transformation", None)
        save(RUN_JSON, run)

        # polyfem throws std::runtime_error on benign exit paths (e.g. "Reached
        # iteration limit" when max_iters is small) which then SIGABRTs.
        # Treat SIGABRT (exit -6 / 134) as a non-fatal "polyfem-quirk" and continue
        # — the iter mesh/vtu were saved before the throw.
        SIGABRT_CODES = (-6, 134)
        if ret.returncode not in (0,) + SIGABRT_CODES:
            print(f"   polyfem exit {ret.returncode}; check {log_path}", flush=True)
            break
        elif ret.returncode in SIGABRT_CODES:
            print(f"   polyfem SIGABRT (likely benign max-iters throw); continuing.", flush=True)

        if energies.get("Minus-Jacobian", 0) < -1e-7:
            print("   inversion (Minus-Jacobian < -1e-7); stopping for external inspection.", flush=True)
            break


def get_smoothing_weight(opt):
    for f in opt.get("functionals", []):
        if f.get("type") == "boundary_smoothing":
            return f["weight"]


def _recompute_weight(opt, base, nb):
    old = get_smoothing_weight(opt)
    return old * ADJUSTMENT_COEFF * base / max(nb, 1)


if __name__ == "__main__":
    sys.exit(main())
