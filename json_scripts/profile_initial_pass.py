#!/usr/bin/env python3
"""Profile one initial forward/adjoint evaluation, with no outer trial step."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PY = Path(
    os.environ.get("ISAAC_PYTHON", "/home/bowen/miniconda3/envs/env_isaaclab/bin/python")
)
CASES = {"fig1": "fig1_bunnies", "fig15": "fig15_tentacles",
         "fig18": "fig18_static_cube", "fig21": "fig21_friction_bunny"}
# Unified: max_i ||d_i||_inf < threshold * dt * sqrt(bboxDiagSize2).
# PolyFEM Linf effective is x_delta_tol * L, L = mesh AABB diagonal.
# Fig.15 L matches Unified; Fig.1/21 compensate so the logged ‖Δx‖ matches.
UNIFIED_LINF = {
    "fig1": dict(threshold=1e-4, dt=0.05, unified_bbox_diag2=5.097503,
                 polyfem_L=28.308548942644247),
    "fig15": dict(threshold=1e-2, dt=0.005, unified_bbox_diag2=5.151646,
                  polyfem_L=2.2697236953242457),
    "fig21": dict(threshold=1e-3, dt=0.05, unified_bbox_diag2=58.847218,
                  polyfem_L=0.9968419191469463),
}
NUM = r"[-+0-9.eE]+"
PHASES = ("forward_linear", "forward_ccd", "forward_hessian", "forward_line_search",
          "backward_hessian", "backward_linear")
FORWARD_PHASES = PHASES[:4]
BACKWARD_PHASES = PHASES[4:]
OTHER_FIELDS = ("forward_other", "backward_other")


def _sum_phases(phases, names):
    return sum(phases.get(name) or 0.0 for name in names)


def attach_totals_and_other(row):
    phases = row.get("phase_seconds") or {}
    if "forward_seconds" in row:
        forward_total = row["forward_seconds"]
    else:
        forward_total = row.get("forward_batch_seconds")
    if "backward_seconds" in row:
        backward_total = row["backward_seconds"]
    else:
        backward_total = row.get("outer_gradient_seconds")
    cache = row.get("hessian_cache_seconds") or 0.0
    # Cache is logged during the forward solve but counted as backward Hessian.
    # Move it from the forward wall into Backward Total so Other = Total - parts.
    if forward_total is not None:
        forward_total = forward_total - cache
    if backward_total is not None:
        backward_total = backward_total + cache
    row["hessian_cache_seconds"] = cache
    row["forward_total_seconds"] = forward_total
    row["backward_total_seconds"] = backward_total
    row["forward_other_seconds"] = (
        None if forward_total is None else forward_total - _sum_phases(phases, FORWARD_PHASES))
    row["backward_other_seconds"] = (
        None if backward_total is None else backward_total - _sum_phases(phases, BACKWARD_PHASES))
    if phases:
        phases["forward_other"] = row["forward_other_seconds"]
        phases["backward_other"] = row["backward_other_seconds"]
    return row


def parse_log(path):
    path = Path(path)
    totals = dict.fromkeys(PHASES, 0.0)
    counts = dict.fromkeys(PHASES, 0)
    objectives, gradients, finished = [], [], []
    forward_batches = 0
    result = {}
    outer_gradient_calls = 0
    completed_rollouts = 0
    newton_linf_tolerances = set()
    forward_solver_stops = []
    hessian_cache = 0.0
    hessian_cache_events = 0
    for raw in path.open(errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        if "Run simulations in " in line:
            forward_batches += 1
        frame = re.search(r"\[info\] (\d+)/(\d+)\s+t=", line)
        if frame and frame[1] == frame[2]:
            completed_rollouts += 1
        if "[adjoint-polyfem]" in line:
            m = re.search(r"objective (" + NUM + r")", line)
            if m:
                objectives.append(float(m[1]))
            if re.search(r"\[trace\] gradient \[", line):
                gradients.append(line.strip())
            if "Finished:" in line:
                finished.append(line.strip())
            if "[timing] compute gradient " in line:
                outer_gradient_calls += 1
                timing = re.search(r"compute gradient (" + NUM + r")s", line)
                if timing:
                    result["outer_gradient_seconds"] = float(timing[1])
            if "update_direction:" in line and "line_search:" in line:
                for name, field in (("update_direction", "outer_update_direction_seconds"),
                                    ("line_search", "outer_line_search_seconds"),
                                    ("constraint_set_update", "forward_batch_seconds")):
                    m = re.search(name + r": (" + NUM + r")s", line)
                    if m:
                        result[field] = float(m[1])
            continue
        if line.startswith("SINGLE_PASS_RESULT "):
            result.update(json.loads(line.split(" ", 1)[1]))
        if line.startswith("PolyFEM build info: "):
            result["polyfem_build_info"] = json.loads(line.split(": ", 1)[1])
        tolerance = re.search(r"Newton direction stopping: Linf absolute tolerance=(" + NUM + r")", line)
        if tolerance:
            newton_linf_tolerances.add(float(tolerance[1]))
        if "[polyfem]" in line and "Finished:" in line:
            forward_solver_stops.append(line.strip())
            if "stopping criteria:" in line:
                criteria = line.split("stopping criteria:", 1)[1]
                step_tol = re.search(r"‖Δx‖=(" + NUM + r")", criteria)
                if step_tol:
                    result.setdefault("native_x_delta_tolerances", []).append(float(step_tol[1]))
        # Hessian matches Unified's fused gradient+Hessian timer. Search
        # includes LS begin (re-eval f and ∇f) and the trial-step contact
        # rebuild inside classical_line_search, matching Unified lineSearch.
        # CCD is only the sibling collision-free step-size work; the nested
        # constraint_set_update stays in Search so the columns are disjoint.
        # Backward Hessian also includes shape and other parameter VJPs
        # (dJ_shape/material/friction/v0/Dirichlet/pressure) and the
        # post-adjoint objective/parameter assembly (`gradient assembly`),
        # matching Unified hessian_assembling (including objective → seed).
        timer_names = {"linear solve": "forward_linear",
                       "assembly time": "forward_hessian",
                       "compute gradient": "forward_hessian",
                       "LS begin": "forward_line_search",
                       "backward hessian assembly": "backward_hessian",
                       "hessian cache": "backward_hessian",
                       "gradient assembly": "backward_hessian",
                       "backward linear solver": "backward_linear"}
        for label, key in timer_names.items():
            m = re.search(r"\[timing\] " + label + r" (" + NUM + r")s", line)
            if m:
                value = float(m[1])
                totals[key] += value
                counts[key] += 1
                if label == "hessian cache":
                    hessian_cache += value
                    hessian_cache_events += 1
        if "[timing][" in line and "classical_line_search" in line:
            values = dict((k, float(v)) for k, v in re.findall(r"(\w+) (" + NUM + r")s", line))
            totals["forward_ccd"] += sum(values.get(k, 0) for k in (
                "checking_for_nan_inf", "broad_phase_ccd", "narrow_phase_ccd"))
            totals["forward_line_search"] += values["classical_line_search"]
            counts["forward_ccd"] += 1; counts["forward_line_search"] += 1
    result.update(phase_seconds={k: totals[k] if counts[k] else None for k in PHASES},
                  phase_events=counts, initial_loss=objectives[0] if objectives else result.get("loss"),
                  forward_batches=forward_batches, outer_finished=finished,
                  gradient_log=gradients, outer_gradient_calls=outer_gradient_calls,
                  completed_rollouts=completed_rollouts,
                  hessian_cache_seconds=hessian_cache,
                  hessian_cache_events=hessian_cache_events)
    result.update(newton_linf_absolute_tolerances=sorted(newton_linf_tolerances),
                  forward_solver_stops=forward_solver_stops)
    return attach_totals_and_other(result)


def unified_linf_x_delta_tol(spec):
    bbox = spec["unified_bbox_diag2"] ** 0.5
    return spec["threshold"] * spec["dt"] * bbox / spec["polyfem_L"]


def apply_unified_linf(nonlinear, case):
    spec = UNIFIED_LINF[case]
    nonlinear.update(
        norm_type="Linf",
        x_delta_tol=unified_linf_x_delta_tol(spec),
        grad_norm_tol=0,
        rel_grad_norm_tol=0,
        rel_x_delta_tol=0,
        newton_decrement_tol=0,
        first_grad_norm_tol=0,
    )
    advanced = nonlinear.setdefault("advanced", {})
    advanced["f_delta_tol"] = 0
    advanced["derivative_along_delta_x_tol"] = 0
    return spec["threshold"] * spec["dt"] * spec["unified_bbox_diag2"] ** 0.5


def prepare(case, work, threads, broad_phase=None, skip_target=False):
    src = ROOT / "json_scripts" / CASES[case]
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("result*", "current", "target", "__pycache__"))
    for path in work.glob("*.json"):
        data = json.loads(path.read_text())
        if "geometry" in data:
            data.setdefault("space", {}).setdefault("advanced", {})["quadrature_order"] = 1
            if case in UNIFIED_LINF:
                apply_unified_linf(data.setdefault("solver", {}).setdefault("nonlinear", {}), case)
            if broad_phase:
                data.setdefault("solver", {}).setdefault("contact", {}).setdefault("CCD", {})["broad_phase"] = broad_phase
            output = data.setdefault("output", {})
            output.setdefault("log", {})["level"] = "trace"
            output.setdefault("paraview", {}).update(skip_frame=10**9, volume=False, surface=False)
            output.setdefault("advanced", {})["save_time_sequence"] = False
            path.write_text(json.dumps(data, indent=2) + "\n")
    path = work / "opt.json"
    opt = json.loads(path.read_text())
    opt.pop("compute_objective", None)
    if skip_target and len(opt.get("states", [])) > 1:
        opt["states"] = opt["states"][:1]
        for functional in opt.get("functionals", []):
            objective = functional.get("static_objective")
            if not objective or "target_state" not in objective:
                continue
            functional["static_objective"] = {
                "type": "position",
                "state": objective.get("state", 0),
                "dim": 2,
                "volume_selection": objective.get("volume_selection", [1]),
            }
    solver = opt.setdefault("solver", {})
    solver["max_threads"] = threads
    solver.setdefault("advanced", {})["solve_in_parallel"] = False
    # PolySolve Solver::minimize computes gradient before checkConvergence,
    # and checks firstGradNorm before calculating a direction or line search.
    solver.setdefault("nonlinear", {}).update(first_grad_norm_tol=1e100,
        max_iterations=0, allow_out_of_iterations=True)
    solver["nonlinear"].setdefault("advanced", {})["apply_gradient_fd"] = "None"
    opt.setdefault("output", {}).setdefault("log", {})["level"] = "trace"
    path.write_text(json.dumps(opt, indent=2) + "\n")
    return len(opt["states"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", default="fig18,fig1,fig21,fig15,fig10")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fig10-newton-stopping", choices=("gradient", "unified-linf"), default="unified-linf")
    p.add_argument("--fig10-backend", choices=("ren-bowen", "pinned"), default="ren-bowen")
    # The current ContactForm.hpp enum mapping omits sweep_and_prune even
    # though the JSON schema lists it; it silently falls back to hash_grid.
    p.add_argument("--broad-phase", choices=["hash_grid", "bvh"])
    p.add_argument("--fig21-skip-target", action=argparse.BooleanOptionalAction, default=True,
                   help="Fig21 target is a second full bunny rollout and can OOM; skip it by default.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_DYNAMIC="FALSE")
    results = {}
    csv_fields = ["case", "complete", "wall_seconds", "forward_total", "backward_total",
                  *FORWARD_PHASES, "forward_other", *BACKWARD_PHASES, "backward_other"]
    for case in args.cases.split(","):
        work = args.out / case
        if work.exists():
            raise RuntimeError(f"Refusing to overwrite existing results: {work}")
        print(f"START {case} {time.strftime('%F %T')}", flush=True)
        skip_target = case == "fig21" and args.fig21_skip_target
        broad_phase = args.broad_phase
        if broad_phase is None:
            broad_phase = "bvh"
        if case == "fig10":
            work.mkdir()
            states = 1
            cmd = ["/home/bowen/miniconda3/envs/env_isaaclab/bin/python", "-u",
                   str(ROOT / "python_scripts/fig10_hanger/optimize.py"),
                   "--eval-init-gradient", "--skip-vtu", "--max-threads", str(args.threads)]
            cmd += ["--polyfem-backend", args.fig10_backend]
            cmd += ["--output-dir", str(work.resolve() / "native-output")]
            if args.fig10_backend == "pinned":
                cmd.append("--legacy-polysolve-schema")
            cmd += ["--newton-stopping", args.fig10_newton_stopping]
            cwd = ROOT / "python_scripts/fig10_hanger"
        else:
            states = prepare(case, work, args.threads, broad_phase, skip_target=skip_target)
            if case == "fig18":
                subprocess.run(
                    [str(ISAAC_PY), str(work / "generate_target.py")],
                    cwd=work,
                    env=env,
                    check=True,
                )
            cmd = ["/home/bowen/polyfem/build/PolyFEM_bin", "-j", str(work / "opt.json"),
                   "--ns", "-o", str(work / "output"), "--max_threads", str(args.threads)]
            cwd = work
        newton_stopping = None
        if case == "fig10":
            newton_stopping = args.fig10_newton_stopping
        elif case in UNIFIED_LINF:
            newton_stopping = "unified-linf"
        elif case == "fig18":
            newton_stopping = "unified-linf"
        (work / "manifest.json").write_text(json.dumps({"command": cmd, "states": states,
            "broad_phase_override": broad_phase,
            "skip_target": skip_target,
            "newton_stopping": newton_stopping,
            "threads": args.threads, "target_note": "One initial objective/gradient evaluation. Fig21 skips the target rollout to avoid a second-sim OOM.",
            "six_phase_convention": "CCD: finite-energy checks plus broad/narrow CCD; Hessian: assembly time plus Newton compute gradient; Search: classical_line_search (including trial-step contact rebuild) plus LS begin; Backward Hessian: cached/applied residual Jacobian plus shape and other parameter derivatives; Other: independently timed total minus the reported parts",
            "outer_stop": "first_grad_norm_tol=1e100, checked after initial gradient and before direction/trial"}, indent=2))
        started = time.monotonic()
        with (work / "run.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = parse_log(work / "run.log")
        row.update(case=case, exit_code=proc.returncode, wall_seconds=time.monotonic()-started, states=states)
        row["broad_phase"] = broad_phase or "bvh"
        row["skip_target"] = skip_target
        if case == "fig10":
            valid = row.get("backward_calls") == 1 and row.get("parameter_updates") == 0
            row["newton_stopping"] = args.fig10_newton_stopping
            row["polyfem_backend"] = args.fig10_backend
            if args.fig10_newton_stopping == "unified-linf":
                if args.fig10_backend == "pinned":
                    valid = valid and bool(row["newton_linf_absolute_tolerances"])
                else:
                    valid = valid and bool(row.get("native_x_delta_tolerances"))
                valid = valid and len(row["forward_solver_stops"]) == row.get("completed_frames")
                valid = valid and all("Change in parameter vector too small" in line
                                      for line in row["forward_solver_stops"])
        else:
            valid = row["forward_batches"] == 1 and len(row["outer_finished"]) == 1 and "iters=0" in row["outer_finished"][0]
            if case in UNIFIED_LINF or case == "fig18":
                row["newton_stopping"] = "unified-linf"
        row["complete"] = proc.returncode == 0 and valid and all(row["phase_events"].values())
        attach_totals_and_other(row)
        (work / "result.json").write_text(json.dumps(row, indent=2))
        results[case] = row
        (args.out / "results.json").write_text(json.dumps(results, indent=2))
        with (args.out / "timing.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for v in results.values():
                phases = v.get("phase_seconds") or {}
                writer.writerow({
                    "case": v["case"], "complete": v["complete"], "wall_seconds": v["wall_seconds"],
                    "forward_total": v.get("forward_total_seconds"),
                    "backward_total": v.get("backward_total_seconds"),
                    **{k: phases.get(k) for k in (*FORWARD_PHASES, *BACKWARD_PHASES, *OTHER_FIELDS)},
                })
        print(f"DONE {case} complete={row['complete']} wall={row['wall_seconds']:.1f}s "
              f"fwd_total={row.get('forward_total_seconds')} bwd_total={row.get('backward_total_seconds')} "
              f"fwd_other={row.get('forward_other_seconds')} bwd_other={row.get('backward_other_seconds')}",
              flush=True)


if __name__ == "__main__":
    main()
