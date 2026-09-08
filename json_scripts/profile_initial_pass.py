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
CASES = {"fig1": "fig1_bunnies", "fig15": "fig15_tentacles",
         "fig18": "fig18_static_cube", "fig21": "fig21_friction_bunny"}
NUM = r"[-+0-9.eE]+"
PHASES = ("forward_linear", "forward_ccd", "forward_hessian", "forward_line_search",
          "backward_hessian", "backward_linear")


def parse_log(path):
    totals = dict.fromkeys(PHASES, 0.0)
    counts = dict.fromkeys(PHASES, 0)
    objectives, gradients, finished = [], [], []
    forward_batches = 0
    result = {}
    outer_gradient_calls = 0
    completed_rollouts = 0
    newton_linf_tolerances = set()
    forward_solver_stops = []
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
        # includes LS begin (re-eval f and ∇f) to match Unified lineSearch.
        timer_names = {"linear solve": "forward_linear",
                       "assembly time": "forward_hessian",
                       "compute gradient": "forward_hessian",
                       "LS begin": "forward_line_search",
                       "backward hessian assembly": "backward_hessian",
                       "hessian cache": "backward_hessian",
                       "backward linear solver": "backward_linear"}
        for label, key in timer_names.items():
            m = re.search(r"\[timing\] " + label + r" (" + NUM + r")s", line)
            if m:
                totals[key] += float(m[1]); counts[key] += 1
        if "[timing][" in line and "classical_line_search" in line:
            values = dict((k, float(v)) for k, v in re.findall(r"(\w+) (" + NUM + r")s", line))
            totals["forward_ccd"] += sum(values.get(k, 0) for k in (
                "constraint_set_update", "checking_for_nan_inf", "broad_phase_ccd", "narrow_phase_ccd"))
            totals["forward_line_search"] += values["classical_line_search"]
            counts["forward_ccd"] += 1; counts["forward_line_search"] += 1
    result.update(phase_seconds={k: totals[k] if counts[k] else None for k in PHASES},
                  phase_events=counts, initial_loss=objectives[0] if objectives else result.get("loss"),
                  forward_batches=forward_batches, outer_finished=finished,
                  gradient_log=gradients, outer_gradient_calls=outer_gradient_calls,
                  completed_rollouts=completed_rollouts)
    result.update(newton_linf_absolute_tolerances=sorted(newton_linf_tolerances),
                  forward_solver_stops=forward_solver_stops)
    return result


def prepare(case, work, threads, broad_phase=None):
    src = ROOT / "json_scripts" / CASES[case]
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("result*", "current", "target", "__pycache__"))
    for path in work.glob("*.json"):
        data = json.loads(path.read_text())
        if "geometry" in data:
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
    p.add_argument("--fig10-newton-stopping", choices=("gradient", "unified-linf"), default="gradient")
    p.add_argument("--fig10-backend", choices=("ren-bowen", "pinned"), default="ren-bowen")
    # The current ContactForm.hpp enum mapping omits sweep_and_prune even
    # though the JSON schema lists it; it silently falls back to hash_grid.
    p.add_argument("--broad-phase", choices=["hash_grid", "bvh"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_DYNAMIC="FALSE")
    results = {}
    for case in args.cases.split(","):
        work = args.out / case
        if work.exists():
            raise RuntimeError(f"Refusing to overwrite existing results: {work}")
        print(f"START {case} {time.strftime('%F %T')}", flush=True)
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
            states = prepare(case, work, args.threads, args.broad_phase)
            cmd = ["/home/bowen/polyfem/build/PolyFEM_bin", "-j", str(work / "opt.json"),
                   "--ns", "-o", str(work / "output"), "--max_threads", str(args.threads)]
            cwd = work
        (work / "manifest.json").write_text(json.dumps({"command": cmd, "states": states,
            "broad_phase_override": args.broad_phase,
            "threads": args.threads, "target_note": "Forward totals include target state if present; one initial objective/gradient evaluation.",
            "six_phase_convention": "CCD: constraint update and finite-energy checks; Hessian: assembly time plus Newton compute gradient; Search: classical_line_search plus LS begin",
            "outer_stop": "first_grad_norm_tol=1e100, checked after initial gradient and before direction/trial"}, indent=2))
        started = time.monotonic()
        with (work / "run.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = parse_log(work / "run.log")
        row.update(case=case, exit_code=proc.returncode, wall_seconds=time.monotonic()-started, states=states)
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
        row["complete"] = proc.returncode == 0 and valid and all(row["phase_events"].values())
        (work / "result.json").write_text(json.dumps(row, indent=2))
        results[case] = row
        (args.out / "results.json").write_text(json.dumps(results, indent=2))
        with (args.out / "timing.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "complete", "wall_seconds", *PHASES])
            writer.writeheader()
            for v in results.values():
                writer.writerow({"case": v["case"], "complete": v["complete"], "wall_seconds": v["wall_seconds"], **v["phase_seconds"]})
        print(f"DONE {case} {json.dumps(row)}", flush=True)


if __name__ == "__main__":
    main()
