#!/usr/bin/env python3
"""Evaluate initial losses for the five aligned DiffIPC cases vs Unified."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = Path("/home/bowen/Unified_GIPC")
POLYFEM_BIN = Path(os.environ.get("POLYFEM_BIN", "/home/bowen/polyfem/build/PolyFEM_bin"))
ISAAC_PY = Path(
    os.environ.get(
        "ISAAC_PYTHON",
        "/home/bowen/miniconda3/envs/env_isaaclab/bin/python",
    )
)
UNIFIED_PY = Path(
    os.environ.get(
        "UNIFIED_PYTHON",
        "/home/bowen/miniconda3/envs/magnet/bin/python",
    )
)
THREADS = int(os.environ.get("MAX_THREADS", "8"))
OBJECTIVE_RE = re.compile(
    r"\[adjoint-polyfem\].*\bobjective\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
OBJECTIVE_IS_RE = re.compile(
    r"Objective is\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
INIT_LOSS_RE = re.compile(r"INIT_LOSS\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

CASES = {
    "fig1": {
        "dir": ROOT / "json_scripts/fig1_bunnies",
        "opt": "opt.json",
    },
    "fig15": {
        "dir": ROOT / "json_scripts/fig15_tentacles",
        "opt": "opt.json",
    },
    "fig18": {
        "dir": ROOT / "json_scripts/fig18_static_cube",
        "opt": "opt.json",
    },
    "fig21": {
        "dir": ROOT / "json_scripts/fig21_friction_bunny",
        "opt": "opt.json",
    },
    "fig10": {
        "script": ROOT / "python_scripts/fig10_hanger/optimize.py",
        "cwd": ROOT / "python_scripts/fig10_hanger",
    },
}


def _disable_state_output(state: dict) -> None:
    output = state.setdefault("output", {})
    output.setdefault("log", {})["level"] = "info"
    paraview = output.setdefault("paraview", {})
    paraview["skip_frame"] = 10**9
    paraview["volume"] = False
    paraview["surface"] = False
    output.setdefault("advanced", {})["save_time_sequence"] = False


def _prepare_opt(src_dir: Path, work: Path, opt_name: str) -> Path:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src_dir, work, ignore=shutil.ignore_patterns("result*", "current", "target", "__pycache__"))
    for path in work.glob("*.json"):
        if path.name == opt_name:
            continue
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "geometry" not in state:
            continue
        _disable_state_output(state)
        path.write_text(json.dumps(state, indent=4) + "\n")
    opt_path = work / opt_name
    opt = json.loads(opt_path.read_text())
    # Official PolyFEM path: solve the initial states once, print the
    # functional, and exit. No L-BFGS/ADAM step and no adjoint.
    opt["compute_objective"] = True
    opt.setdefault("solver", {})["max_threads"] = THREADS
    opt.setdefault("output", {}).setdefault("log", {})["level"] = "trace"
    opt_path.write_text(json.dumps(opt, indent=4) + "\n")
    return opt_path


def _parse_polyfem_objective(log_text: str) -> float:
    match = OBJECTIVE_IS_RE.search(log_text)
    if match:
        return float(match.group(1))
    matches = OBJECTIVE_RE.findall(log_text)
    if matches:
        return float(matches[0])
    raise RuntimeError("no objective value in PolyFEM log")


def run_polyfem_json(name: str, out_root: Path) -> float:
    spec = CASES[name]
    work = out_root / name
    opt_path = _prepare_opt(spec["dir"], work, spec["opt"])
    log_path = work / "init_loss.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    cmd = [
        str(POLYFEM_BIN),
        "-j",
        str(opt_path),
        "--ns",
        "-o",
        str(work / "polyfem_out"),
        "--max_threads",
        str(THREADS),
    ]
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(f"{name} PolyFEM exited {proc.returncode}\n{tail}")
    value = _parse_polyfem_objective(text)
    print(f"POLYFEM {name} {value:.17g}", flush=True)
    return value


def run_polyfem_fig10(out_root: Path) -> float:
    spec = CASES["fig10"]
    work = out_root / "fig10"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "init_loss.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    cmd = [
        str(ISAAC_PY),
        "-u",
        str(spec["script"]),
        "--eval-init-loss",
        "--skip-vtu",
        "--n-iters",
        "0",
        "--output-dir",
        str(work / "opt"),
    ]
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd, cwd=spec["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT
        )
    text = log_path.read_text(errors="replace")
    match = INIT_LOSS_RE.search(text)
    if match is None:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(f"fig10 PolyFEM missing INIT_LOSS (rc={proc.returncode})\n{tail}")
    value = float(match.group(1))
    print(f"POLYFEM fig10 {value:.17g}", flush=True)
    return value


def run_unified(names: list[str], out_root: Path) -> dict[str, float]:
    helper = out_root / "_eval_unified.py"
    helper.write_text(
        r'''
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

UNIFIED = Path("/home/bowen/Unified_GIPC")
sys.path[:0] = [
    str(UNIFIED / "python"),
    str(UNIFIED / "python_examples/diff_sim/tools"),
    str(UNIFIED / "python_examples/diff_sim"),
]


def load_example(name: str, rel: str):
    path = UNIFIED / rel / "main.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def eval_fig1():
    m = load_example("fig1_main", "python_examples/diff_sim/1_fig1_bunny_init_velocity")
    target = m.generate_target(m.default_bunny_mesh(), m.DEFAULT_FRAMES)
    world, vertex_count, tets = m.make_world()
    module = m.build_fig1_module(world, vertex_count, m.DEFAULT_FRAMES, m.INITIAL_VELOCITY.copy())
    return float(m._center_loss(module(), tets, target).detach().cpu())


def eval_fig10():
    m = load_example("fig10_main", "python_examples/diff_sim/2_fig10_hanger_rest_shape")
    from fig10_hanger_loss import loss_tensors
    world, metadata = m.make_world(m.MESH)
    module, _ = m.build_fig10_module(world, metadata, m.FRAMES)
    vertices = module().reshape(-1, 3)
    selected = torch.as_tensor(metadata["selected_ids"], dtype=torch.long)
    rest_base = torch.as_tensor(metadata["rest"], dtype=torch.float64)
    rest = rest_base.index_copy(0, selected, module.physical_parameter().reshape(-1, 3))
    total, stress, lap = loss_tensors(
        vertices,
        rest,
        metadata["tets"],
        metadata["faces"],
        youngs=m.YOUNGS,
        poisson=m.POISSON,
        laplacian_weight=m.LAPLACIAN_WEIGHT,
        y_lower=m.Y_LOWER,
        y_upper=m.Y_UPPER,
        laplacian_mode="legacy_uniform",
    )
    print(f"UNIFIED_PARTS fig10 stress={float(stress):.17g} laplacian={float(lap):.17g}", flush=True)
    return float(total.detach().cpu())


def eval_fig15():
    m = load_example("fig15_main", "python_examples/diff_sim/3_fig15_tentacles_init_velocity")
    target = m._load_or_generate_target(m.DEFAULT_MESH, m.DEFAULT_TARGET, m.FRAMES)
    world, obj1_count, weights = m.make_world(m.DEFAULT_MESH)
    module = m.build_fig15_module(world, obj1_count, m.FRAMES, m.OBJ1_OPT_V0.copy())
    return float(m._trajectory_loss(module(), weights, target, m.FRAMES, obj1_count).detach().cpu())


def eval_fig18():
    m = load_example("fig18_main", "python_examples/diff_sim/4_fig18_cube_material_parameter")
    world, active_ids = m.make_world()
    target = m.make_target(world, active_ids)
    module = m.build_fig18_module(world, m.INITIAL_LOG_LAME.copy())
    return float(m._loss_tensor(module(), target, active_ids).detach().cpu())


def eval_fig21():
    m = load_example("fig21_main", "python_examples/diff_sim/5_fig21_bunny_friction_rate")
    frames = m.FRAMES
    world, slide_count, bunny_count, weights, velocities = m.make_world()
    target_axis = m.generate_target(world, slide_count, bunny_count, weights, velocities, frames)
    world.reset()
    world.set_parameters({"friction": np.array([m.INITIAL_FRICTION]), "v0": velocities})
    module = m.build_fig21_module(world, slide_count, bunny_count, frames, m.INITIAL_FRICTION)
    return float(
        m._terminal_loss_tensor(module().reshape(bunny_count, 3), weights, target_axis).detach().cpu()
    )


dispatch = {
    "fig1": eval_fig1,
    "fig10": eval_fig10,
    "fig15": eval_fig15,
    "fig18": eval_fig18,
    "fig21": eval_fig21,
}
name = sys.argv[1]
value = dispatch[name]()
print(f"UNIFIED {name} {value:.17g}", flush=True)
'''
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name in names:
        log_path = out_root / f"unified_{name}.log"
        cmd = [str(UNIFIED_PY), "-u", str(helper), name]
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=str(UNIFIED), env=env, stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text(errors="replace")
        match = None
        for line in text.splitlines():
            if line.startswith(f"UNIFIED {name} "):
                match = line
        if match is not None:
            values[name] = float(match.split(" ", 2)[2])
            print(match, flush=True)
            continue
        tail = "\n".join(text.splitlines()[-20:])
        errors[name] = f"rc={proc.returncode}: {tail[-500:]}"
        print(f"UNIFIED {name} ERROR rc={proc.returncode}", flush=True)
    if errors and not values:
        raise RuntimeError(json.dumps(errors)[:800])
    if errors:
        print("UNIFIED_PARTIAL_ERRORS " + json.dumps(errors)[:800], flush=True)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("polyfem", "unified", "both"), default="both")
    parser.add_argument(
        "--cases",
        default="fig18,fig1,fig10,fig21,fig15",
        help="comma-separated case names",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "init_loss_compare_20260904"),
    )
    args = parser.parse_args()
    names = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    results: dict[str, dict] = {}
    if summary_path.exists():
        try:
            for row in json.loads(summary_path.read_text()):
                case = row.get("case")
                if case:
                    results[case] = {k: v for k, v in row.items() if k != "case"}
        except json.JSONDecodeError:
            pass
    for name in names:
        results.setdefault(name, {})

    if args.side in ("polyfem", "both"):
        for name in names:
            try:
                if name == "fig10":
                    results[name]["polyfem"] = run_polyfem_fig10(out_root)
                else:
                    results[name]["polyfem"] = run_polyfem_json(name, out_root)
            except Exception as exc:
                results[name]["polyfem_error"] = str(exc)
                print(f"POLYFEM {name} ERROR {exc}", flush=True)

    if args.side in ("unified", "both"):
        try:
            unified = run_unified(names, out_root)
            for name, value in unified.items():
                results[name]["unified"] = value
                results[name].pop("unified_error", None)
        except Exception as exc:
            print(f"UNIFIED ERROR {exc}", flush=True)
            for name in names:
                results[name].setdefault("unified_error", str(exc))

    ordered = []
    seen = set()
    for name in list(names) + [row_name for row_name in results if row_name not in names]:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    summary = []
    for name in ordered:
        row = {"case": name, **results[name]}
        poly = row.get("polyfem")
        uni = row.get("unified")
        if isinstance(poly, (int, float)) and isinstance(uni, (int, float)):
            if abs(uni) > 0:
                row["rel_err"] = abs(poly - uni) / abs(uni)
            else:
                row["rel_err"] = abs(poly - uni)
        summary.append(row)
        print(json.dumps(row), flush=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
