#!/usr/bin/env python3
"""Rerun fig21/fig15/fig10 with one outer iteration and grad_norm_tol=1e-3."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/bowen/DiffIPC-data-original")
POLYFEM_BIN = Path("/home/bowen/polyfem/build/PolyFEM_bin")
ISAAC_PY = Path("/home/bowen/miniconda3/envs/env_isaaclab/bin/python")
THREADS = int(os.environ.get("MAX_THREADS", "8"))
OUT_ROOT = ROOT / "polyfem_one_iter_grad1e-3_20260904"

OBJECTIVE_RE = re.compile(
    r"\[adjoint-polyfem\].*\bobjective\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
FIG10_ITER_RE = re.compile(
    r"Iter\s+(\d+)\s+\|\s+loss\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)

JSON_CASES = {
    "fig21": {
        "dir": ROOT / "json_scripts/fig21_friction_bunny",
        "opt": "opt.json",
    },
    "fig15": {
        "dir": ROOT / "json_scripts/fig15_tentacles",
        "opt": "opt.json",
    },
}


def _disable_state_output(state: dict) -> None:
    output = state.setdefault("output", {})
    paraview = output.setdefault("paraview", {})
    paraview["skip_frame"] = 10**9
    paraview["volume"] = False
    paraview["surface"] = False
    output.setdefault("advanced", {})["save_time_sequence"] = False


def _prepare_json_case(name: str) -> Path:
    spec = JSON_CASES[name]
    work = OUT_ROOT / name
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(
        spec["dir"],
        work,
        ignore=shutil.ignore_patterns("result*", "current", "target", "__pycache__"),
    )
    for path in work.glob("*.json"):
        if path.name == spec["opt"]:
            continue
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "geometry" not in state:
            continue
        _disable_state_output(state)
        path.write_text(json.dumps(state, indent=4) + "\n")
    opt_path = work / spec["opt"]
    opt = json.loads(opt_path.read_text())
    opt.pop("compute_objective", None)
    solver = opt.setdefault("solver", {})
    solver["max_threads"] = THREADS
    solver.setdefault("nonlinear", {})["max_iterations"] = 1
    opt.setdefault("output", {}).setdefault("log", {})["level"] = "info"
    opt_path.write_text(json.dumps(opt, indent=4) + "\n")
    return work


def _unique_changed(values: list[float]) -> list[float]:
    history: list[float] = []
    for value in values:
        if not history or abs(value - history[-1]) > max(1e-15, 1e-12 * abs(history[-1])):
            history.append(value)
    return history


def _summarize(history: list[float]) -> dict:
    if not history:
        return {"history": [], "init": None, "best": None, "final": None, "n": 0}
    return {
        "history": history,
        "init": history[0],
        "best": min(history),
        "final": history[-1],
        "n": len(history),
    }


def _run(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    text = log_path.read_text(errors="replace")
    return proc.returncode, elapsed, text


def run_json_case(name: str) -> dict:
    work = _prepare_json_case(name)
    log_path = work / "opt.log"
    print(f"=== start {name} ===", flush=True)
    rc, elapsed, text = _run(
        [
            str(POLYFEM_BIN),
            "-j",
            str(work / JSON_CASES[name]["opt"]),
            "--ns",
            "-o",
            str(work / "polyfem_out"),
            "--max_threads",
            str(THREADS),
        ],
        work,
        log_path,
    )
    history = _unique_changed([float(x) for x in OBJECTIVE_RE.findall(text)])
    summary = _summarize(history)
    summary.update({"case": name, "exit_code": rc, "elapsed_s": elapsed, "log": str(log_path)})
    if rc != 0:
        summary["error"] = "\n".join(text.splitlines()[-30:])
    (work / "polyfem_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"POLYFEM {name} rc={rc} init={summary.get('init')} "
        f"best={summary.get('best')} final={summary.get('final')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def run_fig10() -> dict:
    work = OUT_ROOT / "fig10"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log_path = work / "opt.log"
    print("=== start fig10 ===", flush=True)
    rc, elapsed, text = _run(
        [
            str(ISAAC_PY),
            "-u",
            str(ROOT / "python_scripts/fig10_hanger/optimize.py"),
            "--skip-vtu",
            "--n-iters",
            "1",
            "--output-dir",
            str(work / "opt"),
        ],
        ROOT / "python_scripts/fig10_hanger",
        log_path,
    )
    history = [float(match.group(2)) for match in FIG10_ITER_RE.finditer(text)]
    loss_file = work / "opt" / "loss_history.txt"
    if loss_file.exists():
        history = [float(line) for line in loss_file.read_text().split() if line.strip()]
    summary = _summarize(history)
    summary.update({"case": "fig10", "exit_code": rc, "elapsed_s": elapsed, "log": str(log_path)})
    if rc != 0:
        summary["error"] = "\n".join(text.splitlines()[-30:])
    (work / "polyfem_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"POLYFEM fig10 rc={rc} init={summary.get('init')} "
        f"best={summary.get('best')} final={summary.get('final')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name in ("fig21", "fig15", "fig10"):
        results[name] = run_fig10() if name == "fig10" else run_json_case(name)
        (OUT_ROOT / "partial_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
