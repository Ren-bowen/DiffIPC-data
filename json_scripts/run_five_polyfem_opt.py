#!/usr/bin/env python3
"""Run the five aligned DiffIPC PolyFEM optimizations and compare with stage61."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLYFEM_BIN = Path(os.environ.get("POLYFEM_BIN", "/home/bowen/polyfem/build/PolyFEM_bin"))
ISAAC_PY = Path(
    os.environ.get("ISAAC_PYTHON", "/home/bowen/miniconda3/envs/env_isaaclab/bin/python")
)
THREADS = int(os.environ.get("MAX_THREADS", "8"))
OUT_ROOT = Path(
    os.environ.get(
        "POLYFEM_OPT_OUT",
        str(ROOT / "polyfem_opt_vs_stage61_20260904"),
    )
)

OBJECTIVE_RE = re.compile(
    r"\[adjoint-polyfem\].*\bobjective\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)
FIG10_ITER_RE = re.compile(
    r"Iter\s+(\d+)\s+\|\s+loss\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)

STAGE61 = {
    "fig1": {
        "unified_init": 4.764047,
        "unified_best": 1.295239e-06,
        "unified_final": 1.295239e-06,
        "steps": 20,
        "source": "stage61_eleven_validation_20260903_current/1_fig1_bunny_init_velocity",
        "note": "stage61 记录 4.764047 -> 1.295239e-6",
    },
    "fig10": {
        "unified_init": 97.38676244584,
        "unified_best": 45.819641,
        "unified_final": 498.872882,
        "steps": 100,
        "source": "stage61_eleven_validation_20260903_fig10_retry/2_fig10_hanger_rest_shape",
        "note": "按 raw loss 验收：初值 97.38676，最佳 45.81964，末步 498.87",
    },
    "fig15": {
        "unified_init": 0.01119760,
        "unified_best": 2.218199e-06,
        "unified_final": 2.218199e-06,
        "steps": 10,
        "source": "stage61_eleven_validation_20260903_current/3_fig15_tentacles_init_velocity",
        "note": "stage61 记录 1.119760e-2 -> 2.218199e-6",
    },
    "fig18": {
        "unified_init": 8.867660,
        "unified_best": 7.835828e-12,
        "unified_final": 7.835828e-12,
        "steps": 20,
        "source": "stage61_eleven_validation_20260903_current/4_fig18_cube_material_parameter",
        "note": "stage61 记录 8.867660 -> 7.835828e-12",
    },
    "fig21": {
        "unified_init": 54.65497,
        "unified_best": 1.395564e-06,
        "unified_final": 1.395564e-06,
        "steps": 5,
        "source": "stage61_eleven_validation_20260903_current/5_fig21_bunny_friction_rate",
        "note": "stage61 记录 54.65497 -> 1.395564e-6",
    },
}

JSON_CASES = {
    "fig18": {
        "dir": ROOT / "json_scripts/fig18_static_cube",
        "opt": "opt.json",
    },
    "fig1": {
        "dir": ROOT / "json_scripts/fig1_bunnies",
        "opt": "opt.json",
    },
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
    opt.setdefault("solver", {})["max_threads"] = THREADS
    opt.setdefault("output", {}).setdefault("log", {})["level"] = "trace"
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
        return {"history": [], "init": None, "best": None, "final": None}
    return {
        "history": history,
        "init": history[0],
        "best": min(history),
        "final": history[-1],
        "n": len(history),
    }


def _regenerate_fig18_target(work: Path, env: dict) -> None:
    script = work / "generate_target.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    subprocess.run([str(ISAAC_PY), str(script)], cwd=work, env=env, check=True)


def run_json_case(name: str) -> dict:
    work = _prepare_json_case(name)
    log_path = work / "opt.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    if name == "fig18":
        _regenerate_fig18_target(work, env)
    cmd = [
        str(POLYFEM_BIN),
        "-j",
        str(work / JSON_CASES[name]["opt"]),
        "--ns",
        "-o",
        str(work / "polyfem_out"),
        "--max_threads",
        str(THREADS),
    ]
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    text = log_path.read_text(errors="replace")
    history = _unique_changed([float(x) for x in OBJECTIVE_RE.findall(text)])
    summary = _summarize(history)
    summary.update(
        {
            "case": name,
            "exit_code": proc.returncode,
            "elapsed_s": elapsed,
            "log": str(log_path),
        }
    )
    if proc.returncode != 0:
        summary["error"] = "\n".join(text.splitlines()[-20:])
    (work / "polyfem_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"POLYFEM {name} rc={proc.returncode} init={summary.get('init')} "
        f"best={summary.get('best')} final={summary.get('final')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def run_fig10() -> dict:
    work = OUT_ROOT / "fig10"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "opt.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS)
    env["MKL_NUM_THREADS"] = str(THREADS)
    cmd = [
        str(ISAAC_PY),
        "-u",
        str(ROOT / "python_scripts/fig10_hanger/optimize.py"),
        "--skip-vtu",
        "--n-iters",
        "100",
        "--output-dir",
        str(work / "opt"),
    ]
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT / "python_scripts/fig10_hanger"),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    text = log_path.read_text(errors="replace")
    history = [float(match.group(2)) for match in FIG10_ITER_RE.finditer(text)]
    loss_file = work / "opt" / "loss_history.txt"
    if loss_file.exists():
        history = [
            float(line)
            for line in loss_file.read_text().split()
            if line.strip()
        ]
    summary = _summarize(history)
    summary.update(
        {
            "case": "fig10",
            "exit_code": proc.returncode,
            "elapsed_s": elapsed,
            "log": str(log_path),
        }
    )
    if proc.returncode != 0:
        summary["error"] = "\n".join(text.splitlines()[-20:])
    (work / "polyfem_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"POLYFEM fig10 rc={proc.returncode} init={summary.get('init')} "
        f"best={summary.get('best')} final={summary.get('final')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def _rel_err(a, b) -> float | None:
    if a is None or b is None:
        return None
    denom = abs(b)
    if denom == 0:
        return abs(a - b)
    return abs(a - b) / denom


def _near_zero_agree(poly, uni) -> bool:
    if poly is None or uni is None:
        return False
    return abs(poly) < 1e-3 and abs(uni) < 1e-3


def compare(results: dict[str, dict]) -> list[dict]:
    rows = []
    for name, ref in STAGE61.items():
        poly = results.get(name, {})
        init_rel = _rel_err(poly.get("init"), ref["unified_init"])
        best_rel = _rel_err(poly.get("best"), ref["unified_best"])
        if name == "fig10":
            consistent = (
                poly.get("best") is not None
                and poly["best"] < 80
                and ref["unified_best"] < 80
            )
            verdict = "最佳 raw/loss 同量级" if consistent else "不一致或未完成"
        else:
            consistent = _near_zero_agree(poly.get("best"), ref["unified_best"])
            verdict = "终值都接近 0" if consistent else "终值未对齐"
        row = {
            "case": name,
            "polyfem_init": poly.get("init"),
            "unified_init": ref["unified_init"],
            "init_rel_err": init_rel,
            "polyfem_best": poly.get("best"),
            "unified_best": ref["unified_best"],
            "best_rel_err": best_rel,
            "polyfem_final": poly.get("final"),
            "unified_final": ref["unified_final"],
            "polyfem_elapsed_s": poly.get("elapsed_s"),
            "exit_code": poly.get("exit_code"),
            "consistent": consistent,
            "verdict": verdict,
            "stage61": ref["note"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "stage61_reference.json").write_text(json.dumps(STAGE61, indent=2) + "\n")
    order = ["fig18", "fig1", "fig21", "fig15", "fig10"]
    results: dict[str, dict] = {}
    partial_path = OUT_ROOT / "partial_results.json"
    if partial_path.exists():
        try:
            results = json.loads(partial_path.read_text())
        except json.JSONDecodeError:
            results = {}
    for name in order:
        existing = OUT_ROOT / name / "polyfem_result.json"
        if name in results and existing.exists() and results[name].get("exit_code") == 0:
            print(f"=== skip finished {name} ===", flush=True)
            continue
        print(f"=== start {name} ===", flush=True)
        if name == "fig10":
            results[name] = run_fig10()
        else:
            results[name] = run_json_case(name)
        partial_path.write_text(json.dumps(results, indent=2) + "\n")
    rows = compare(results)
    (OUT_ROOT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {OUT_ROOT / 'comparison.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
