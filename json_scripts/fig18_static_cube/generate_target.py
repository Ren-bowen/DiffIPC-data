#!/usr/bin/env python3
"""Generate Fig.18 +X-face markers from a PolyFEM target solve.

This is the PolyFEM counterpart of Unified ``make_target``: every official
opt / init-loss / profile launch must call this first so ``data.txt``
matches the current ``target.json``. ``node-target`` can only read that
file; the PolyFEM kernel has no live ``target_state`` for these 24 markers.

Matches Unified ``4_fig18_cube_material_parameter``:

* target material ``E=1e6``, ``nu=0.45``
* same 10-step dynamic solve as ``run.json``, with full Z-cap
  Dirichlet applied from the first step (interpolation ``none``)
* self-contact enabled (physical dhat ``relative_dhat * bbox``)
* the same 24 rest-frame +X markers, written as ``rest_xyz def_xyz``
  for PolyFEM ``node-target``

``--skip-solve`` only rebuilds markers from an existing last frame.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np


HERE = Path(__file__).resolve().parent
POLYFEM_BIN = Path(os.environ.get("POLYFEM_BIN", "/home/bowen/polyfem/build/PolyFEM_bin"))
TARGET_JSON = HERE / "target.json"
SOLVE_JSON = HERE / "target.generate.json"
RESULT_DIR = HERE / "result-target"
LAST_FRAME = RESULT_DIR / f"step_{int(json.loads(TARGET_JSON.read_text())['time']['time_steps'])}.vtu"
DATA_TXT = HERE / "data.txt"

# Unified rest-frame markers after scale 0.004 and translation -0.02.
TARGET_MARKERS = np.array(
    [
        [0.02, -0.020, -0.020],
        [0.02, -0.012, -0.020],
        [0.02, 0.000, -0.020],
        [0.02, 0.020, -0.020],
        [0.02, -0.020, -0.012],
        [0.02, -0.010, -0.010],
        [0.02, 0.000, -0.008],
        [0.02, 0.010, -0.010],
        [0.02, 0.020, -0.012],
        [0.02, -0.020, 0.000],
        [0.02, -0.008, 0.000],
        [0.02, 0.000, 0.000],
        [0.02, 0.008, 0.000],
        [0.02, 0.020, 0.000],
        [0.02, -0.020, 0.012],
        [0.02, -0.010, 0.010],
        [0.02, 0.000, 0.008],
        [0.02, 0.010, 0.010],
        [0.02, 0.020, 0.012],
        [0.02, -0.020, 0.020],
        [0.02, -0.012, 0.020],
        [0.02, 0.000, 0.020],
        [0.02, 0.012, 0.020],
        [0.02, 0.020, 0.020],
    ],
    dtype=np.float64,
)


def _write_solve_json() -> Path:
    data = json.loads(TARGET_JSON.read_text())
    output = data.setdefault("output", {})
    output["directory"] = str(RESULT_DIR)
    paraview = output.setdefault("paraview", {})
    paraview["volume"] = True
    paraview["surface"] = False
    paraview["skip_frame"] = 1
    output.setdefault("advanced", {})["save_time_sequence"] = True
    SOLVE_JSON.write_text(json.dumps(data, indent=4) + "\n")
    return SOLVE_JSON


def run_target(skip_solve: bool) -> None:
    if skip_solve and LAST_FRAME.is_file():
        print(f"Reusing existing {LAST_FRAME}", flush=True)
        return
    if not POLYFEM_BIN.is_file():
        raise FileNotFoundError(f"PolyFEM_bin not found: {POLYFEM_BIN}")
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    solve_json = _write_solve_json()
    cmd = [str(POLYFEM_BIN), "-j", str(solve_json), "-o", str(RESULT_DIR), "--max_threads", "32"]
    print("Running:", " ".join(cmd), flush=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "32"
    env["MKL_DYNAMIC"] = "FALSE"
    subprocess.run(cmd, cwd=HERE, env=env, check=True)


def match_markers(rest_pts: np.ndarray, markers: np.ndarray) -> np.ndarray:
    ids = []
    used: set[int] = set()
    for marker in markers:
        dist = np.linalg.norm(rest_pts - marker[None, :], axis=1)
        for candidate in np.argsort(dist):
            idx = int(candidate)
            if idx in used:
                continue
            if dist[idx] > 1e-10:
                raise RuntimeError(f"marker {marker} not on mesh; nearest dist={dist[idx]}")
            ids.append(idx)
            used.add(idx)
            break
    return np.asarray(ids, dtype=np.int64)


def write_data_txt() -> Path:
    if not LAST_FRAME.is_file():
        raise FileNotFoundError(f"missing last target frame: {LAST_FRAME}")

    deformed = meshio.read(LAST_FRAME)
    rest_vis = deformed.points.astype(np.float64)
    if "displacement" not in deformed.point_data:
        raise RuntimeError(f"{LAST_FRAME} has no displacement field")
    disp = np.asarray(deformed.point_data["displacement"], dtype=np.float64)
    ids = match_markers(rest_vis, TARGET_MARKERS)

    rows = []
    for marker, i in zip(TARGET_MARKERS, ids):
        def_xyz = marker + disp[i]
        rows.append(
            f"{marker[0]:.16e} {marker[1]:.16e} {marker[2]:.16e} "
            f"{def_xyz[0]:.16e} {def_xyz[1]:.16e} {def_xyz[2]:.16e}"
        )
    DATA_TXT.write_text("\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} markers to {DATA_TXT}", flush=True)
    return DATA_TXT


def ensure_target(*, skip_solve: bool = False) -> Path:
    run_target(skip_solve)
    return write_data_txt()


def main() -> int:
    ensure_target(skip_solve="--skip-solve" in sys.argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
