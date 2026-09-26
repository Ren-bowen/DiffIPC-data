#!/usr/bin/env python3
"""Generate the Fig.1 moving-bunny target with a PolyFEM forward solve.

This script is the source of the Fig.1 target. It does not copy a Unified
centroid. It:

1. Runs ``target.json`` (same setup as ``run.json``, but the moving bunny
   starts at ``v0 = [3.6, 0.5, 0]``).
2. Reads the last frame and reports the yellow bunny (body 3) volume-
   weighted XZ centroid for inspection.
3. Leaves ``result-target/`` in place so ``opt.json`` can use
   ``center-target`` against that simulated state.

The optimizer itself recomputes the same target through ``target.json``;
the numbers written here are only a record.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np


HERE = Path(__file__).resolve().parent
POLYFEM_BIN = Path(os.environ.get("POLYFEM_BIN", "/home/bowen/polyfem/build/PolyFEM_bin"))
TARGET_JSON = HERE / "target.json"
RESULT_DIR = HERE / "result-target"
LAST_FRAME = RESULT_DIR / "step_40.vtu"
RECORD_JSON = HERE / "target_center.json"

# Unified generate_target() historically used these numbers as a hardcoded
# SoftBound. They are recorded only so the PolyFEM-generated centroid can
# be compared; they are not used by opt.json.
UNIFIED_HARDCODED_XZ = [4.1206334421, 0.1263607470]
YELLOW_BODY_ID = 3


def run_target(skip_solve: bool) -> None:
    if skip_solve and LAST_FRAME.is_file():
        print(f"Reusing existing {LAST_FRAME}", flush=True)
        return
    if not POLYFEM_BIN.is_file():
        raise FileNotFoundError(f"PolyFEM_bin not found: {POLYFEM_BIN}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(POLYFEM_BIN), "-j", str(TARGET_JSON), "--max_threads", "32"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True, env=dict(os.environ, OMP_NUM_THREADS="32", MKL_NUM_THREADS="32", OPENBLAS_NUM_THREADS="32", MKL_DYNAMIC="FALSE"))


def tet_volume_weighted_xz(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    v0, v1, v2, v3 = [points[cells[:, i]] for i in range(4)]
    vols = np.abs(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    centroids = 0.25 * (v0 + v1 + v2 + v3)
    return (vols[:, None] * centroids).sum(axis=0)[:3:2] / vols.sum()


def yellow_bunny_xz_from_vtu(mesh: meshio.Mesh) -> np.ndarray:
    body = mesh.point_data["body_ids"].reshape(-1)
    points = mesh.points.astype(np.float64)
    # Some PolyFEM exports keep rest coordinates in `points` and store the
    # motion in `displacement` / `solution`.
    if "displacement" in mesh.point_data:
        disp = np.asarray(mesh.point_data["displacement"], dtype=np.float64)
        if np.linalg.norm(disp) > 0:
            points = points + disp
    yellow = points[body == YELLOW_BODY_ID]
    if yellow.shape[0] % 4 != 0:
        raise RuntimeError(f"expected 4 visualization points per tet, got {yellow.shape[0]}")
    tets = np.arange(yellow.shape[0], dtype=np.int64).reshape(-1, 4)
    return tet_volume_weighted_xz(yellow, tets)


def main() -> int:
    run_target(skip_solve="--skip-solve" in sys.argv)
    if not LAST_FRAME.is_file():
        raise FileNotFoundError(f"missing last target frame: {LAST_FRAME}")

    mesh = meshio.read(LAST_FRAME)
    xz = yellow_bunny_xz_from_vtu(mesh)
    record = {
        "source": "polyfem_forward_target_json",
        "target_json": "target.json",
        "last_frame": "result-target/step_40.vtu",
        "yellow_body_id": YELLOW_BODY_ID,
        "xz_centroid": [float(xz[0]), float(xz[1])],
        "unified_hardcoded_xz_not_used": UNIFIED_HARDCODED_XZ,
        "delta_vs_unified_hardcoded": [
            float(xz[0] - UNIFIED_HARDCODED_XZ[0]),
            float(xz[1] - UNIFIED_HARDCODED_XZ[1]),
        ],
    }
    RECORD_JSON.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Wrote {RECORD_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
