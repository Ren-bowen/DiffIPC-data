#!/usr/bin/env python3
"""Official Fig.18 PolyFEM entry: regenerate the target, then optimize.

``opt.json`` ``node-target`` still reads ``data.txt``. This script always
solves the current ``target.json`` first, matching Unified ``make_target``.
Do not invoke ``PolyFEM_bin -j opt.json`` alone; that reuses a stale file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
POLYFEM_BIN = Path(os.environ.get("POLYFEM_BIN", "/home/bowen/polyfem/build/PolyFEM_bin"))
PYTHON = Path(
    os.environ.get("ISAAC_PYTHON", "/home/bowen/miniconda3/envs/env_isaaclab/bin/python")
)


def regenerate_target() -> None:
    subprocess.run(
        [str(PYTHON), str(HERE / "generate_target.py")],
        cwd=HERE,
        check=True,
    )


def run_opt() -> None:
    if not POLYFEM_BIN.is_file():
        raise FileNotFoundError(f"PolyFEM_bin not found: {POLYFEM_BIN}")
    subprocess.run(
        [
            str(POLYFEM_BIN),
            "-j",
            str(HERE / "opt.json"),
            "--ns",
            "-o",
            str(HERE / "result"),
        ],
        cwd=HERE,
        check=True,
    )


def main() -> int:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "32"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    regenerate_target()
    run_opt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
