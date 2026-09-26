#!/usr/bin/env python3
"""Safe 20x hanger remesh: original mesh, 4 threads, abort if #t > 300k."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/bowen/DiffIPC-data-original/python_scripts/src")
from tet_remesh import _outward_boundary_faces, _read_tet_mesh, _resolve_binary, _write_surface_obj

SRC = Path("/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/middle_cleaned.mesh")
OUT_DIR = Path("/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/remesh_initial_res20x")
DEST = Path("/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/middle_cleaned_res20x.msh")
ORIGINAL_MEDIAN_EDGE = 0.06051315496768646
RESOLUTION_FACTOR = 20.0
TARGET_LA = ORIGINAL_MEDIAN_EDGE / (RESOLUTION_FACTOR ** (1.0 / 3.0))
MAX_THREADS = 4
MAX_TETS = 300_000
MAX_RSS_GB = 20.0
TET_RE = re.compile(r"#t\s*=\s*(\d+)")


def read_medit(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    i = 0
    verts = tets = None
    while i < len(lines):
        tag = lines[i].strip()
        if tag == "Vertices":
            n = int(lines[i + 1])
            i += 2
            verts = np.array(
                [list(map(float, lines[i + j].split()[:3])) for j in range(n)],
                dtype=np.float64,
            )
            i += n
            continue
        if tag in ("Tetrahedra", "Tetrahedron"):
            n = int(lines[i + 1])
            i += 2
            tets = np.array(
                [list(map(int, lines[i + j].split()[:4])) for j in range(n)],
                dtype=np.int64,
            )
            if tets.min() >= 1:
                tets -= 1
            i += n
            continue
        i += 1
    if verts is None or tets is None:
        raise RuntimeError(f"failed to parse {path}")
    return verts, tets


def rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return 0.0
    return 0.0


def kill_tree(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verts, tets = read_medit(SRC)
    before_obj = OUT_DIR / "ftetwild_before_r000_try1.obj"
    after_msh = OUT_DIR / "ftetwild_after_r000_try1.msh"
    log_path = OUT_DIR / "ftetwild_r000_try1.log"
    after_msh.unlink(missing_ok=True)
    faces = _outward_boundary_faces(verts, tets)
    _write_surface_obj(before_obj, verts, faces)
    cmd = [
        _resolve_binary(),
        "-i",
        str(before_obj),
        "-o",
        str(after_msh),
        "--la",
        f"{TARGET_LA:.8e}",
        "--no-binary",
        "--max-threads",
        str(MAX_THREADS),
    ]
    print(
        f"input verts={len(verts)} tets={len(tets)} target_la={TARGET_LA:.8e} "
        f"threads={MAX_THREADS} abort_tets>{MAX_TETS}",
        flush=True,
    )
    print("command", cmd, flush=True)
    env = os.environ.copy()
    env["FTETWILD_BIN"] = "/home/bowen/bin/fTetWild"
    env["TBB_NUM_THREADS"] = str(MAX_THREADS)
    env["OMP_NUM_THREADS"] = str(MAX_THREADS)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        peak_tets = 0
        abort_reason = None
        while proc.poll() is None:
            time.sleep(2.0)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            found = [int(x) for x in TET_RE.findall(text)]
            if found:
                peak_tets = max(peak_tets, max(found))
                latest = found[-1]
                print(f"watch latest_t={latest} peak_t={peak_tets} rss={rss_gb(proc.pid):.2f}GB", flush=True)
                if latest > MAX_TETS or peak_tets > MAX_TETS:
                    abort_reason = f"intermediate tets {max(latest, peak_tets)} exceeded {MAX_TETS}"
                    kill_tree(proc.pid)
                    break
            mem = rss_gb(proc.pid)
            if mem > MAX_RSS_GB:
                abort_reason = f"fTetWild RSS {mem:.2f}GB exceeded {MAX_RSS_GB}GB"
                kill_tree(proc.pid)
                break
        rc = proc.wait()
    if abort_reason:
        print(f"ABORTED {abort_reason}", flush=True)
        return 2
    if rc != 0 or not after_msh.is_file():
        print(f"FAILED rc={rc} after={after_msh.is_file()}", flush=True)
        return 1
    points, remeshed = _read_tet_mesh(after_msh)
    if len(remeshed) > MAX_TETS:
        print(f"ABORTED output tets {len(remeshed)} exceeded {MAX_TETS}", flush=True)
        return 2
    shutil.copy2(after_msh, DEST)
    meta = {
        "source": str(SRC),
        "output": str(DEST),
        "ftetwild_after": str(after_msh),
        "original_median_edge": ORIGINAL_MEDIAN_EDGE,
        "target_la": TARGET_LA,
        "resolution_factor_elements": RESOLUTION_FACTOR,
        "max_threads": MAX_THREADS,
        "max_tets_abort": MAX_TETS,
        "input_vertices": int(len(verts)),
        "input_tets": int(len(tets)),
        "output_vertices": int(len(points)),
        "output_tets": int(len(remeshed)),
        "peak_logged_tets": peak_tets,
        "tet_ratio": len(remeshed) / len(tets),
        "vertex_ratio": len(points) / len(verts),
        "command": cmd,
    }
    (OUT_DIR / "remesh_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", DEST, flush=True)
    print(json.dumps({k: v for k, v in meta.items() if k != "command"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
