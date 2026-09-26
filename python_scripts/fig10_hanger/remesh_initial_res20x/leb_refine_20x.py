#!/usr/bin/env python3
"""Conforming longest-edge bisection of the 5x hanger mesh to ~20x original tets."""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from pathlib import Path

import meshio
import numpy as np

SRC = Path("/home/bowen/Unified_GIPC/Assets/diff_sim/fig10_hanger/middle_cleaned_res5x.msh")
DEST = Path("/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/middle_cleaned_res20x.msh")
OUT_DIR = Path("/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/remesh_initial_res20x")
ORIG_TETS = 7593
ORIG_VERTS = 2080
TARGET_TETS = int(np.ceil(20.0 * ORIG_TETS))


def signed_volumes(pts: np.ndarray, tet: np.ndarray) -> np.ndarray:
    p = pts[tet]
    return np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    ) / 6.0


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def tet_edges(t: list[int]) -> tuple[tuple[int, int], ...]:
    a, b, c, d = t
    return (
        edge_key(a, b),
        edge_key(a, c),
        edge_key(a, d),
        edge_key(b, c),
        edge_key(b, d),
        edge_key(c, d),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = meshio.read(str(SRC))
    pts0 = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
    tets0 = np.asarray(mesh.cells_dict["tetra"], dtype=np.int64).reshape(-1, 4)
    vol0 = signed_volumes(pts0, tets0)
    if np.any(vol0 < 0):
        flip = vol0 < 0
        tets0[flip] = tets0[flip][:, [0, 2, 1, 3]]
        print(f"flipped {int(flip.sum())} inverted input tets", flush=True)
    if np.any(signed_volumes(pts0, tets0) <= 0):
        raise RuntimeError("input has non-positive volumes")

    points = [tuple(map(float, p)) for p in pts0]
    tets = [list(map(int, t)) for t in tets0]
    edge_to_tets: dict[tuple[int, int], set[int]] = defaultdict(set)
    heap: list[tuple[float, int, int]] = []

    def length2(e: tuple[int, int]) -> float:
        pa, pb = points[e[0]], points[e[1]]
        return (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2

    def push_edge(e: tuple[int, int]) -> None:
        heapq.heappush(heap, (-length2(e), e[0], e[1]))

    for i, t in enumerate(tets):
        for e in tet_edges(t):
            if i not in edge_to_tets[e]:
                if len(edge_to_tets[e]) == 0:
                    push_edge(e)
                edge_to_tets[e].add(i)

    print(
        f"input verts={len(points)} tets={len(tets)} target_tets={TARGET_TETS}",
        flush=True,
    )
    n_split = 0
    while len(tets) < TARGET_TETS:
        while True:
            if not heap:
                raise RuntimeError("heap emptied before reaching target")
            neg_l2, a, b = heapq.heappop(heap)
            e = edge_key(a, b)
            if e in edge_to_tets:
                break
        mid = len(points)
        pa, pb = points[a], points[b]
        points.append(((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5))
        incident = sorted(edge_to_tets.pop(e))
        new_edges: set[tuple[int, int]] = set()
        for ti in incident:
            t = tets[ti]
            others = [v for v in t if v != a and v != b]
            if len(others) != 2:
                raise RuntimeError(f"edge {e} not in tet {t}")
            c, d = others
            new1 = [a, mid, c, d]
            new2 = [mid, b, c, d]
            for old in tet_edges(t):
                ids = edge_to_tets.get(old)
                if ids is None:
                    continue
                ids.discard(ti)
                if not ids:
                    del edge_to_tets[old]
            tets[ti] = new1
            new_i = len(tets)
            tets.append(new2)
            for ne in tet_edges(new1):
                edge_to_tets[ne].add(ti)
                new_edges.add(ne)
            for ne in tet_edges(new2):
                edge_to_tets[ne].add(new_i)
                new_edges.add(ne)
        for ne in new_edges:
            push_edge(ne)
        n_split += 1
        if n_split % 5000 == 0 or len(tets) >= TARGET_TETS:
            print(
                f"splits={n_split} verts={len(points)} tets={len(tets)} "
                f"longest={(-neg_l2) ** 0.5:.5f}",
                flush=True,
            )

    pts = np.asarray(points, dtype=np.float64)
    tet = np.asarray(tets, dtype=np.int64)
    vol = signed_volumes(pts, tet)
    if np.any(vol < 0):
        flip = vol < 0
        tet[flip] = tet[flip][:, [0, 2, 1, 3]]
        vol = signed_volumes(pts, tet)
        print(f"flipped {int(flip.sum())} after refine", flush=True)
    if np.any(vol <= 0):
        raise RuntimeError(f"non-positive volumes remain: {int((vol <= 0).sum())}")

    p = pts[tet]
    den = (
        np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        * np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
        * np.linalg.norm(p[:, 3] - p[:, 0], axis=1)
    )
    q = np.abs(
        np.linalg.det(np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=-1))
    ) / np.maximum(den, 1.0e-30)
    quality = {
        "min": float(q.min()),
        "p01": float(np.quantile(q, 0.01)),
        "p05": float(np.quantile(q, 0.05)),
        "median": float(np.median(q)),
    }
    print("quality", quality, flush=True)
    meshio.write(
        str(DEST),
        meshio.Mesh(points=pts, cells=[("tetra", tet)]),
        file_format="gmsh22",
        binary=False,
    )
    meta = {
        "method": "conforming_longest_edge_bisection",
        "source": str(SRC),
        "original_source": "/home/bowen/DiffIPC-data-original/python_scripts/fig10_hanger/middle_cleaned.mesh",
        "output": str(DEST),
        "why_not_ftetwild": (
            "fTetWild --la 0.02229 from original/5x surface splits to ~3.8e6 "
            "intermediate tets; aborted to avoid another freeze"
        ),
        "resolution_factor_elements": 20.0,
        "input_vertices": 8172,
        "input_tets": 35843,
        "output_vertices": int(len(pts)),
        "output_tets": int(len(tet)),
        "splits": n_split,
        "output_quality": quality,
        "tet_ratio_vs_5x": len(tet) / 35843.0,
        "vertex_ratio_vs_5x": len(pts) / 8172.0,
        "tet_ratio_vs_original": len(tet) / ORIG_TETS,
        "vertex_ratio_vs_original": len(pts) / ORIG_VERTS,
        "min_signed_volume": float(vol.min()),
    }
    (OUT_DIR / "remesh_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", DEST, "bytes", DEST.stat().st_size, flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
