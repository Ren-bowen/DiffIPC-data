# Sphere shape optimization — reproduction package

End-to-end shape-optimization of a 3D sphere falling onto an inclined plank
under IPC contact, minimizing time-soft-max of L⁸ stress over the sphere
volume while keeping the volume above a soft bound and smoothing the
boundary. **Frictionless** transient contact (μ = 0), using the *fixed*
transient adjoint (the polyfem `[!shouldfail]` regime had a 10⁵× gradient
bug; this run validates the fix).

## What this folder contains

### Inputs (everything needed to start a run)
- `sphere600_clean.msh` — starting sphere (unit ball, will be scaled 0.5×
  and translated to `(0, 0.5, -0.2)` by `run-shape.json`)
- `plank.msh` — fixed obstacle (rotated, scaled 2× by `run-shape.json`)
- `run-shape.json` — forward-simulation config (P2, μ=0, κ=1e8,
  use_convergent_formulation, 12 time steps × dt=0.02, BDF1, gravity 9.81,
  PardisoLLT-first linear solver chain)
- `opt-shape.json` — optimization config (3 functionals: stress (1e-5),
  volume soft-constraint (1e4), boundary smoothing (0.5); L-BFGS,
  `max_iterations=20`, Backtracking line search with
  `default_init_step_size=0.5` and `min_step_size=1e-4`)
- `adaptive_smoothing.py` — chunked driver: probe per-term gradients →
  L-BFGS for `PERIOD=20` iters → check mesh quality → wmtk-remesh sphere
  interior if needed → swap mesh back into `run-shape.json` → repeat for
  `TOTAL_CHUNKS=10` (= 200 total L-BFGS iters)

### `reference/` — outputs from one prior run (for diffing / validation only, not inputs)
- `metrics_final_reference.csv` — per-chunk metrics table
- `run_reference.log` — driver stdout from the original run
- `final_sphere_chunk010.msh` — optimized sphere after chunk 10
- `final_sphere_chunk010_remeshed.msh` — same, post wmtk retet

## Dependencies (external binaries used by the driver)
The driver shells out to two external binaries, configured via the
`POLYFEM` and `WMTK_BIN` constants near the top of `adaptive_smoothing.py`.
Set them to the locations of:
- **PolyFEM** — a build of polyfem with the transient frictionless shape
  adjoint fix applied (see "Required polyfem state" below).
- **wmtk_interior_tetopt_bin** — the interior-tet-optimization app from the
  Wildmeshing Toolkit, https://github.com/wildmeshing/wildmeshing-toolkit
  (the `app/interior_tet_opt` target). The driver calls it to retet the
  sphere interior between L-BFGS chunks while holding the boundary fixed.

## Run
```
python3 adaptive_smoothing.py 2>&1 | tee run.log
```
- Wall-clock: ~3.5 h on a 16-thread machine.
- Per-chunk: ~20 min (probe ≈ 30 s + 20 L-BFGS forward+adjoint passes ≈ 19 min).
- Output: `metrics.csv` (live table) and stdout.

## Expected results (matching `metrics_final_reference.csv`)
Initial state (chunk 0): maxσ ≈ 9.04e+04, L⁸σ ≈ 4.17e+04, weighted stress ≈
0.156, weighted smooth ≈ 0.0126.

| chunk | maxσ | L⁸σ | stress | smooth | sj_tet_post |
|------:|-----:|----:|-------:|-------:|------------:|
| 1 | 1.33e+04 | 6.40e+03 | 0.0603 | 0.0259 | 0.16 |
| 2 | 2.11e+04 | 7.76e+03 | 0.0510 | 0.0199 | 0.10 |
| 3 | 1.06e+04 | 4.29e+03 | 0.0465 | 0.0177 | 0.24 |
| 4 | 6.59e+03 | 2.44e+03 | 0.0423 | 0.0155 | 0.20 |
| 5 | 6.93e+03 | 2.65e+03 | 0.0415 | 0.0131 | 0.18 |
| 6 | 5.79e+03 | 2.40e+03 | 0.0404 | 0.0119 | 0.20 |
| 7 | 6.75e+03 | 2.50e+03 | 0.0394 | 0.0113 | 0.20 |
| 8 | 6.45e+03 | 2.57e+03 | 0.0388 | 0.0103 | 0.19 |
| 9 | 7.35e+03 | 2.56e+03 | 0.0382 | 0.0096 | 0.19 |
| 10 | 6.76e+03 | 2.47e+03 | 0.0377 | 0.0091 | 0.21 |

Cumulative reduction: **maxσ −92.5 %, L⁸σ −94 %, weighted stress −76 %,
‖∇str‖ −94 %**. Chunk 1 alone takes maxσ from 9.0e+04 → 1.3e+04 (−85 %)
because the L-BFGS step is no longer artificially clamped (we use
`default_init_step_size = 0.5` rather than the very conservative 0.0625).

Polyfem is non-deterministic across threads (per the upstream test_diff
preamble: "non-associative floating point reduction"), so your numbers may
differ slightly. Trends should match.

## Inspecting the final shape (Paraview)
After a run, the per-iter rest mesh + stress field overlay is in
`run1_sim/opt_state_0_iter_N.vtu`. The deformed-state outputs (used to
compute the stress diagnostics in the metrics table) are in
`run1_sim/step_T.vtu` for forward time step T ∈ [0, 12]. Open the latest
`opt_state_0_iter_*.vtu` to see the optimized sphere with `cauchy_stess_*`
and `von_mises` fields.

## Tuning knobs to play with
- `adaptive_smoothing.py`:
  - `PERIOD` — L-BFGS iters per chunk (higher = more quasi-Newton history,
    but a remesh at chunk boundary still resets the (s, y) history because
    each chunk is a separate polyfem subprocess).
  - `TOTAL_CHUNKS` — outer iterations.
  - `REMESH_PERIOD_ITERS` (default 10) — force a wmtk retet every N
    cumulative iters even if `sj_tet ≥ REMESH_TOLERANCE`.
  - `REBALANCE_SMOOTHING` (False) — re-enable the legacy ShapeProblem
    boundary-smoothing weight rebalance.
- `opt-shape.json`:
  - `solver.nonlinear.line_search.default_init_step_size` — start step for
    Backtracking. 0.5 is the value used here; the standard L-BFGS default
    is 1.0 (slightly more aggressive, may produce more element flips at
    iter 0 of each chunk).
  - Per-functional `weight` — the stress / volume / smoothing weights here
    are what `run-new.json` (legacy) had.
