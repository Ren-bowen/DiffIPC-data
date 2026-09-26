# Fig10 with the Ren-bowen PolyFEM checkout

> This file also preserves dated validation notes. The current `optimize.py`
> defaults to the `ren-bowen` binding and `unified-linf` Newton stopping;
> `run.json` explicitly enables forward PSD projection. Earlier defaults
> described below apply only to their dated experiments.

Fig10 defaults to `/home/bowen/polyfem`, whose origin is
`git@github.com:Ren-bowen/polyfem.git`. It does not fetch or reset that checkout:
local uncommitted core changes are included. The module reports `build_info`
(source directory, configure-time revision/dirty marker, and adapter API).

The adapter in `polyfem-python/src/modern/fig10_binding.cpp` implements the APIs
used by Fig10, not the entire historical polyfempy API. It calls the same
`legacy::State`, `DiffCache::cache_transient`, `solve_adjoint_cached`, and
`AdjointTools` implementations used by the modern command-line optimization
backend. `legacy::State` is a namespace inside the current Ren-bowen core;
it does not refer to the separate `polyfem-pinned` checkout.

The Medit importer lacks complete input edge/face ordering metadata. The core
now reconstructs P1 input-vertex-to-node ordering from the basis map in that
case, and guards empty maps. Fig10 has 2078 non-identity entries among 2080
vertices. The adapter verifies the permutation and BC map, returns cached
displacements in vertex order, and converts incoming adjoint RHSs to node
order. Native shape derivatives already return vertex order and are not
permuted a second time. This corrects the old Python path's ordering assumption;
unchanged loss formulas do not imply unchanged historical loss values.

## Build

The optional `POLYFEM_PYTHON_BINDINGS_SOURCE_DIR` CMake setting adds the binding
to the existing core build and links the actual `polyfem` target. The module
output is separate from both the pinned build and the installed conda package.
The current machine's vcpkg wrapper also needs explicit Python3 internal hints
to avoid mixing its Python 3.12 headers with conda Python 3.11.

```sh
cmake -S /home/bowen/polyfem -B /home/bowen/polyfem/build \
  -DPOLYFEM_PYTHON_BINDINGS_SOURCE_DIR=/home/bowen/polyfem-python \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/bowen/polyfem-python/build-renbowen/polyfempy \
  -DPython_EXECUTABLE=/home/bowen/miniconda3/envs/env_isaaclab/bin/python \
  -DPYTHON_EXECUTABLE=/home/bowen/miniconda3/envs/env_isaaclab/bin/python \
  -DPython3_EXECUTABLE=/home/bowen/miniconda3/envs/env_isaaclab/bin/python \
  -DPython3_INCLUDE_DIR=/home/bowen/miniconda3/envs/env_isaaclab/include/python3.11 \
  -DPython3_LIBRARY=/home/bowen/miniconda3/envs/env_isaaclab/lib/libpython3.11.so \
  -D_Python3_INCLUDE_DIR=/home/bowen/miniconda3/envs/env_isaaclab/include/python3.11 \
  -D_Python3_LIBRARY_RELEASE=/home/bowen/miniconda3/envs/env_isaaclab/lib/libpython3.11.so \
  -D_Python3_LIBRARY_DEBUG=/home/bowen/miniconda3/envs/env_isaaclab/lib/libpython3.11.so
cmake --build /home/bowen/polyfem/build --target polyfempy -j 2
```

This updates the existing core build's embedded-Python configuration to 3.11.
It does not install over the conda package. The old pinned build is retained;
when reconfiguring it, explicitly pass
`-DPOLYFEMPY_POLYFEM_SOURCE_DIR=/home/bowen/polyfem-pinned` because standalone
binding builds now default to the Ren-bowen source.

## Run one forward and backward

```sh
/home/bowen/miniconda3/envs/env_isaaclab/bin/python \
  /home/bowen/DiffIPC-data-original/json_scripts/profile_initial_pass.py \
  --out /absolute/path/to/new-results \
  --cases fig10 --threads 4 --fig10-backend ren-bowen \
  --fig10-newton-stopping unified-linf
```

This performs one 20-frame forward and one backward, with no parameter update
or trial step. Fig10 explicitly keeps `Eigen::SimplicialLDLT` for both solves,
even though the new core also supports MKL/Pardiso. The modern Newton settings
use `norm_type=Linf`, `x_delta_tol=1e-3 * 0.02`, and disable the other tolerances.
Native `NLProblem` multiplies this by the bbox diagonal once, giving
`5.4188686482758236e-05` for the initial hanger. Solver limits still guard failure.
The loss definition is unchanged by the adapter migration.

Default Newton mode remains `gradient`; explicitly request `unified-linf` for
the aligned benchmark. For historical reproduction use `--fig10-backend pinned`
and leave `--fig10-newton-stopping gradient`; the runner adds the old schema
flag. Direct `optimize.py` calls use `--polyfem-backend pinned
--legacy-polysolve-schema`. Pinned Linf mode additionally requires the earlier
native patch. No paper CSV or historical result is overwritten automatically.

## Validation, 2026-09-07

The successful one-pass run is recorded in
`/home/bowen/DiffIPC-data-original/initial_profile_fig10_renbowen_20260907_6QALGt/mapped/fig10/result.json`.
It completed all 20 frames, one backward, and zero updates, with finite loss
102.00941131046018 and gradient norm 443.5962474472942. All 20 native stopping
records use the displacement criterion and report tolerance 5.41887e-05
(rounded by the logger). Forward took 6.1197 s and backward 36.1077 s.
The six native phase times are in the associated `timing.csv`; they do not
include all wall-clock overhead. Other rendering work was present during this
session, so this is functional validation, not a certified idle timing run.

The parent directory retains a pre-time-step segfault record and its GDB
backtrace; `verified/` retains a second preflight rejected by the permutation
guard. Neither is a successful simulation or part of the reported timing.
Nine Python configuration/backend-selection tests pass. Finite-difference
gradient accuracy and complete optimization trajectories were not retested.

## Default Newton strategy retest, 2026-09-07

At the user's request, `run.json` now uses `"Newton": {}`. PolySolve injects
its defaults: ordinary Newton first, projected Newton as fallback, then
projected regularized Newton (initial weight 1e-8), and gradient descent.
This removes the previous forced projected-regularized strategy with weight
0.1. The change is to Fig10's strategy only, not the library-wide defaults.
Newton stopping mode is a separate option: the aligned test still explicitly
uses `--fig10-newton-stopping unified-linf`.

The single-pass retest is in
`/home/bowen/DiffIPC-data-original/initial_profile_fig10_default_newton_20260907_yCJQ6E/`.
It preserves `previous_run.json` and `tested_run.json`; they differ only in
the Newton strategy object. All 20 frames started and finished with
`SparseNewton`; no strategy fallback occurred. There were 22 forward linear
solves, 20 backward solves, two Newton position updates (one in each of the
first two frames), one forward/backward evaluation and no optimization update.

Six phase totals, seconds: forward Linear 0.5578, CCD 0.046281, Hessian 1.3995,
line search 0.0473; backward Hessian 0.011434, Linear 0.5399. Complete Python
forward/backward times were 6.215402 / 35.419480 seconds. Six phases are not
an exhaustive decomposition of those totals. Timing is a single-run
observation, not a statistically established speedup.

Finite loss: 103.70890217660565; gradient norm: 437.693003936624. The first
frame's terminal direction norm was 3.31317e-7, below the unchanged 5.41887e-5
threshold. The result is not equal to the earlier regularized trajectory or
the Unified trajectory. Ten Python config/backend tests pass. No native
rebuild was needed, and historical profiles/paper tables were not replaced.

## Fig10 正式应力目标默认值（2026-09-15）

`optimize.py` 默认 `--stress-power 4 --stress-weight 5e-5`，目标为 `5e-5 * Σ V_e ||1e-4 P_e||_F^4 + L_lap`，不取四次根；平滑权重1，默认300步。权重显式传入初始评估、优化、线搜索与FD路径。旧p=2目标可用 `--stress-power 2 --stress-weight 1` 复现。此前50步配对使用权重5.5452218053470574e-5，保留原实验记录。
