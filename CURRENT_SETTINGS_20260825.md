# Current DiffIPC settings snapshot

Snapshot date: 2026-08-25. This file records the DiffIPC settings that are
currently used by the five-Fig comparison and the updated Fig.10 script. The
JSON files and source files listed below are the authoritative settings.

## Fig.1

Configuration files:

- `json_scripts/fig1_bunnies/run.json`
- `json_scripts/fig1_bunnies/opt.json`

The scene is aligned with the current Unified new-diff-sim Fig.1 entry point:
the target bunny translation is `[1, 0, 0]` and the initial velocity is
`[1, 0.5, 0]`. The dynamic solve uses BDF1 with `dt=0.05`, 40 frames, and
physical time 2.0. The material is Stable Neo-Hookean with
`E=1e6`, `nu=0.48`, and `rho=1240`; contact uses `dHat=1e-3`, friction `0`,
and barrier stiffness `1e5`.

The outer optimizer is L-BFGS-B with 20 iterations, history size 6, maximum
parameter change `0.5`, bounds `[-5, 5]`, no external line search, and gradient
tolerance `1e-3`. The latest run is in
`optimization_runs/20260825_unified_new_diff_sim_fig1/fig1/`. It saved all 20
outer updates. PolyFEM returned `134` only after saving
`opt_state_0_iter_20` because of an iteration-limit exception; the final loss
is `0.582157`, and the best saved loss is `0.0938361` at zero-based step 3.
The measured wall time is `2238.227 s`.

## Fig.10

Configuration and implementation files:

- `python_scripts/fig10_hanger/run.json`
- `python_scripts/fig10_hanger/optimize.py`
- `python_scripts/src/tet_remesh.py`
- `python_scripts/fig10_hanger/middle_cleaned.mesh`

The saved run uses a dynamic 20-frame Implicit Euler solve with `dt=0.02`,
Stable Neo-Hookean material `E=1e9`, `nu=0.49`, `rho=1000`, contact `dHat=3e-4`,
and 16 threads. The optimizer defaults are 100 iterations, learning rate
`5e-3`, normalized direct gradient descent, stress power 2, and Laplacian
weight 1.0. The trust-region optimizer is opt-in. Every volume remesh uses
fTetWild with `--la 0.05 --no-binary`; the saved 100-step result completed 14
remesh operations.

The selected result is
`optimization_runs/20260825_dynamic_unified_remesh/fig10_exact_unified_loss/`.
Its loss changed from `240.49657136769122` to `14544543.324701158`; divergence
is retained because this is the requested 100-step result. The final mesh has
5029 vertices and 19745 tetrahedra and has no negative-volume elements.

The selected run directory does not contain an exact wall-clock timer. The
Fig.10 DiffIPC phase values in `timing_tables.tex` therefore use the earlier
36-record profiler log at
`optimization_runs/20260825_original_settings/fig10/run.log`; this is stated
explicitly in the table footnote and is not presented as the 100-step total
runtime.

## Other saved comparison settings

The current edits to the comparison settings are also preserved in:

- `json_scripts/fig15_tentacles/opt.json`
- `json_scripts/fig15_tentacles/state.json`
- `json_scripts/fig15_tentacles/state-target.json`
- `json_scripts/fig18_static_cube/opt.json`
- `json_scripts/fig18_static_cube/run.json`
- `json_scripts/fig18_static_cube/data.txt`
- `json_scripts/fig21_friction_bunny/run-new.json`
- `json_scripts/fig21_friction_bunny/target.json`

## Timing documents

- `optimization_runs/20260825_original_settings/five_fig_results.md` contains
  the updated five-Fig result summary.
- `timing_tables.tex` is a standalone, portrait-orientation LaTeX document
  that compiles directly in Overleaf. It contains the complete nine-case
  Unified GIPC table and the five-Fig two-sided phase comparison, including
  the Unified GIPC speedup for every comparable item.
- The Unified timing source is
  `/home/bowen/Unified_GIPC_new_diff/agent_check/diff_sim/stage61_nine_main_timing_profile.md`.
