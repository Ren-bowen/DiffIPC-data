# Current DiffIPC settings snapshot

Snapshot updated: 2026-09-03 (the historical filename is retained). This file records the DiffIPC settings that are
currently used by the five-Fig comparison and the updated Fig.10 script. The
JSON files and source files listed below are the authoritative settings.

## Fig.1

Configuration files:

- `json_scripts/fig1_bunnies/run.json`
- `json_scripts/fig1_bunnies/target.json`
- `json_scripts/fig1_bunnies/opt.json`
- `json_scripts/fig1_bunnies/generate_target.py`
- `json_scripts/fig1_bunnies/target_center.json`

The scene is aligned with the current numbered Unified Fig.1 entry point
`/home/bowen/Unified_GIPC/python_examples/diff_sim/1_fig1_bunny_init_velocity/main.py`.
The target bunny translation is `[1, 0, 0]`, the optimized initial velocity
starts from `[1, 0.5, 0]`, and the target rollout uses reference velocity
`[3, 0.5, 0]`. The target is no longer a hardcoded Unified centroid. Run
`generate_target.py` to solve `target.json` with PolyFEM; `opt.json` then
compares the design and target states through `center-target` on body 3,
XZ only, volume-normalized, last frame only. `target_center.json` records
the PolyFEM yellow-bunny XZ centroid
`[3.9743406348577177, 0.4190920289159835]`. This differs from the old
Unified hardcoded pair `[4.1206334421, 0.1263607470]` because the target
is now a PolyFEM rollout, not a copied Unified number. The dynamic solve
uses BDF1 with `dt=0.05`, 40 frames, and
physical time 2.0. The material is Stable Neo-Hookean with
`E=1e6`, `nu=0.48`, and `rho=1240`; contact uses `dHat=1e-3`, friction `0`,
and barrier stiffness `1e5`.

The outer optimizer is L-BFGS-B with 20 iterations, history size 6, maximum
parameter change `0.5`, bounds `[-5, 5]`, no external line search, and gradient
tolerance `1e-3`. The 2026-08-31 rerun in
`optimization_runs/20260831_unified_current_fig1_target/fig1/` still used the
hardcoded Unified centroid as `soft_bound`. It saved the
initial state plus all 20 outer updates. PolyFEM returned `134` only after
saving `opt_state_0_iter_20` because of its iteration-limit exception. The
loss changed from `4.9411173878` to `0.5700637509`; the best saved loss is
`0.3490623137` at zero-based iteration 7. The measured wall time is
`2252.84 s`. The final mesh has 2061 vertices and 7562 tetrahedra, all values
are finite, and it has no negative or zero-volume tetrahedra. See
`summary.json`, `loss_history.txt`, `run.log`, and `runtime.txt` in the run
directory.

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

## Fig.18

Configuration files:

- `json_scripts/fig18_static_cube/run.json`
- `json_scripts/fig18_static_cube/target.json`
- `json_scripts/fig18_static_cube/opt.json`
- `json_scripts/fig18_static_cube/data.txt`
- `json_scripts/fig18_static_cube/generate_target.py`

The material and loading now match the current numbered Unified Fig.18 entry
point
`/home/bowen/Unified_GIPC/python_examples/diff_sim/4_fig18_cube_material_parameter/main.py`:
initial `E=1e6`, `nu=0.15`, target `E=1e6`, `nu=0.4`, and `rho=1000`. The
final Z-cap displacements are `-0.00551` on `+z` and `+0.00646` on `-z`,
applied as a 19-step quasistatic linear ramp (`t0=0`, `tend=1`,
`time_steps=19`). `generate_target.py` solves `target.json` and writes the
same 24 Unified +X-face markers to `data.txt`.

The outer optimizer is PolyFEM ADAM with `alpha=0.15`, no line search, 20
iterations, and `allow_out_of_iterations`. Because `node-target` as a static
form would read time step 0, it is wrapped in `transient_integral` with
`integral_type=final`. Unified also clamps `log(lambda), log(mu)` to
`[4, 16]`; PolyFEM ADAM does not project onto that box, but `exp` still
keeps the Lamé parameters positive.

Unified Stable-NH v1 consumes rates `length=mu` and `volume=lambda+mu`, while
the legacy PolyFEM StableNeoHookean implementation uses different internal
rates. The existing `stable-nh-v1-to-legacy` composition maps the physical
Unified Lamé values before assigning PolyFEM element parameters:
`lambda_legacy=lambda+3*mu/8-1e-4` and `mu_legacy=3*mu/4`. The optimized
variables remain physical `log(lambda), log(mu)`; their initial values for
`nu=0.15` are `[12.135348811191877, 12.98260143502911]`.

A smoke run parsed `opt.json` with `--no_strict_validation`, completed the
19-step quasistatic forward solve, and evaluated a finite marker objective
`12.417`. The current PolyFEM material adjoint for quasistatic incremental
loading returns a NaN gradient, so a new full 20-step optimization was not
completed. The old saved Fig.18 result is historical and must not be
interpreted as a result produced by these new settings. Run the JSON files
with `PolyFEM_bin --no_strict_validation`; that flag is also required by
the other DiffIPC opt JSON files whose functional objects are not listed
in the root opt spec.

## Other saved comparison settings

The current edits to the comparison settings are also preserved in:

- `json_scripts/fig15_tentacles/opt.json`
- `json_scripts/fig15_tentacles/state.json`
- `json_scripts/fig15_tentacles/state-target.json`
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
