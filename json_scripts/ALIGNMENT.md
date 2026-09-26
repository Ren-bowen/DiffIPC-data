# Five-example alignment (2026-09-10)

> Historical alignment notes. Several settings below describe the September 10–11
> experiments, not the current production defaults. For current defaults and
> selected results, see `Unified_GIPC/agent_check/DiffIPC-comparason/README.md`
> and `04_alignment_and_differences.md`. In particular, the current Fig.1
> target velocity is `[3.6, 0.5, 0]`.

These defaults align PolyFEM's absolute contact activation distance to the
current Unified_GIPC source, which computes
`dHat = relative_dhat^2 * bbox_diagonal^2`. The physical gap is therefore
`relative_dhat * bbox_diagonal`, not `relative_dhat` alone.

- Fig.1: bbox diagonal 2.257765078839169; Unified frontend
  `relative_dhat=0.001` (backend multiplies once by the bbox diagonal);
  PolyFEM run/target dhat 0.002257765078839169.
  `opt.json` computes its own target through `target.json` using v0=[5,0.5,0].
  `target_center.json` is diagnostic output, never an optimization input.
- Fig.10: both defaults use `middle_cleaned_res5x.msh` (8172 vertices / 35843 tets).
  Initial bbox diagonal 2.709101388874178 and absolute dhat
  0.0008127304166622534. `python_scripts/fig10_hanger/optimize.py` recomputes
  this distance when initializing each remeshed solver. Default Newton
  stopping is `unified-linf`: absolute movement tolerance
  `1e-3 * 0.02 * bbox_diagonal`, initially 5.418202777748356e-5.
  Default maximum remesh count is 100.
- Fig.15: default `diffipc_aligned` profile; bbox diagonal
  2.2697236953242457, absolute dhat 0.0022697236953242456. Both optimizers use
  energy-decrease Backtracking with initial step 1, reduction factor 0.5,
  minimum step 1e-10 and 30 trials. Unified retains Torch L-BFGS directions;
  PolyFEM uses PolySolve L-BFGS. Both sides clip accepted updates componentwise to global bounds
  intersected with the previous parameter +/- max_change.
- Fig.18: bbox diagonal 0.06928203230275509; Unified frontend
  `relative_dhat=0.001`; PolyFEM run/target dhat 6.928203230275509e-5.
  Both sides run 19 **dynamic** steps at `dt=1/19` (`quasistatic: false`
  / `DiffSimModule(dynamic=True)`). Self-contact is enabled on both
  sides; there is no ground. Unified AL probe now CCD-checks the
  rest→target snap, matching PolyFEM `is_step_collision_free`. Both
  sides use constant barrier stiffness 1e5. The PolyFEM
  composition is `clamp(4,16) -> exp -> per-body-to-per-elem` (shared
  nominal-Lame mapping). This requires the ClampMap support in
  `/home/bowen/polyfem` and its rebuilt `build/PolyFEM_bin`; arbitrary
  older binaries do not accept `type: clamp`. Gradients pass through at
  the endpoints and are zero outside, matching `torch.clamp`.
- Fig.21: bbox diagonal 7.671194045503013; absolute dhat
  0.007671194045503013. PolyFEM keeps kappa=1e4; Unified keeps adaptive kappa.
  Both simulate 40 steps at dt=0.05 (2 seconds, excluding the initial state).

The gaps above use the default meshes and transforms in the numbered Unified
examples. Recompute them if those meshes/transforms change. All existing
convergent-formulation and friction-smoothing settings are retained.

Validation: native ClampMap boundary and finite-difference tests; six existing
hanger Newton configuration tests; earlier native res20x hanger initialization; three
Backtracking tests and two native five-frame Fig.15 updates; regenerated
40-step PolyFEM Fig.1 target; complete 20-update Unified Fig.18 run and exports.
The PolyFEM Fig.18 integration check accepted the new JSON and completed its
initial 19-stage forward and adjoint. Its slower post-Adam-update solve was
stopped; no complete PolyFEM optimization convergence is claimed here.
This is not a full optimization or derivative-accuracy validation of all five cases.

Follow-up: Fig.10 defaults switched to res5x on both sides. Vertex coordinates
and tetrahedral connectivity match exactly between the two assets. Its bbox
is identical to res20x, so the physical dhat and Newton threshold are unchanged.

## L-BFGS optimizer selection

Fig.1 and Fig.15 PolyFEM now select `L-BFGS`, retaining history sizes 6 and
3 respectively. Fig.21 already selected `L-BFGS` (default history size 6).
The L-BFGS-B-only `box_constraints` entries were replaced by PolyFEM's
`parameter_clip` configuration. Both sides use ordinary L-BFGS directions,
without projecting directions or limiting the common step by parameter bounds.
Physical parameters during trial evaluations are clamped to global bounds,
with endpoint gradients enabled and outside gradients zero. After line search,
the accepted raw parameter is clipped componentwise to global bounds intersected
with the previous parameter +/- max_change. Fig.1 uses [-5,5] and 0.5;
Fig.15 uses [-0.2,-2,-0.2] to [0.2,0,0.2] and 0.5;
Fig.21 uses [0,1] and 0.1. L-BFGS history uses the actual clipped displacement.
Fig.15 recomputes its reported loss at the clipped iterate.

The PolyFEM build requires the PolySolve `clip_update` hook. The focused patch
is saved in `/home/bowen/polyfem/patches/polysolve-post-update-clip.patch`;
apply it with `git apply` inside a fresh PolySolve checkout before rebuilding.
The active dependency is `/home/bowen/.cache/CPM/polysolve/2c08`.
This hook is disabled by default for other problems and configurations.

Fig.10 retains normalized gradient descent and Fig.18 retains Adam.
The previous line-search, learning-rate, history-size and stopping-condition
differences across implementations are not changed by this constraint update.

Unified Fig.1/15/21 now use `AlignedLBFGS` (Fig.15 through its Backtracking
subclass): initial trial step equals the configured lr, cancelling Torch's
first-update L1-gradient scaling. Gradient stopping is L2 norm <= 1e-3 for Unified Fig.1,
and <= 1e-4 for Unified Fig.15/21. The outer iteration loops remain fixed-count; a converged
call returns without updating. PolyFEM gradient tolerances are 1e-3, 1e-4,
and 1e-4 respectively. Other line-search and history settings are unchanged.
Validation: five CPU optimizer tests cover first trial steps with/without
Strong-Wolfe, L2 versus max-norm stopping, clipping, and Backtracking rollback.

Unified Fig.21 now uses Backtracking with lr=1, history size 6, L2 gradient
threshold 1e-4, and no initial gradient normalization. Backtracking starts at 1,
halves to a minimum of 1e-10, and permits 30 trials. Accepted updates are clipped
to [0,1] intersected with previous +/-0.1; the loss is reevaluated after clipping.
Fig.15 now uses L2 threshold 1e-4 on both sides. Six CPU optimizer tests pass, including a
Fig.21 check that gradient 5e-4 updates while gradient 1e-4 stops.

PolyFEM Fig.1 default_init_step_size is now 0.5, matching Unified lr=0.5.

PolyFEM Fig.10 defaults to feasibility Backtracking: preserve reference tet
orientation with signed quality >1e-10, require a successful finite forward
solve, accept without comparing loss decrease. Try scale 1 and at most 24
halvings. Failed trials restore the original shape and rerun it after exhaustion.
Remesh is requested for low quality or accepted scale <=0.25 (including zero),
subject to the existing remesh budget and failure guard. Legacy --line-search
and --trust-region modes remain explicit overrides. The feasibility evaluator
propagates solve failures rather than using a partially completed rollout.
Validation: four CPU tests cover increased-loss acceptance, orientation rejection,
failed-rollout shrinking/restoration, and small-step remesh triggering.

Fig.18 now uses plain GradientDescent (PolyFEM) / Torch SGD without momentum
(Unified), fixed lr=1e-6 and no line search, with 20 updates as before. Both
physical log-material mappings retain clamp [4,16]. The export utility follows
the same optimizer and shared Unified learning-rate constant. Existing Adam
export files are historical and were not regenerated. The initial lr is based
on PolyFEM's recorded initial objective 6.20849 and gradient approximately
[-2361.6884,2361.6932], giving a first coordinate change about 0.00236.
This is a starting value, not a validated convergence guarantee.

Fig.1 Unified now computes both target and optimization barycenters using
fixed reference-tet volumes, matching PolyFEM center-target reference integration.
No derivative of deformed tet volume enters the loss. Headless retains the last
clipped update, evaluates its final loss, and writes final_velocity/final_loss
with selection=last_update rather than best-iterate fields. Validation covers
nonuniform deformation, analytic position gradients and finite differences;
no full native rollout was rerun for this change.

Fig.18 material convention update: Unified now uses a=4*mu/3,
b=lambda+5*mu/6 in target generation and parameter adapter, including the
correct transpose Jacobian. PolyFEM opt.json no longer uses stable-nh-v1-to-legacy;
its target.json already uses this native nominal mapping (with +1e-4 in b).
See fig18_static_cube/MATERIAL_MAPPING.md for the effective-modulus distinction.
The energy formula remains SNK1. Fig.18 diagnostic tests explicitly used exact adjoint
Hessians: the previous projected adjoint had ~14.4% FD error at the stable
nu=0.35 test point. With strict forward settings and exact adjoints, all four
coordinate/epsilon tests had relative error <1.7e-7.
The unchanged default initial nu=0.15 flips a tet at loading stage 16 after
this physical-response change (min det(F) about -0.00184), also with tighter
Newton/AL settings. Thus default-initial-point FD and full optimization are
not validated; no determinant safety check was disabled or initial material
silently retuned. The stable probe uses E=1e6, nu=0.35 and the full 19 stages.

Unified optimization defaults are projected Hessians in all five cases.
Fig.1's exact-elastic override and Fig.18's implicit exact override were removed
from the optimization defaults. Exact adjoints remain explicit test options.
This policy update does not change PolyFEM's adjoint implementation.

Fig.1 audit correction: min_step_size and min_step_size_final are 1e-10,
so fixed step 0.5 is not rejected by PolySolve generic line-search guards.
The previous retained min_step_size=1 conflicted with that fixed step.

Latest consolidated alignment report (split into four files):
/home/bowen/Unified_GIPC/agent_check/DiffIPC-comparason/polyfem_unified_alignment_20260910.md
Fig.1 now exits on absolute L2 convergence; Fig.15 returns final parameters.
Fig.1/15 outer PolyFEM L-BFGS uses an explicit single-strategy list without GD
fallback. Fig.1/15/18 forward run/target states explicitly use Backtracking
with energy-based acceptance (use_grad_norm_tol=0). Native validation: 20
assertions passed, including no fallback after a failed L-BFGS search.

Fig.15 Unified headless now exits when the L2 gradient norm is <=1e-4,
retaining the final parameter and evaluating its final loss. Ten calls are
a maximum budget rather than a mandatory fixed number.

2026-09-11 Fig.15 default change (both sides): loss weight 20 -> 200,
outer optimizer GradientDescent / StoppingSGD, lr=1.0, no line search,
grad L2 tol 1e-2, clip after the raw gradient step. The in-progress BVH run
fig15_polyfem_bvh_20260910_233613 still uses the previous L-BFGS / weight=20
/ 1e-4 settings.
