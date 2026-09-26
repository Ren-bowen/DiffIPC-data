# Fig.18 material convention

Both implementations now follow the same nominal-Lame convention as the other
four examples. Let q = (log(lambda), log(mu)); clamp q to [4,16], then exponentiate.
Unified uses a=4*mu/3 and b=lambda+5*mu/6. PolyFEM native StableNeoHookean uses the
same mapping with an extra 1e-4 in b. This tiny legacy term is retained, as in the
other examples. The energy formula remains SNK1.

These are nominal material labels, not the effective small-strain moduli:
mu_effective=4*mu/3 and lambda_effective=lambda-mu/2 (plus PolyFEM's 1e-4).
Consequently switching this convention changes the response for fixed E/nu.

The opt.json composition deliberately omits stable-nh-v1-to-legacy. Both the
optimization and the independently generated target.json now use native
PolyFEM coefficients directly. E and nu differ only as the intended optimized
material values; do not map the target a second time. Unified target generation
and its parameter adapter likewise share the new coefficient convention.

The Unified pullback before the exp/clamp chain is:
dL/dlambda = sum(dL/db)
dL/dmu = (5/6)*sum(dL/db) + (4/3)*sum(dL/da).

Existing simulation exports are historical. The previous default initial
material nu=0.15 no longer gives the previously tuned near-flat shape.

## Native derivative validation

The current numbered diff-sim entrypoint was tested, not experiment/main.py.
At nominal E=1e6, nu=0.35, both log-parameter coordinate derivatives were checked
by centered finite differences at epsilon=1e-4 and 3e-5 after all 19 loading stages.
With an explicitly enabled exact adjoint and normal forward settings, maximum relative
error was 4.46e-5; with tighter forward settings it was 1.70e-7.
Test: Unified_GIPC/python/tests/diff_sim/test_fig18_current_mapping_fd.py --stable-probe.
Logs: /tmp/fig18_current_mapping_fd_final.log and /tmp/fig18_current_mapping_fd_probe.log.

The unchanged nu=0.15 initial point failed the forward det(F) safeguard at
stage 16 (approximately -0.00184), including under tighter tolerances. This
point therefore has no validated derivative or full optimization result.

Optimization defaults to projected Hessians, as requested. The exact-Hessian
FD results above are diagnostic runs only (--exact); the projected adjoint had
approximately 14.4% relative error at the same stable probe. make_world() does
not force exact Hessians. The default nu=0.15 forward limitation still applies.

Initial nominal nu is 0.258 on both sides (E=1e6). That is the measured
nearly-flat +X-face point under the current mapping and 19-step cap load
(center dx ≈ -0.008 mm, all 19 steps complete). nu=1/6 still gives
small-strain nu_eff≈0, but the free face sinks in and Unified inverts
near step 17; scaling E at nu=1/6 does not change that. Target nominal
nu is 0.45. PolyFEM opt.json / run.json initial values match 0.258.

Outer step: both sides use `lr=0.02`. The old `0.02/361` pairing is
retired; see `OPT_LR.md`. PolyFEM still omits `form.weight()` on
`force_material_derivative`, and Unified still multiplies the material
VJP by `dt^2`. That is a leftover backend difference, not a change to
the loss.
