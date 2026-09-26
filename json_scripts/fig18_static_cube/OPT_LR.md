# Fig.18 PolyFEM outer learning rate

`opt.json` → `solver.nonlinear.line_search.default_init_step_size = 0.02`

This matches Unified `OPTIMIZER_LR = 0.02`. There is no `/361` pairing.

The old `0.02/361` value cancelled a quasistatic coefficient mismatch
when Unified used `dt=1` and PolyFEM used `dt=1/19`. Both sides are now
dynamic with `dt=1/19`. PolyFEM's material VJP still omits
`form.weight()` and Unified still multiplies the material VJP by
`dt^2`; that leftover backend difference is no longer absorbed into the
outer step.

JSON schema rejects unknown keys inside `line_search`, so this note lives
next to `opt.json` instead of inside it.

`opt.json` `node-target` still reads `data.txt`. The official launch is
`python run_opt.py`, which always solves the current `target.json` first
(same role as Unified `make_target`). `run_five_polyfem_opt.py`,
`profile_initial_pass.py`, and `compare_init_loss.py` do the same for
Fig.18. Do not invoke `PolyFEM_bin -j opt.json` alone.
