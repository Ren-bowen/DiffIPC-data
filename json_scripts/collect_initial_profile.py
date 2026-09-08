#!/usr/bin/env python3
"""Collect saved initial-pass logs without launching any simulation."""
import csv
import json
import math
from pathlib import Path
from profile_initial_pass import ROOT, PHASES, parse_log


def main():
    output = ROOT / "initial_profile_20260905"
    rows = []
    for case in ("fig1", "fig10", "fig15", "fig18", "fig21"):
        folder = {"fig10": "initial_profile_fig10_20260905",
                  "fig15": "initial_profile_fig15_bvh_20260905",
                  "fig21": "initial_profile_fig21_sap_20260905"}.get(case, "initial_profile_20260905")
        work = ROOT / folder / case
        if not (work / "result.json").exists():
            rows.append({"case": case, "status": "pending", "log": str(work / "run.log")})
            continue
        row = json.loads((work / "result.json").read_text())
        row.update(parse_log(work / "run.log"))
        manifest = json.loads((work / "manifest.json").read_text())
        requested_broad_phase = manifest.get("broad_phase_override") or "hash_grid"
        row["requested_broad_phase"] = requested_broad_phase
        row["broad_phase"] = "hash_grid" if requested_broad_phase == "sweep_and_prune" else requested_broad_phase
        if requested_broad_phase == "sweep_and_prune":
            row["broad_phase_note"] = "ContactForm.hpp omits the sweep_and_prune JSON enum mapping; verified hash-grid events in native log. Effective algorithm is hash_grid."
        row["log"] = str(work / "run.log")
        if case == "fig10":
            valid = row.get("forward_calls") == row.get("backward_calls") == 1 and row.get("parameter_updates") == 0
        else:
            valid = (row["forward_batches"] == row["outer_gradient_calls"] == 1
                     and len(row["outer_finished"]) == 1
                     and "iters=0" in row["outer_finished"][0]
                     and row.get("outer_update_direction_seconds") == 0
                     and row.get("outer_line_search_seconds") == 0)
        valid = valid and row["completed_rollouts"] == row["states"]
        valid = valid and row["initial_loss"] is not None and math.isfinite(row["initial_loss"])
        for gradient_line in row["gradient_log"]:
            gradient = json.loads(gradient_line.split("gradient ", 1)[1])
            valid = valid and all(math.isfinite(v) for v in gradient)
            row["initial_gradient"] = gradient
        row["complete"] = row["exit_code"] == 0 and valid and all(row["phase_events"].values())
        row["status"] = "complete" if row["complete"] else "failed_or_unverified"
        row["phase_average_per_state_seconds"] = {
            k: (v / row["states"] if v is not None else None) for k, v in row["phase_seconds"].items()}
        rows.append(row)
    failed_attempts = [json.loads(path.read_text()) for case in ("fig15", "fig21")
                       if (path := output / case / "result.json").exists()]
    failed_attempts.append({"case": "fig15", "complete": False,
        "requested_broad_phase": "sweep_and_prune", "effective_broad_phase": "hash_grid",
        "status": "systemd_oom_kill", "killed_at": "2026-09-05 05:04:20 Asia/Shanghai",
        "log": str(ROOT / "initial_profile_fig15_sap_20260905/fig15/run.log"),
        "note": "Service journal confirms OOM; runner was stopped before writing result.json."})
    payload = {
        "date": "2026-09-05", "threads_per_process": 4,
        "semantics": "One initial objective/gradient evaluation. No parameter update or outer trial. Multi-state cases include both current and target rollouts/adjoints; states records the count.",
        "timing_convention": "Raw six phase sums over the entire evaluation. Per-state averages also provided. CCD includes constraint_set_update, checking_for_nan_inf, broad_phase_ccd and narrow_phase_ccd. Hessian is assembly time plus Newton compute gradient. Search is classical_line_search plus LS begin. Logged timers are rounded and do not cover all overhead.",
        "caveats": ["Fig15/Fig21 initial hash_grid attempts failed with OOM. Fig21 retry completed with hash_grid (requested sweep_and_prune silently fell back because of a missing C++ enum mapping). Fig15 selected retry uses bvh. Do not compare its CCD time as unchanged hash_grid performance.",
                    "Fig10 uses the installed legacy Python binding with grad_norm=1e-3; other cases use the current PolyFEM binary.",
                    "Fig1/Fig18 retain Linf displacement stopping; Fig10/Fig15/Fig21 retain the requested gradient stopping. This is not a claim of identical Newton criteria across cases.",
                    "Each case ran once successfully if marked complete; failed preflight/OOM attempts are excluded from successful timing totals."],
        "cases": rows, "failed_attempts": failed_attempts,
    }
    (output / "profile_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["case", "status", "states", "broad_phase", "wall_seconds", *PHASES]
    with (output / "six_phase_times.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in fields if k not in PHASES}, **row.get("phase_seconds", {})})
    with (output / "six_phase_average_per_state.csv").open("w") as f:
        fields = ["case", "status", "states", "broad_phase", *PHASES]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in fields if k not in PHASES},
                             **row.get("phase_average_per_state_seconds", {})})
    print(json.dumps([{k: row.get(k) for k in ("case", "status", "wall_seconds", "initial_loss")} for row in rows]))


if __name__ == "__main__":
    main()
