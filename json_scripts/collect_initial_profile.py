#!/usr/bin/env python3
"""Collect saved initial-pass logs without launching any simulation."""
import argparse
import csv
import json
import math
from pathlib import Path
from profile_initial_pass import (
    BACKWARD_PHASES,
    FORWARD_PHASES,
    OTHER_FIELDS,
    PHASES,
    ROOT,
    attach_totals_and_other,
    parse_log,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "initial_profile_20260908")
    args = parser.parse_args()
    output = args.input
    rows = []
    for case in ("fig1", "fig10", "fig15", "fig18", "fig21"):
        work = output / case
        if not (work / "result.json").exists():
            rows.append({"case": case, "status": "pending", "log": str(work / "run.log")})
            continue
        row = json.loads((work / "result.json").read_text())
        row.update(parse_log(work / "run.log"))
        manifest = json.loads((work / "manifest.json").read_text())
        requested_broad_phase = manifest.get("broad_phase_override") or row.get("broad_phase") or "bvh"
        row["requested_broad_phase"] = requested_broad_phase
        row["broad_phase"] = requested_broad_phase
        row["skip_target"] = manifest.get("skip_target", False)
        row["log"] = str(work / "run.log")
        attach_totals_and_other(row)
        if case == "fig10":
            valid = row.get("forward_calls") == row.get("backward_calls") == 1 and row.get("parameter_updates") == 0
        else:
            valid = (row["forward_batches"] == row.get("outer_gradient_calls") == 1
                     and len(row["outer_finished"]) == 1
                     and "iters=0" in row["outer_finished"][0]
                     and row.get("outer_update_direction_seconds") == 0
                     and row.get("outer_line_search_seconds") == 0)
        valid = valid and row["completed_rollouts"] == row["states"]
        valid = valid and row["initial_loss"] is not None and math.isfinite(row["initial_loss"])
        for gradient_line in row["gradient_log"]:
            if "gradient [" not in gradient_line:
                continue
            gradient = json.loads(gradient_line.split("gradient ", 1)[1])
            valid = valid and all(math.isfinite(v) for v in gradient)
            row["initial_gradient"] = gradient
        row["complete"] = row["exit_code"] == 0 and valid and all(row["phase_events"].values())
        row["status"] = "complete" if row["complete"] else "failed_or_unverified"
        row["phase_average_per_state_seconds"] = {
            k: (v / row["states"] if v is not None else None)
            for k, v in row["phase_seconds"].items()}
        row["forward_total_average_per_state_seconds"] = (
            None if row.get("forward_total_seconds") is None
            else row["forward_total_seconds"] / row["states"])
        row["backward_total_average_per_state_seconds"] = (
            None if row.get("backward_total_seconds") is None
            else row["backward_total_seconds"] / row["states"])
        rows.append(row)
    payload = {
        "date": "2026-09-08", "threads_per_process": 4,
        "semantics": "One initial objective/gradient evaluation. No parameter update or outer trial. Fig21 skips the target rollout. Multi-state cases include current and target unless skip_target is set.",
        "timing_convention": "Raw phase sums over the entire evaluation. Per-state averages also provided. CCD includes checking_for_nan_inf, broad_phase_ccd and narrow_phase_ccd. Trial-step constraint_set_update stays inside Search (classical_line_search plus LS begin) so it is not counted twice. Hessian is assembly time plus Newton compute gradient. Backward Hessian includes residual-Jacobian apply/cache plus shape and other parameter derivatives. Hessian cache is moved from the forward wall into Backward Total. Other is independently timed total minus the reported parts. Logged timers are rounded and do not cover all overhead.",
        "caveats": [
            "Contact cases default to bvh. Fig21 runs only the current state.",
            "Fig10 uses the Ren-bowen Python binding with unified Linf Newton stopping.",
            "Fig1/Fig18 retain Linf displacement stopping; Fig15/Fig21 retain the requested gradient stopping.",
            "Each case ran once; no retries.",
        ],
        "cases": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["case", "status", "states", "broad_phase", "wall_seconds",
              "forward_total", "backward_total",
              *FORWARD_PHASES, "forward_other", *BACKWARD_PHASES, "backward_other"]
    with (output / "six_phase_times.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            phases = row.get("phase_seconds") or {}
            writer.writerow({
                **{k: row.get(k) for k in ("case", "status", "states", "broad_phase", "wall_seconds")},
                "forward_total": row.get("forward_total_seconds"),
                "backward_total": row.get("backward_total_seconds"),
                **{k: phases.get(k) for k in (*PHASES, *OTHER_FIELDS)},
            })
    with (output / "six_phase_average_per_state.csv").open("w") as f:
        fields = ["case", "status", "states", "broad_phase",
                  "forward_total", "backward_total",
                  *FORWARD_PHASES, "forward_other", *BACKWARD_PHASES, "backward_other"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            averages = row.get("phase_average_per_state_seconds") or {}
            writer.writerow({
                **{k: row.get(k) for k in ("case", "status", "states", "broad_phase")},
                "forward_total": row.get("forward_total_average_per_state_seconds"),
                "backward_total": row.get("backward_total_average_per_state_seconds"),
                **{k: averages.get(k) for k in (*PHASES, *OTHER_FIELDS)},
            })
    print(json.dumps([{k: row.get(k) for k in (
        "case", "status", "wall_seconds", "initial_loss",
        "forward_total_seconds", "backward_total_seconds",
        "forward_other_seconds", "backward_other_seconds")} for row in rows]))


if __name__ == "__main__":
    main()
