# DiffIPC-data-original 五个 Fig 优化结果（更新）

记录更新日期：2026-09-03。Fig.1 的 2026-08-31 重跑仍是已保存结果，但其 target 当时抄自 Unified 质心；当前设置改为由 PolyFEM `target.json` 生成，完整优化尚未按新 target 重跑。Fig.10 使用此前完成的 100 次动态仿真+反传结果；Fig.15、Fig.21 沿用 `20260825_original_settings` 中已经保存的有效结果。Fig.18 的旧结果仅作为历史结果保留，因为材料、19 步加载和 ADAM 设置已更新但尚未重跑完整优化。所有结果均使用本地 `/home/bowen/polyfem/build/PolyFEM_bin` 或 Fig.10 的本地 `polyfempy` 后端。

## Fig.1

- 运行目录：`optimization_runs/20260831_unified_current_fig1_target/fig1/`
- 场景：目标平移 `[1, 0, 0]`，初始速度 `[1, 0.5, 0]`。该次重跑的 target 仍是当时写入 `soft_bound` 的 Unified 质心 `[4.1206334421, 0.1263607470]`。当前仓库设置已改为 `target.json` + `center-target`，不再使用这两个写死数字。
- 动力学：`dt=0.05`、40 帧、BDF1、物理时间 2.0。
- 状态：20 次外层更新均已保存；PolyFEM 在保存 `opt_state_0_iter_20` 后因 iteration-limit 异常返回 `134`，因此不把异常退出误记为正常结束。
- 时间：`2252.84 s`（`/usr/bin/time -v` 墙钟时间）。
- loss：初始 `4.9411173878`，最终 `0.5700637509`，最低 `0.3490623137`（zero-based iteration `7`）。
- 最终检查点：`opt_state_0_iter_20`；最终网格 2061 点、7562 个四面体，全部有限，负体积和零体积单元均为 0。
- 结果文件：`optimization_runs/20260831_unified_current_fig1_target/fig1/summary.json`、`run.log`、`runtime.txt`、`loss_history.txt`。

## Fig.10

- 运行目录：`optimization_runs/20260825_dynamic_unified_remesh/fig10_exact_unified_loss/`
- 设置：动态 20 帧、`dt=0.02`、Implicit Euler、`n_iters=100`、默认 `lr=5e-3`、归一化直接梯度下降、stress power `2`、Laplacian weight `1.0`。
- 材料与 remesh：Stable Neo-Hookean `E=1e9`、`nu=0.49`、`rho=1000`；fTetWild，`--la 0.05 --no-binary`。
- 状态：完成 100 次优化记录，14 次 remesh；使用当前默认设置，不再继续重跑。
- 时间：本次 100 次结果目录没有保存可追溯的总墙钟计时，因此不虚构一个总时间；逐阶段对比表中的 DiffIPC 基准使用旧 `fig10/run.log` 的 36 条记录，详见 `timing_tables.tex` 的注释。
- loss：初始 `2.4049657e2`，最终 `1.4544543e7`；loss 发散不影响本次结果保存。
- 最终网格：5029 个顶点、19745 个四面体；最终 remesh 网格无负体积单元，最小质量约 `8.89383e-2`。
- 结果文件：`optimization_runs/20260825_dynamic_unified_remesh/fig10_exact_unified_loss/run_summary.json`、`loss_history.txt`、`final_rest_mesh.msh`。

## Fig.15

- 运行目录：`fig15/`
- 状态：按已接受的外层更新记录；iteration 6 之后的 line-search 试探没有作为正式结果计入。
- 初始 loss：`4.703760468429425e-2`
- 已接受 loss：
  `4.703760e-2 → 4.056581e-2 → 1.654309e-2 → 3.120064e-3 → 1.056137e-3 → 4.957311e-4 → 2.004622e-5`
- 最后有效检查点：`opt_state_0_iter_6`，loss `2.004622e-5`。
- 到 iteration 6 的墙钟时间：约 `6976.604 s`（`1:56:16.604`）。之后日志继续运行到约 `3:47:57.908`，停在未接受的 line search 中。
- 结果日志：`fig15/run.log`；该日志没有被新的运行覆盖。

## Fig.18

- 历史运行目录：`fig18/`（以下数据来自 2026-08-25 旧设置，不是 2026-08-31 新材料/target 设置的结果）。
- 历史状态：进程正常结束，14 次外层更新。
- 时间：`117.878 s`
- loss：`6.86747 → 3.26412e-11`
- 最终梯度范数：`4.32888e-9`
- 最终检查点：`fig18/final_opt_iter.txt`，`state=0 iter=14`
- 结果文件：`fig18/loss_history.txt`、`fig18/loss_curve.png`、`fig18/summary.json`
- 当前设置：初始 `E=1e6, nu=0.15`，target `E=1e6, nu=0.4`，`+z/-z` cap 位移 `-0.00551/+0.00646`，19 步准静态线性加载；24 个 marker 由 `generate_target.py` 从 PolyFEM `target.json` 写出；外层为 ADAM `alpha=0.15`。准静态材料伴随的 NaN 已在 PolyFEM `ada5f8f92` 修好；冒烟梯度为有限值 `[-4723.39, 4723.39]`。完整 20 步优化尚未按新设置重跑。完整说明见仓库根目录 `CURRENT_SETTINGS_20260825.md`。

## Fig.21

- 使用运行目录：`/home/bowen/DiffIPC-data-original/optimization_runs/20260825_fig21_no_normalization_original_optimizer_rerun/`
- 状态：只计入 OOM 前已经接受的 iteration 1；之后的 resume 试跑已停止，不计入本汇总。
- 参数：`0.5 → 0.006234603303580388`
- loss：`1.3513505629601483 → 0.22724204246682031`
- 到最后有效 iteration 的时间：约 `1179.292 s`（`19:39.292`）。
- OOM：下一次未接受 line-search 试探期间，内核于 `14:46:47` 杀掉 `PolyFEM_bin`，内核报告 RSS 约 `50.3 GB`。
- 原始记录：`fig21_no_normalization_original_optimizer_rerun/fig21/run.log` 和 `RESULT_RECORD.md`。

## 五个 Fig 的逐阶段时间和加速比

以下平均值均为一次完整仿真的六个阶段之和：前向线性求解、前向 CCD、前向 Hessian assembly、前向 line search、反传 Hessian assembly、反传线性求解。Unified GIPC 的时间来自 `Unified_GIPC_new_diff/agent_check/diff_sim/stage61_nine_main_timing_profile.md`；加速比定义为
`DiffIPC 平均六阶段时间 / Unified GIPC 平均六阶段时间`，因此大于 `1` 表示 Unified GIPC 更快。逐阶段的完整 LaTeX 对照表见仓库根目录的 `timing_tables.tex`。

| Fig | DiffIPC 平均仿真+反传 (s) | Unified GIPC 平均仿真+反传 (s) | Unified GIPC 加速比 |
|---|---:|---:|---:|
| Fig.1 | 59.405308 | 0.915166 | 64.91x |
| Fig.10 | 1.233343 | 2.760356 | 0.447x |
| Fig.15 | 451.990614 | 6.224459 | 72.62x |
| Fig.18 | 6.330310 | 0.041197 | 153.66x |
| Fig.21 | 314.534118 | 7.818286 | 40.23x |

Fig.10 的 `1.233343 s` 是旧 `20260825_original_settings/fig10/run.log` 的逐阶段 profiler 平均值，不是 100 次动态结果的总运行时间；100 次结果的 loss、remesh 数量和最终网格以本节上方的最新 summary 为准。

## 备注

- 本文档只汇总已保存且可追溯的结果；Fig.21 的未完成试探和 `fig21_resume_from_iter1/` 的中途运行不覆盖、不替代上一次有效结果。
- Fig.10 的默认步长在第一次 remesh 前可以将 loss 降到约 `1.10e2`，但随后网格质量和 loss 都恶化，因此不能标记为完整收敛。
