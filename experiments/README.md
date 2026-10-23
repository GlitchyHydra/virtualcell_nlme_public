# Experiments

One script per tested idea, named after its ID in `docs/EXPERIMENT_LOG.md` (H.. hypotheses, E.. H1 scorer runs, F.. final
phase). Every script uses the pipeline (`src/vcell`): tables through `etl.stage1.table`, pooling through `model.pool`,
scores through `evaluation.panel` / `evaluation.harness` / `evaluation.h1_benchmark`. Only the idea being tested is
written in the script itself.

- `experiments/` the ideas that are in the final model, and the experiments that validated or tuned them
- `experiments/analysis/` diagnostics, ceilings and identity checks that did not propose a model change
- `experiments/failed/` tested model changes that were not adopted: rejected, neutral, or lost on the leaderboard

Run a script from the repository root, e.g. `.venv/Scripts/python.exe experiments/h38_combined.py`. Each one sets its
snapshot at the top: `abc` (the A/B/C validation tables in `out/val_tables`, on which most experiments ran) or `def` (the
D/E/F final tables in `out/`). Results go to `out/experiments/`. The header of each script gives the question, the method,
the logged verdict with its numbers, the script it replaces (`eda/NN_*.py` at git tag `final-def-build`) and the runtime.
Heavy runs (the essential full axis, the learners, the H1 scorer) must run one at a time.

**Checked** says how the port was verified against the old script on 2026-10-23:
`= old` the new script prints exactly the numbers of the old one on the same snapshot (tests/golden/run_old.py,
compare_logs.py); `= log` it reproduces the logged values; `golden` covered by a regression test in tests/regression;
`smoke` runs to the end on a subset; `ported` code moved and linted, not re-run against the old script (the check was stopped on request, or it takes hours,
network or the official scorer); `run_old.py` against the tag can still check any of them. Logs: `out/verify_logs/`.

## Adopted, or validation of what was adopted

| ID | Question | Verdict | File | Was | Checked |
|---|---|---|---|---|---|
| H1, H11, E1-E4 | Pooling beats the generic response; the standardised effect beats raw v3 (official H1 scorer) | Supported: E1 -0.180, E2 0.076, E4 0.121 | `h01_pooling_h1_scorer.py` | 15 | golden (E4 file), ported |
| H11 | Standardised z vs raw v3, scales, on the leaderboard scale | Supported: z x1.5 (L2 0.160 vs L1 0.092) | `h11_standardised_effect.py` | 20 | = old |
| H12 | Best scale by target strength | Supported: 1 / 1.5 / 2-3 by DE count; 1.5 overall | `h12_effect_regime.py` | 21 | = old |
| H21 | z x1.5 on lines never tuned on (Jurkat, HepG2, RPE1) | Supported: best scale 1.5 in all three | `h21_essential_holdout.py` | 30 | = old |
| H26 | Direct optimisation of the effect family | The uploaded model is the optimum | `h26_direct_opt.py` | 35 | = old |
| H28 | KOLF2.1J as a sixth line | Not for differentiated lines; later in at 0.25 (H38) | `h28_kolf.py` | 37 | = old |
| H29 | Tune c, kappa, floor, KOLF on 8 lines | Within noise; floor 0.02 kept | `h29_tune_all.py` | 38 | = old |
| H36 | Effect scale per expression bin | Adopted: [1.5, 1.25, 1, 1, 0.8] (+0.0027) | `h36_bin_scale.py` | 43 (H36 part) | = old |
| H37 | Promoter neighbours | Adopted: cap 0.15 (+0.0011) | `h37_promoter.py` | 44 | = old |
| H38 | The small gains together | Adopted (L5 recipe): +0.005 | `h38_combined.py` | 45 | = old, = log |
| H40 | Pan_H1 and Feng_iPSC as stem sources | Pan_H1 0.25 adopted, Feng no | `h40_stem_sources.py` | 49 | = old |
| H44 | Weight of a same-line source | 4 is best (the same-line rule) | `h44_same_line_weight.py` | none (interactive) | new |
| H47, E6 | The same-line rule on the official scorer | Supported: 0.190 vs 0.121-0.131 | `h47_same_line_h1_scorer.py` | 15 | ported |
| H53 | Weight 4 for a noisy same-line source (U87) | Weight 4 still best | `h53_noisy_same_line.py` | 69 | = old |
| H54 | U87 drug arms with the drug effect removed | Adopted as the F18 variant | `h54_u87_drug_adjusted.py` | 70 | golden (table) |
| H55 | Held-out check of the drug-adjusted U87 data | Weakly supported | `h55_u87_heldout.py` | 71 | = old |

## Analysis (no model change proposed)

| ID | Question | Result | File | Was | Checked |
|---|---|---|---|---|---|
| EDA phase 2 | Do effects transfer K562 -> CD4+ T? | Weakly, through a few genes | `analysis/concordance_k562_cd4.py` | 04 | ported (network) |
| R1 | The scorer's top anchor (real fresh cells) | 1.067 vs -0.303 | `analysis/r1_positive_control.py` | 07 | ported |
| H3 | First Stage 2 proxy on held-out H1 | Pooling adds signal where sources see the target | `analysis/h03_ladder_first_proxy.py` | 09 | ported |
| | Scoring only well-expressed genes | Diagnostic | `analysis/eval_by_cpm.py` | 12 | ported |
| H3, H4 | Add a line; H1 as a source | Each line helps a little; H1 neutral | `analysis/h03_h04_add_line.py` | 14 | ported |
| H6 | The mse proxy and its best scale | Proxy validated; mse not worth chasing | `analysis/mse_proxy_validation.py` | 17 | ported |
| H11, H7, H8, H14 | The pds proxy and transforms | z wins; JS, hard cut-offs, low rank lose | `analysis/proxy_pds.py` | 18 | ported |
| | DE proxy validation, single-line transfer | Proxies within 0.01-0.02 of the scorer | `analysis/proxy_directions.py` | 19 | ported |
| H39 | A line's control covariance and its global direction | Supported (direction in the control PCs) | `analysis/h39_control_covariance.py` | 46 | ported |
| H42, H42b | The generator with the true effects; one line's real profile | Generator nearly lossless (0.972) | `analysis/h42_generator_oracle.py` | 15 --oracle | ported |
| H43 | Ceiling of a better shared effect | 0.06-0.15; only the same cell type moves it | `analysis/h43_lines_ceiling.py` | 51 | ported |
| H46, F4 | Identity of A-F; match to source lines and HPA | D MOLM-13, E U-87 MG, F hTERT fibroblast | `analysis/h46_context_identity.py` | 52, 64, 65 | ported (pipeline: qc.identity) |
| H51c | What the shared patterns are | Gene length, stress, line-dependent response | `analysis/h51c_shared_patterns.py` | 60 | ported |
| F10 | Is the D/E/F panel as predictable as A/B/C? | Yes (0.140 -> 0.136) | `analysis/f10_proxy_panels.py` | 68 | golden (def), = log |
| H56 | U87 tables as ordinary sources | Neutral overall | `analysis/h56_u87_as_source.py` | 72 | = old |
| | Figures of the mse trade-off | | `analysis/mse_tradeoff_figures.py` | 56 | ported |

## Failed (not adopted)

| ID | Idea | Verdict | File | Was | Checked |
|---|---|---|---|---|---|
| H2, E3 | Sparse prior (v3b) | Rejected: -0.142 vs 0.076 | `failed/h02_sparse_prior.py` | 15 --rung v3b | ported |
| | Dropping low-expressed genes from training | Changes nothing | `failed/train_mask.py` | 13 | ported |
| H13, H15 | Line groups; per-target scale | Rejected | `failed/h13_h15_line_groups_adaptive_scale.py` | 22 | = old |
| H16, H17 | Visibility switch; stacked meta-predictor | Rejected | `failed/h16_h17_visibility_and_stack.py`, `failed/visibility_generator.py` | 23, generator.py | = old |
| H18 | Target's own expression predicts strength | Rejected | `failed/h18_target_expression.py` | 24 | = old |
| H19 | Gene weights by transferability | Rejected | `failed/h19_gene_transfer.py` | 25 | = old |
| H20 | K562 from single cells | Rejected (0.145 vs 0.147) | `failed/h20_k562_single_cells.py` | 26 | = old |
| H22 | Nearest-neighbour prior | Rejected | `failed/h22_neighbour_prior.py` | 31 | = old |
| H23 | Knockdown-aware pooling | Neutral | `failed/h23_knockdown_pooling.py` | 32 | = old |
| H24 | Shrinkage through gene programs | Rejected | `failed/h24_programs.py` | 34 | ported |
| H25 | Gradient boosting on the pool | Not supported | `failed/h25_gbm.py`, `failed/gbm_features.py` | 33, gbm_features.py | ported (hours) |
| H27 | Ranking learner (LambdaRank) | Not adopted | `failed/h27_rank_gbm.py`, `failed/h27b_rank_effects_upload.py` | 36, 41 | ported (hours) |
| H30, H31 | Panel-like regime; stimulated CD4+ T | Not supported / marginal | `failed/h30_h31_panel_regime_stim_cd4.py` | 39 | ported |
| H32 | Non-linear meta-regression | Rejected | `failed/h32_meta_regression.py` | 40 | ported |
| H33 | Cell-to-cell variation of the response | Within noise | `failed/h33_cell_noise.py` | 15 --cell-noise | ported |
| H34 | A structure that brings mse under the baseline | Not reachable | `failed/h34_mse_structure.py` | 42 | ported |
| H35 | Line weights by control-profile similarity | Rejected | `failed/h35_similarity_weights.py` | 43 (H35 part) | ported |
| H39b | Strength x covariance-adjusted global direction | Rejected | `failed/h39b_global_direction.py` | 47 | ported |
| H41 | Same platform beats same cell type | Not visible | `failed/h41_platform.py` | 50 | ported |
| H45 | Line adapter from the line's essential screen | Rejected | `failed/h45_line_adapter.py` | 53 | ported |
| H48 | Upweight the most related cell type | Rejected | `failed/h48_related_type.py` | 54 | ported |
| H50 | Context control covariance predicts responses | Rejected | `failed/h50_context_covariance.py` | 57 | ported |
| H51, H51b, H51d, L6 | Remove the patterns all predictions share | Lost on the leaderboard (0.1683 < 0.1705) | `failed/h51_shared_pattern_removal.py`, `h51b_shared_pattern_harness.py`, `h51d_shared_pattern_variants.py` | 58, 59, 61 | ported |
| H52 | Cell-level QC of the source screens | Rejected (-0.0013) | `failed/h52_source_qc.py` | 63 | ported |

## Where the old scripts went

Pipeline (no longer experiments): 01 `vcell.qc.controls`; 02, 05, 10, 11 `vcell.etl.download`; 03 `vcell.etl.coverage`;
06, 66 `vcell.etl.universe`; 08 and stage1.py `vcell.etl.stage1`; 27 `etl.stage1.kolf`; 28, 29 `etl.stage1.essential`;
48 `etl.stage1.hku`; 67, 70 `etl.stage1.u87`; stage2.py `vcell.model.pool`; promoter.py `vcell.model.promoter`;
16 `vcell.model.effects` + `model.generate` + `model.package` (`python -m vcell build`); 15 `vcell.evaluation.h1_benchmark`;
proxy.py `vcell.evaluation.panel` + `evaluation.proxy`; tune_harness.py `vcell.evaluation.harness`; qc.py and 62
`vcell.qc.sources`; 52, 64, 65 `vcell.qc.identity` + `vcell.model.sources`; test_stage2.py `tests/unit/test_pool.py`.
H5 (Stage 1 s is calibrated) has no script of its own: the robust sd of beta / s is printed by `failed/h20_k562_single_cells.py`
and `h28_kolf.py`.
