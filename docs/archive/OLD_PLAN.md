# Old upload plan and resume notes (archived 2026-10-22 from EXPERIMENT_LOG.md; superseded)

## Plan

Two scored uploads are allowed per day (renewing at midnight UTC), so the order is:

| Step | What | Why | Status |
|---|---|---|---|
| P0 | Build the A/B/C generator (`eda/16_generate_abc.py`, new): per context, the 300 targets x 400 cells from that context's own control cells, v3 effects fitted on the four lines **plus H1**, on the 18,533-gene axis; then `vcc sample` + `vcc prep --perts data/controls/pert_counts.csv -g data/controls/gene_names.csv` as a dry run | Packaging and disk (30 to 40 GB temp) must be proven before the real file | **done 2026-10-19**: `eda/16_generate_abc.py --tag v3_h1` (3 min, 16.8 GB h5ad, streamed) then `vcc prep` (5 min) gave `data/submission/v3_h1.vcc` (3.1 GB). Set `TEMP`/`TMP` to a folder on D: first, because C: has 16 GB free and prep fails with "No space left" |
| P1 | **Upload 1: v3 with H1 included**, generator exactly as in E2 | First leaderboard number; the cleanest test of H1 and H10; an unmodified reference for later variants | to do (user submits) |
| P2 | Tune the generator on H1 with v3, no H1 in the fit: effect scale (H6), effect floor and responder cutoff (H8), partial shrinkage (H7); 2 to 3 seeds per setting | Each scored run takes about 25 min and needs 3.5 GB of disk; use `PYTHONIOENCODING=utf-8` when logging `vcc-h1` output | to do, can run while P1 uploads |
| P3a | Upload-2 file built 2026-10-19: `data/submission/v3z15_h1.vcc` (E4's settings, H1 included in the fit), `vcc prep` passed | | submitted: L2 |
| P3c | Upload A of 2026-10-20 built: `data/submission/v3z15_kolf.vcc` (v3, z x1.5, six lines incl. KOLF2.1J; effect \|log2\| 50/99/99.99% = 0.083 / 0.485 / 2.156), `vcc prep` passed. Tests H28 on A/B/C: the proxy says KOLF is neutral to slightly negative for differentiated lines and helps the stem line | Note: `vcc prep` alone commits about 31 GB; run nothing heavy beside it | ready (user submits) |
| P3d | Upload built 2026-10-20: `data/submission/rank15.vcc`, the uploaded model (v3, z x1.5, five lines) with per-context gene order mixed 85% pool / 15% LambdaRank learner (`eda/41_rank_effects.py`, `16_generate_abc.py --effects eda/out/effects_rank15.npz`); `vcc prep` passed. Tests H27 on A/B/C | proxy: ties overall, +0.003 to +0.004 on the differentiated lines | ready (user submits) |
| P3b | Scale probe built 2026-10-19: `data/submission/v3z20_h1.vcc` (`--transform z --scale 2`, H1 in the fit; effect \|log2\| 50/99/99.99% = 0.112 / 0.661 / 3.117), `vcc prep` passed | Tests whether A/B/C rewards larger magnitudes (L1 to L2 nmae +0.16). If it beats 0.160, try x2.5 | ready (user submits) |
| P3 | **Upload 2: the best variant from P2** (only if it beats E2 on H1 by clearly more than 0.05 overall and does not collapse any metric), with H1 included | A second leaderboard point; comparing it with P1 tests H10 | to do (user submits) |
| P4 | ~~Add Jurkat, HepG2, RPE1 single cells as more lines~~ | **Dropped 2026-10-19**: these essential-gene screens cover 0 of the 300 panel targets (`DATASET_PREPARATION.md` 3). The only further genome-wide line is KOLF2.1J iPSC (282 targets, one 189 GB column-stored file); too costly unless it can be streamed | dropped |
| P5 | Per-cell variability and a noise-aware generator (H9) | Lifts the floor of -0.30 | later |
| P6 | 2026-10-22: download D/E/F, re-run coverage, rebuild with the best pipeline; 2026-11-03 final upload | Deadline 2026-11-05 23:59 UTC | later |

## How to resume

- **State.** Downloads are complete (`data/sources/`: K562, CD4+ T, H1, HCT116, HEK293T, Jurkat, HepG2, RPE1, K562 single cell). Stage 1 tables `eda/out/stage1_{K562,H1,CD4_Rest,HCT116,HEK293T}.npz` exist; nothing is running in the background. No submission has been made.
- **Run.** `.venv\Scripts\python.exe eda\<script>.py`; `eda/08_stage1_table.py` builds tables, `09_eval_stage2.py`, `12_eval_by_cpm.py`, `13_train_mask.py`, `14_add_line.py` are the proxy checks, `15_generate_h1.py --rung v1|v2|v3|v3b|zero` writes `data/h1_benchmark/pred_gen_<rung>.h5ad`, then `C:\Users\glitc\.local\bin\vcc-h1.exe score <file> --data-dir data/h1_benchmark --output eda/out/h1_score_<tag>`. Delete the 3.5 GB prediction file after scoring.
- **vcc CLI** (`C:\Users\glitc\.local\bin\vcc.exe`, v0.2.2): the user runs `vcc login` and `vcc submit`; never run them unprompted.
- **Notebook** `eda/EDA.ipynb` sections 11 to 11c hold the figures for the proxy findings.
