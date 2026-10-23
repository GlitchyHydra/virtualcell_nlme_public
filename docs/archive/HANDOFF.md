# Handoff: state on 2026-10-22 (start here in a new session)

Test contexts D/E/F are released today. The final deadline is **2026-11-05 23:59 UTC**. The final leaderboard is hidden, so
nothing can be tuned on D/E/F. Details live in `docs/EXPERIMENT_LOG.md` (H1–H51d, L1–L6). The model is explained step by
step in `docs/ALGORITHM_WALKTHROUGH.md`, and the build steps are in `docs/FINAL_RUNBOOK.md`.

## 1. The final model (L5, best on the A/B/C leaderboard: 0.1705, entry kwEQQeuqzyVCXUgzQMmU)

- **Stage 1:** per-line log2FC β and its SE s for 575 targets × 11,961 genes, from public Perturb-seq screens.
  - Source lines: K562, CD4_Rest, HCT116, HEK293T, H1, KOLF, Pan_H1.
- **Stage 2:** DerSimonian–Laird random-effects pooling over the lines (`eda/stage2.py`).
  - τ² is estimated per expression bin.
  - Line weights: K562, CD4_Rest, HCT116, HEK293T and H1 at 1; KOLF at 0.25; Pan_H1 at 0.25.
- **Standardised effect:** z = θ̂ / sd × median(sd), scaled by 1.5.
- **Per-bin multipliers** by the context's control CPM (edges 15/40/120/400): [1.5, 1.25, 1, 1, 0.8].
- **Promoter-neighbour cap:** 0.15. **Effect floor:** 0.02.
- **Generator:** resample 400 control cells per target; binomial thinning for down effects, Poisson additions for up effects.
  It is nearly lossless: with the true effects it scores 0.97.
- **Per-context source weights** come from `eda/52_context_match.py`, which writes `eda/out/ctx_weights.json`:
  - a context that matches a line with panel data gets that line at weight 4;
  - a stem-like context (POU5F1 ≥ 100 CPM) gets Pan_H1 at weight 4.

The build command is in FINAL_RUNBOOK section 5. **Do not add `--center-k`.**

## 2. What worked (keep)

| Step | Gain | Ref |
|---|---|---|
| Pooling effects across lines (v3), instead of the generic response | +0.26 on H1 | H1 |
| Standardising z at scale 1.5 | Leaderboard 0.092 → 0.160 | H11, L2 |
| Per-bin scale + neighbour cap + floor + KOLF 0.25 | +0.005 | H36–H38 |
| Pan_H1 at 0.25 | +0.002 proxy | H40 |
| Combined (L5) | Leaderboard 0.1705 | L5 |
| **Same-cell-type source at weight 4** | **H1 official scorer 0.126 → 0.190 at only 37% target coverage** | H44, H47 |

The main lesson: **same-cell-type data is the only big lever.**
- Information from other cell types has a ceiling: their effects correlate with the target cell type's at only 0.06–0.15 (H43).
- The generator is not the bottleneck (H42).
- Teams ranked 61–100 match us on the per-cell DE metrics. Their lead comes from mse (probably the noise-allowance loophole, see section 5) and from pds.

## 3. Tried and rejected (do not repeat)

- **Shrinkage and denoising:**
  - shrinking weak effects toward zero (H2);
  - James–Stein shrinkage (H7);
  - keeping only confident genes (H8);
  - low-rank denoising (H14);
  - Wiener filter on gene programs (H24);
  - a stacked meta-predictor (H17).
- **Scale and strength:**
  - a separate scale per target (H15, H30);
  - a direct parametric optimum, which turned out to be the current model already (H26);
  - knockdown-aware pooling, neutral (H23);
  - the target's own expression, which does not predict strength (H18).
- **Line weighting:**
  - by similarity group (H13);
  - by similarity of control profiles (H35);
  - by platform, since 10x Flex showed no edge (H41);
  - upweighting the most related lineage when no same-type data exists (H48);
  - a line adapter learned from the line's other targets (H45);
  - Feng_iPSC as a source (H40);
  - stimulated CD4 conditions, marginal (H31).
- **Learners:**
  - LightGBM on the pool (H25);
  - LambdaRank, which ranks genes better but scores worse (H27);
  - per-gene transfer weights (H19);
  - non-linear meta-regression (H32);
  - a nearest-neighbour target prior (H22, cheap form);
  - K562 effects from single cells instead of pseudobulk (H20).
- **Generator tricks:**
  - decoupling significance from magnitude (H16);
  - per-cell gamma noise, within noise (H33).
- **mse and the global direction:**
  - getting raw mse under the baseline with transferable information (H34, H39b);
  - using the control-cell covariance as a response predictor (H50).
- **Shared-pattern removal (H51, H51d):** removing the top 3 SVD components across targets.
  - It helped on the proxy and on the H1 scorer.
  - It lost on the leaderboard: L6 0.1683 vs 0.1705, and pds did not rise.
  - Variants did not beat it: per-line removal, damping, gene-length removal.
- **Score hacking: rejected by the user, never use.**
  - Pinned or decoupled pseudobulk emission.
  - The mse noise-allowance exploit.
  - Tuning only for A/B/C.

## 4. To do now (D/E/F released)

1. **Get the data. DONE 2026-10-22.** `vcc` upgraded to 0.3.0 (required for final submissions);
   `vcc datasets download controls-final` → `data/downloads/vcc_2026_controls_final.zip` (checksum verified), extracted
   to `data/controls_final/`. Manifest: partition `test`, panel `vcc2026-test-1`, same counts as A/B/C. Each context is
   18,400 × 18,533 with obs `target_gene`/`context`/`ntc_id`, 46 NTC ids.
2. **Check the panel. DONE: the targets changed.**
   - The gene list is identical to A/B/C.
   - Only 129 of the 300 targets are shared with A/B/C. The current Stage 1 tables cover 141 of 300.
   - So the Stage 1 rebuild (FINAL_RUNBOOK section 3) is required. KOLF takes about 3 h.
3. **Identify D/E/F** (FINAL_RUNBOOK section 4): `eda/52_context_match.py --dir data/controls_final --contexts D E F`. Read the markers.
4. **The highest-value step: look for same-cell-type public screens for each D/E/F identity.**
   - A match at weight 4 was worth +0.06 even at 37% coverage.
   - Recheck X-Atlas/Pisces (Jurkat and HepG2, genome-wide). On 2026-10-21 its Hugging Face repo still had only a README and LICENSE.
   - Check the Illumina Billion Cell Atlas (genome-wide CRISPR in 200+ lines) for released data.
   - Search for other Perturb-seq or CRISPRi screens in the identified line.
   - If one is found: build its Stage 1 table on the 575-target × 11,961-gene axis (as `eda/48_hku_tables.py` does), then add it to the pool at weight 4 for that context.
5. **Build** (section 5, about 1 h, 17 GB) with the L5 settings + `--ctx-weights`. Check that the log says `0 panel targets without a Stage 1 row`.
6. **Package** with `vcc prep` (section 6). It must run alone and commits about 31 GB. Use `PYTHONIOENCODING=utf-8` and TEMP on D:.
   - The user submits: `PYTHONIOENCODING=utf-8 vcc submit data/submission/final.vcc`.

## 5. Constraints

- Never run `vcc login` or `submit`, and never read `~/.config/vcc/state.json`.
- Do not use `-I` with the project scripts.
- Do not send the user's email to any external service.
- The upload limit is 2 per day and resets at 00:00 UTC.
- A low-memory reaper stops jobs: do not restart a stopped job without asking. Run heavy jobs (build, prep) one at a time.
- Without the env prefix, vcc crashes on Windows cp1252 with a UnicodeEncodeError before it uploads anything. Always set `PYTHONIOENCODING=utf-8`.
- Hardware: 32 GB RAM, 8 GB GPU. D: had about 136 GB free on 2026-10-21.
- The mse score is 0 for us. The "unbiased" mse subtracts a noise allowance estimated from the submitted cells' own spread. That is why pinned submissions score high on mse. The user has rejected exploiting this.
