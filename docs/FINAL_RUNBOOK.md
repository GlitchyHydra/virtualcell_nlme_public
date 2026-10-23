# Final runbook: contexts D/E/F (released 2026-10-22, deadline 2026-11-05 23:59 UTC)

Dry-run on 2026-10-21 with a simulated folder (context A copied as D, the H1 benchmark controls as E): every step below ran,
the detector gave D the defaults and E `{"Pan_H1": 4, "H1": 4}`. The leaderboard is hidden in the final phase, so nothing
here is tuned on D/E/F: the settings are L5 (H38 + H40) and the same-line rule (H44, H47).

Since 2026-10-23 the steps are pipeline commands (`python -m vcell ...`, package `src/vcell`, settings in `configs/`); run
them from the repository root with `.venv/Scripts/python.exe -m vcell ...`. The old scripts that built F13-F18 are kept at
git tag `final-def-build`; the commands of that version are in the "Old commands" note under each step. To run them, check
the tag out next to the repository and link the data and the results folder in (they are not in git; the old code expects
the results in `eda/out`), from PowerShell:

```powershell
git worktree add ..\vc-final final-def-build
cmd /c mklink /J ..\vc-final\data data
cmd /c mklink /J ..\vc-final\eda\out out
cd ..\vc-final; & "..\Virtual Cell\.venv\Scripts\python.exe" eda\16_generate_abc.py ...
``` `tests/regression/test_release.py` checks that the pipeline still rebuilds the
uploaded files bit for bit.

## 1. Put the files in their own folder

`data/controls_final/`: `context_D.h5ad`, `context_E.h5ad`, `context_F.h5ad`, `manifest.json`, `pert_counts.csv`,
`gene_names.csv`. Never overwrite `data/controls/` (A/B/C, the proxy and the H1 tools read it). The configs name their
controls folder (`controls = "data/controls_final"` in `configs/final_def.toml`); other commands read `VCC_CONTROLS`
(default `data/controls`).

```bash
export VCC_CONTROLS=data/controls_final PYTHONIOENCODING=utf-8
```

Check the manifest (`contexts`, `pert_col`, `context_col`, `cells_per_pert`) against the A/B/C one, and run the control QC:
`python -m vcell qc controls --contexts D E F` (report in `out/qc/controls_DEF.csv`; F2: no filtering).

## 2. Same targets and genes as validation?

```bash
diff <(sort data/controls/pert_counts.csv) <(sort data/controls_final/pert_counts.csv) | head
diff data/controls/gene_names.csv data/controls_final/gene_names.csv | head
```

- **Same 300 targets and gene list**: go to step 4.
- **New targets**: step 3. HCT116 and HEK293T screen about 18,000 targets, so every new target should still have data.
- **New gene list**: the generator asserts the context axis equals `gene_names.csv`. Genes on only in the new contexts get
  no effect unless the universe is widened (`python -m vcell etl universe --widen --controls data/controls_final
  --contexts DEF`, F6), which needs a re-stream of X-Atlas and KOLF.

## 3. Only if the targets changed: rebuild the Stage 1 tables

Back up first: `mkdir -p out/val_tables && cp out/stage1_*.npz out/val_tables/`. The tables cover the new panel plus every
H1 target, so the H1 benchmark keeps working.

| Step | Command | Time |
|---|---|---|
| K562 | `python -m vcell etl stage1 k562` | 1 min |
| HCT116, HEK293T | `python -m vcell etl stage1 hct116acc hek293tacc` (from the all-target accumulators) | minutes |
| CD4+ T | `python -m vcell etl stage1 cd4` (remote rows) | about 20 min |
| H1 | `python -m vcell etl stage1 h1` | about 10 min |
| Pan_H1, Feng_iPSC | `python -m vcell etl stage1 hku` (needs K562 and H1 first) | minutes |
| KOLF | move `data/sources/kolf/accum.npz` away (it holds the old target list), then `python -m vcell etl stage1 kolf` | about 3 h (75 GB by HTTP range) |
| U-87 MG (context E) | `python -m vcell etl stage1 u87 u87-strata u87x` | minutes from the cached passes |
| CD4 control profile | `python -m vcell etl stage1 cd4-ctrl` (for the identity step) | 5 min |

Old commands: `eda/08_stage1_table.py k562` etc., `eda/48_hku_tables.py`, `eda/27_kolf_stream.py`, `eda/67_u87_table.py`.

## 4. Identity of D/E/F and the per-context source weights

```bash
python -m vcell qc identity --controls data/controls_final --contexts D E F --hpa
```

Writes `out/ctx_weights.json` and prints the markers and the HPA match. A context that matches a line with panel data
(contradiction <= 0.10) gets that line at 4; a stem-like one (POU5F1 >= 100 CPM) gets Pan_H1 at 4 (H47: +0.06-0.07 on the
official H1 scorer with 37% coverage). Related but different lineages get nothing extra (H48). Copy the weights into the
config (`[model.context_weights.<ctx>]`); a same-line table found by a search, like the U87 table for E (F8), is added there
by hand. Old command: `eda/52_context_match.py --dir data/controls_final --contexts D E F`.

## 5. Build (alone; about 10 minutes, writes about 17 GB)

```bash
python -m vcell build --config configs/final_def.toml          # data/submission/pred_final.h5ad
```

`configs/final_def.toml` holds the L5 settings (z x1.5, floor 0.02, bins [1.5, 1.25, 1, 1, 0.8], neighbour 0.15, KOLF 0.25,
Pan_H1 0.25) and `[model.context_weights.E] U87 = 4`; `configs/final_def_u87x.toml` is the variant with the drug-adjusted
U87 table (F18). The log must print `0 panel targets without a Stage 1 row`. Do NOT add a centring step: the shared-component
removal (H51) scored 0.1683 on A/B/C against 0.1705 without it (EXPERIMENT_LOG L6); `tests/unit/test_effects.py` fails if a
config gains one. Old command: `eda/16_generate_abc.py --rung v3 --transform z --scale 1.5 --min-effect 0.02 --bin-mult
1.5 1.25 1 1 0.8 --neighbour 0.15 --line-weight KOLF=0.25 --line-weight Pan_H1=0.25 --ctx-weights
eda/out/ctx_weights_def.json --tag final`.

## 6. Package (alone, close other apps: commits about 31 GB)

```bash
python -m vcell package --config configs/final_def.toml        # local checks, then vcc prep -> data/submission/final.vcc
```

It checks the file against the submission rules (docs/SCORING.md section 1), then runs `vcc prep` with
`PYTHONIOENCODING=utf-8` and TEMP/TMP on `D:/tmp_vcc` (log `out/prep_final.log`). Expect `prep complete`, 3 x 120,000 cells,
`targets: verified against the official list`. Then the user submits (`PYTHONIOENCODING=utf-8 vcc submit
data/submission/final.vcc`); never run `login` or `submit` without being asked.
