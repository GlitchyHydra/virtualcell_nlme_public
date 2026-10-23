# qc: check the inputs before using them

The model writes its predictions into the new context's own control cells and borrows effects from other cell lines, so
three things must be right before anything is built: the control cells are what the organisers say they are, the public
source cells are clean enough to trust, and we know which cell line each new context is. Nothing in this stage changes
data; it reports, and writes masks and suggested weights that the later stages may use.

```
data/controls*/context_<X>.h5ad ──► controls.py ──► out/context_summary*.npz, out/null_calibration*.csv, out/qc/controls_<X>.csv
data/sources/<screen>/          ──► sources.py  ──► out/qc/<source>.npz (keep mask + reason per cell)
controls + Stage 1 tables + HPA ──► identity.py ──► out/ctx_weights.json (suggested same-line weights), identity report
```

## 1. The organisers' control cells (`controls.py`)

`python -m vcell qc controls --contexts D E F`

- **Summary**: depth (UMI per cell), genes per cell, the balance of the 46 non-targeting guides (`ntc_id`, 400 cells each),
  which genes pass the scorer's 5 CPM filter (only those are tested for DE), how strongly the panel targets are expressed
  (a knockdown of a silent gene cannot be seen), and the Spearman correlation between contexts.
- **Null calibration**: each `ntc_id` group is treated as a fake knockdown against the other 45. Its z = beta / s should be
  N(0, 1); sd(z) of 1.01-1.10 on A/B/C showed that the Stage 1 standard error (`vcell.stats.log2fc_and_se`) is honest with
  the organisers' guide structure.
- **Per-cell audit**: depth, genes, mitochondrial share and 5-MAD outliers, with the same rules as the source QC.

Result for D/E/F (EXPERIMENT_LOG F2): already curated (median depth about 20,000 UMI, 400 cells per guide). D is looser
(min 1,001 UMI, 153 low-depth outliers, mito median 1.9%), but **the controls are not filtered**: the real knockdown cells
come through the same pipeline, so filtering what the generator resamples would move our cells away from the cells we are
scored against.

## 2. Cell QC of the public screens (`sources.py`)

`python -m vcell qc sources --sources HCT116 HEK293T H1 KOLF`

Per batch: low depth or few genes (5 MADs below the median), high mitochondrial share (3 MADs above the median and above a
5% floor, or above 20%), doublet-like depth, and cells with guides of more than one construct. Thresholds are computed per
batch over all cells, so a knockdown that really changes depth is judged against its batch, not against the controls. The
mito floor keeps the rule from firing on probe-based 10x Flex data, where mito is near zero.

The masks are written but **not used** for the final tables: rebuilding X-Atlas HCT116 and HEK293T without the flagged
cells was neutral to slightly negative (H52, `experiments/failed/h52_source_qc.py`). The removed high-mito cells sit in
some knockdowns and carry response the lines share; standard single-cell QC can delete the phenotype.

## 3. Which cell line is each context? (`identity.py`, rule in `vcell.model.sources`)

`python -m vcell qc identity --controls data/controls_final --contexts D E F --hpa`

Whole-profile correlation mostly measures the lab (Arc's H1 against another lab's H1: 0.75, the same as unrelated lines),
so identity is judged on on/off features:

| Feature | Meaning |
|---|---|
| markers | stem, T cell, epithelial, mesenchymal, liver, colon genes |
| sex | Y-chromosome genes |
| fingerprint | per source line, genes on there and off in every other line; the share of them on in the context |
| contradiction | per source line, the share of its silent genes that the context expresses (same cell type across labs 0.00-0.06, different lines 0.26-0.74) |
| HPA match | Pearson with the 1,206 cell lines of the Human Protein Atlas, on centred log expression of the most variable genes |

**The rule** (`model/sources.py`): a context that matches a source line with panel data (contradiction <= 0.10) gets that
line at weight 4 in the pool; a stem-like context (POU5F1 >= 100 CPM) gets Pan_H1 at 4. Weight 4 was best in the proxy
(H44) and on the official scorer (+0.06-0.07, H47), and still best for a noisy source (H53). Related but different
lineages get nothing extra (H48).

Result: A = Jurkat, B = HeLa, C = CAL-33, D = MOLM-13, E = U-87 MG, F = an hTERT fibroblast (F4). No source line matched
D/E/F; the one same-line screen, a U-87 MG kinome CRISPRi screen for E, was found by a search afterwards and added to
`configs/final_def.toml` by hand.

## Files

| File | Was | Contents |
|---|---|---|
| `controls.py` | eda/01 + the F2 audit | `summary()`, `cell_audit()` |
| `sources.py` | eda/qc.py, eda/62 | `cell_mask()`, `report()`, `audit()` per source |
| `identity.py` | eda/52, 64, 65 | `source_match()`, `marker_panel()`, `hpa_match()` |
