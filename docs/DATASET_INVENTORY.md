# Dataset inventory (2026-10-19, updated 2026-10-20)

Every dataset considered for this challenge, where it lives, whether it is used, and whether to re-test it when the final
panel (contexts D, E, F) arrives on 2026-10-22. Coverage numbers are for the 300 validation (A/B/C) targets, from
`archive/DATASET_PREPARATION.md` section 3. Challenge FAQ: any data may be used if you have the appropriate permissions.

"Essential library" below means the roughly 2,393 DepMap common-essential genes that the Replogle essential screen, RPE1, and
the two Nadig screens (HepG2, Jurkat) all chose as perturbation targets. The four screens knock down the same genes; none of
the 300 validation targets is among them. For those screens the target rows are therefore useless for the panel, but their
response-gene columns, their between-line spread and their effect sizes may still be informative (see "Why not used").

Re-test flag: **YES** = rerun coverage and a proxy check on 2026-10-22; **MAYBE** = only if the final panel changes the coverage;
**NO** = not useful for any panel.

| # | Dataset | Line(s) | Panel targets covered | Size and format | Location | Licence | Status | Why not used / note | Re-test for final |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Replogle K562 genome-wide, pseudobulk | K562 | 272 | 0.37 GB h5ad | `data/sources/replogle/K562_gwps_raw_bulk_01.h5ad`; https://plus.figshare.com/ndownloader/files/35774443 | CC BY 4.0 | **Used** (source line) | | YES |
| 2 | Replogle K562 genome-wide, single cells | K562 | 272 | 65.8 GB h5ad | `data/sources/replogle/K562_gwps_raw_singlecell_01.h5ad`; https://plus.figshare.com/ndownloader/files/35775507 | CC BY 4.0 | Built (`stage1_K562sc.npz`), **not used** | Proxy 0.145 vs 0.147 for pseudobulk (H20); pseudobulk s already calibrated | NO |
| 3 | Replogle K562 essential, pseudobulk | K562 | 0 | 0.08 GB | `data/sources/replogle/K562_essential_raw_bulk_01.h5ad` | CC BY 4.0 | Not used | Essential library; no panel target. Possible use: between-line variance and effect scale for essential targets | MAYBE |
| 4 | Replogle K562 essential, single cells | K562 | 0 | 9.9 GB | not downloaded; https://plus.figshare.com/ndownloader/files/35775507 (page lists one id for both; check the article) | CC BY 4.0 | Not used | Same as 3 | MAYBE |
| 5 | Replogle RPE1, pseudobulk | RPE1 | 0 | 0.10 GB | `data/sources/replogle/rpe1_raw_bulk_01.h5ad` | CC BY 4.0 | Not used | Essential library | MAYBE |
| 6 | Replogle RPE1, single cells | RPE1 | 0 | 8.1 GB | `data/sources/replogle/rpe1_raw_singlecell_01.h5ad`; https://plus.figshare.com/ndownloader/files/35775606 | CC BY 4.0 | **Table built** (`stage1_ess_rpe1.npz`, 2,394 essential targets) | Used as a held-out check line (H21, H26, H29) | MAYBE |
| 7 | Nadig HepG2, single cells | HepG2 | 0 | 5.2 GB (dense float32) | `data/sources/nadig/GSE264667_hepg2_raw_singlecell_01.h5ad`; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667 | not stated | **Table built** (`stage1_ess_hepg2.npz`) | Used as a held-out check line (H21, H26, H29) | MAYBE |
| 8 | Nadig Jurkat, single cells | Jurkat | 0 | 8.7 GB (dense float32) | `data/sources/nadig/GSE264667_jurkat_raw_singlecell_01.h5ad`; same GEO page | not stated | **Table built** (`stage1_ess_jurkat.npz`) | Used as a held-out check line (H21, H26, H29). Control profile is the closest match to context A (Spearman 0.71) | MAYBE |
| 9 | Zhu et al., CD4+ T genome-scale (published in Cell 2026: about 22 million cells, rest and re-stimulation; https://www.cell.com/cell/fulltext/S0092-8674(26)00929-3) | CD4+ T, 4 donors | 291 with DE | DE statistics 16.8 GB, read remotely row by row | `out/stage1_CD4_Rest.npz`, `stage1_CD4_Stim8hr.npz`, `stage1_CD4_Stim48hr.npz`, `stage1_ess_CD4_Rest.npz` (tables only); https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad | MIT | **Used** (resting condition) | Stim48hr as an extra source: pds up on all held-out lines, overall marginal (0.146 vs 0.145, H31); Stim8hr hurts K562. Same donors as Rest, so not independent. Candidate for a T-cell-like context only | YES |
| 10 | X-Atlas/Orion HCT116 | HCT116 | 300 | 46.6 GB Parquet | `data/sources/xatlas/HCT116/`; https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion | CC BY-NC-SA 4.0 | **Used** (source line) | Licence vs prize terms is open | YES |
| 11 | X-Atlas/Orion HEK293T | HEK293T | 300 | 79.7 GB Parquet | `data/sources/xatlas/HEK293T/` (same repo) | CC BY-NC-SA 4.0 | **Used** (source line) | Median knockdown only 52%; weighting by knockdown not yet tested. Licence open | YES |
| 12 | VCC 2025 H1 (train, validation, test) | H1 stem cells | 25 | ~15 GB train file + benchmark bundle | `data/sources/h1/`, `data/h1_benchmark/` | organisers' | **Used** (held-out benchmark, and a source in the uploads) | | YES |
| 13 | KOLF2.1J iPSC atlas (Nature Biotechnology 2026: https://www.nature.com/articles/s41587-026-03199-w; raw reads SRA PRJNA1173491) | KOLF2.1J | 282 | 189 GB CSC h5ad, 2.66M cells x 37,567 genes | `data/sources/kolf/` (metadata, accumulators); `out/stage1_KOLF.npz`; https://ndownloader.figshare.com/files/64650261; article https://plus.figshare.com/articles/dataset/27261219 | CC BY 4.0 | **Table built** (streamed, `vcell.etl.stage1.kolf`); tested | Leaderboard with KOLF 0.157 vs 0.160 without (L4); in the proxy it helps only the other stem line, H1 (H28). Controls pooled over batches (s about 16% too small). Keep for a stem-like final context and as the licence-clean fallback for 10 and 11 | YES (for a stem-like context) |
| 14 | KOLF2.1J strong perturbations, chromatin modifiers, metabolic enzymes | KOLF2.1J | subsets | 46.7 GB, 4.6 GB, 7.0 GB, plus two small | figshare files 64650852, 64680606, 59541377, 64680708, 64680753 | CC BY 4.0 | Not used | Subsets of 13 | NO |
| 15 | Jiang et al. 2025 | A549, MCF7, HT29, HAP1, BxPC3, K562 | 9 | Seurat objects, DE tables 0.32 GB | https://zenodo.org/records/14518762 (earlier note lists 10520190) | CC BY 4.0 | Not used | Cells stimulated with cytokines for 24 h; 9 panel targets; R objects. DE tables could check between-line variance across six lines | MAYBE |
| 16 | McFaline-Figueroa et al. 2024 | A172, T98G, U87MG | 23 (kinases); **D/E/F panel: 21** | sci-Plex (three-level combinatorial indexing), low depth; CRISPRi sample GSM7056149: 6.0 GB UMI matrix (gz), 115 MB cell annotations, 3,163 guides for 523 kinases | guide list in `data/sources/mcfaline/`; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE225775 | not stated | Not used | Combined with drug exposure; kinases only. **Context E is U-87 MG** (F4), so this is the only same-line source for any final context: vehicle-treated U87MG cells, 21 of 300 targets | **YES (E)** |
| 17 | Srivatsan et al. 2020 (sci-Plex) | A549, MCF7, K562 | 0 | | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944 | not stated | Not used | 188 compounds, no gene knockdown | NO |
| 18 | Tahoe-100M, scBaseCount | many | 0 | very large | | | Not used | Small-molecule / unperturbed; no CRISPRi targets | NO |
| 19 | VIPerturb-seq (Bradu, Satija et al., bioRxiv 2026.02.12.705613) | not stated in the abstract | unknown | an unbiased genome-wide CRISPRi screen of about 880,000 cells, plus enriched screens | https://www.biorxiv.org/content/10.64898/2026.02.12.705613v1; https://satijalab.org/viperturb-seq/ | preprint CC BY-NC-ND 4.0; data licence unknown | **Not available** | Same chemistry as the challenge (10x Flex) and genome-wide CRISPRi (GuEST-List library). The preprint and lab pages state neither the cell line nor a data deposit; check the bioRxiv "Data/Code" tab or ask the authors | MAYBE (if data appears) |
| 20 | Genome-scale single-cell CRISPRi map across human pluripotent stem cell lines (Cell Genomics 2025/26) | several hPSC lines | not checked | not checked | https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00332-5 (page blocks automatic reading) | not checked | Not used | Stem cells only; probably overlaps the KOLF atlas. Worth a look only if a final context is stem-like | MAYBE (stem context) |
| 21 | X-Atlas/Pisces (Xaira, preprint bioRxiv 10.64898/2026.03.18.712807; X-Cell model) | 16 contexts incl. HCT116, HEK293T, **HepG2 genome-wide (9,735 perturbations)**, iPSCs, **Jurkat resting and CD3/CD28-activated**, differentiating iPSCs; 25.6M cells | probably most | not yet uploaded (24 kB on 2026-10-20) | https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Pisces | CC BY-NC-SA 4.0 | **"Coming soon"**, checked 2026-10-20 | Would add genome-wide Jurkat (closest to context A) and HepG2 as source lines. Recheck before 2026-11-05 | YES (if uploaded) |
| 22 | Feng et al., Cell Genomics 2026: iPSC CRISPRi (34 lines, 26 donors), via the HKU reprocessing (row 23): Feng-gwsf + Feng-gwsnf + Feng-ts | iPSC | 30 + 100 + 5 (axis: 270 of 575) | effect files 1.38 + 2.17 + 0.40 GB | `data/sources/hku/effect_Feng-*_moments.h5ad`; table `out/stage1_Feng_iPSC.npz` (`vcell.etl.stage1.hku`) | CC BY-NC-SA 4.0 (reprocessing) | **Table built** 2026-10-20; tested in H40 | Stem source for a stem-like D/E/F context | YES (stem context) |
| 23 | HKU uniformly processed CRISPRi screens (Liu ... Wang, bioRxiv 2026.10.05.756839; doi 10.25442/hku.34002315): 12 screens, effect / DE / single-cell / split-half files | see rows 1-13, 22, 24 | | 6.8 GB downloaded (effect files of VCC, Pan, Feng) of ~1 TB | `data/sources/hku/` (README.md, MANIFEST.csv, MD5-checked) | CC BY-NC-SA 4.0 | Partly used | Effects are differences of mean log1p(1e4) expression; converted to our log2FC with a per-bin factor calibrated on VCC-H1 (r 0.94-0.99 for expressed genes, s within 1-5%). Nourreddine-iPSC = KOLF2.1J (row 13), Huang = X-Atlas (rows 10, 11) | |
| 24 | Pan et al., bioRxiv 2026: H1 hESC CRISPRi, 2,975 growth/pluripotency targets (HKU `Pan-GW-hESC`) | H1 | 66 (axis: 188 of 575) | effect file 2.83 GB | `data/sources/hku/effect_Pan-GW-hESC_moments.h5ad`; table `out/stage1_Pan_H1.npz` | CC BY-NC-SA 4.0 (reprocessing) | **Table built** 2026-10-20; tested in H40 | Same cell line as VCC-H1, another lab. If a final context is H1-like this is a direct source | YES (stem context) |
| 25 | ELEM-1200RS (ELEM Biotech, Hugging Face `ELEM-Bio/ELEM-1200RS`, v1.0; checked 2026-10-22) | 60 lines after name clean-up, mostly HAP1 (about 820 of 1,171 samples); also HCT116, HEK293/HEK293T, A549, Caco-2, HaCaT, PANC-1, T24, MRC-5 and others | unknown: 908 targets, list in the gated metadata xlsx | 4.9 GB: four TPM h5ad (samples x genes) plus per-sample read counts in tar archives | not downloaded (gated: accept the terms on the page) | custom: AI training, fine-tuning and research; attribution required in any competition disclosure; derivatives under the same licence | **Not used** | **Bulk RNA-seq of CRISPR knockout** (RNP, KO pools > 70% or homozygous clones), not CRISPRi single cells. About one sample per target (833 of 908 targets). Controls are unedited wild-type cells, no non-targeting guides: HAP1 has one control for 543 samples, 13 lines (incl. HCT116) have none (promised for 2026-10-23). Samples pooled from 2-3 years of projects with varying time points. So no usable standard error and clone and batch effects mixed into every effect | MAYBE: only if a D/E/F context is one of its lines, that line has controls, and it covers enough panel targets. 2026-10-22: gated "manual" (owner approval), no HF login on this machine, so the metadata sheet (targets per line) is unread. The public README names **U87MG** (= context E) and **THP-1** (AML like D, but not MOLM-13) and BJ-5ta (hTERT fibroblast, male; F has no Y) among the lines; U87MG is not among the 13 lines without controls. README answers: all CRISPR KO by RNP (not CRISPRi), WT controls unedited, 2-3 clones per target only where replicated, time points vary by line, about 38% of samples show < 1.4-fold target mRNA loss. Worth a look only for U87MG: needs access to count its panel targets. **2026-10-23, metadata read (access granted; `data/sources/elem/`: sheet, README, P1-P4 TPM h5ad, 190 MB): NO.** 1,209 samples, 61 line names; panel targets: U87MG 0 (9 KO clones + 2 WT), BJ-5ta 0 (5 + 2), THP-1 0 (8 + 4); 35 of 300 panel targets in total, 32 of them in HAP1 only (not a test line), plus RAB26 (BEAS-2B), SLC25A51 (HepG2, SH-SY5Y), APEX1 (TE-1). Single bulk KO samples with no noise estimate; as another-cell-type source the expected gain is about 0 (H48, H56) |
| 26 | Human Protein Atlas cell-line RNA atlas (`rna_celline.tsv`, HPA v23, 1,206 lines, nTPM) | reference only | n/a | 0.2 GB zip, 1.07 GB tsv | `data/sources/hpa/`; https://www.proteinatlas.org/download/tsv/rna_celline.tsv.zip | CC BY-SA 3.0 | **Used for identification** (`vcell.qc.identity`) | Names the contexts: A Jurkat, B HeLa, C CAL-33, D MOLM-13, E U-87 MG, F hTERT fibroblast | |

Methods seen in public repos (2026-10-20): github.com/mjh72849-cpu/Virtual-Cell-Challenge-2026 (0.1546: weighted average of
five sources after subtracting each source's mean response, promoter-neighbour rule, exact pseudobulk pinning);
github.com/mmyen/virtual-cell-challenge-2026 (Replogle + Nadig + STRING embeddings; a "target scaling" baseline after
Vollenweider and Bühlmann 2026, PCA linear regression after Ahlmann-Eltze 2025, and SLIM, Hu et al. bioRxiv
10.64898/2026.08.07.743481: STRING embeddings with ridge regression).

## What Arc says about the 2026 data (https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge-2026)

- Six cell lines chosen for diversity in tissue of origin, lineage and disease status, of three kinds: stem cells, immortalised
  lines and cancer lines. Three are validation (A, B, C), three are test (D, E, F); the lines are not named.
- Our marker check of the controls: A is T-cell-like (CD3E 760 CPM), B mesenchymal or epithelial (VIM, KRT8), C squamous
  epithelial (KRT5, EPCAM, SOX2). None expresses POU5F1 or NANOG, so **no validation context is a stem line; a stem line is
  probably among D, E, F.**
- Perturbations kept only if the median on-target knockdown was at least 80%. About 500 cells per perturbation, subsampled to 400;
  about 50,000 non-targeting cells per experiment; median about 20,000 UMIs per cell; 10x Flex probes, Ultima UG100 sequencing.

## Why "same essential library" is not a reason to throw 3 to 8 away

The earlier verdict (skip) was too quick. The argument against: the effect for a given (target, gene) pair can only be
transferred from lines that perturbed that target, and these screens perturbed none of the 300 validation targets. The argument
for, which is yours and is right in part:

- **Response-gene coverage.** Each screen measures its own set of genes; every extra line fills in more of the 18,533-gene
  panel for the shared genes (K562 covers about 71% of testable genes).
- **Stable error.** The essential targets are measured in up to eight lines (the six above plus HCT116, HEK293T, CD4+ T). That
  is the best data we have for the between-line variance (`tau^2`), for how effect size varies from line to line, and for
  whether a scale such as 1.5 holds. The weakness is the regime: essential genes have strong effects (median 156 DE genes in K562
  against 5 for panel targets), so values learned there need a transfer step.
- **A cheap test now.** The five lines already on disk (3 to 8) can be turned into Stage 1 tables for their essential targets
  and used as extra held-out lines for the proxy (Jurkat especially for context A), and to check the z scale and `tau^2` outside
  the four lines we tuned on. That is a small job (single-cell streaming, 5 to 9 GB each).
- **The final panel may change this.** If the D/E/F targets include essential genes, these screens become real sources
  (the 2025 H1 panel overlapped the library: 50 of 150 training targets).

## To do on 2026-10-22

0. Run the per-context source selector (`docs/archive/IDEAS.md`, "Per-context source selector") on the D/E/F control cells.
1. Rerun the coverage count of the new 300 targets against rows 1 to 20.
2. Rows flagged YES are rechecked in the proxy; rows flagged MAYBE are promoted to sources if they cover three or more targets.
3. Ask the organisers whether CC BY-NC-SA data may be used in a prize-eligible entry (rows 10 and 11).
