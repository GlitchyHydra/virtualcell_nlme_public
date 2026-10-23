"""What cell type is each context? (was eda/52_context_match.py, eda/64_identify_contexts.py and eda/65_hpa_match.py)

Whole-profile correlation mostly measures the lab and pipeline (Arc's H1 vs Pan's H1: Spearman 0.75, the same as unrelated
lines; Pan vs Feng, one pipeline, 0.93), so the match to a source line is judged on on/off features:
  markers       stem (POU5F1, DNMT3B, SOX2), T cell (CD3E, PTPRC, CD4), epithelial (EPCAM, KRT8, KRT18, KRT5), VIM
  sex           Y-chromosome genes (RPS4Y1, DDX3Y, UTY, EIF1AY, KDM5D); XIST is not on the panel axis
  fingerprint   for each source line, genes at >= 20 CPM there and < 2 CPM in every other line that measures them;
                the share of them at >= 10 CPM in the new context
  contradiction for each line, the share of its silent genes (< 1 CPM, measured; >= 20 CPM in >= 2 lines) that the context
                expresses at >= 20 CPM. Not measurable for K562, Jurkat, HepG2, RPE1 (their tables list expressed genes only)
The weights that follow from a match are model.sources.suggest (same-line rule).

    source_match()   per context: markers, sex, fingerprint shares, contradictions -> suggested ctx weights (OUT/ctx_weights.json)
    marker_panel()   a broad lineage marker panel for A-F and the genes most specific to each -> OUT/context_pseudobulk_cpm.csv
    hpa_match()      Pearson of each context with the 1,206 lines of the Human Protein Atlas (needs marker_panel first)

    python -m vcell qc identity [--controls data/controls_final --contexts D E F] [--hpa]
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from vcell import config, stats
from vcell.etl.stage1.table import load
from vcell.model import sources

SEX = {"K562": "F", "HCT116": "M", "HEK293T": "F", "H1": "M", "KOLF": "M", "Pan_H1": "M", "Feng_iPSC": "mixed",
       "jurkat": "M", "hepg2": "M", "rpe1": "F", "CD4_Rest": "donors"}
MARKERS = ["POU5F1", "DNMT3B", "SOX2", "LIN28B", "CD3E", "PTPRC", "CD4", "HBG1", "HBG2", "GATA1", "ALB", "AFP", "APOA2",
           "CDX2", "EPCAM", "KRT8", "KRT18", "KRT5", "VIM"]          # stem, T, K562, HepG2, colon, epithelial, mesenchymal
YGENES = ["RPS4Y1", "DDX3Y", "UTY", "EIF1AY", "KDM5D"]


def cd4_ctrl_profile(tables=None) -> np.ndarray:
    """CD4+ T control CPM on the axis of the tables folder (etl cd4-ctrl writes it next to the tables)."""
    return np.load(Path(tables or config.TABLES) / "cd4_rest_ctrl_cpm.npy").astype(float)


def source_profiles() -> tuple[list[str], dict]:
    """Control CPM of every source line on the Stage 1 gene axis (CD4+ T: the DESeq2 baseMean profile of etl cd4-ctrl)."""
    genes = list(load("K562")["genes"])
    prof = {n: load(n)["ctrl_cpm"].astype(float) for n in ("K562", "HCT116", "HEK293T", "H1", "KOLF", "Pan_H1", "Feng_iPSC")}
    for n in ("jurkat", "hepg2", "rpe1"):
        prof[n] = load(f"ess_{n}", config.OUT)["ctrl_cpm"].astype(float)
    prof["CD4_Rest"] = cd4_ctrl_profile()
    return genes, prof


def fingerprints(prof: dict, n_genes: int, min_on=20.0, max_off=2.0) -> dict:
    fp = {}
    for n, p in prof.items():
        others = [q for m, q in prof.items() if m != n]
        off = np.ones(n_genes, bool)
        for q in others:
            off &= ~(np.isfinite(q) & (q >= max_off))
        measured = sum(np.isfinite(q) for q in others) >= 3
        fp[n] = np.flatnonzero(np.isfinite(p) & (p >= min_on) & off & measured)
    return fp


def contradiction(p: np.ndarray, on2: np.ndarray, v: np.ndarray) -> float:
    sil = np.isfinite(p) & (p < 1) & on2 & np.isfinite(v)
    return float(np.mean(v[sil] >= 20)) if sil.sum() >= 50 else np.nan


def context_cpm(path) -> dict:
    a = ad.read_h5ad(path)
    return dict(zip(a.var_names.astype(str), stats.context_gene_cpm(a.X.tocsr())))


def source_match(controls=None, contexts=None, out=None) -> dict:
    """Identity report per context and the suggested per-context weights, written as JSON (default OUT/ctx_weights.json).
    The final configs copy the weights they use (configs/final_def.toml); the U87 entry for E was added by hand (F8),
    because the U87 table has too few genes to fingerprint."""
    warnings.filterwarnings("ignore")
    controls = config._repo_path(controls) if controls else config.CONTROLS
    genes, prof = source_profiles()
    on2 = sum(np.nan_to_num(q) >= 20 for q in prof.values()) >= 2
    fp = fingerprints(prof, len(genes))
    print("fingerprint genes per source line: " + ", ".join(f"{n} {len(v)}" for n, v in fp.items()))
    ctxs = contexts or config.manifest(controls)["contexts"]
    report, weights = [], {}
    for ctx in ctxs:
        cp = context_cpm(controls / f"context_{ctx}.h5ad")
        v = np.array([cp.get(g, np.nan) for g in genes])
        mk = {g: round(cp.get(g, np.nan), 1) for g in MARKERS}
        xist, y = cp.get("XIST", np.nan), np.nansum([cp.get(g, 0.0) for g in YGENES])
        sex = "M" if y > 20 else "F or Y lost" if y < 2 else "?"
        fps = {n: float(np.mean(v[idx] >= 10)) if len(idx) else np.nan for n, idx in fp.items()}
        con = {n: contradiction(prof[n], on2, v) for n in prof}
        stem = mk["POU5F1"] >= sources.STEM_POU5F1_CPM
        tcell = mk["CD3E"] >= 50 or mk["PTPRC"] >= 50
        report.append({"context": ctx, "sex (XIST, Y)": f"{sex} ({xist:.0f}, {y:.0f})", "stem": stem, "T cell": tcell,
                       **{f"fp {n} ({len(fp[n])})": round(s, 2) for n, s in fps.items()},
                       **{f"contra {n}": round(c, 3) for n, c in con.items()}, **mk})
        w = sources.suggest(con, fps, {n: len(i) for n, i in fp.items()}, mk["POU5F1"])
        if w:
            weights[ctx] = w
            print(f"{ctx}: suggested weights {w}", flush=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame(report).set_index("context").T.to_string())
    print("\nsex of the source lines:", SEX)
    out = config._repo_path(out) if out else config.OUT / "ctx_weights.json"
    out.write_text(json.dumps(weights, indent=1))
    print(f"\nper-context weights -> {out}: {weights or 'none (every context keeps the defaults)'}")
    return weights


PANEL = {
    "stem": ["POU5F1", "NANOG", "LIN28A", "DNMT3B", "SOX2", "TDGF1"],
    "T/NK": ["CD3E", "CD3D", "CD2", "CD7", "LCK", "GNLY", "NKG7"],
    "B": ["CD19", "MS4A1", "CD79A", "PAX5"],
    "myeloid": ["PTPRC", "CD4", "LYZ", "CD14", "CD68", "CSF1R", "SPI1", "ITGAM", "CD33", "MPO", "TYROBP", "FCER1G", "S100A8"],
    "erythroid/mega": ["HBG1", "GATA1", "GYPA", "ITGA2B", "TAL1"],
    "fibroblast/MSC": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "PDGFRB", "THY1", "NT5E", "ENG", "FAP", "TWIST1", "PRRX1"],
    "smooth muscle": ["ACTA2", "TAGLN", "CNN1", "MYH11", "DES"],
    "RPE/eye": ["RPE65", "BEST1", "RLBP1", "TTR", "SERPINF1", "PAX6", "OTX2", "SIX3"],
    "melanocyte": ["MITF", "PMEL", "TYR", "MLANA", "SOX10", "DCT"],
    "neural/glial": ["NES", "GFAP", "OLIG2", "S100B", "SOX9", "STMN2", "TUBB3", "ELAVL4", "PHOX2B", "MYCN"],
    "endothelial": ["PECAM1", "CDH5", "VWF", "KDR"],
    "epithelial": ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT7", "KRT5", "KRT14", "TP63"],
    "liver/gut": ["ALB", "APOA1", "TF", "AFP", "CDX2", "VIL1", "LGALS4"],
    "kidney/lung/breast": ["PAX8", "PAX2", "NKX2-1", "SFTPC", "ESR1", "GATA3", "AR"],
    "bone/muscle": ["RUNX2", "SP7", "ALPL", "MYOD1", "MYOG"],
    "sex": ["XIST", "RPS4Y1", "DDX3Y", "UTY", "KDM5D"],
    "immortal/tumour": ["TERT", "CDKN2A", "MKI67", "MYC"],
}
DIRS = {**{c: config.REPO / "data" / "controls" for c in "ABC"}, **{c: config.REPO / "data" / "controls_final" for c in "DEF"}}


def pseudobulk_cpm(ctx: str) -> pd.Series:
    a = ad.read_h5ad(DIRS[ctx] / f"context_{ctx}.h5ad")
    tot = np.asarray(a.X.sum(axis=0)).ravel().astype(float)
    return pd.Series(tot / tot.sum() * 1e6, index=a.var_names)


def marker_panel(contexts=tuple(DIRS)) -> pd.DataFrame:
    cpm = pd.DataFrame({c: pseudobulk_cpm(c) for c in contexts})
    pd.set_option("display.width", 200)
    print("mean CPM (pseudobulk over 18,400 control cells)")
    for group, gs in PANEL.items():
        gs = [g for g in gs if g in cpm.index]
        print(f"\n[{group}]")
        print(cpm.loc[gs].round(1).to_string())
    print("\n\ngenes most specific to each context: log2((CPM + 1) / (max of the other five + 1)), CPM >= 50")
    for c in cpm.columns:
        others = cpm.drop(columns=c).max(axis=1)
        lr = np.log2((cpm[c] + 1) / (others + 1))
        top = lr[cpm[c] >= 50].sort_values(ascending=False).head(40)
        print(f"\n{c}: " + ", ".join(f"{g} {cpm.at[g, c]:.0f} ({v:.1f})" for g, v in top.items()))
    cpm.to_csv(config.OUT / "context_pseudobulk_cpm.csv")
    return cpm


def hpa_match(top: int = 12, n_genes: int = 3000) -> None:
    """HPA nTPM (bulk RNA-seq, another lab): both sides log2(x + 1) on the n_genes most variable HPA genes, centred by the
    HPA mean of each gene to cancel gene-wise platform bias; Pearson of each centred context with each centred HPA line."""
    hpa = pd.read_csv(config.SOURCES / "hpa" / "rna_celline.tsv", sep="\t", usecols=["Gene name", "Cell line", "nTPM"])
    hpa = hpa.pivot_table(index="Gene name", columns="Cell line", values="nTPM", aggfunc="sum")
    ctx = pd.read_csv(config.OUT / "context_pseudobulk_cpm.csv", index_col=0)
    genes = hpa.index.intersection(ctx.index)
    H = np.log2(hpa.loc[genes] + 1)
    X = np.log2(ctx.loc[genes] + 1)
    var = H.var(axis=1)
    sel = var[(H.max(axis=1) >= 3)].sort_values(ascending=False).index[:n_genes]
    Hc = H.loc[sel].sub(H.loc[sel].mean(axis=1), axis=0)
    Xc = X.loc[sel].sub(H.loc[sel].mean(axis=1), axis=0)
    Hz = (Hc - Hc.mean()) / Hc.std()
    print(f"{hpa.shape[1]} HPA lines, {len(genes)} shared genes, {len(sel)} most variable used\n")
    for c in Xc.columns:
        x = (Xc[c] - Xc[c].mean()) / Xc[c].std()
        r = (Hz.mul(x, axis=0)).sum() / (len(sel) - 1)
        best = r.sort_values(ascending=False).head(top)
        print(f"{c}: " + ", ".join(f"{n} {v:.2f}" for n, v in best.items()))
        for line in ("HAP1", "hTERT-RPE1", "THP-1", "MOLM-13", "MV4-11", "U-87 MG ATCC", "U-251 MG", "BJ hTERT+", "HL-60", "NB-4"):
            if line in r.index:
                print(f"     {line}: {r[line]:.2f} (rank {int((r > r[line]).sum()) + 1})")
