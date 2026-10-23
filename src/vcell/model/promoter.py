"""Promoter neighbours of CRISPRi targets (H37; was eda/promoter.py): genes whose TSS lies within a few kb of the
target's are silenced too.

    tss_table()                       GENCODE v47 gene TSS for uniquely named genes (cached in eda/out)
    neighbour_pairs(targets, genes)   (ti, gj, distance) for every target and axis gene within 5 kb on the same chromosome
    neighbour_cap(pairs, f)           log2 cap: f of the expression left at <= 500 bp, rising log-linearly to 1 at 5 kb
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd

from vcell import config

GTF = config.SOURCES / "gencode" / "gencode.v47.annotation.gtf.gz"
TSS = config.OUT / "gene_tss_gencode47.csv"


def tss_table() -> pd.DataFrame:
    if TSS.exists():
        return pd.read_csv(TSS)
    rows = []
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]*)"', f[8])
            if m:
                rows.append((m.group(1), f[0], f[6], int(f[3] if f[6] == "+" else f[4])))
    t = pd.DataFrame(rows, columns=["gene", "chrom", "strand", "tss"])
    t = t[~t.gene.duplicated(keep=False)]
    t.to_csv(TSS, index=False)
    return t


def neighbour_pairs(targets, genes, max_d: int = 5000, tss: pd.DataFrame | None = None) -> pd.DataFrame:
    """(ti, gj, distance) for every target ti and axis gene gj within max_d bp of its TSS (tss: a table like tss_table())."""
    t = tss_table() if tss is None else tss
    gi = {g: j for j, g in enumerate(genes)}
    by_chrom = {c: d for c, d in t[t.gene.isin(gi)].groupby("chrom")}
    tt = t.set_index("gene")
    rows = []
    for i, tg in enumerate(targets):
        if tg not in tt.index or tt.loc[tg].chrom not in by_chrom:
            continue
        a = tt.loc[tg]
        d = by_chrom[a.chrom]
        for _, b in d[(d.tss - a.tss).abs() <= max_d].iterrows():
            if b.gene != tg:
                rows.append((i, gi[b.gene], abs(int(a.tss) - int(b.tss))))
    return pd.DataFrame(rows, columns=["ti", "gj", "distance"])


def neighbour_cap(pairs: pd.DataFrame, f: float) -> np.ndarray:
    ramp = np.clip(np.log(np.maximum(pairs.distance.to_numpy(float), 500) / 500) / np.log(10), 0, 1)
    return np.log2(f + (1 - f) * ramp)


LENGTH = config.OUT / "gene_length_gencode47.csv"


def gene_lengths() -> dict:
    """Genomic length (end - start) of every GENCODE v47 gene by name, the first entry of a repeated name (cached)."""
    if LENGTH.exists():
        d = pd.read_csv(LENGTH)
        return dict(zip(d.gene, d.length))
    out = {}
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            m = re.search(r'gene_name "([^"]+)"', f[8])
            if m:
                out.setdefault(m.group(1), int(f[4]) - int(f[3]))
    pd.DataFrame({"gene": list(out), "length": list(out.values())}).to_csv(LENGTH, index=False)
    return out
