"""Readers for the AnnData (HDF5) files of the organisers and the public sources, with h5py only (no full anndata load)."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config


def decode(a) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a).ravel()]


def read_df_group(g: h5py.Group) -> pd.DataFrame:
    """Read an AnnData obs/var group (encoding 0.1 and 0.2: categoricals, nullable strings, plain arrays)."""
    index_key = g.attrs.get("_index", "_index")
    out = {}
    for k in g.keys():
        if k in ("__categories",):
            continue
        x = g[k]
        if isinstance(x, h5py.Group) and "categories" in x:
            cats = np.array(decode(x["categories"][...]), dtype=object)
            codes = x["codes"][...]
            out[k] = pd.Categorical.from_codes(codes, categories=cats)
        elif isinstance(x, h5py.Group) and "values" in x:
            out[k] = np.array(decode(x["values"][...]), dtype=object)
        elif isinstance(x, h5py.Dataset):
            v = x[...]
            out[k] = np.array(decode(v), dtype=object) if v.dtype.kind in "SO" else v
    df = pd.DataFrame(out)
    if index_key in df.columns:
        df = df.set_index(index_key)
        df.index.name = None
    return df


def load_context(ctx: str, controls=None) -> tuple[sp.csr_matrix, pd.DataFrame, list[str]]:
    """Return (counts CSR cells x genes, obs, gene names) for one control context. About 1 GB in memory."""
    folder = Path(controls) if controls is not None else config.CONTROLS
    with h5py.File(folder / f"context_{ctx}.h5ad", "r") as f:
        X = f["X"]
        shape = tuple(int(v) for v in X.attrs["shape"])
        mat = sp.csr_matrix((X["data"][...], X["indices"][...], X["indptr"][...]), shape=shape)
        obs = read_df_group(f["obs"])
        var = read_df_group(f["var"])
    genes = list(var.index) if len(var) else config.gene_names(folder)
    return mat, obs, genes


def read_replogle_bulk(path: Path) -> dict:
    """Read a Replogle pseudobulk h5ad. Returns X (rows x genes, mean counts per cell), per-row metadata and gene symbols."""
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        gt = decode(obs["gene_transcript"][...])
        meta = pd.DataFrame({
            "gene_transcript": gt,
            "cells": obs["num_cells_filtered"][...].astype(float),
            "umi": obs["UMI_count_unfiltered"][...].astype(float),
            "n_de": obs["anderson_darling_counts"][...],
            "kd": -obs["pct_expr"][...],           # knockdown fraction of the target itself
        })
        parts = [g.split("_") for g in gt]
        meta["symbol"] = ["_".join(p[1:-2]) for p in parts]
        meta["ensembl"] = [p[-1] for p in parts]
        meta["is_nt"] = meta.symbol.str.contains("non-targeting")
        var = f["var"]
        cats = decode(var["__categories"]["gene_name"][...])
        genes = [cats[c] for c in var["gene_name"][...]]
        X = f["X"][...].astype(np.float64)
    return {"X": X, "meta": meta, "genes": genes}


def csr_rows(path: Path, start: int, stop: int) -> sp.csr_matrix:
    """Rows start..stop of the CSR X of an h5ad (a prediction file, for instance), without reading the rest."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        ip = X["indptr"][start: stop + 1].astype(np.int64)
        lo, hi = int(ip[0]), int(ip[-1])
        return sp.csr_matrix((X["data"][lo:hi], X["indices"][lo:hi], ip - lo), shape=(stop - start, int(X.attrs["shape"][1])))
