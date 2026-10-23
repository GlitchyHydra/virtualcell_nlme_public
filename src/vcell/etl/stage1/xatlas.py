"""X-Atlas/Orion (Xaira; HCT116, HEK293T): Parquet files, one cell per row, streamed once into per-target accumulators
(data/sources/xatlas/<line>_accum.npz), from which etl.stage1.table.from_accum builds a table for any target list."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from vcell import config, stats
from vcell.etl.stage1.table import qc_keep, save


def xatlas(line: str, targets, genes, max_files: int | None = None, name: str | None = None, keep_accum: bool = True,
           qc: bool = False, accum_targets=None, ckpt_every: int = 20) -> None:
    """Stream the Parquet files of one X-Atlas/Orion line (HCT116 or HEK293T) once.
    Accumulates, for EVERY target seen (about 18,000; or only `accum_targets`, which needs about 20x less memory) and the
    Non-Targeting cells, the number of cells and the sum and sum of squares of CPM over the gene universe, using cells with
    pass_guide_filter == 1 (two guides of one pair). qc=True also drops the cells eda/qc.py flags (default name <line>_qc).
    Every `ckpt_every` files the sums go to data/sources/xatlas/<name>_partial.npz; a rerun with the same targets resumes
    there. Writes stage1_<name>.npz for `targets`; the accumulators go to data/sources/xatlas/<name>_accum.npz."""
    import pyarrow.parquet as pq

    d = config.SOURCES / "xatlas"
    gm = pd.read_parquet(d / "gene_metadata.parquet")
    gix = {g: i for i, g in enumerate(genes)}
    tok2col = np.full(int(gm.gene_token_id.max()) + 1, -1, np.int32)
    seen = set()
    for tok, gname in zip(gm.gene_token_id.to_numpy(), gm.gene_name.to_numpy()):
        if gname in gix and gname not in seen:                       # first token wins when a symbol repeats
            tok2col[tok] = gix[gname]
            seen.add(gname)
    calls = pd.read_csv(d / f"{line}_filtered_guide_calls_per_cell.csv.gz", usecols=["gene_target"])
    all_t = sorted(set(calls.gene_target.dropna()) - {"Non-Targeting"})
    if accum_targets is not None:
        all_t = sorted(set(all_t) & set(accum_targets))
    code = {t: i for i, t in enumerate(all_t)}
    nt = len(all_t)
    G = len(genes)
    n = np.zeros(nt + 1)
    S = np.zeros((nt + 1, G))
    SS = np.zeros((nt + 1, G))
    files = sorted((d / line).glob(f"{line}_Batch*.parquet"), key=lambda f: int(f.stem.split("Batch")[1]))
    if max_files:
        files = files[:max_files]
    keep_all, offs = qc_keep(line) if qc else (None, None)
    name = name or (f"{line}_qc" if qc else line)
    part = d / f"{name}_partial.npz"
    start = 0
    if part.exists():
        with np.load(part, allow_pickle=True) as z:                  # closed again, so Windows can replace the file later
            if list(z["targets"]) == all_t and list(z["genes"]) == list(genes):
                n, S, SS, start = np.array(z["n"]), np.array(z["S"]), np.array(z["SS"]), int(z["done"])
                print(f"resuming from {part.name} after file {start}", flush=True)
            else:
                print(f"{part.name} holds another target or gene list: starting over", flush=True)
    print(f"{line}: {len(files)} files, {nt} targets accumulated, {int((tok2col >= 0).sum())} of {G} universe genes mapped"
          + (f"; QC drops {int((~keep_all).sum())} rows" if qc else ""), flush=True)
    for k, f in enumerate(files, 1):
        if k <= start:
            continue
        tb = pq.read_table(f, columns=["gene_token_id", "gene_expression", "gene_target", "pass_guide_filter"])
        gt = tb["gene_target"].to_pylist()
        codes = np.array([code.get(t, nt if t == "Non-Targeting" else -1) for t in gt], dtype=np.int64)
        codes[tb["pass_guide_filter"].to_numpy() != 1] = -1
        if qc:                                                       # same file order as eda/62_qc_audit.py
            kp = keep_all[offs[k - 1]:offs[k]]
            assert len(kp) == len(codes), f"QC mask does not match {f.name}"
            codes[~kp] = -1
        ids = tb["gene_token_id"].combine_chunks()
        off = ids.offsets.to_numpy()
        tok = ids.values.to_numpy()
        val = tb["gene_expression"].combine_chunks().values.to_numpy()
        ncell = len(off) - 1
        for a in range(0, ncell, 2000):                              # blocks keep the peak memory small
            b = min(a + 2000, ncell)
            lo, hi = int(off[a]), int(off[b])
            nb = b - a
            row = np.repeat(np.arange(nb), np.diff(off[a:b + 1]))
            tk, vl = tok[lo:hi], val[lo:hi]
            cb = codes[a:b]
            tot = np.bincount(row, weights=vl, minlength=nb)
            keep = (tok2col[tk] >= 0) & (cb[row] >= 0) & (tot[row] > 0)
            r, c, v = row[keep], tok2col[tk[keep]], vl[keep] / tot[row[keep]] * 1e6
            cells = np.flatnonzero((cb >= 0) & (tot > 0))
            if cells.size == 0:
                continue
            remap = np.full(nb, -1)
            remap[cells] = np.arange(len(cells))
            X = sp.csr_matrix((v, (remap[r], c)), shape=(len(cells), G))
            uniq, loc = np.unique(cb[cells], return_inverse=True)
            Ind = sp.csr_matrix((np.ones(len(cells)), (loc, np.arange(len(cells)))), shape=(len(uniq), len(cells)))
            S[uniq] += (Ind @ X).toarray()
            X.data **= 2
            SS[uniq] += (Ind @ X).toarray()
            n[uniq] += np.bincount(loc, minlength=len(uniq))
        if k % 5 == 0 or k == len(files):
            print(f"   file {k}/{len(files)}: {int(n[:nt].sum())} perturbed and {int(n[nt])} control cells so far", flush=True)
        if ckpt_every and k % ckpt_every == 0 and k < len(files):     # write then rename, so a kill never leaves half a file
            tmp = d / f"{name}_partial.tmp.npz"
            np.savez(tmp, n=n, S=S, SS=SS, done=k, targets=np.array(all_t), genes=np.array(genes))
            tmp.replace(part)

    tix = {t: i for i, t in enumerate(targets)}
    have = [t for t in targets if t in code]
    rows = np.array([code[t] for t in have])
    ctrl = S[nt] / n[nt]
    beta_s, se_s = stats.log2fc_and_se(n[rows, None], S[rows], SS[rows], n[nt], S[nt], SS[nt])
    T = len(targets)
    beta = np.full((T, G), np.nan)
    s = np.full((T, G), np.nan)
    nn = np.full(T, np.nan)
    kd = np.full(T, np.nan)
    for j, t in enumerate(have):
        i = tix[t]
        beta[i], s[i], nn[i] = beta_s[j], se_s[j], n[rows[j]]
        if t in gix and ctrl[gix[t]] > 0:
            kd[i] = 1 - (S[rows[j], gix[t]] / n[rows[j]]) / ctrl[gix[t]]
    save(name, targets, genes, beta, s, nn, ctrl, kd)
    if keep_accum:
        np.savez(d / f"{name}_accum.npz", targets=np.array(all_t), genes=np.array(genes), n=n, S=S, SS=SS)
        print("saved accumulators", d / f"{name}_accum.npz")
    part.unlink(missing_ok=True)

