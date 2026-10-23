"""KOLF2.1J iPSC genome-scale CRISPRi atlas (figshare 27261219, CC BY 4.0), streamed by gene column (was eda/27).

The 189 GB h5ad is stored by gene column (CSC, uncompressed), so it is never downloaded whole: for each universe gene its
column (cell indices and counts) is fetched with HTTP range requests, counts become CPM with the cell's total_counts, and
they are added to per-target sums and sums of squares in data/sources/kolf/accum.npz (resumable). Then the usual estimator.

    stream()            the full pass (hours; 76.5 GB on the 12,816-gene axis, F12), then the table
    stream(probe=True)  metadata, byte offsets and the estimated download only
    table_from_accum()  the table again from a finished accum.npz (no network)

figshare's download link redirects to a signed S3 link that expires quickly, so every request resolves a fresh one.
Measured 2026-10-19: 4.5 MB/s single stream, about 7 MB/s with 8 parallel requests.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
import urllib.request

import fsspec
import h5py
import numpy as np

from vcell import config, io, stats
from vcell.etl.stage1.table import all_targets, save, universe_genes

URL = "https://ndownloader.figshare.com/files/64650261"
DIR = config.SOURCES / "kolf"
ACC = DIR / "accum.npz"


def fetch(a: int, n: int, tries: int = 8) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
                link = r.geturl()
            rq = urllib.request.Request(link, headers={"Range": f"bytes={a}-{a + n - 1}"})
            with urllib.request.urlopen(rq, timeout=600) as r:
                out = r.read()
            if len(out) == n:
                return out
        except Exception as e:                                        # network hiccup or an expired link: retry
            print(f"   retry {k + 1} at byte {a}: {e!r}", flush=True)
        time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"range {a}+{n} failed {tries} times")


def meta() -> dict:
    """Cell labels, totals, gene names, column pointers and the chunk tables of the remote file (cached in meta.npz)."""
    DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DIR / "meta.npz"
    if not meta_path.exists():
        t0 = time.time()
        fs = fsspec.filesystem("https")
        with fs.open(URL, "rb", cache_type="blockcache", block_size=4 << 20) as fo, h5py.File(fo, "r") as f:
            X = f["X"]
            print("X", dict(X.attrs), {k: (X[k].shape, str(X[k].dtype), X[k].chunks, X[k].compression) for k in X}, flush=True)
            obs = f["obs"]
            print("obs columns", list(obs.keys()), flush=True)
            g = obs["gene_target"]
            if isinstance(g, h5py.Group):
                cats = np.array(io.decode(g["categories"][...]), dtype=object)
                tg = cats[g["codes"][...]]
            else:
                tg = np.array(io.decode(g[...]), dtype=object)
            tot = obs["total_counts"][...].astype(float)
            v = f["var"]
            idx = v.attrs.get("_index", "_index")
            vg = v[idx]
            if isinstance(vg, h5py.Group):
                genes_k = np.array(io.decode(vg["categories"][...]), dtype=object)[vg["codes"][...]]
            else:
                genes_k = np.array(io.decode(vg[...]), dtype=object)
            print("var index sample", genes_k[:5], "other var columns", list(v.keys())[:12], flush=True)
            indptr = X["indptr"][...]
            tables = {}
            for k in ("data", "indices"):                             # uncompressed chunks: element offset -> byte offset
                ds = X[k]
                rec = []
                ds.id.chunk_iter(lambda c: rec.append((c.chunk_offset[0], c.byte_offset, c.size)))
                tables[k] = np.array(sorted(rec), dtype=np.int64)
                print(f"{k}: {len(rec)} chunks of {ds.chunks[0]} elements", flush=True)
            dt_d, dt_i = str(X["data"].dtype), str(X["indices"].dtype)
            cs = np.array([X["data"].chunks[0], X["indices"].chunks[0]])
        np.savez(meta_path, tg=tg.astype(str), tot=tot, genes=genes_k.astype(str), indptr=indptr,
                 ch_d=tables["data"], ch_i=tables["indices"], cs=cs, dtypes=np.array([dt_d, dt_i]))
        print(f"metadata read in {time.time() - t0:.0f}s", flush=True)
    m = np.load(meta_path, allow_pickle=False)
    return {"tg": m["tg"], "tot": m["tot"], "genes": m["genes"], "indptr": m["indptr"].astype(np.int64),
            "ch_d": m["ch_d"], "ch_i": m["ch_i"], "cs": m["cs"], "dtypes": [np.dtype(x) for x in m["dtypes"]]}


def cell_codes(tg, targets) -> tuple[np.ndarray, list[str]]:
    """Group code per cell: index into targets, len(targets) for the controls, -1 for every other cell."""
    tix = {t: i for i, t in enumerate(targets)}
    labels = sorted(set(tg))
    nt_lab = [l for l in labels if l == "NTC" or ("non" in l.lower() and "target" in l.lower())]
    nt_code = len(targets)
    return np.array([tix.get(t, nt_code if t in nt_lab else -1) for t in tg]), nt_lab


def read_elems(ch: np.ndarray, csize: int, dt: np.dtype, lo: int, hi: int) -> np.ndarray:
    """Elements lo..hi of a chunked, uncompressed 1-D dataset; adjacent chunks are fetched in one range request."""
    c0, c1 = lo // csize, (hi - 1) // csize
    parts, a, end = [], int(ch[c0, 1]), int(ch[c0, 1] + ch[c0, 2])
    for c in range(c0 + 1, c1 + 1):
        if ch[c, 1] == end:
            end += int(ch[c, 2])
        else:
            parts.append(fetch(a, end - a))
            a, end = int(ch[c, 1]), int(ch[c, 1] + ch[c, 2])
    parts.append(fetch(a, end - a))
    x = np.frombuffer(b"".join(parts), dtype=dt)
    return x[lo - c0 * csize: hi - c0 * csize]


def finish(n, S, SS, name, targets, genes, genes_k) -> None:
    """The Stage 1 table from per-target sums (rows: targets, then the controls)."""
    nt_code = len(targets)
    tix = {t: i for i, t in enumerate(targets)}
    gk = set(genes_k)
    missing = [u for u, g in enumerate(genes) if g not in gk]
    gix = {g: u for u, g in enumerate(genes)}
    beta, s = stats.log2fc_and_se(n[:nt_code, None], S[:nt_code], SS[:nt_code], n[nt_code], S[nt_code], SS[nt_code])
    ctrl = S[nt_code] / n[nt_code]
    beta[:, missing] = np.nan
    s[:, missing] = np.nan
    ctrl[missing] = np.nan
    nn = np.where(n[:nt_code] > 0, n[:nt_code], np.nan)
    beta[~np.isfinite(nn)] = np.nan
    kd = np.full(nt_code, np.nan)
    for t, r in tix.items():
        if t in gix and n[r] > 0 and ctrl[gix[t]] > 0:
            kd[r] = 1 - (S[r, gix[t]] / n[r]) / ctrl[gix[t]]
    save(name, targets, genes, beta, s, nn, ctrl, kd)


def stream(probe: bool = False, workers: int = 8, block_mb: float = 24.0) -> None:
    m = meta()
    tg, tot, genes_k, indptr = m["tg"], m["tot"], m["genes"], m["indptr"]
    ch_d, ch_i, cs = m["ch_d"], m["ch_i"], m["cs"]
    dt_d, dt_i = m["dtypes"]
    for nm, ch in (("data", ch_d), ("indices", ch_i)):
        gaps = np.diff(ch[:, 1]) != ch[:-1, 2]
        print(f"{nm}: {len(ch)} chunks, {int(gaps.sum())} breaks in file order", flush=True)
    targets, genes = all_targets(), universe_genes()
    codes, nt_lab = cell_codes(tg, targets)
    nt_code = len(targets)
    print(f"{len(tg)} cells x {len(genes_k)} genes; control labels {nt_lab}; "
          f"{sum(t in set(tg) for t in targets)} of {len(targets)} targets present "
          f"({sum(t in set(tg) for t in config.panel_targets())} of 300 panel)", flush=True)
    QCF = config.OUT / "qc" / "KOLF.npz"                       # cell QC (qc.sources): a second, cleaned table
    codes_q = None
    if QCF.exists():
        qkeep = np.load(QCF)["keep"]
        assert len(qkeep) == len(codes), "QC mask does not match the cells"
        codes_q = np.where(qkeep, codes, -1)
        print(f"QC: {int((~qkeep).sum())} cells dropped in the stage1_KOLF_qc table", flush=True)
    gk = {g: j for j, g in enumerate(genes_k)}
    pairs = sorted((gk[g], u) for u, g in enumerate(genes) if g in gk)      # (column in the file, universe index)
    cols = np.array([p[0] for p in pairs])
    nnz = indptr[cols + 1] - indptr[cols]
    est = nnz.sum() * (dt_d.itemsize + dt_i.itemsize)
    print(f"{len(cols)} of {len(genes)} universe genes in the file; {nnz.sum() / 1e9:.2f}G values, "
          f"{est / 1e9:.0f} GB to fetch (~{est / 7e6 / 3600:.1f} h at 7 MB/s)", flush=True)
    if probe:
        return

    blocks, cur, size = [], [], 0                             # consecutive file columns, about block-mb of data each
    for k, c in enumerate(cols):
        if cur and (c != cols[cur[-1]] + 1 or size > block_mb * 1e6):
            blocks.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += nnz[k] * dt_d.itemsize
    blocks.append(cur)
    G = len(genes)
    if ACC.exists():
        a = np.load(ACC)
        if "targets" in a.files:
            assert list(a["targets"]) == list(targets), f"{ACC} holds another target list: move it away to start over"
        else:
            assert a["S"].shape[0] == nt_code + 1, f"{ACC} holds another target list: move it away to start over"
        S, SS, done = a["S"], a["SS"], int(a["done"])
        if codes_q is not None:
            assert "Sq" in a.files or done == 0, f"{ACC} was started without QC: move it away to start over"
            Sq, SSq = (a["Sq"], a["SSq"]) if "Sq" in a.files else (np.zeros_like(S), np.zeros_like(S))
        print(f"resuming after block {done} of {len(blocks)}", flush=True)
    else:
        S, SS, done = np.zeros((nt_code + 1, G)), np.zeros((nt_code + 1, G)), 0
        if codes_q is not None:
            Sq, SSq = np.zeros_like(S), np.zeros_like(S)
    n = np.bincount(codes[codes >= 0], minlength=nt_code + 1).astype(float)
    scale = 1e6 / np.maximum(tot, 1)

    def load_block(b):
        lo, hi = int(indptr[cols[b[0]]]), int(indptr[cols[b[-1]] + 1])
        d = read_elems(ch_d, int(cs[0]), dt_d, lo, hi)
        i = read_elems(ch_i, int(cs[1]), dt_i, lo, hi)
        return b, lo, d, i

    t0, got = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as ex:
        pending = {}
        nxt = done
        while nxt < len(blocks) and len(pending) < 2 * workers:
            pending[nxt] = ex.submit(load_block, blocks[nxt])
            nxt += 1
        while pending:
            k = min(pending)
            b, lo, d, i = pending.pop(k).result()
            if nxt < len(blocks):
                pending[nxt] = ex.submit(load_block, blocks[nxt])
                nxt += 1
            for kk in b:
                c = cols[kk]
                a_, z_ = int(indptr[c]) - lo, int(indptr[c + 1]) - lo
                rows_, vals = i[a_:z_], d[a_:z_].astype(float)
                cd = codes[rows_]
                ok = cd >= 0
                cpm = vals[ok] * scale[rows_[ok]]
                u = pairs[kk][1]
                S[:, u] += np.bincount(cd[ok], weights=cpm, minlength=nt_code + 1)
                SS[:, u] += np.bincount(cd[ok], weights=cpm ** 2, minlength=nt_code + 1)
                if codes_q is not None:
                    cq = codes_q[rows_]
                    oq = cq >= 0
                    cpq = vals[oq] * scale[rows_[oq]]
                    Sq[:, u] += np.bincount(cq[oq], weights=cpq, minlength=nt_code + 1)
                    SSq[:, u] += np.bincount(cq[oq], weights=cpq ** 2, minlength=nt_code + 1)
            got += (len(d) * dt_d.itemsize + len(i) * dt_i.itemsize)
            done = k + 1
            if done % 25 == 0 or done == len(blocks):
                extra = {"Sq": Sq, "SSq": SSq} if codes_q is not None else {}
                np.savez(ACC, S=S, SS=SS, done=done, targets=np.array(targets), **extra)
                el = time.time() - t0
                print(f"   block {done}/{len(blocks)}: {got / 1e9:.1f} GB in {el / 60:.0f} min ({got / 1e6 / el:.1f} MB/s)",
                      flush=True)

    finish(n, S, SS, "KOLF", targets, genes, genes_k)
    if codes_q is not None:
        finish(np.bincount(codes_q[codes_q >= 0], minlength=nt_code + 1).astype(float), Sq, SSq, "KOLF_qc", targets, genes,
               genes_k)


def table_from_accum(name: str = "KOLF", acc=ACC) -> None:
    """Rebuild stage1_KOLF from a finished accumulator (the cell counts per target come from the cached metadata)."""
    m = meta()
    a = np.load(acc)
    targets = [str(t) for t in a["targets"]]
    codes, _ = cell_codes(m["tg"], targets)
    n = np.bincount(codes[codes >= 0], minlength=len(targets) + 1).astype(float)
    finish(n, a["S"], a["SS"], name, targets, universe_genes(), m["genes"])
