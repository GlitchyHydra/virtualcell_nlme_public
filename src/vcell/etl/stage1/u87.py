"""U-87 MG (= context E, EXPERIMENT_LOG F4) from the McFaline-Figueroa et al. 2024 sci-Plex CRISPRi kinome screen
(GSE225775, sample GSM7056149; dCas9-KRAB, 3,163 guides for 523 kinases + 300 NTC; A172, T98G, U87MG pooled and exposed to
four kinase inhibitors, sci-RNA-seq3 readout). Was eda/67_u87_table.py (vehicle table, F8) and eda/70_u87_drug_adjusted.py
(all drug arms with the average drug effect removed, H54) plus the drug-only table of eda/71 (H55).

Raw GEO files in data/sources/mcfaline/ (the 19 GB processed R object is not needed). Cell filters follow the authors'
notebooks (github cole-trapnell-lab/sci-Plex-GxE, Kinome_GxE_screen):
  cells      >= 500 UMI (Notebook 1)
  hash       top hash oligo >= 5 UMI and top / second >= 2.5 (Notebook 2); the oligo plus the replicate (RT barcode 1-192 ->
             replicate_1, 193-384 -> replicate_2) gives cell line, treatment and dose via hash_whitelist_rep1/2
  sgRNA      guide reads per cell from the enrichment library, matched on the P5/RT/ligation barcode (the guide library has a
             single PCR plate letter, so the RNA plate letter cannot be matched; barcodes shared by two cells are dropped);
             a cell keeps its top guide if it has >= 2 reads and is alone or >= 2x the second (Notebook 1); NTC guides are
             the controls

    build_vehicle()          vehicle wells only -> U87_accum.npz -> stage1_U87 (F8; the uploaded entry F15 uses it)
    build_strata()           every treatment x dose x replicate stratum -> U87_strata_accum.npz
    table_drug_adjusted()    stage1_U87x: per stratum and gene f = mean CPM of all guide-assigned cells / vehicle mean,
                             cells divided by f and pooled over vehicle and every drug arm that passes the interaction rule
                             (mean z^2 vs vehicle <= 1.2 x the null of vehicle rep 1 vs rep 2; F18 uses it)
    table_drug_only()        stage1_U87d: the drug arms alone, the independent source of the H55 held-out check
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from vcell import config, stats
from vcell.etl.stage1.table import all_targets, from_accum, load, save, universe_genes

D = config.SOURCES / "mcfaline"
CHUNK = 20_000_000
MIN_UMI, HASH_MIN, HASH_RATIO, GUIDE_MIN, GUIDE_RATIO = 500, 5, 2.5, 2, 2.0
RULE, MIN_CPM = 1.2, 20.0
STRATA_ACC = D / "U87_strata_accum.npz"


def matrix_chunks():
    return pd.read_csv(D / "UMI.count.matrix.gz", sep="\t", header=None, names=["g", "c", "x"],
                       dtype={"g": np.int32, "c": np.int32, "x": np.int32}, chunksize=CHUNK)


def cell_umi() -> np.ndarray:
    """UMI total per cell (1-based cell index), cached in cell_umi.npy after one pass over the 6 GB matrix."""
    if (D / "cell_umi.npy").exists():
        return np.load(D / "cell_umi.npy")
    n_cells = sum(1 for _ in gzip.open(D / "cell.annotations.txt.gz", "rt"))
    tot = np.zeros(n_cells + 1, np.int64)
    for k, ch in enumerate(matrix_chunks()):
        tot += np.bincount(ch.c.to_numpy(), weights=ch.x.to_numpy(), minlength=len(tot)).astype(np.int64)
        print(f"   pass 1: {(k + 1) * CHUNK / 1e6:.0f}M entries", flush=True)
    np.save(D / "cell_umi.npy", tot)
    return tot


def select_cells(tot: np.ndarray, vehicle_only: bool) -> pd.DataFrame:
    """U87MG cells with an unambiguous hash and one guide: index = cell barcode; columns idx (matrix cell index), treatment,
    dose, replicate, guide, target. vehicle_only keeps the vehicle wells (eda/67), otherwise every arm (eda/70)."""
    idx = np.flatnonzero(tot >= MIN_UMI)
    print(f"{len(tot) - 1} barcodes, {len(idx)} with >= {MIN_UMI} UMI", flush=True)
    parts, off = [], 0                                         # names of the kept barcodes only (43M names would take GBs)
    for ch in pd.read_csv(D / "cell.annotations.txt.gz", sep="\t", header=None, usecols=[0], names=["cell"], chunksize=CHUNK):
        k = idx[(idx > off) & (idx <= off + len(ch))]
        parts.append(pd.DataFrame({"idx": k, "cell": ch.cell.to_numpy()[k - off - 1]}))
        off += len(ch)
    cells = pd.concat(parts).set_index("cell")
    cells["rt"] = cells.index.str.extract(r"RT_BC_(\d+)_", expand=False).astype(int).to_numpy()
    # The guide library was amplified with a single PCR plate letter (E), so it cannot tell the four RNA PCR plates (A-D)
    # apart: guides are matched on the rest of the barcode (P5 well + RT + ligation), and a rest shared by two cells is dropped.
    rest_all = cells.index.to_series().str.slice(4)
    ambiguous = set(rest_all[rest_all.duplicated(keep=False)])
    print(f"{len(ambiguous)} barcode rests shared by two or more cells (dropped from guide matching)", flush=True)
    cells["replicate"] = np.where(cells.rt <= 192, "replicate_1", "replicate_2")

    hs = []                                                    # hash -> cell line, treatment
    for ch in pd.read_csv(D / "hashTable.out.txt.gz", sep="\t", header=None, usecols=[1, 2, 4], names=["cell", "oligo", "umi"],
                          chunksize=CHUNK):
        hs.append(ch[ch.cell.isin(cells.index)])
    hs = pd.concat(hs).groupby(["cell", "oligo"], as_index=False).umi.sum().sort_values(["cell", "umi"], ascending=[True, False])
    top = hs.groupby("cell").head(2).copy()
    top["rank"] = top.groupby("cell").cumcount()
    first = top[top["rank"] == 0].set_index("cell")
    second = top[top["rank"] == 1].set_index("cell").umi.reindex(first.index).fillna(0)
    ok = (first.umi >= HASH_MIN) & (first.umi >= HASH_RATIO * second)
    cells = cells.join(first.loc[ok, ["oligo"]], how="inner")
    wl = pd.concat([pd.read_csv(D / f"hash_whitelist_rep{r}.txt.gz", sep="\t") for r in (1, 2)])
    cells = cells.reset_index().merge(wl, left_on=["oligo", "replicate"], right_on=["hash_oligo", "replicate"]).set_index("cell")
    print("hashed cells by line and treatment:\n" + cells.groupby(["cell_line", "treatment"]).size().unstack().to_string(),
          flush=True)
    keep = cells.cell_line == "U87MG"
    if vehicle_only:
        keep &= cells.treatment == "vehicle"
    cells = cells[keep].copy()
    print(f"U87MG {'vehicle ' if vehicle_only else ''}cells: {len(cells)}", flush=True)

    cells["key"] = cells.index.to_series().str.slice(4)       # sgRNA assignment on P5 well + RT + ligation barcodes
    cells = cells[~cells.key.isin(ambiguous)]
    wanted = set(cells.key)
    gr = []
    for ch in pd.read_csv(D / "gRNATable_reads.out.txt.gz", sep="\t", header=None, usecols=[1, 2, 4],
                          names=["cell", "guide", "reads"], chunksize=CHUNK):
        ch["key"] = ch.cell.str.slice(4)
        gr.append(ch[ch.key.isin(wanted)])
    gr = pd.concat(gr)
    print(f"cells with an unambiguous barcode: {len(cells)}; guide-library plate letters: "
          f"{gr.cell.str.slice(2, 3).value_counts().to_dict()}", flush=True)
    g = gr.groupby(["key", "guide"], as_index=False).reads.sum().sort_values(["key", "reads"], ascending=[True, False])
    g["rank"] = g.groupby("key").cumcount()
    g1 = g[g["rank"] == 0].set_index("key")
    g2 = g[g["rank"] == 1].set_index("key").reads.reindex(g1.index).fillna(0)
    g1 = g1[(g1.reads >= GUIDE_MIN) & (g1.reads >= GUIDE_RATIO * g2)]
    cells = cells.join(g1[["guide"]], on="key", how="inner")
    cells["target"] = cells.guide.str.replace(r"_\d+$", "", regex=True)
    return cells


def gene_columns(genes) -> np.ndarray:
    """1-based gene index of the matrix -> universe column (-1 if off the universe; the first symbol wins)."""
    ga = pd.read_csv(D / "gene.annotations.txt.gz", sep="\t", header=None, names=["id", "sym"])
    gix = {gname: j for j, gname in enumerate(genes)}
    col = np.full(len(ga) + 1, -1, np.int32)
    seen = set()
    for i, s in enumerate(ga.sym, start=1):
        if s in gix and s not in seen:
            col[i] = gix[s]
            seen.add(s)
    print(f"{len(seen)} of {len(genes)} universe genes mapped", flush=True)
    return col


def accumulate(tot, grp, col, n_groups: int, G: int, every: int = 1):
    """n, sum and sum of squares of CPM per (group, universe gene) in one pass over the matrix."""
    n = np.bincount(grp[grp >= 0], minlength=n_groups).astype(float)
    nb = n_groups * G
    S, SS = np.zeros(nb), np.zeros(nb)
    for k, ch in enumerate(matrix_chunks()):
        c, gg, x = ch.c.to_numpy(), ch.g.to_numpy(), ch.x.to_numpy()
        m = (grp[c] >= 0) & (col[gg] >= 0)
        c, gg, x = c[m], gg[m], x[m]
        cpm = x / tot[c] * 1e6
        b = grp[c].astype(np.int64) * G + col[gg]
        S += np.bincount(b, weights=cpm, minlength=nb)
        SS += np.bincount(b, weights=cpm ** 2, minlength=nb)
        if k % every == every - 1:
            print(f"   pass: {(k + 1) * CHUNK / 1e6:.0f}M entries", flush=True)
    return n, S, SS


def build_vehicle() -> None:
    tot = cell_umi()
    cells = select_cells(tot, vehicle_only=True)
    cells = cells[cells.target != "random_region"]
    print(f"U87MG vehicle cells with one guide: {len(cells)} ({int((cells.target == 'NTC').sum())} NTC, "
          f"{cells.target.nunique() - 1} kinases)", flush=True)
    targets, genes = all_targets(), universe_genes()
    acc_t = sorted(set(cells.target) & set(targets))
    code = {t: i for i, t in enumerate(acc_t)}
    nt = len(acc_t)
    grp = np.full(len(tot), -1, np.int32)
    grp[cells.idx.to_numpy()] = [nt if t == "NTC" else code.get(t, -1) for t in cells.target]
    col = gene_columns(genes)
    G = len(genes)
    n, S, SS = accumulate(tot, grp, col, nt + 1, G)
    S, SS = S.reshape(nt + 1, G), SS.reshape(nt + 1, G)
    np.savez(D / "U87_accum.npz", targets=np.array(acc_t), genes=np.array(genes), n=n, S=S, SS=SS)
    print("cells per panel target: " + ", ".join(f"{t} {int(n[code[t]])}" for t in acc_t if t in set(config.panel_targets())),
          flush=True)
    print(f"NTC cells: {int(n[nt])}", flush=True)
    from_accum("U87", targets, genes, "U87", path=D / "U87_accum.npz")
    z = load("U87")
    kd = pd.Series(z["kd"], index=z["targets"]).dropna()
    print("self-knockdown (1 - target CPM / NTC CPM) of the panel targets:\n"
          + kd[kd.index.isin(config.panel_targets())].round(2).to_string())


def build_strata() -> None:
    tot = cell_umi()
    cells = select_cells(tot, vehicle_only=False)
    cells["stratum"] = cells.treatment + "_" + cells.dose.astype(str) + "_" + cells.replicate.str[-1]
    print("U87MG cells with one guide, by stratum:\n" + cells.groupby("stratum").agg(
        cells=("target", "size"), ntc=("target", lambda x: int((x == "NTC").sum()))).to_string(), flush=True)
    targets, genes = all_targets(), universe_genes()
    G = len(genes)
    strata = sorted(cells.stratum.unique())
    acc_t = sorted(set(cells.target) & set(targets))
    code = {t: i for i, t in enumerate(acc_t)}
    nt = len(acc_t)
    K = nt + 2                                                 # targets, NTC (nt), any other guide (nt + 1)
    hix = {h: i for i, h in enumerate(strata)}
    grp = np.full(len(tot), -1, np.int32)
    tc = np.array([nt if t == "NTC" else code.get(t, nt + 1) for t in cells.target])
    grp[cells.idx.to_numpy()] = cells.stratum.map(hix).to_numpy() * K + tc
    col = gene_columns(genes)
    n, S, SS = accumulate(tot, grp, col, len(strata) * K, G, every=5)
    np.savez(STRATA_ACC, strata=np.array(strata), targets=np.array(acc_t), genes=np.array(genes),
             n=n.reshape(len(strata), K), S=S.reshape(len(strata), K, G), SS=SS.reshape(len(strata), K, G))


def _strata():
    z = np.load(STRATA_ACC, allow_pickle=True)
    strata, acc_t, n, S, SS = list(z["strata"]), list(z["targets"]), z["n"], z["S"], z["SS"]
    veh = [i for i, h in enumerate(strata) if h.startswith("vehicle")]
    mean_all = S.sum(1) / n.sum(1)[:, None]                    # stratum x gene, all guide-assigned cells
    ref = S[veh].sum((0, 1)) / n[veh].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((mean_all > 0) & (ref > 0), mean_all / ref, 1.0)
    return strata, acc_t, n, S, SS, veh, f, ref >= MIN_CPM


def _pooled(sel, n, S, SS, f, nt):
    """log2FC and s of every target vs NTC from the adjusted, pooled cells of strata `sel` (order kept: it fixes the sums)."""
    w = 1.0 / f[sel]
    ns = n[sel].sum(0)
    s1 = (S[sel] * w[:, None, :]).sum(0)
    s2 = (SS[sel] * (w ** 2)[:, None, :]).sum(0)
    b, s = stats.log2fc_and_se(ns[:nt, None], s1[:nt], s2[:nt], ns[nt], s1[nt], s2[nt])
    return b, s, ns, s1, s2


def table_drug_adjusted(name: str = "U87x") -> None:
    strata, acc_t, n, S, SS, veh, f, good = _strata()
    nt = len(acc_t)
    print(f"{len(strata)} strata; average drug effect |log2 f| (genes >= {MIN_CPM:.0f} CPM), median by stratum: " +
          ", ".join(f"{h} {np.median(np.abs(np.log2(f[i, good]))):.3f}" for i, h in enumerate(strata)), flush=True)

    def z2(a, b):
        (b1, s1), (b2, s2) = a[:2], b[:2]
        q = (b1 - b2) ** 2 / (s1 ** 2 + s2 ** 2)
        return q[:, good]

    panel = set(config.panel_targets())
    pr = [i for i, t in enumerate(acc_t) if t in panel]
    v1 = _pooled([i for i in veh if strata[i].endswith("_1")], n, S, SS, f, nt)
    v2 = _pooled([i for i in veh if strata[i].endswith("_2")], n, S, SS, f, nt)
    vv = _pooled(veh, n, S, SS, f, nt)
    null = np.nanmean(z2(v1, v2)[pr])
    print(f"null (vehicle rep 1 vs rep 2): mean z^2 {null:.3f} over {len(pr)} panel targets", flush=True)
    keep = list(veh)
    pathway = {"trametinib": ["MAP2K2", "MAP2K3"], "zstk474": ["AKT2"], "lapatinib": [], "nintedanib": []}
    for d in ["lapatinib", "nintedanib", "trametinib", "zstk474"]:
        sel = [i for i, h in enumerate(strata) if h.startswith(d + "_")]
        q = z2(_pooled(sel, n, S, SS, f, nt), vv)
        m = np.nanmean(q[pr])
        per = {acc_t[i]: round(float(np.nanmean(q[i])), 2) for i in pr if acc_t[i] in pathway[d]}
        verdict = m <= RULE * null
        if verdict:
            keep += sel
        print(f"{d}: {int(n[sel].sum())} cells; mean z^2 vs vehicle {m:.3f} = {m / null:.2f} x null -> "
              f"{'pooled' if verdict else 'left out'}; own-pathway targets {per}", flush=True)

    b, s, ns, s1, s2 = _pooled(keep, n, S, SS, f, nt)
    targets, genes = all_targets(), universe_genes()
    tix, gix = {t: i for i, t in enumerate(targets)}, {g: j for j, g in enumerate(genes)}
    T, G = len(targets), len(genes)
    beta, se, nn, kd = np.full((T, G), np.nan), np.full((T, G), np.nan), np.full(T, np.nan), np.full(T, np.nan)
    ctrl = vv[3][nt] / vv[2][nt]                               # vehicle NTC CPM
    for j, t in enumerate(acc_t):
        i = tix[t]
        beta[i], se[i], nn[i] = b[j], s[j], ns[j]
        if t in gix and s1[nt, gix[t]] > 0:
            kd[i] = 1 - (s1[j, gix[t]] / ns[j]) / (s1[nt, gix[t]] / ns[nt])
    save(name, targets, genes, beta, se, nn, ctrl, kd)
    old = load("U87")
    rows = [tix[acc_t[i]] for i in pr]
    print(f"pooled strata: {len(keep)} of {len(strata)}; panel targets {len(pr)}: median cells {np.median(nn[rows]):.0f} "
          f"(vehicle only {np.median(old['n'][rows]):.0f}); median s {np.nanmedian(se[rows]):.3f} (vehicle only "
          f"{np.nanmedian(old['s'][rows]):.3f})", flush=True)
    both = np.isfinite(beta[rows]) & np.isfinite(old["beta"][rows]) & good[None, :]
    print(f"agreement with the vehicle-only table, genes >= {MIN_CPM:.0f} CPM: r = "
          f"{np.corrcoef(beta[rows][both], old['beta'][rows][both])[0, 1]:.3f}", flush=True)
    print("self-knockdown, pooled vs vehicle only:\n" + pd.DataFrame(
        {"pooled": kd[rows], "vehicle": old["kd"][rows]}, index=[targets[i] for i in rows]).round(2).to_string())


def table_drug_only(name: str = "U87d") -> None:
    strata, acc_t, n, S, SS, veh, f, _ = _strata()
    nt = len(acc_t)
    drug = [i for i in range(len(strata)) if i not in veh]
    b, s, ns, s1, s2 = _pooled(drug, n, S, SS, f, nt)
    targets, genes = all_targets(), universe_genes()
    tix = {t: i for i, t in enumerate(targets)}
    T, G = len(targets), len(genes)
    beta, se, nn = np.full((T, G), np.nan), np.full((T, G), np.nan), np.full(T, np.nan)
    for j, t in enumerate(acc_t):
        beta[tix[t]], se[tix[t]], nn[tix[t]] = b[j], s[j], ns[j]
    save(name, targets, genes, beta, se, nn, s1[nt] / ns[nt])
