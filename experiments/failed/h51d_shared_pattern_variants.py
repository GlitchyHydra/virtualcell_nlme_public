"""H51d (rejected on the leaderboard, L6): better forms of the shared-pattern removal (H51) on top of the candidate, all 8
held-out lines of the harness:
    post k a     remove a x the top k components of the POOLED effect matrix over the scored targets (H51, H51b)
    line k       remove the top k components of EACH SOURCE LINE's own beta over the scored targets, before pooling (a
                 public entry subtracts each source's mean response; pattern 1 of H51c is mostly HEK293T's gene-length trend)
    damp g       after the removal, scale each gene by min(1, ref / median_t |eff|)^g: genes that move for most targets count less
    len          remove each target's linear trend on log gene length (GENCODE v47), the direct form of pattern 1
Order: pool -> z x1.5 -> [len] -> [post] -> [damp] -> per-bin scale -> neighbour cap 0.15 -> floor 0.02. KOLF 0.25.

Result (EXPERIMENT_LOG H51d): strength 0.75 is best (+0.0101, 8 of 8 lines up, worst line +0.003); per-line removal,
damping and the gene-length trend are not better. Uploaded as cand_c3.vcc (L6): leaderboard 0.1683 vs 0.1705 for L5, so
the removal is not in the final model (configs/final_def.toml has no centring step, and a unit test keeps it that way).
Was eda/61_shared_variants.py. Snapshot abc. About 12 minutes.

    python experiments/failed/h51d_shared_pattern_variants.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import EDGES, KI, Harness
from vcell.evaluation.panel import standardised
from vcell.model.promoter import gene_lengths, neighbour_cap

warnings.filterwarnings("ignore")
config.use_snapshot("abc")
MULT = np.array([1.5, 1.25, 1.0, 1.0, 0.8], np.float32)
VARIANTS = {                       # name: (line_k, post_k, post_alpha, damp, length)
    "candidate": (0, 0, 0.0, 0.0, False),
    "post k3 a1": (0, 3, 1.0, 0.0, False),
    "post k3 a0.5": (0, 3, 0.5, 0.0, False),
    "post k3 a0.75": (0, 3, 0.75, 0.0, False),
    "line k1": (1, 0, 0.0, 0.0, False),
    "line k3": (3, 0, 0.0, 0.0, False),
    "line k3 + post k3 a0.5": (3, 3, 0.5, 0.0, False),
    "post k3 a1 + damp 0.5": (0, 3, 1.0, 0.5, False),
    "post k3 a1 + damp 1": (0, 3, 1.0, 1.0, False),
    "len": (0, 0, 0.0, 0.0, True),
    "len + post k3 a1": (0, 3, 1.0, 0.0, True),
}
H = Harness()


def remove_top(Z: np.ndarray, k: int, alpha: float) -> np.ndarray:
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z - alpha * (U[:, :k] * S[:k]) @ Vt[:k]


def line_cleaned_pool(axis, P, hi, rows, k):
    """z effect from a pool whose source lines each had their own top k components removed over the scored targets."""
    if axis == "main":
        tr = [i for i in range(5) if i != hi] + [KI]
        lam = [1.0] * (len(tr) - 1) + [0.25]
    else:
        tr = [i for i in range(len(P.lines)) if i != hi]
        lam = None
    Bc = P.B.copy()
    for l in tr:
        b = Bc[l, rows]
        okr = np.isfinite(b).sum(1) > 100
        Z = remove_top(np.nan_to_num(b[okr]), k, 1.0)
        Z[~np.isfinite(b[okr])] = np.nan
        b[okr] = Z
        Bc[l, rows] = b
    return standardised(P.with_arrays(B=Bc).rungs(tr, kappa=1.0, lam=lam))


if __name__ == "__main__":
    LEN = gene_lengths()
    recs = []
    for name, (lk, pk, pa, damp, use_len) in VARIANTS.items():
        for axis, P, hi, rows, pairs in H.cases():
            loglen = np.log(np.array([LEN.get(g, np.nan) for g in P.genes], float))
            loglen = np.where(np.isfinite(loglen), loglen - np.nanmean(loglen), 0.0)
            base = line_cleaned_pool(axis, P, hi, rows, lk) if lk else H.pooled(axis, hi, 1.0, 0.25)
            eff = 1.5 * base
            Zr = np.nan_to_num(eff[rows]).astype(float)
            if use_len:
                slope = Zr @ loglen / (loglen @ loglen)
                Zr = Zr - slope[:, None] * loglen[None, :]
            if pk:
                Zr = remove_top(Zr, pk, pa)
            if damp:
                med = np.median(np.abs(Zr), axis=0)
                ref = np.median(med[med > 0])
                Zr = Zr * (np.minimum(1.0, ref / np.maximum(med, 1e-12)) ** damp)[None, :]
            eff = eff.copy()
            eff[rows] = Zr.astype(np.float32)
            eff = eff * MULT[np.digitize(P.ctrl_cpm(hi), EDGES)][None, :]
            ti, gj = pairs.ti.to_numpy(), pairs.gj.to_numpy()
            eff[ti, gj] = np.minimum(eff[ti, gj], neighbour_cap(pairs, 0.15))
            d = P.de(eff, hi, rows, per_target=True, min_effect=0.02)
            pv = P.pds(eff, hi, rows)
            recs.append({"variant": name, "line": P.lines[hi], "est": proxy.est(pv, d), "pds_raw": pv,
                         "fid_raw": float(np.nanmean(d["fid"])), "jac_raw": float(np.nanmean(d["jac"])),
                         "reach_raw": float(np.nanmean(d["reach"])), "nmae_raw": float(np.nanmean(d["nmae"]))})
        v = pd.DataFrame([r for r in recs if r["variant"] == name])
        print(f"{name:24s} mean est {v.est.mean():.4f}  " + " ".join(f"{l} {x:.4f}" for l, x in zip(v.line, v.est)), flush=True)

    R = pd.DataFrame(recs)
    R.to_csv(config.results("h51d_shared_pattern_variants.csv"), index=False)
    t = R.pivot(index="variant", columns="line", values="est").reindex(list(VARIANTS))
    d = t.sub(t.loc["candidate"], axis=1)
    out = pd.DataFrame({"mean8": d.mean(axis=1), "lines up": (d > 0).sum(axis=1), "worst line": d.min(axis=1)})
    for m in ("pds_raw", "fid_raw", "jac_raw", "reach_raw", "nmae_raw"):
        p = R.pivot(index="variant", columns="line", values=m).reindex(list(VARIANTS))
        out[m] = p.sub(p.loc["candidate"], axis=1).mean(axis=1)
    print("\nchange vs the candidate (proxy.est; raw metrics, nmae lower is better)")
    print(out.round(4).to_string())
