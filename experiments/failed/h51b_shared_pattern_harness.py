"""H51b (rejected on the leaderboard, L6): the shared-component removal of H51 on top of the candidate (L5 without Pan_H1:
z x1.5, bins, neighbour cap 0.15, floor 0.02, KOLF 0.25), all 8 held-out lines of the harness. Order as the generator
would apply it: z x1.5 -> remove alpha x the top k components of the effect matrix over the scored targets (uncentred SVD)
-> per-bin scale -> neighbour cap -> floor.

Result (EXPERIMENT_LOG H51): in the proxy k 3 / 5 at alpha 0.5 +0.0066 / +0.0068 (8 of 8 lines up), k 3 alpha 1 +0.0087;
k >= 10 at alpha 1 negative. Confirmed on the official H1 scorer (alpha 0.5 +0.010, alpha 1 +0.017 on two seeds), but the
leaderboard upload (L6, alpha 0.75) scored 0.1683 against 0.1705 for L5: the shared part of the response is context
dependent. Not in the final model.
Was eda/59_centering_harness.py. Snapshot abc. About 8 minutes.

    python experiments/failed/h51b_shared_pattern_harness.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import EDGES, Harness
from vcell.model.promoter import neighbour_cap

warnings.filterwarnings("ignore")
config.use_snapshot("abc")
MULT = np.array([1.5, 1.25, 1.0, 1.0, 0.8], np.float32)
KS = [0, 1, 3, 5, 10, 20, 40]
ALPHAS = [0.5, 1.0]
H = Harness()


def run(k, alpha):
    out = {}
    for axis, P, hi, rows, pairs in H.cases():
        eff = 1.5 * H.pooled(axis, hi, 1.0, 0.25)
        if k:
            Zr = np.nan_to_num(eff[rows]).astype(float)
            U, S, Vt = np.linalg.svd(Zr, full_matrices=False)
            eff = eff.copy()
            eff[rows] = (Zr - alpha * (U[:, :k] * S[:k]) @ Vt[:k]).astype(np.float32)
        eff = eff * MULT[np.digitize(P.ctrl_cpm(hi), EDGES)][None, :]
        ti, gj = pairs.ti.to_numpy(), pairs.gj.to_numpy()
        eff[ti, gj] = np.minimum(eff[ti, gj], neighbour_cap(pairs, 0.15))
        d = P.de(eff, hi, rows, per_target=True, min_effect=0.02)
        pv = P.pds(eff, hi, rows)
        out[P.lines[hi]] = {"est": proxy.est(pv, d), "pds_raw": pv, "fid_raw": float(np.nanmean(d["fid"])),
                            "jac_raw": float(np.nanmean(d["jac"]))}
    return out


if __name__ == "__main__":
    recs = []
    for k in KS:
        for alpha in (ALPHAS if k else [0.0]):
            v = run(k, alpha)
            for line, r in v.items():
                recs.append({"k": k, "alpha": alpha, "line": line, **r})
            e = {l: r["est"] for l, r in v.items()}
            print(f"k {k:2d} alpha {alpha:.1f}: mean est {np.mean(list(e.values())):.4f}  " +
                  " ".join(f"{l} {x:.4f}" for l, x in e.items()), flush=True)
    R = pd.DataFrame(recs)
    R.to_csv(config.results("h51b_shared_pattern_harness.csv"), index=False)
    t = R.pivot_table(index=["k", "alpha"], columns="line", values="est")
    base = t.loc[(0, 0.0)]
    print("\nchange in proxy.est vs the candidate (k 0)")
    d = t - base
    d["mean8"] = d.mean(1)
    d["lines up"] = (t - base > 0).sum(1)
    print(d.round(4).to_string())
