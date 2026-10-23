"""H52 (rejected): does cell-level QC of the source screens (qc.sources rules; tables stage1_<line>_qc, built with
`python -m vcell etl stage1 hct116acc hek293tacc --qc`) improve the transfer? Leave one main line out with the L5 settings
(z x1.5, bins, neighbour cap 0.15, floor 0.02, KOLF 0.25, Pan_H1 0.25). The pool uses either the published tables or the
cleaned ones for the lines that have one; the held-out line's truth is the published table (the scored truth cells only
went through the organisers' QC). Paired over the same DE noise seeds.
--truth cleaned scores the X-Atlas lines against their cleaned tables instead (a sensitivity check).

Verdict (EXPERIMENT_LOG H52): neutral to slightly negative, mean -0.0013 (K562 +0.0019, CD4 +0.0002, H1 +0.0012, HCT116
-0.0022, HEK293T -0.0075); the same with the truth cleaned. The removed high-mito cells carry response the two X-Atlas lines
share. The final model uses the published tables; the QC code and masks stay in qc.sources.
Was eda/63_qc_eval.py. Snapshot abc (the cleaned tables are on the A/B/C axis). About 5 minutes.

    python experiments/failed/h52_source_qc.py [--truth published|cleaned]
"""
import argparse
from dataclasses import replace

import numpy as np

from vcell import config
from vcell.etl.stage1.table import load, table_path
from vcell.evaluation import proxy
from vcell.evaluation.harness import EDGES, Harness
from vcell.evaluation.panel import standardised
from vcell.model.promoter import neighbour_cap

config.use_snapshot("abc")
ap = argparse.ArgumentParser()
ap.add_argument("--truth", choices=["published", "cleaned"], default="published")
args = ap.parse_args()
MULT = np.array([1.5, 1.25, 1, 1, 0.8], np.float32)
SEEDS = (0, 1, 2)
H = Harness()
M = H.main
lines = list(M.lines)
qc_lines = [l for l in lines if table_path(f"{l}_qc").exists()]
print("lines with a cleaned table:", qc_lines, flush=True)

RAW = {k: getattr(M, k) for k in ("B", "S", "NC")}
QCD = {k: v.copy() for k, v in RAW.items()}
for l in qc_lines:
    t, i = load(f"{l}_qc"), lines.index(l)
    assert list(t["targets"]) == M.targets and list(t["genes"]) == M.genes, f"{l}_qc is on another axis"
    QCD["B"][i], QCD["S"][i], QCD["NC"][i] = t["beta"], t["s"], t["n"]
    b0, b1, s0, s1 = RAW["B"][i], QCD["B"][i], RAW["S"][i], QCD["S"][i]
    m = np.isfinite(b0) & np.isfinite(b1)
    n0, n1 = RAW["NC"][i], QCD["NC"][i]
    k = np.isfinite(n0) & np.isfinite(n1)
    print(f"{l}: beta r {np.corrcoef(b0[m], b1[m])[0, 1]:.4f}; median s cleaned/published {np.nanmedian(s1[m] / s0[m]):.3f}; "
          f"median cells per target {np.median(n0[k]):.0f} -> {np.median(n1[k]):.0f}", flush=True)


def line_score(hi, tables, seed):
    arrays = {k: v.copy() for k, v in tables.items()}
    truth = QCD if args.truth == "cleaned" else RAW
    for k in ("B", "S", "NC"):                                       # the held-out line's truth: same in both arms
        arrays[k][hi] = truth[k][hi]
    P = replace(M, **arrays)
    rows = P.rows_for(hi)
    tr, lam = H.training("main", hi, 0.25, 0.25)
    eff = 1.5 * standardised(P.rungs(tr, kappa=1.0, lam=lam))
    eff = eff * MULT[np.digitize(P.ctrl_cpm(hi), EDGES)][None, :]
    pairs = H.pairs["main"]
    eff = eff.copy()
    ti, gj = pairs.ti.to_numpy(), pairs.gj.to_numpy()
    eff[ti, gj] = np.minimum(eff[ti, gj], neighbour_cap(pairs, 0.15))
    p = P.pds(eff, hi, rows)
    d = P.de(eff, hi, rows, seed=seed, per_target=True, min_effect=0.02)
    return proxy.est(p, d), (p - 0.5) / proxy.SPAN["pds"] / 6, proxy.mixed(d) / 6


if __name__ == "__main__":
    print(f"\n{'held out':10s} {'published':>10s} {'cleaned':>10s} {'delta':>8s}   (pds part / DE part of the delta)")
    deltas = []
    for hi in range(5):
        r = np.array([line_score(hi, RAW, s) for s in SEEDS]).mean(0)
        q = np.array([line_score(hi, QCD, s) for s in SEEDS]).mean(0)
        deltas.append(q[0] - r[0])
        print(f"{lines[hi]:10s} {r[0]:10.4f} {q[0]:10.4f} {q[0] - r[0]:+8.4f}   ({q[1] - r[1]:+.4f} / {q[2] - r[2]:+.4f})", flush=True)
    print(f"mean delta {np.mean(deltas):+.4f}; lines up {sum(d > 0 for d in deltas)} of 5")
