"""H37: promoter neighbours. CRISPRi represses the target's promoter, and genes whose TSS lies within a few kb (divergent,
head-to-head pairs above all) are often silenced too. The scorer drops only the target's own column, so a silenced neighbour
is scored, and the mechanism is physical, so it should transfer between lines (idea from a public entry,
github.com/mjh72849-cpu/Virtual-Cell-Challenge-2026). Pairs: GENCODE v47 TSS, every target with every axis gene within
10 kb on the same chromosome. Leave one main line out:
  A. how far a neighbour really drops, by distance bin and orientation, and what the pooled z x1.5 already predicts;
  B. a neighbour prior learned from the other lines (min or replacement) and the fixed rule (15% left at <= 500 bp, rising
     log-linearly to 100% at 5 kb = model.promoter.neighbour_cap); proxy.est and the mse ratio at a fitted scale.

Verdict (EXPERIMENT_LOG H37): real but mostly captured already; the rule gains +0.0011 (CD4 +0.0035, K562 +0.0022, HCT116
-0.0008) and was adopted (neighbour = 0.15 in configs/final_def.toml). Neighbours <= 0.5 kb drop by a median -0.54 to
-0.70 log2; H1 (Arc's protocol) much harder (-3.03 at <= 1 kb). The neighbour channel explains 0.2% of the effect energy.
Was eda/44_promoter.py. Snapshot abc. About 3 minutes.

    python experiments/h37_promoter.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.mse_proxy import fitted_ratio
from vcell.evaluation.panel import Panel, standardised
from vcell.model.promoter import tss_table

warnings.filterwarnings("ignore")
config.use_snapshot("abc")
DBINS = [0, 500, 1000, 2000, 5000, 10001]
DLAB = ["<=0.5k", "0.5-1k", "1-2k", "2-5k", "5-10k"]
X = Panel.load()


def annotated_pairs(t: pd.DataFrame, max_d: int = 10000) -> pd.DataFrame:
    """model.promoter.neighbour_pairs with names, orientation (divergent = head to head) and a distance bin, to 10 kb."""
    gi = {g: j for j, g in enumerate(X.genes)}
    on_axis = t[t.gene.isin(gi)]
    by_chrom = {c: d.sort_values("tss") for c, d in on_axis.groupby("chrom")}
    tt = t.set_index("gene")
    rows = []
    for i, tg in enumerate(X.targets):
        if tg not in tt.index:
            continue
        a = tt.loc[tg]
        d = by_chrom.get(a.chrom)
        if d is None:
            continue
        near = d[(d.tss - a.tss).abs() <= max_d]
        for _, b in near.iterrows():
            if b.gene == tg:
                continue
            div = a.strand != b.strand and ((a.strand == "+" and a.tss > b.tss) or (a.strand == "-" and a.tss < b.tss))
            rows.append((i, gi[b.gene], tg, b.gene, abs(int(a.tss) - int(b.tss)), bool(div)))
    p = pd.DataFrame(rows, columns=["ti", "gj", "target", "neighbour", "distance", "divergent"])
    p["dbin"] = pd.cut(p.distance, DBINS, labels=DLAB, include_lowest=True)
    return p


if __name__ == "__main__":
    P = annotated_pairs(tss_table())
    print(f"{len(P)} target-neighbour pairs within 10 kb ({P.target.nunique()} of {len(X.targets)} targets; "
          f"{int(P.divergent.sum())} divergent); panel targets with a pair: "
          f"{P[P.target.isin(X.panel)].target.nunique()} of {len(X.panel)}", flush=True)
    P.to_csv(config.results("h37_promoter_pairs.csv"), index=False)
    TI, GJ = P.ti.to_numpy(), P.gj.to_numpy()
    LINES = range(len(X.lines))
    pred = {hi: standardised(X.rungs([i for i in LINES if i != hi]), 1.5) for hi in LINES}

    print("\nA. neighbour log2FC by distance and orientation: real (median, share with z < -2) | pooled z x1.5 from the other lines")
    recs = []
    for hi in LINES:
        b, s = X.B[hi][TI, GJ], X.S[hi][TI, GJ]
        recs.append(pd.DataFrame({"line": X.lines[hi], "dbin": P.dbin, "div": P.divergent, "beta": b, "z": b / s,
                                  "pred": pred[hi][TI, GJ], "ti": P.ti, "gj": P.gj}))
    A = pd.concat(recs)
    A = A[np.isfinite(A.beta)]
    tab = A.groupby(["dbin", "div"], observed=True).agg(n=("beta", "size"), real=("beta", "median"),
                                                        down=("z", lambda z: float(np.mean(z < -2))),
                                                        pooled=("pred", "median"))
    print(tab.round(3).to_string())
    print("\n   by line, pairs <= 1 kb: median real | median pooled | share down")
    for ln, d in A[A.dbin.isin(DLAB[:2])].groupby("line"):
        print(f"   {ln:9s} n {len(d):4d}  real {d.beta.median():6.2f}  pooled {d.pred.median():6.2f}  down {np.mean(d.z < -2):.2f}")

    print("\nB. variants (proxy.est per line, five metrics; mse ratio at a transferred least-squares scale)")

    def prior_from(tr):
        """Median log2FC of neighbour pairs in lines `tr`, per (distance bin, orientation); NaN-free, capped at 0."""
        d = A[A.line.isin([X.lines[i] for i in tr])]
        med = d.groupby(["dbin", "div"], observed=True).beta.median().clip(upper=0.0)
        return np.array([med.get((db, dv), 0.0) for db, dv in zip(P.dbin, P.divergent)])

    def rule_mjh(f=0.15):
        ramp = np.clip(np.log(np.maximum(P.distance.to_numpy(float), 500) / 500) / np.log(10), 0, 1)
        return np.log2(f + (1 - f) * ramp)

    def variants(hi):
        tr = [i for i in LINES if i != hi]
        base = pred[hi]
        pri = prior_from(tr)
        near = P.distance.to_numpy() <= 5000
        out = {"uploaded z x1.5": base}
        for name, val in (("min(pred, learned prior)", pri), ("min(pred, rule 15%)", rule_mjh())):
            e = base.copy()
            e[TI[near], GJ[near]] = np.minimum(e[TI[near], GJ[near]], val[near])
            out[name] = e
        e = base.copy()
        e[TI[near], GJ[near]] = pri[near]
        out["replace by learned prior"] = e
        nb = np.zeros_like(base)
        nb[TI[near], GJ[near]] = pri[near]
        return out, nb

    res = []
    for hi in LINES:
        rows = X.rows_for(hi)
        vs, nb = variants(hi)
        r_nb, _ = fitted_ratio(X, [nb], hi, rows)
        r_two, _ = fitted_ratio(X, [vs["uploaded z x1.5"], nb], hi, rows)
        for name, e in vs.items():
            d = X.de(e, hi, rows, per_target=True)
            r, _ = fitted_ratio(X, [e], hi, rows)
            res.append({"line": X.lines[hi], "variant": name, "est": proxy.est(X.pds(e, hi, rows), d),
                        "mse_fit": r, "fid": np.mean(d["fid"]), "jac": np.mean(d["jac"])})
        res.append({"line": X.lines[hi], "variant": "mse: neighbour prior alone", "mse_fit": r_nb})
        res.append({"line": X.lines[hi], "variant": "mse: z + neighbour prior (2 coefs)", "mse_fit": r_two})
        print(f"   {X.lines[hi]} done", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(config.results("h37_promoter_variants.csv"), index=False)
    pd.set_option("display.width", 200)
    for col in ("est", "mse_fit"):
        t = R.pivot(index="variant", columns="line", values=col)[list(X.lines)]
        t["mean"] = t.mean(1)
        print(f"\n{col}  ({'higher' if col == 'est' else 'lower'} is better; mse_fit is the ratio at the best scale)")
        print(t.dropna(how="all").round(4).to_string())
