"""H28: KOLF2.1J iPSC (etl.stage1.kolf) as a sixth source line, leave one line out.
  1. calibration: robust sd of beta / s (about 1 if s is right; most pairs are null), cells per target, knockdown;
  2. value as a source: each of the five main lines held out, predicted from the other four with and without KOLF (z x1.5);
  3. KOLF held out: how well the five lines predict an iPSC line.

Verdict (EXPERIMENT_LOG H28): not for differentiated lines (K562 -0.008, HEK293T -0.010, HCT116 -0.002, CD4 0.000) but
H1 +0.005 (the other stem line); robust sd of beta/s 1.16 (batch noise from pooled controls). The leaderboard agreed (L4).
Later adopted at weight 0.25 with the clean weighting (H38), which is in the final model.
Was eda/37_kolf_proxy.py. Snapshot abc. About 3 minutes.

    python experiments/h28_kolf.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

warnings.filterwarnings("ignore")
config.use_snapshot("abc")

if __name__ == "__main__":
    P = Panel.load()
    k = load("KOLF")
    b, s = k["beta"], k["s"]
    z = (b / s)[np.isfinite(b) & np.isfinite(s)]
    print(f"KOLF: robust sd(beta/s) {np.median(np.abs(z)) / 0.6745:.3f}; targets {int(np.isfinite(k['n']).sum())} "
          f"(panel {sum(t in P.panel and np.isfinite(n) for t, n in zip(P.targets, k['n']))}); median cells "
          f"{np.nanmedian(k['n']):.0f}; median knockdown {np.nanmedian(k['kd']):.2f}")
    P = P.add("KOLF", k)
    KI = P.index("KOLF")
    recs = []

    def run(hi, tr, tag):
        rows = P.rows_for(hi)
        R = P.rungs(tr)
        eff = 1.5 * np.nanmedian(np.where(R["has"], R["sd"], np.nan)) * R["v3"] / R["sd"]
        d = P.de(eff, hi, rows, per_target=True)
        p = P.pds(eff, hi, rows)
        recs.append({"held_out": P.lines[hi], "sources": tag, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                     "fid": d["fid"].mean(), "jac": d["jac"].mean(), "nmae": np.nanmean(d["nmae"])})

    for hi in range(5):
        base = [i for i in range(5) if i != hi]
        run(hi, base, "4 lines")
        run(hi, base + [KI], "4 lines + KOLF")
        print(f"{P.lines[hi]} done", flush=True)
    run(KI, list(range(5)), "5 lines (KOLF held out)")
    df = pd.DataFrame(recs)
    df.to_csv(config.results("h28_kolf.csv"), index=False)
    for col in ["est", "pds", "de_mix", "fid", "jac", "nmae"]:
        pv = df.pivot_table(index="sources", columns="held_out", values=col, sort=False)
        pv = pv[[c for c in P.lines if c in pv.columns]]
        print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
