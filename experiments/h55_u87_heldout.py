"""H55: held-out check of the drug-adjusted U87 data (H54) on independent cells. Truth = the vehicle-only U87 table
(stage1_U87); source = the drug arms only, vehicle strata excluded, drug effect removed (stage1_U87d, etl.stage1.u87).
U87 held out on its 21 panel targets; the final pool (K562, CD4_Rest, HCT116, HEK293T, H1 at 1, KOLF 0.25, Pan_H1 0.25;
L5 settings) without and with U87d at weight 1 / 4. The truth has ~80 cells per target, so it is noisy, but the noise is
the same for every candidate.

Verdict (EXPERIMENT_LOG H55): weakly supported. U87d weight 0 / 1 / 4: pds raw 0.410 / 0.417 / 0.455, est -0.054 /
-0.052 / -0.038; right direction and weight 4 best, on a weak test. The only direct test available for E.
Was eda/71_u87_heldout.py (which also built stage1_U87d, now `python -m vcell etl stage1 u87d`). Snapshot def. 2 minutes.

    python experiments/h55_u87_heldout.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.harness import model_effect
from vcell.evaluation.panel import Panel
from vcell.model.promoter import neighbour_pairs

warnings.filterwarnings("ignore")
config.use_snapshot("def")
P = Panel.load()
for name in ["KOLF", "Pan_H1", "U87", "U87d"]:                  # indices 5, 6, 7 (held out), 8
    P = P.add(name)
P.min_effect = 0.02
HI = 7

if __name__ == "__main__":
    rows = P.rows_for(HI)
    nn = load("U87d")["n"].astype(float)
    print(f"U87 held out: {len(rows)} panel targets; truth median cells {np.nanmedian(P.NC[HI][rows]):.0f}, "
          f"source U87d median cells {np.nanmedian(nn[rows]):.0f}", flush=True)
    pairs = neighbour_pairs(P.targets, P.genes)
    out = []
    for wt in [0, 1, 4]:
        tr = [0, 1, 2, 3, 4, 5, 6] + ([8] if wt else [])
        R = P.rungs(tr, lam=[1.0] * 5 + [0.25, 0.25] + ([float(wt)] if wt else []))
        eff = model_effect(P, R, HI, pairs)
        p = P.pds(eff, HI, rows)
        for seed in range(5):
            d = P.de(eff, HI, rows, seed=seed, per_target=True)
            out.append({"U87d weight": wt, "seed": seed, "pds raw": p, "DE part": proxy.mixed(d) / 6, "est": proxy.est(p, d)})
    df = pd.DataFrame(out)
    df.to_csv(config.results("h55_u87_heldout.csv"), index=False)
    print(df.groupby("U87d weight")[["pds raw", "DE part", "est"]].agg(["mean", "std"]).round(4).to_string())
