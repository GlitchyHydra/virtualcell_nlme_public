"""H21: do z x1.5 and the pooled pipeline hold up on lines that were never tuned on? Jurkat, HepG2 and RPE1 held out on the
essential-target axis (all 2,394 targets; lines K562 genome-wide, CD4+ T Rest, HCT116, HEK293T and the three essential
screens).
  1. scale check: train on the six other lines; v3 raw and z x {1, 1.5, 2, 3};
  2. value of extra sources: the four panel-covering lines only, then plus one essential screen;
  3. tau^2 per expression bin learned with or without the essential screens (diagnostic).
A strong-effect regime (3,000-4,700 real DE genes per target): read differences between variants, not levels.

Verdict (EXPERIMENT_LOG H21): supported. Estimated overall v3 raw 0.12-0.18, z x1.5 0.16-0.22; best scale 1.5 in all three
(x2 within 0.01-0.02, x3 clearly worse). Extra essential screens help (Jurkat 0.203 -> 0.214 -> 0.216); tau^2 is 2-3x larger
with them in the pool, so lines differ more than four panel-covering lines suggested.
Was eda/30_essential_proxy.py. Essential axis (the same in both snapshots). About 10 minutes and ~12 GB of memory.

    python experiments/h21_essential_holdout.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import essential_panel
from vcell.model import pool

warnings.filterwarnings("ignore")
HELD = ["jurkat", "hepg2", "rpe1"]
SRC4 = ["K562", "CD4_Rest", "HCT116", "HEK293T"]

if __name__ == "__main__":
    X = essential_panel(None)
    ix = {n: i for i, n in enumerate(X.lines)}
    recs = []

    def run(hname, tr_names, tag, scales=(1.0, 1.5, 2.0, 3.0)):
        hi, tr = ix[hname], [ix[n] for n in tr_names]
        rows = X.rows_for(hi)
        R = X.rungs(tr)
        ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
        variants = {"v3 raw": R["v3"]}
        for c in scales:
            variants[f"z x{c:g}"] = c * ref * R["v3"] / R["sd"]
        for vn, eff in variants.items():
            d = X.de(eff, hi, rows, per_target=True)
            p = X.pds(eff, hi, rows)
            recs.append({"held_out": hname, "train": tag, "variant": vn, "pds": p, "fid": d["fid"].mean(),
                         "jac": d["jac"].mean(), "reach": np.nanmean(d["reach"]), "nmae": np.nanmean(d["nmae"]),
                         "median_n_real": np.median(d["n_real"]), "est": proxy.est(p, d)})
        print(f"{hname} <- {tag} done", flush=True)

    for h in HELD:
        run(h, [n for n in X.lines if n != h], "all 6 others")
        run(h, SRC4, "4 panel lines")
        run(h, SRC4 + [x for x in HELD if x != h][:1], "4 + 1 essential")
    df = pd.DataFrame(recs)
    df.to_csv(config.results("h21_essential_holdout.csv"), index=False)
    for col in ["est", "pds", "fid", "jac", "reach", "nmae", "median_n_real"]:
        print(f"\n{col}")
        print(df.pivot_table(index=["train", "variant"], columns="held_out", values=col, sort=False).to_string(
            float_format=lambda x: f"{x:.3f}"))
    for name, tr in (("4 panel lines", [ix[n] for n in SRC4]), ("all 7", list(range(len(X.lines))))):
        P0 = pool.pool(X.B[tr], X.S[tr], tau2=0.0)
        bins = pool.gene_bins(X.max_cpm)
        t2 = pool.tau2_by_bin(P0["Q"], P0["K"], P0["C"], bins)
        print(f"tau2 by bin ({name}):", np.round([t2[bins == b][0] for b in np.unique(bins)], 4))
