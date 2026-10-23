"""H48 (rejected): without a same-type screen, does upweighting the most RELATED cell type help? Context A is Jurkat-like,
but no Jurkat screen covers the panel; the nearest lineage with panel data is primary CD4+ T cells. Essential axis of the
harness (600 random targets, 7 lines): hold out each line, pool the other six with one of them at weight w (combination
only, tau^2 unweighted), proxy.est minus the equal-weight pool. Jurkat with CD4_Rest upweighted maps onto context A.

Verdict (EXPERIMENT_LOG H48): rejected. Jurkat held out: CD4_Rest at 2 -0.008, at 4 -0.022; 36 of 42 upweightings hurt (to
-0.032); the few gains (<= +0.007) do not follow lineage. Only the same cell type is a lever (H44, H47); related lines keep
weight 1, as the same-line rule does (model.sources).
Was eda/54_related_type.py. Essential axis. About 6 minutes.

    python experiments/failed/h48_related_type.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import essential_panel

warnings.filterwarnings("ignore")
X = essential_panel()
L = X.lines


def est(hi, up=None, w=1.0, c=1.5):
    tr = [i for i in range(len(L)) if i != hi]
    lam = None if up is None else [w if i == up else 1.0 for i in tr]
    R = X.rungs(tr, lam=lam)
    ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
    eff = (c * ref * R["v3"] / R["sd"]).astype(np.float32)
    rows = X.rows_for(hi)
    d = X.de(eff, hi, rows, per_target=True)
    return proxy.est(X.pds(eff, hi, rows), d), X.pds(eff, hi, rows)


if __name__ == "__main__":
    recs = []
    for hi in range(len(L)):
        base, bp = est(hi)
        recs.append({"held": L[hi], "up": "-", "w": 1.0, "d_est": 0.0, "d_pds": 0.0, "est": base})
        for up in range(len(L)):
            if up == hi:
                continue
            for w in ((2.0, 4.0) if (L[hi], L[up]) == ("jurkat", "CD4_Rest") else (4.0,)):
                e, p = est(hi, up, w)
                recs.append({"held": L[hi], "up": L[up], "w": w, "d_est": e - base, "d_pds": p - bp, "est": e})
        print(f"{L[hi]} done (base {base:.4f})", flush=True)
    R = pd.DataFrame(recs)
    R.to_csv(config.results("h48_related_type.csv"), index=False)
    M = R[R.w == 4.0].pivot(index="held", columns="up", values="d_est")
    print("\nproxy.est change when one source line gets weight 4 (rows: held-out line)")
    print(M.round(4).to_string())
    print("\nJurkat held out, CD4_Rest weight 2 / 4:")
    print(R[(R.held == "jurkat") & (R.up == "CD4_Rest")].round(4).to_string(index=False))
