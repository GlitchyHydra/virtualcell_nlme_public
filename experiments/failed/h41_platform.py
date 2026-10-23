"""H41 (not visible): Does the measurement platform matter more than the cell type? A competitor reports "same-platform data beats
same-cell-type data five to one". Arc's 2026 panel uses 10x Flex probes, as do the X-Atlas/Orion lines (HCT116, HEK293T); the
other sources are 10x 3' (Replogle K562), DESeq2 tables (CD4+ T) or Arc's 2025 H1 screen.

Leave one main line out (uploaded setting: z x1.5, floor 0.05); the X-Atlas lines enter the combination with weight lam_x
(tau^2 unweighted, proxy.rungs(lam=...)), every other source with 1. For HCT116 or HEK293T held out, the other X-Atlas line is
the same-platform source.

    python experiments/failed/h41_platform.py

Verdict (EXPERIMENT_LOG H41): up-weighting the X-Atlas (10x Flex) lines lowers est on every held-out line (x4: -0.001 to
-0.007; x8: -0.003 to -0.019), including HCT116 and HEK293T; equal weights stay best.
Was eda/50_platform.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
XATLAS = {"HCT116", "HEK293T"}
recs = []
for hi, line in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    rows = X.rows_for(hi)
    for lam_x in (0.25, 1.0, 2.0, 4.0, 8.0):
        lam = [lam_x if X.lines[i] in XATLAS else 1.0 for i in tr]
        R = X.rungs(tr, lam=None if lam_x == 1.0 else lam)
        ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
        eff = 1.5 * ref * R["v3"] / R["sd"]
        d = X.de(eff, hi, rows, per_target=True)
        recs.append({"line": line, "lam_xatlas": lam_x, "est": proxy.est(X.pds(eff, hi, rows), d),
                     "pds": X.pds(eff, hi, rows)})
    print(f"{line} done", flush=True)
R = pd.DataFrame(recs)
R.to_csv(config.results("h41_platform.csv"), index=False)
for col in ("est", "pds"):
    t = R.pivot(index="lam_xatlas", columns="line", values=col)[list(X.lines)]
    t = t - t.loc[1.0]
    print(f"\n{col}, change against equal weights (X-Atlas weight on the rows)")
    print(t.round(4).to_string())
