"""H44: should a same-line source dominate the pool, and at what weight? Proxy, H1 held out; Pan_H1 (H1 hESC from another
lab) as its same-cell-type source, the other four main lines at 1, KOLF 0.25, L5 recipe (z x1.5, bins, neighbour cap 0.15,
floor 0.02). Weight of Pan_H1 swept 0.25 to 16, and Pan_H1 alone; scored on all 126 benchmark targets and on the 46 that
Pan_H1 covers. Stem case: KOLF (iPSC) held out, the five main lines at 1 plus Pan_H1 and/or Feng_iPSC at weight w.

Verdict (EXPERIMENT_LOG H44): weight 4 is best. All 126 targets, w 0.25 / 1 / 2 / 4 / 8 / 16 / alone: 0.209 / 0.221 / 0.226 /
0.229 / 0.229 / 0.226 / -0.087; the 46 covered 0.263 / 0.314 / 0.344 / 0.365 / 0.359 / 0.364 / 0.348 (alone fails where it
has no data). KOLF held out: no stem source 0.029; Pan_H1 0.25 / 1 / 4: 0.033 / 0.038 / 0.043; Feng 1 / 4: 0.022 / 0.025.
Rule (model.sources): a context matching a line with panel data gets it at 4; a stem-like context gets Pan_H1 at 4.
No script existed for H44 (it was run interactively); this rebuilds it on the pipeline. Snapshot abc. About 5 minutes.

    python experiments/h44_same_line_weight.py
"""
import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation import proxy
from vcell.evaluation.harness import model_effect
from vcell.evaluation.panel import Panel
from vcell.model.promoter import neighbour_pairs

warnings.filterwarnings("ignore")
config.use_snapshot("abc")
X = Panel.load().add("KOLF").add("Pan_H1").add("Feng_iPSC")     # indices 5, 6, 7
X.min_effect = 0.02
KO, PAN, FENG, H1 = 5, 6, 7, 4
pairs = neighbour_pairs(X.targets, X.genes)


def score(tr, lam, hi, rows):
    R = X.rungs(tr, lam=None if all(x == 1.0 for x in lam) else lam)
    eff = model_effect(X, R, hi, pairs)
    return proxy.est(X.pds(eff, hi, rows), X.de(eff, hi, rows, per_target=True))


if __name__ == "__main__":
    rows = X.rows_for(H1)
    cov = np.array([i for i in rows if np.isfinite(X.B[PAN, i]).sum() > 100])
    recs = []
    print(f"H1 held out: {len(rows)} benchmark targets, Pan_H1 covers {len(cov)}", flush=True)
    for w in [0, 0.25, 1, 2, 4, 8, 16, "alone"]:
        if w == "alone":
            tr, lam = [PAN], [1.0]
        else:
            tr, lam = [0, 1, 2, 3, KO] + ([PAN] if w else []), [1.0] * 4 + [0.25] + ([float(w)] if w else [])
        r = {"case": "H1 held out", "source": f"Pan_H1 {w}", "est all": score(tr, lam, H1, rows), "est covered": score(tr, lam, H1, cov)}
        recs.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    krows = X.rows_for(KO)
    print(f"\nKOLF held out: {len(krows)} panel targets", flush=True)
    for name, extra in [("no stem source", []), ("Pan_H1 0.25", [(PAN, 0.25)]), ("Pan_H1 1", [(PAN, 1.0)]),
                        ("Pan_H1 4", [(PAN, 4.0)]), ("Feng 1", [(FENG, 1.0)]), ("Feng 4", [(FENG, 4.0)]),
                        ("Pan_H1 4 + Feng 4", [(PAN, 4.0), (FENG, 4.0)])]:
        tr = [0, 1, 2, 3, 4] + [i for i, _ in extra]
        lam = [1.0] * 5 + [w for _, w in extra]
        r = {"case": "KOLF held out", "source": name, "est all": score(tr, lam, KO, krows)}
        recs.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(recs).to_csv(config.results("h44_same_line_weight.csv"), index=False)
