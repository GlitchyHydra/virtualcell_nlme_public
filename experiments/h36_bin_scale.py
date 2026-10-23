"""H36: a separate effect scale per expression bin of the response gene, bins from the new context's own control CPM
(<15, 15-40, 40-120, 120-400, >400; known for any context from its control cells). Effect = c x m_b x z, coordinate search
over the multipliers m_b in two passes, all 8 held-out lines of the harness, against the uploaded model (c 1.5, kappa 1,
floor 0.05, five main lines at equal weight).

Verdict (EXPERIMENT_LOG H36): small gain, adopted: [1.5, 1.25, 1, 1, 0.8] gives 0.1703 vs 0.1676 (+0.0027; 6 of 8 lines
up, HEK293T -0.003), stable over a second pass. Weakly expressed genes, whose standardised effects are shrunk most, need
more scale to be called; highly expressed genes are over-called. Picked from about 40 trials, so part may be selection.
Was the H36 part of eda/43_similarity_bins.py (the H35 part is experiments/failed/h35_similarity_weights.py).
Snapshot abc. About 15 minutes.

    python experiments/h36_bin_scale.py
"""
import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation.harness import BIN_LAB, Harness

config.use_snapshot("abc")
H = Harness()
recs, REF = [], None


def run(label, c=1.5, mult=None):
    global REF
    v = H.objective(c=c, mult=mult)
    m = float(np.mean(list(v.values())))
    REF = REF or v
    dl = {k: v[k] - REF[k] for k in v}
    up = sum(x > 0.0005 for x in dl.values())
    down = sum(x < -0.0005 for x in dl.values())
    recs.append({"label": label, "c": c, "mult": None if mult is None else list(mult), **v, "mean_all8": m})
    pd.DataFrame(recs).to_csv(config.results("h36_bin_scale.csv"), index=False)
    print(f"{label:38s} all8 {m:.4f}  ({m - np.mean(list(REF.values())):+.4f}; lines up {up}, down {down})   "
          + " ".join(f"{k} {dl[k]:+.4f}" for k in v), flush=True)
    return m


if __name__ == "__main__":
    base = run("uploaded (equal, no KOLF)")
    print("\nH36 per-bin scale multipliers, bins of control CPM " + ", ".join(BIN_LAB))
    mult = [1.0] * 5
    cur = base
    for p in range(2):
        for b in range(5):
            for val in (0.6, 0.8, 1.0, 1.25, 1.5):
                if val == mult[b]:
                    continue
                trial = mult.copy()
                trial[b] = val
                s = run(f"pass {p + 1} bin {BIN_LAB[b]} x{val}", mult=trial)
                if s > cur + 0.0002:
                    mult, cur = trial, s
        print(f"   after pass {p + 1}: {dict(zip(BIN_LAB, mult))} -> {cur:.4f}", flush=True)
    print(f"\nbest bin multipliers {mult} ({cur:.4f}); uploaded {base:.4f}")
