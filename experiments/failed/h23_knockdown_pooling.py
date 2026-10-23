"""H23 (neutral, not adopted): Knockdown-aware pooling (docs/archive/IDEAS.md idea 5).

A target knocked down weakly in one line (HEK293T median 50%, HCT116 72%, K562 86%) shows a smaller effect there, which looks
like between-line disagreement. Two corrections, per (line, target) with a measured knockdown kd (K562, HCT116, HEK293T; CD4+
T and H1 have none and stay unchanged):
    rescale:  beta and s times kd_ref / kd (kd clipped to [floor, 1], kd_ref = 0.85, factor capped at 3)
    weight:   s divided by sqrt(w), w = clip(kd / kd_ref, floor, 1): weak knockdowns count less in the pool
    drop:     the line's row is removed for that target when kd < cut
Scored as z x1.5 in the leave-one-line-out proxy (A/B/C-like mix).

    python experiments/failed/h23_knockdown_pooling.py

Verdict (EXPERIMENT_LOG H23): neutral: 0.138-0.146 vs 0.146; rescaling helps K562 and CD4+ T slightly and hurts H1.
Was eda/32_knockdown.py. Snapshot abc (the A/B/C validation tables). 4 minutes.
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
KD = np.stack([X.tabs[n]["kd"] for n in X.lines]).astype(float)        # (L, T)
REF = 0.85
B0, S0 = X.B.copy(), X.S.copy()


def variant(kind, par):
    B, S = B0.copy(), S0.copy()
    kd = KD[:, :, None]
    ok = np.isfinite(kd)
    if kind == "rescale":
        f = np.where(ok, np.minimum(REF / np.clip(kd, par, 1.0), 3.0), 1.0)
        B, S = B * f, S * f
    elif kind == "weight":
        w = np.where(ok, np.clip(kd / REF, par, 1.0), 1.0)
        S = S / np.sqrt(w)
    elif kind == "drop":
        bad = ok & (kd < par)
        B, S = np.where(bad, np.nan, B), np.where(bad, np.nan, S)
    return B, S


recs = []
for kind, pars in (("none", [0]), ("rescale", [0.3, 0.5]), ("weight", [0.1, 0.3]), ("drop", [0.3, 0.5])):
    for par in pars:
        Bv, Sv = variant(kind, par)
        for hi, name in enumerate(X.lines):
            tr = [i for i in range(len(X.lines)) if i != hi]
            X.B, X.S = Bv, Sv
            R = X.rungs(tr)
            X.B, X.S = B0, S0                                         # truth stays the original table
            eff = 1.5 * np.nanmedian(np.where(R["has"], R["sd"], np.nan)) * R["v3"] / R["sd"]
            rows = X.rows_for(hi)
            d = X.de(eff, hi, rows, per_target=True)
            p = X.pds(eff, hi, rows)
            recs.append({"variant": f"{kind} {par}" if kind != "none" else "none (z x1.5)", "held_out": name,
                         "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d)})
        print(f"{kind} {par} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h23_knockdown_pooling.csv"), index=False)
for col in ["est", "pds", "de_mix"]:
    pv = df.pivot_table(index="variant", columns="held_out", values=col, sort=False)[X.lines]
    pv["mean"] = pv.mean(1)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
