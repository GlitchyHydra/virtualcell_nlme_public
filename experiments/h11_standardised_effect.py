"""H11 (standardised effect, score sweep): Estimated leaderboard score of effect variants, leave one line out, with the proxies in evaluation.panel.

Each held-out line gets its own baseline (its mean real effect over the scored targets, as the scorer does); spans come
from H1. Overall = mean of the five proxied metrics and mse at 0.
Check: the H1 row for v3 should land near the scored 0.076 (E2) and v1 near -0.18 (E1).

    python experiments/h11_standardised_effect.py

Verdict (EXPERIMENT_LOG H11): supported in the proxy on all five held-out lines (at scale 1.5 the estimated overall rises
+0.01 to +0.04 per line) and on the official H1 scorer (E4 0.121 vs E2 0.076); leaderboard L2 0.160 vs L1 0.092. The
standardised effect x1.5 is the core of the final model (model.pool.z_effect).
Was eda/20_score_sweep.py. Snapshot abc (the A/B/C validation tables). 3 minutes.
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


def variants(R: dict) -> dict:
    v3, sd, g = R["v3"], R["sd"], R["g"]
    ref = np.nanmedian(sd)
    prec = v3 * np.minimum(1.0, ref / sd)
    z = v3 / sd * ref
    out = {"zero": np.zeros_like(v3), "v1": R["v1"]}
    for k in (1.0, 1.5, 2.0, 3.0):
        out[f"v3 x{k:g}"] = k * v3
        out[f"prec x{k:g}"] = k * prec
        out[f"z x{k:g}"] = k * z
    out["v3-generic x2"] = 2 * (v3 - g[None, :])
    out["v3 x2, cap 1.5"] = np.clip(2 * v3, -1.5, 1.5)
    return out


rows_out = []
for hi, name in enumerate(X.lines):
    tr = [i for i in range(len(X.lines)) if i != hi]
    R = X.rungs(tr)
    rows = X.rows_for(hi)
    base_raw = X.evaluate(X.baseline_eff(hi, rows), hi, rows)
    base_raw["pds"] = 0.5                                              # one shared prediction cannot rank
    rows_out.append({"held_out": name, "variant": "baseline (proxy)", **{k: base_raw[k] for k in proxy.SPAN}})
    for vname, eff in variants(R).items():
        raw = X.evaluate(eff, hi, rows)
        sc = proxy.score(raw, base_raw)
        rows_out.append({"held_out": name, "variant": vname, **{f"raw_{k}": raw[k] for k in proxy.SPAN},
                         **sc, "n_ours": raw["n_ours"], "n_real": raw["n_real"]})
    print(f"{name}: baseline raw " + " ".join(f"{k} {base_raw[k]:.3f}" for k in proxy.SPAN), flush=True)

df = pd.DataFrame(rows_out)
df.to_csv(config.results("h11_standardised_effect.csv"), index=False)
d = df[df.variant != "baseline (proxy)"]
print("\nH1 baseline from the scorer: " + " ".join(f"{k} {v:.3f}" for k, v in proxy.H1_BASE.items()))
for metric in ["overall", "pds", "fid", "reach", "jac", "nmae"]:
    pv = d.pivot(index="variant", columns="held_out", values=metric)[X.lines]
    pv["mean"] = pv.mean(1)
    pv["mean w/o CD4"] = pv.drop(columns="CD4_Rest").iloc[:, :4].mean(1)
    print(f"\n{metric} (leaderboard scale)")
    print(pv.sort_values("mean", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))
print("\nmedian significant genes per target, ours / real")
pv = d.pivot(index="variant", columns="held_out", values="n_ours")[X.lines]
print(pv.to_string(float_format=lambda x: f"{x:.0f}"))
print("real:", d.groupby("held_out").n_real.first().reindex(X.lines).to_dict())
