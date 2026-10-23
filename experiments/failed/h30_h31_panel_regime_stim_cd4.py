"""H30 / H31 (not supported / marginal): Evaluate in a regime closer to the 2026 panel (Arc kept targets with >= 80% median knockdown).

Held-out K562, HCT116, HEK293T (the lines with a measured knockdown), scored on two row sets: all panel targets, and only
targets knocked down >= 80% in the held-out line. Sources: the other lines of the main five.
Variants:
    z x1.5 (uploaded), z x2
    knockdown rescale: source effects times 0.9 / kd (kd clipped to [floor, 1], factor capped at 3), as failed/h23_knockdown_pooling.py
    + CD4+ T stimulated 8 h, + 48 h, + both (Zhu tables built by vcell.etl.stage1, never pooled before)

    python experiments/failed/h30_h31_panel_regime_stim_cd4.py

Verdict (EXPERIMENT_LOG H30, H31): in a panel-like regime (targets knocked down >= 80%) neither z x2 (K562 +0.006, HCT116
-0.007) nor knockdown rescaling (+0.001 / -0.008) wins; the stimulated CD4+ T conditions are marginal (+ Stim48hr
raises pds, 0.739 vs 0.729, but lowers the DE part; 0.146 vs 0.145) and share donors with Rest.
Was eda/39_vcc_regime.py. Snapshot abc (the A/B/C validation tables). 4 minutes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
extra = {n: load(f"CD4_{n}") for n in ("Stim8hr", "Stim48hr")}
for n, t in extra.items():
    assert list(t["genes"]) == X.genes and list(t["targets"]) == X.targets
    X = X.add(f"CD4_{n}", t)
I8, I48 = X.lines.index("CD4_Stim8hr"), X.lines.index("CD4_Stim48hr")
KD = np.stack([X.tabs[n]["kd"] for n in X.lines]).astype(float)
B0, S0 = X.B.copy(), X.S.copy()
recs = []
for name in ("K562", "HCT116", "HEK293T"):
    hi = X.lines.index(name)
    base = [i for i in range(5) if i != hi]
    rows_all = X.rows_for(hi)
    kd_h = KD[hi, rows_all]
    rows_kd = rows_all[np.isfinite(kd_h) & (kd_h >= 0.8)]
    print(f"{name}: {len(rows_all)} panel targets, {len(rows_kd)} with knockdown >= 0.8", flush=True)
    variants = {"z x1.5 (uploaded)": (base, 1.5, None), "z x2": (base, 2.0, None),
                "kd rescale 0.5": (base, 1.5, 0.5), "kd rescale 0.3": (base, 1.5, 0.3),
                "+ CD4 Stim8hr": (base + [I8], 1.5, None), "+ CD4 Stim48hr": (base + [I48], 1.5, None),
                "+ CD4 both stim": (base + [I8, I48], 1.5, None)}
    for vn, (tr, c, floor) in variants.items():
        Bv, Sv = B0, S0
        if floor is not None:
            f = np.where(np.isfinite(KD), np.minimum(0.9 / np.clip(KD, floor, 1.0), 3.0), 1.0)[:, :, None]
            f[hi] = 1.0                                                        # never touch the truth
            Bv, Sv = B0 * f, S0 * f
        X.B, X.S = Bv, Sv
        R = X.rungs(tr)
        X.B, X.S = B0, S0
        eff = c * np.nanmedian(np.where(R["has"], R["sd"], np.nan)) * R["v3"] / R["sd"]
        for rs, rows in (("all targets", rows_all), ("kd >= 0.8", rows_kd)):
            d = X.de(eff, hi, rows, per_target=True)
            p = X.pds(eff, hi, rows)
            recs.append({"held_out": name, "rows": rs, "variant": vn, "pds": p, "de_mix": proxy.mixed(d), "est": proxy.est(p, d),
                         "nmae": np.nanmean(d["nmae"]), "fid": d["fid"].mean()})
    print(f"{name} done", flush=True)

df = pd.DataFrame(recs)
df.to_csv(config.results("h30_h31_panel_regime_stim_cd4.csv"), index=False)
for col in ["est", "pds", "de_mix", "nmae"]:
    pv = df.pivot_table(index="variant", columns=["rows", "held_out"], values=col, sort=False)
    for rs in ("all targets", "kd >= 0.8"):
        pv[(rs, "mean")] = pv[rs].mean(1)
    pv = pv.sort_index(axis=1, level=0)
    print(f"\n{col}\n" + pv.to_string(float_format=lambda x: f"{x:.3f}"))
