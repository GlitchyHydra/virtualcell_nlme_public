"""`python -m vcell evaluate proxy|harness|h1 --config ...`: the three offline evaluations of a run config.

    proxy     each main line (K562, CD4_Rest, HCT116, HEK293T, H1) held out, predicted from the other four plus the config's
              extra lines (line_weights) with the config's recipe; proxy.est and its pds / DE parts (as eda/68, F10)
    harness   the 8-line objective (evaluation.harness) at the config's settings
    h1        the official scorer on the H1 benchmark (evaluation.h1_benchmark; ~25 min)
Results go to OUT/eval/<tag>_<what>.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vcell import config
from vcell.config import ModelConfig, RunConfig
from vcell.evaluation import proxy
from vcell.evaluation.harness import EDGES, Harness
from vcell.evaluation.panel import LINES, Panel, standardised
from vcell.model.promoter import neighbour_cap, neighbour_pairs


def extra_lines(m: ModelConfig) -> list[tuple[str, float]]:
    """Source lines beyond the five main ones, with their weights, in config order."""
    return [(n, float(w)) for n, w in m.weights().items() if n not in LINES and w > 0]


def proxy_lines(m: ModelConfig, seed: int = 0) -> pd.DataFrame:
    """eda/68_proxy_panels.py for any recipe: z x scale (transform z only), bins, neighbour cap, floor; per held-out line."""
    assert m.transform == "z" and m.rung == "v3", "the proxy evaluates the standardised v3 recipe"
    extra = extra_lines(m)
    P = Panel.load()
    for n, _ in extra:
        P = P.add(n)
    P.min_effect = m.min_effect
    pairs = neighbour_pairs(P.targets, P.genes) if m.neighbour is not None else None
    mult = None if m.bin_mult is None else np.array(m.bin_mult, np.float32)
    rows_out = []
    for hi in range(5):
        tr = [i for i in range(5) if i != hi] + list(range(5, 5 + len(extra)))
        lam = [1.0] * 4 + [w for _, w in extra]
        R = P.rungs(tr, kappa=m.kappa, lam=None if all(x == 1.0 for x in lam) else lam)
        eff = standardised(R, m.scale)
        if mult is not None:
            eff = eff * mult[np.digitize(P.ctrl_cpm(hi), EDGES)][None, :]
        if pairs is not None:
            ti, gj = pairs.ti.to_numpy(), pairs.gj.to_numpy()
            eff[ti, gj] = np.minimum(eff[ti, gj], neighbour_cap(pairs, m.neighbour))
        rows = P.rows_for(hi)
        p = P.pds(eff, hi, rows)
        d = P.de(eff, hi, rows, seed=seed, per_target=True)
        rows_out.append({"held out": P.lines[hi], "targets": len(rows), "pds raw": p,
                         "pds part": (p - 0.5) / proxy.SPAN["pds"] / 6, "DE part": proxy.mixed(d) / 6, "est": proxy.est(p, d)})
    return pd.DataFrame(rows_out)


def harness_settings(m: ModelConfig) -> dict:
    """Harness.objective arguments for a recipe (extra lines other than KOLF, Pan_H1, Feng_iPSC are not on the harness)."""
    w = m.weights()
    other = [n for n, _ in extra_lines(m) if n not in ("KOLF", "Pan_H1", "Feng_iPSC")]
    assert not other, f"the harness has no {other}"
    return {"c": m.scale, "mult": None if m.bin_mult is None else list(m.bin_mult), "neighbour": m.neighbour,
            "floor": m.min_effect, "kappa": m.kappa, "w_kolf": w.get("KOLF", 0.0), "w_pan": w.get("Pan_H1", 0.0),
            "w_feng": w.get("Feng_iPSC", 0.0)}


def main(what: str, cfg: RunConfig, argv) -> int:
    out = config.OUT / "eval"
    out.mkdir(parents=True, exist_ok=True)
    if what == "h1":
        from vcell.evaluation import h1_benchmark
        return h1_benchmark.main(cfg, argv)
    if what == "proxy":
        df = proxy_lines(cfg.model)
        print(df.round(4).to_string(index=False))
        print(f"\nmean est over the 4 panel lines (H1 excluded, it scores its own benchmark targets): "
              f"{df[df['held out'] != 'H1'].est.mean():.4f}")
    else:
        vals = Harness().objective(**harness_settings(cfg.model))
        df = pd.DataFrame({"held out": list(vals), "est": list(vals.values())})
        print(df.round(4).to_string(index=False))
        print(f"\nmean est over the 8 lines: {df.est.mean():.4f}")
    df.to_csv(out / f"{cfg.tag}_{what}.csv", index=False)
    return 0
