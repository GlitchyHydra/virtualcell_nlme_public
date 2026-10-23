"""H47: does the same-line rule (a same-cell-type source at weight 4 in the pool, H44) hold on the official scorer, not
only in the proxy? E6 = E4 (v3, z x1.5, four lines, H1 held out) + Pan_H1 (H1 hESC from another lab) at weight 4.

Verdict (EXPERIMENT_LOG H47, E6): supported: overall 0.190 vs 0.121 / 0.131 for E4 (pds 0.781 vs 0.668 / 0.707, reach 0.183,
nmae 0.095, fid 0.125). Pan_H1 covers 46 of the 126 targets: on them the pool + Pan beats Pan alone on every metric; on the
other 80 it equals the pool. The largest gain of the project on the official scorer, and the rule behind the U87 weight for
context E (configs/final_def.toml).
Was eda/15_generate_h1.py --rung v3 --transform z --scale 1.5 --line-weight Pan_H1=4 (log eda/out/pan4.log). Snapshot abc.
About 30 minutes (vcc-h1 score ~25 min).

    python experiments/h47_same_line_h1_scorer.py [--weight 4] [--snapshot abc|def] [--seed 0]
"""
import argparse
import json
from dataclasses import replace

from vcell import config
from vcell.evaluation import h1_benchmark as H

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, default=4.0)
    ap.add_argument("--snapshot", default="abc", choices=["abc", "def"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = config.load_config("configs/e4_h1.toml").on(a.snapshot)
    m = replace(base.model, line_weights={**base.model.line_weights, "Pan_H1": a.weight})
    cfg = replace(base, tag=f"e6_pan{a.weight:g}_{a.snapshot}_s{a.seed}", seed=a.seed, model=m)
    print(json.dumps({k: round(v, 4) for k, v in H.run(cfg).items()}))
