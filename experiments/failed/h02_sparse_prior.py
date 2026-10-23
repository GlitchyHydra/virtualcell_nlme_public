"""H2 (rejected): shrinking weak effects toward zero with a sparse prior (v3b: point mass at 0 plus zero-centred normals,
fitted by EM; model.pool.ash_fit / ash_post) reduces noise and improves the score. E3 on the official H1 scorer.

Verdict (EXPERIMENT_LOG H2, E3): rejected under the real scorer: overall -0.142 vs 0.076 for v3 (E2), although the proxy
metrics of the time (rank correlation, sign accuracy) preferred it. The prior puts about 99.5% of its weight on "no effect",
so the generated cells look like unchanged controls on fid and jac; the DE metrics need the weak but real effects. This
mismatch is why every later decision used proxies of the scored metrics themselves (evaluation.panel).
Was eda/15_generate_h1.py --rung v3b. Snapshot abc. About 35 minutes (the prior fit adds ~5 min).

    python experiments/failed/h02_sparse_prior.py [--snapshot abc|def]
"""
import argparse
import json
from dataclasses import replace

from vcell import config
from vcell.evaluation import h1_benchmark as H

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="abc", choices=["abc", "def"])
    a = ap.parse_args()
    base = config.load_config("configs/e4_h1.toml").on(a.snapshot)
    cfg = replace(base, tag=f"e3_v3b_{a.snapshot}", model=replace(base.model, rung="v3b", transform="none", scale=1.0))
    print(json.dumps({k: round(v, 4) for k, v in H.run(cfg).items()}))
