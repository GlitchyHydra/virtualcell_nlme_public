"""H1 (pooling across lines carries target-specific information beyond the generic response) and H11 on the official
scorer: the E-series of the public H1 benchmark (126 targets x 400 cells, cell-eval2 0.16.0), H1 held out of every fit.
    R0  unchanged control resamples (rung zero): the scorer's lower anchor
    E1  v1, the generic response of the four lines K562, CD4_Rest, HCT116, HEK293T
    E2  v3, their random-effects pool
    E4  v3, standardised effect x1.5 (configs/e4_h1.toml)
The variants with a rejected idea are failed/h02_sparse_prior.py (E3) and failed/h33_cell_noise.py; the same-line run is
h47_same_line_h1_scorer.py (E6); the generator ceiling is analysis/h42_generator_oracle.py.

Verdict (EXPERIMENT_LOG H1, H11, E-series): R0 -0.303, E1 -0.180, E2 0.076 (+0.26 over v1: H1 supported), E4 0.121
(seed 1: 0.131; H11 supported). One draw per row; seeds differ by about 0.01. On the def tables E4 scores 0.130 / 0.124
(F11, OUT/h1_score_axis12816_e4_s0/1), and tests/regression/test_h1_golden.py pins that file bit for bit.
Was eda/15_generate_h1.py with --rung zero / v1 / v3 [--transform z --scale 1.5]. Snapshot abc by default.
About 30 minutes per run (generation ~2 min, vcc-h1 score ~25 min); the 3.5 GB predictions are deleted after scoring.

    python experiments/h01_pooling_h1_scorer.py [R0 E1 E2 E4] [--snapshot abc|def] [--seed 0]
"""
import argparse
import json
from dataclasses import replace

from vcell import config
from vcell.evaluation import h1_benchmark as H

RUNS = {"R0": {"rung": "zero"}, "E1": {"rung": "v1", "transform": "none", "scale": 1.0},
        "E2": {"rung": "v3", "transform": "none", "scale": 1.0}, "E4": {}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=list(RUNS), choices=list(RUNS))
    ap.add_argument("--snapshot", default="abc", choices=["abc", "def"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = config.load_config("configs/e4_h1.toml").on(a.snapshot)
    for name in a.runs:
        cfg = replace(base, tag=f"{name.lower()}_{a.snapshot}_s{a.seed}", seed=a.seed, model=replace(base.model, **RUNS[name]))
        res = H.run(cfg)
        print(name, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
