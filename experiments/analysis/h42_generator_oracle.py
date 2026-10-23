"""H42 / H42b: how much does the generator lose when the biology is right, and how far does one real profile from another
line carry? The generator fed a Stage 1 table's own effects (evaluation.h1_benchmark.oracle_effects: beta x scale, no
pooling, clipped to +-10 log2), on the official H1 scorer:
    H42   the TRUE H1 effects (stage1_H1, Arc 2025 training cells; optimistic, they overlap the reference cells)
    H42b  the real effects of K562, HCT116 or Pan_H1 (another H1 lab), unchanged

Result (EXPERIMENT_LOG H42, H42b): the generator is nearly lossless: true effects score 0.972 vs 1.067 for real fresh cells
(R1) and 0.12-0.13 for the pool (E4); mse 0.858. So the gap is information about the effects, not emission. One other
line's real profile is worse than the standardised pool: K562 -> H1 0.063, HCT116 -> H1 0.014; Pan_H1 alone -0.067 overall
(46 of 126 targets covered) but it beats the pool on its covered targets, the evidence behind the same-line rule (H44, H47).
Was eda/15_generate_h1.py --oracle <table> (eda/out/run_oracle.sh, run_oracle2.sh, run_day.sh). Snapshot abc.
About 30 minutes per run.

    python experiments/analysis/h42_generator_oracle.py [H1 K562 HCT116 Pan_H1] [--floor0] [--seed 0]
"""
import argparse
import json
from dataclasses import replace

from vcell import config
from vcell.evaluation import h1_benchmark as H

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="*", default=["H1", "K562", "HCT116", "Pan_H1"])
    ap.add_argument("--floor0", action="store_true", help="min_effect 0 instead of 0.05 (oracle_h1_x1_floor0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", default="abc", choices=["abc", "def"])
    a = ap.parse_args()
    base = config.load_config("configs/e4_h1.toml").on(a.snapshot)
    for name in a.tables:
        m = replace(base.model, min_effect=0.0) if a.floor0 else base.model
        cfg = replace(base, tag=f"oracle_{name.lower()}_{a.snapshot}_s{a.seed}", seed=a.seed, model=m)
        cfg.activate()
        res = H.run(cfg, P=H.oracle_effects(name, 1.0))
        print(name, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
