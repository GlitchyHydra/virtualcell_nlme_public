"""H38: do the small gains add up? Per-bin scale (H36), promoter-neighbour rule (H37), effect floor and kappa (H29), and
KOLF rechecked with the clean weighting (H35's fix), on all 8 held-out lines of the harness.

Verdict (EXPERIMENT_LOG H38): yes, +0.005 in total. Best: c 1.5, bins [1.5, 1.25, 1, 1, 0.8], neighbour 0.15, floor 0.02,
kappa 1, KOLF 0.25; with Pan_H1 0.25 (H40) this became the L5 upload (configs/l5.toml, leaderboard 0.1705).
Logged (A/B/C tables): uploaded 0.1676; bins 0.1703; neighbour 0.15 0.1685; bins + neighbour 0.1711; + floor 0.02 0.1712;
+ kappa 2 0.1707; c 1.375 / 1.625 0.1700 / 0.1708; + KOLF 0.25 0.1725, KOLF 0.5 0.1720.
Was eda/45_combined.py. Snapshot abc. About 10 minutes.

    python experiments/h38_combined.py
"""
from vcell import config
from vcell.evaluation.harness import Harness, sweep

config.use_snapshot("abc")
BINS = [1.5, 1.25, 1.0, 1.0, 0.8]
RUNS = [
    ("uploaded", {}),
    ("bins", {"mult": BINS}),
    ("neighbour 0.15", {"neighbour": 0.15}),
    ("bins + neighbour", {"mult": BINS, "neighbour": 0.15}),
    ("bins + neighbour + floor 0.02", {"mult": BINS, "neighbour": 0.15, "floor": 0.02}),
    ("bins + neighbour + floor 0.02 + kappa 2", {"mult": BINS, "neighbour": 0.15, "floor": 0.02, "kappa": 2.0}),
    ("c 1.375 + bins + neighbour + floor 0.02", {"c": 1.375, "mult": BINS, "neighbour": 0.15, "floor": 0.02}),
    ("c 1.625 + bins + neighbour + floor 0.02", {"c": 1.625, "mult": BINS, "neighbour": 0.15, "floor": 0.02}),
    ("bins + neighbour + floor 0.02 + KOLF 0.5", {"mult": BINS, "neighbour": 0.15, "floor": 0.02, "w_kolf": 0.5}),
    ("bins + neighbour + floor 0.02 + KOLF 0.25", {"mult": BINS, "neighbour": 0.15, "floor": 0.02, "w_kolf": 0.25}),
    ("neighbour 0.05", {"neighbour": 0.05}),
]

if __name__ == "__main__":
    sweep(Harness(), RUNS, config.results("h38_combined.csv"))
