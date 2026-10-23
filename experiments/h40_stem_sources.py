"""H40: do the new stem-like source lines of the HKU reprocessing (Pan_H1: H1 hESC from another lab; Feng_iPSC) add to
H38's best setting? All 8 held-out lines; H1 is the stem-like one (what a stem-like context would gain), the other seven
show the cost for differentiated contexts.

Verdict (EXPERIMENT_LOG H40): Pan_H1 yes, Feng no. H38 0.1725; + Pan 0.25: 0.1748 (adopted, L5); + Pan 1: 0.1757 (H1
+0.037 but K562 and HCT116 lose); + Feng 0.25: 0.1730; + Feng 1: 0.1706; all three stem lines at 1: 0.1651.
Was eda/49_stem_sources.py. Snapshot abc. About 8 minutes.

    python experiments/h40_stem_sources.py
"""
from vcell import config
from vcell.evaluation.harness import Harness, sweep

config.use_snapshot("abc")
BEST = {"mult": [1.5, 1.25, 1.0, 1.0, 0.8], "neighbour": 0.15, "floor": 0.02}
RUNS = [
    ("uploaded", {}),
    ("H38 best (KOLF 0.25)", {**BEST, "w_kolf": 0.25}),
    ("H38 + Pan 0.25", {**BEST, "w_kolf": 0.25, "w_pan": 0.25}),
    ("H38 + Pan 1", {**BEST, "w_kolf": 0.25, "w_pan": 1.0}),
    ("H38 + Feng 0.25", {**BEST, "w_kolf": 0.25, "w_feng": 0.25}),
    ("H38 + Feng 1", {**BEST, "w_kolf": 0.25, "w_feng": 1.0}),
    ("H38 + Pan 0.25 + Feng 0.25", {**BEST, "w_kolf": 0.25, "w_pan": 0.25, "w_feng": 0.25}),
    ("stem pool: KOLF, Pan, Feng at 1", {**BEST, "w_kolf": 1.0, "w_pan": 1.0, "w_feng": 1.0}),
]

if __name__ == "__main__":
    sweep(Harness(), RUNS, config.results("h40_stem_sources.csv"), width=34, up_down=False, str_settings=True)
