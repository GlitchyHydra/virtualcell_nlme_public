"""H54: can the U87 drug arms of GSE225775 (lapatinib, nintedanib, trametinib, ZSTK474 at 1 and 10 uM; 4x the vehicle
cells) join the U87 table for context E once the average drug effect is removed? (The user's idea: subtract the mean drug
effect as a covariate.) Per stratum (treatment x dose x replicate) and gene, f = mean CPM of all guide-assigned cells /
vehicle mean; cells divided by f, then pooled. Rule fixed before the run: a drug arm is pooled if its mean z^2 against
vehicle is <= 1.2 x the null (vehicle replicate 1 vs 2). The builder is etl.stage1.u87.table_drug_adjusted; this runs it
into a scratch folder and prints the check and the comparison with the vehicle-only table.

Verdict (EXPERIMENT_LOG H54): supported, adopted as the variant final_u87x (F18, configs/final_def_u87x.toml). All four
arms pass (1.01-1.04 x null); median cells per panel target 82 -> 451, median s 0.689 -> 0.350; r = 0.40 with the vehicle
betas. Held-out check: h55_u87_heldout.py; as an ordinary source: analysis/h56_u87_as_source.py.
Was eda/70_u87_drug_adjusted.py (the accumulation over the raw GEO files is `python -m vcell etl stage1 u87-strata`).
Snapshot def. About 1 minute from data/sources/mcfaline/U87_strata_accum.npz.

    python experiments/h54_u87_drug_adjusted.py
"""
import shutil

import numpy as np

from vcell import config
from vcell.etl.stage1 import table, u87

config.use_snapshot("def")

if __name__ == "__main__":
    scratch = config.results("h54_tables")
    scratch.mkdir(exist_ok=True)
    shutil.copy2(table.table_path("U87"), scratch)              # the vehicle table it compares against
    config.use(tables=scratch)
    u87.table_drug_adjusted()
    new, old = table.load("U87x", scratch), table.load("U87x", config.OUT)
    same = all(np.array_equal(new[k], old[k], equal_nan=True) for k in ("beta", "s", "n", "kd"))
    print(f"\nscratch U87x equals the table in OUT (used by F18): {same}")
