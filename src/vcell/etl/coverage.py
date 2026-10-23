"""Which panel targets each source line measures (was eda/03_coverage.py), and the check that every panel target has a
Stage 1 row before a build.

    targets_per_line()   cells per panel target in K562, HCT116, HEK293T and CD4+ T (three conditions), the target's own
                         knockdown where available, lines per target -> OUT/coverage_targets.csv
    missing_rows()       panel targets without a row in the Stage 1 tables (a build must report 0)
"""
from __future__ import annotations

import pandas as pd

from vcell import config, io


def missing_rows(table_targets, panel=None) -> list[str]:
    have = set(table_targets)
    return [t for t in (panel or config.panel_targets()) if t not in have]


def targets_per_line() -> None:
    panel = config.panel_targets()
    S = config.SOURCES
    cells = pd.DataFrame(index=panel)
    kd = pd.DataFrame(index=panel)

    # Replogle K562 genome-wide pseudobulk: cells per target and the target's own knockdown
    b = io.read_replogle_bulk(S / "replogle" / "K562_gwps_raw_bulk_01.h5ad")
    m = b["meta"][~b["meta"].is_nt].sort_values("cells", ascending=False).drop_duplicates("symbol").set_index("symbol")
    cells["K562"] = m.cells.reindex(panel)
    kd["K562"] = m.kd.reindex(panel)
    print(f"K562 genome-wide: {len(m)} targets, {cells.K562.notna().sum()} of the panel; "
          f"{int(b['meta'].is_nt.sum())} non-targeting pseudobulks")

    # X-Atlas/Orion: dual-guide cells per target (the guide calls are per cell, so counting is exact)
    for line in ("HCT116", "HEK293T"):
        g = pd.read_csv(S / "xatlas" / f"{line}_filtered_guide_calls_per_cell.csv.gz", usecols=["gene_target", "pass_guide_filter"])
        g = g[g.pass_guide_filter.astype(str).str.lower() == "true"]
        vc = g.gene_target.value_counts()
        cells[line] = vc.reindex(panel)
        print(f"{line}: {len(vc) - 1} targets, {cells[line].notna().sum()} of the panel, "
              f"{int(vc.get('Non-Targeting', 0))} non-targeting cells")

    # Zhu CD4+ T: guide-level table, three conditions
    z = pd.read_csv(S / "zhu" / "guide_kd_efficiency.suppl_table.csv", index_col=0)
    z["symbol"] = z.index.str.replace(r"-\d+$", "", regex=True)
    z["w"] = z.guide_n * z.guide_mean_expr
    for cond, sub in z.groupby("culture_condition"):
        a = sub.groupby("symbol").agg(cells=("guide_n", "sum"), w=("w", "sum"), ntc=("ntc_mean_expr", "first"))
        cells[f"CD4 T {cond}"] = a.cells.reindex(panel)
        kd[f"CD4 T {cond}"] = (1 - (a.w / a.cells) / a.ntc.where(a.ntc > 0)).reindex(panel)
    cells["CD4 T"] = cells[[c for c in cells if c.startswith("CD4 T ")]].max(axis=1)

    lines = ["K562", "HCT116", "HEK293T", "CD4 T"]
    print("\n=== per line: panel targets covered, cells per covered target (q25/median/q75), share with >= 50 cells ===")
    for c in lines:
        v = cells[c].dropna()
        print(f"{c:9s} {len(v):3d}/300  {v.quantile(.25):6.0f} {v.median():6.0f} {v.quantile(.75):6.0f}   {(v >= 50).mean():.2f}")
    print("\n=== knockdown of the target itself (K562 pseudobulk; CD4+ T guide table) ===")
    print(kd.describe().loc[["count", "50%"]].round(2).to_string())

    k = (cells[lines].fillna(0) >= 1).sum(axis=1)
    k50 = (cells[lines].fillna(0) >= 50).sum(axis=1)
    print("\n=== independent lines per target ===")
    for label, kk in (("any cells", k), ("at least 50 cells", k50)):
        print(f"{label:18s} 0 lines: {(kk == 0).sum():3d}   1-2: {((kk >= 1) & (kk <= 2)).sum():3d}   3+: {(kk >= 3).sum():3d}   mean {kk.mean():.2f}")
    print("targets in fewer than 3 lines:", dict(k[k < 3]))
    print("targets missing from K562:", list(cells.index[cells.K562.isna()]))

    cells["n_lines"] = k
    config.OUT.mkdir(parents=True, exist_ok=True)
    cells.join(kd.add_suffix(" kd")).to_csv(config.OUT / "coverage_targets.csv")
    print("\nsaved", config.OUT / "coverage_targets.csv")
