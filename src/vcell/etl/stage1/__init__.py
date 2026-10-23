"""Stage 1 tables, one per source line, on a common target and gene axis (format: etl.stage1.table).

Targets: the panel of VCC_CONTROLS plus every H1 target (train, validation, test); genes: the universe of OUT/gene_lines.csv.

    python -m vcell etl stage1 k562                    # 1 min, local pseudobulk
    python -m vcell etl stage1 h1                      # one pass over the 15 GB H1 training file, about 10 min
    python -m vcell etl stage1 cd4                     # remote rows of the Zhu DE statistics, about 20 min
    python -m vcell etl stage1 hct116acc hek293tacc    # X-Atlas for a new target list from the accumulators, minutes
    python -m vcell etl stage1 hct116 hek293t          # X-Atlas full stream (hours; checkpoints every 20 files)
    python -m vcell etl stage1 kolf                    # KOLF2.1J streamed by range requests (about 3 h); kolf-table: from accum
    python -m vcell etl stage1 hku                     # Pan_H1 and Feng_iPSC from the HKU moments (needs K562 and H1 first)
    python -m vcell etl stage1 u87 u87-strata u87x u87d  # U-87 MG for context E (vehicle; all arms; drug-adjusted; drug only)
    python -m vcell etl stage1 cd4-ctrl                # CD4+ T control profile (DESeq2 baseMean) for qc.identity
    python -m vcell etl stage1 essential essential-sources   # the essential-target axis of the 8-line harness
    python -m vcell etl stage1 k562sc                  # K562 single cells (exact s; not used, H20)
--qc builds the cleaned variants from the qc.sources masks (h1 and X-Atlas; not used, H52).
"""
from __future__ import annotations

import argparse

import pandas as pd

from vcell import config
from vcell.etl.stage1.table import (all_targets, benchmark_targets, from_accum, load, save, table_path,
                                    universe_genes)

__all__ = ["all_targets", "benchmark_targets", "build", "from_accum", "load", "main", "save", "table_path",
           "universe_genes"]

SOURCES = ["k562", "k562sc", "h1", "cd4", "hct116", "hek293t", "hct116acc", "hek293tacc", "kolf", "kolf-table", "hku",
           "u87", "u87-strata", "u87x", "u87d", "cd4-ctrl", "essential", "essential-sources"]


def build(sources, qc: bool = False, max_files=None, name=None, panels_only: bool = False) -> None:
    from vcell.etl.stage1 import cd4, essential, h1, hku, k562, kolf, u87, xatlas

    targets, genes = all_targets(), universe_genes()
    print(f"{len(targets)} targets x {len(genes)} genes", flush=True)
    acc_t = None
    if panels_only:
        acc_t = set(targets)
        for d in ("controls", "controls_final"):
            p = config.REPO / "data" / d / "pert_counts.csv"
            if p.exists():
                acc_t |= set(pd.read_csv(p).target_gene)
        acc_t = sorted(acc_t)
        print(f"accumulating {len(acc_t)} targets (both panels and H1)", flush=True)
    for src in sources:
        print(f"--- {src}", flush=True)
        if src in ("hct116", "hek293t"):
            xatlas.xatlas(src.upper() if src == "hct116" else "HEK293T", targets, genes, max_files=max_files, name=name,
                          qc=qc, accum_targets=acc_t)
        elif src in ("hct116acc", "hek293tacc"):               # minutes: from the all-target accumulators (same gene axis)
            line = ("HCT116" if src == "hct116acc" else "HEK293T") + ("_qc" if qc else "")
            from_accum(line, targets, genes, name or line)
        elif src == "h1":
            h1.h1(targets, genes, qc=qc)
        elif src == "k562":
            k562.k562(targets, genes)
        elif src == "k562sc":
            k562.k562_sc(targets, genes)
        elif src == "cd4":
            cd4.cd4(targets, genes)
        elif src == "cd4-ctrl":
            cd4.cd4_ctrl_cpm(genes)
        elif src == "kolf":
            kolf.stream()
        elif src == "kolf-table":
            kolf.table_from_accum()
        elif src == "hku":
            hku.build()
        elif src == "u87":
            u87.build_vehicle()
        elif src == "u87-strata":
            u87.build_strata()
        elif src == "u87x":
            u87.table_drug_adjusted()
        elif src == "u87d":
            u87.table_drug_only()
        elif src == "essential":
            essential.build_held_out()
        elif src == "essential-sources":
            essential.build_sources()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vcell etl stage1", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", choices=SOURCES)
    ap.add_argument("--max-files", type=int, default=None, help="X-Atlas only: first n Parquet files (a test run)")
    ap.add_argument("--name", default=None, help="X-Atlas only: output name (default the line)")
    ap.add_argument("--qc", action="store_true", help="h1 and X-Atlas: drop the cells qc.sources flags; output <line>_qc")
    ap.add_argument("--panels-only", action="store_true",
                    help="X-Atlas: accumulate only the A/B/C and D/E/F panels plus H1 targets (about 20x less memory)")
    a = ap.parse_args(argv)
    build(a.sources, qc=a.qc, max_files=a.max_files, name=a.name, panels_only=a.panels_only)
