"""The essential-target axis (was eda/28_essential_tables.py and eda/29_essential_sources.py): the Replogle/Nadig essential
screens (Jurkat, HepG2, RPE1; about 2,400 DepMap common-essential targets, no panel target) and the panel-covering source
lines (K562, HCT116, HEK293T, CD4+ T Rest) on one shared target axis, so Jurkat, HepG2 and RPE1 can be held out and scored
like the main lines (the 8-line harness, evaluation.harness). Tables stage1_ess_<line>.npz.

The essential tables were built on the 11,961-gene universe of A/B/C and have not been rebuilt since the universe grew to
12,816 genes (F6); a rebuild now would put them on the new axis.
"""
from __future__ import annotations

from vcell import config
from vcell.etl.stage1.cd4 import cd4
from vcell.etl.stage1.k562 import k562, k562_sc
from vcell.etl.stage1.table import dense_sc_targets, from_accum, load, universe_genes

FILES = {"jurkat": config.SOURCES / "nadig" / "GSE264667_jurkat_raw_singlecell_01.h5ad",
         "hepg2": config.SOURCES / "nadig" / "GSE264667_hepg2_raw_singlecell_01.h5ad",
         "rpe1": config.SOURCES / "replogle" / "rpe1_raw_singlecell_01.h5ad"}


def build_held_out(lines=tuple(FILES)) -> None:
    """Tables of the three essential screens (single cells, eda/28)."""
    targets = sorted(set().union(*(dense_sc_targets(p) for p in FILES.values())))
    genes = universe_genes()
    print(f"{len(targets)} essential targets x {len(genes)} genes; panel targets among them: "
          f"{len(set(targets) & set(config.panel_targets()))}", flush=True)
    for line in lines:
        print(f"--- {line}", flush=True)
        k562_sc(targets, genes, name=f"ess_{line}", path=FILES[line])


def build_sources(sources=("k562", "hct116", "hek293t")) -> None:
    """The panel-covering lines on the essential axis (eda/29); 'cd4' reads remote rows (about 20 min)."""
    targets = [str(t) for t in load("ess_jurkat")["targets"]]
    genes = universe_genes()
    for src in sources:
        print(f"--- {src}", flush=True)
        if src == "k562":
            k562(targets, genes, name="ess_K562")
        elif src == "hct116":
            from_accum("HCT116", targets, genes, "ess_HCT116")
        elif src == "hek293t":
            from_accum("HEK293T", targets, genes, "ess_HEK293T")
        elif src == "cd4":
            cd4(targets, genes, conditions=("Rest",), prefix="ess_CD4")
