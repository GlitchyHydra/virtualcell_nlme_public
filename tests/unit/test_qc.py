"""Cell QC rules (qc.sources.cell_mask) and the same-line rule (model.sources)."""
import numpy as np

from vcell.model import sources
from vcell.qc.sources import REASONS, cell_mask


def _cells(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    total = rng.lognormal(np.log(20000), 0.2, n)
    genes = total ** 0.8
    mt = rng.uniform(1, 4, n)
    return total, genes, mt


def test_planted_outliers_are_flagged_with_their_reason():
    total, genes, mt = _cells()
    total[0] = 300                                               # low depth (genes follow, but depth is checked first)
    genes[0] = 300 ** 0.8
    total[1] = 400_000                                           # doublet-like depth
    mt[2] = 25                                                   # above the hard mito limit
    multi = np.zeros(len(total), bool)
    multi[3] = True
    keep, reason = cell_mask(total, genes, mt, multi_guide=multi)
    assert [REASONS[r] for r in reason[:4]] == ["low_depth", "high_depth", "high_mito", "multi_guide"]
    assert keep[4:].mean() > 0.99


def test_mito_floor_spares_probe_based_data():
    total, genes, _ = _cells()
    mt = np.full(len(total), 0.02)                               # 10x Flex: mito near zero
    mt[0] = 0.5                                                  # far above the median in MADs, but below the 5% floor
    keep, reason = cell_mask(total, genes, mt)
    assert keep[0] and REASONS[reason[0]] == "ok"


def test_thresholds_are_per_batch():
    total, genes, mt = _cells()
    total[1000:] *= 4                                            # a second, 4x deeper batch
    genes[1000:] *= 4 ** 0.8
    total[1500] = 20_000                                         # shallow for its batch, typical for the other one
    genes[1500] = 20_000 ** 0.8
    batch = np.r_[np.zeros(1000), np.ones(1000)]
    keep_b, reason_b = cell_mask(total, genes, mt, batch)
    keep_all, _ = cell_mask(total, genes, mt)
    assert not keep_b[1500] and REASONS[reason_b[1500]] == "low_depth"
    assert keep_all[1500]                                        # pooled thresholds would miss it
    assert keep_b.mean() > 0.99


def test_same_line_rule():
    con = {"HCT116": 0.05, "K562": np.nan, "H1": 0.4}
    fps = {"K562": 0.9, "HCT116": 0.1}
    size = {"K562": 25, "HCT116": 40}
    assert sources.suggest(con, fps, size, pou5f1_cpm=3.0) == {"HCT116": 4.0, "K562": 4.0}
    assert sources.suggest({"H1": 0.3}, {}, {}, pou5f1_cpm=150.0) == {"Pan_H1": 4.0}
    assert sources.suggest({"HCT116": 0.11}, {"K562": 0.9}, {"K562": 9}, pou5f1_cpm=99.9) == {}
    assert sources.suggest({"KOLF": 0.0}, {}, {}, pou5f1_cpm=0.0) == {}          # no panel data: never upweighted
