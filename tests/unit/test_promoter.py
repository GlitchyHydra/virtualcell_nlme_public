"""The promoter-neighbour rule (model.promoter, H37)."""
import numpy as np
import pandas as pd
import pytest

from vcell.model import promoter


def test_cap_shape():
    pairs = pd.DataFrame({"distance": [100, 500, 1581, 5000, 9000]})
    cap = promoter.neighbour_cap(pairs, 0.15)
    assert cap[0] == pytest.approx(np.log2(0.15)) and cap[1] == pytest.approx(np.log2(0.15))
    assert cap[3] == pytest.approx(0.0) and cap[4] == pytest.approx(0.0)
    assert cap[1] < cap[2] < cap[3]
    assert cap[2] == pytest.approx(np.log2(0.15 + 0.85 * 0.5), abs=1e-3)   # half way on the log scale (sqrt(10) x 500 bp)


def test_pairs_same_chromosome_within_5kb_no_self():
    tss = pd.DataFrame({"gene": ["T", "near", "far", "other_chr", "edge"], "chrom": ["1", "1", "1", "2", "1"],
                        "strand": ["+"] * 5, "tss": [10_000, 10_400, 20_000, 10_000, 15_000]})
    genes = ["near", "far", "other_chr", "edge", "T"]
    p = promoter.neighbour_pairs(["T", "absent"], genes, tss=tss)
    assert sorted(zip(p.ti, p.gj, p.distance)) == [(0, 0, 400), (0, 3, 5000)]
