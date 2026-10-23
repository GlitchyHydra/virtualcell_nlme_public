"""Which source lines a context should lean on: the same-cell-type rule (H44, H47; was the rule part of eda/52).

A context that matches a source line with panel data gets that line at weight 4 in the pool (H44: best on H1 with Pan_H1 as
its same-line source; H53: still best for a source as noisy as the U87 table). A stem-like context gets Pan_H1 at 4 (best for
H1 and for KOLF held out; KOLF and Feng hurt at high weight). Related but different lineages get nothing extra (H48), so every
other context keeps the default weights. The evidence (contradiction, fingerprint share) comes from qc.identity.
"""
from __future__ import annotations

import numpy as np

PANEL_DATA = {"K562", "HCT116", "HEK293T", "H1", "Pan_H1", "CD4_Rest"}   # lines with panel data; KOLF, Feng hurt at high weight
SAME_LINE_WEIGHT = 4.0
MAX_CONTRADICTION = 0.10            # same cell type across labs: 0.00-0.06; different lines 0.26-0.74
MIN_FINGERPRINT_SHARE = 0.8         # where contradiction is not measurable: share of the line's fingerprint genes on
MIN_FINGERPRINT_GENES = 10
STEM_POU5F1_CPM = 100.0


def matches(contradiction: dict, fingerprint_share: dict, fingerprint_size: dict) -> list[str]:
    """Lines with panel data whose profile the context matches."""
    out = []
    for n in PANEL_DATA:
        c = contradiction.get(n, np.nan)
        if np.isfinite(c):
            if c <= MAX_CONTRADICTION:
                out.append(n)
        elif fingerprint_size.get(n, 0) >= MIN_FINGERPRINT_GENES and fingerprint_share.get(n, 0.0) >= MIN_FINGERPRINT_SHARE:
            out.append(n)
    return out


def suggest(contradiction: dict, fingerprint_share: dict, fingerprint_size: dict, pou5f1_cpm: float) -> dict:
    """{line: weight} for one context; empty = the default weights."""
    w = {n: SAME_LINE_WEIGHT for n in matches(contradiction, fingerprint_share, fingerprint_size)}
    if pou5f1_cpm >= STEM_POU5F1_CPM:
        w["Pan_H1"] = SAME_LINE_WEIGHT
    return w
