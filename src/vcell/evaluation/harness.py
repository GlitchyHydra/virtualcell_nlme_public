"""The 8-line harness behind every tuning decision (was eda/tune_harness.py): proxy.est on eight held-out lines.

    main axis (Panel of the current tables)   K562, CD4_Rest, HCT116, HEK293T, H1 held out in turn; KOLF, Pan_H1 and
                                              Feng_iPSC available as extra sources (lines 5, 6, 7)
    essential axis (stage1_ess_* tables)      Jurkat, HepG2, RPE1 held out (sources: the other six lines of the axis);
                                              600 random essential targets (seed 0) for memory; never tuned on before H29

    H = Harness()
    vals = H.objective(c=1.5, mult=[1.5, 1.25, 1, 1, 0.8], neighbour=0.15, floor=0.02, w_kolf=0.25, w_pan=0.25)  # final model

objective() options (defaults = the first upload's model):
    c          scale of the standardised effect
    mult       per-bin multipliers on c, bins of the held-out line's control CPM at EDGES (H36)
    neighbour  f of the promoter rule (H37); None = off
    floor      the generator's min_effect: smaller |log2| effects are left out
    kappa      weight of tau^2 in the standardising sd
    w_kolf, w_pan, w_feng   KOLF, Pan_H1, Feng_iPSC in the main pool at this weight (combination only, tau^2 unweighted)
For other effect recipes, loop over H.cases() and score with H.score() (experiments/ use this).
"""
from __future__ import annotations

import warnings

import numpy as np

from vcell import config
from vcell.etl.stage1.table import load
from vcell.evaluation import proxy
from vcell.evaluation.panel import LINES, Panel, standardised
from vcell.model.promoter import neighbour_cap, neighbour_pairs

warnings.filterwarnings("ignore")
EDGES = [15, 40, 120, 400]
BIN_LAB = ["<15", "15-40", "40-120", "120-400", ">400"]
EXTRA = ["KOLF", "Pan_H1", "Feng_iPSC"]
KI = 5                                          # index of the first extra line on the main axis
ESS_LINES = ["K562", "CD4_Rest", "HCT116", "HEK293T", "jurkat", "hepg2", "rpe1"]
ESS_TARGETS = 600


def essential_panel(n_targets: int | None = ESS_TARGETS, seed: int = 0) -> Panel:
    """The essential-axis Panel: 7 lines, n_targets random targets (None: all 2,394, about 10 GB of pooling temporaries;
    pds compares genes that are no target of the full axis)."""
    tabs = {n: load(f"ess_{n}", config.OUT) for n in ESS_LINES}   # the essential axis is the same in every snapshot
    genes, targets = list(tabs["K562"]["genes"]), list(tabs["K562"]["targets"])
    not_target = np.array([g not in set(targets) for g in genes])
    keep = (np.arange(len(targets)) if n_targets is None
            else np.sort(np.random.default_rng(seed).choice(len(targets), n_targets, replace=False)))
    sub = [targets[i] for i in keep]
    B = np.stack([tabs[n]["beta"] for n in ESS_LINES]).astype(float)[:, keep].copy()
    S = np.stack([tabs[n]["s"] for n in ESS_LINES]).astype(float)[:, keep].copy()
    NC = np.stack([tabs[n]["n"] for n in ESS_LINES]).astype(float)[:, keep].copy()
    tabs = {n: {**tabs[n], "beta": None, "s": None} for n in ESS_LINES}
    max_cpm = np.nanmax(np.stack([np.where(np.isfinite(tabs[n]["ctrl_cpm"]), tabs[n]["ctrl_cpm"], 0) for n in ESS_LINES]), 0)
    return Panel(lines=list(ESS_LINES), tabs=tabs, genes=genes, targets=sub, B=B, S=S, NC=NC, max_cpm=max_cpm,
                 panel=set(sub), bench=set(sub), not_target=not_target)


class Harness:
    def __init__(self, main: Panel | None = None, ess: Panel | None = None):
        main = main or Panel.load(LINES)
        for n in EXTRA:
            if n not in main.lines:
                main = main.add(n)
        self.main = main
        self.ess = ess or essential_panel()
        self.pairs = {"main": neighbour_pairs(self.main.targets, self.main.genes),
                      "ess": neighbour_pairs(self.ess.targets, self.ess.genes)}
        self.states = {"main": (self.main, [0, 1, 2, 3, 4]), "ess": (self.ess, [4, 5, 6])}
        self.cache = {}

    def cases(self):
        """(axis, panel, held-out index, scored rows, promoter pairs) for the eight held-out lines, in the fixed order."""
        for axis, (P, held) in self.states.items():
            for hi in held:
                yield axis, P, hi, P.rows_for(hi), self.pairs[axis]

    def training(self, axis: str, hi: int, w_kolf=0.0, w_pan=0.0, w_feng=0.0):
        """Training line indices and their combination weights (None = all 1) for held-out line hi."""
        if axis == "main":
            extra = [(KI + j, w) for j, w in enumerate((w_kolf, w_pan, w_feng)) if w > 0]
            tr = [i for i in range(5) if i != hi] + [i for i, _ in extra]
            lam = [1.0] * (len(tr) - len(extra)) + [w for _, w in extra]
            return tr, (None if all(x == 1.0 for x in lam) else lam)
        return [i for i in range(len(ESS_LINES)) if i != hi], None

    def pooled(self, axis, hi, kappa=1.0, w_kolf=0.0, w_pan=0.0, w_feng=0.0) -> np.ndarray:
        """The standardised pooled effect (scale 1, float32) for held-out line hi, cached."""
        ws = (w_kolf, w_pan, w_feng) if axis == "main" else (0.0, 0.0, 0.0)
        key = (axis, hi, kappa) + ws
        if key not in self.cache:
            tr, lam = self.training(axis, hi, *ws)
            R = self.states[axis][0].rungs(tr, kappa=kappa, lam=lam)
            self.cache[key] = standardised(R)
        return self.cache[key]

    @staticmethod
    def context_steps(eff, P: Panel, hi, pairs, mult=None, neighbour=None) -> np.ndarray:
        """The per-context steps of the model on a held-out line: bin multipliers by its control CPM, the neighbour cap."""
        if mult is not None:
            eff = eff * np.asarray(mult, np.float32)[np.digitize(P.ctrl_cpm(hi), EDGES)][None, :]
        if neighbour is not None:
            eff = eff.copy()
            ti, gj = pairs.ti.to_numpy(), pairs.gj.to_numpy()
            eff[ti, gj] = np.minimum(eff[ti, gj], neighbour_cap(pairs, neighbour))
        return eff

    @staticmethod
    def score(P: Panel, eff, hi, rows, floor=0.05, seed=0) -> float:
        d = P.de(eff, hi, rows, seed=seed, per_target=True, min_effect=floor)
        return proxy.est(P.pds(eff, hi, rows), d)

    def objective(self, c=1.5, mult=None, neighbour=None, floor=0.05, kappa=1.0, w_kolf=0.0, w_pan=0.0, w_feng=0.0) -> dict:
        """proxy.est per held-out line (8 lines)."""
        vals = {}
        for axis, P, hi, rows, pairs in self.cases():
            eff = c * self.pooled(axis, hi, kappa, w_kolf, w_pan, w_feng)
            eff = self.context_steps(eff, P, hi, pairs, mult, neighbour)
            vals[P.lines[hi]] = self.score(P, eff, hi, rows, floor)
        return vals


def final_settings() -> dict:
    """objective() arguments of the final model (L5: H38 + H40)."""
    return {"c": 1.5, "mult": [1.5, 1.25, 1, 1, 0.8], "neighbour": 0.15, "floor": 0.02, "w_kolf": 0.25, "w_pan": 0.25}


def with_main(h: Harness, main: Panel) -> Harness:
    """A harness on another main panel (e.g. cleaned tables swapped in), sharing the essential axis."""
    return Harness(main=main, ess=h.ess)


FINAL_BINS = [1.5, 1.25, 1, 1, 0.8]


def model_effect(P: Panel, R: dict, hi: int, pairs, scale: float = 1.5, mult=FINAL_BINS, neighbour=0.15) -> np.ndarray:
    """The final recipe on proxy rungs for held-out line hi: standardised effect x scale (float32), the bin multipliers by
    the line's control CPM, the promoter-neighbour cap (the steps of eda/68-72; the floor is Panel.min_effect)."""
    return Harness.context_steps(standardised(R, scale), P, hi, pairs, mult, neighbour)


def sweep(H: Harness, runs, csv=None, width: int = 45, up_down: bool = True, str_settings: bool = False):
    """Run a list of (label, objective kwargs) and print each against the first: the mean over the 8 lines, its change, and
    the change per line (experiments h38, h40 and others). Writes the table to `csv` after every run."""
    import pandas as pd
    recs, ref = [], None
    for label, kw in runs:
        v = H.objective(**kw)
        ref = ref or v
        m = float(np.mean(list(v.values())))
        dl = {k: v[k] - ref[k] for k in v}
        recs.append({"label": label, **({k: str(x) for k, x in kw.items()} if str_settings else kw), **v, "mean_all8": m})
        if csv is not None:
            pd.DataFrame(recs).to_csv(csv, index=False)
        ud = f"; up {sum(x > 0.0005 for x in dl.values())}, down {sum(x < -0.0005 for x in dl.values())})" if up_down else ")"
        print(f"{label:{width}s} all8 {m:.4f} ({m - np.mean(list(ref.values())):+.4f}{ud}  "
              + " ".join(f"{k} {dl[k]:+.4f}" for k in v), flush=True)
    return pd.DataFrame(recs)
