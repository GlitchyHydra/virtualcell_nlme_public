"""Figures of the mse trade-off: Figures of the mse trade-off (OUT/experiments/fig_mse_leaderboard.png, OUT/experiments/fig_mse_attempts.png).

Leaderboard (A/B/C): teams pasted on 2026-10-21 (ranks 33-34, 306, 331-335, 421-426), rank 1 from the log, our L2 and L5.
Attempts: row 1, every official H1 scorer run (OUT/h1_score_*: raw mse from aggregates.csv, scores from scores.csv);
row 2, the proxy leave one line out: the standardised effect z x c for c from 0.05 to 3 (raw mse ratio as failed/h34_mse_structure.py, raw values
of the other metrics as h11_standardised_effect.py; CD4_Rest left out, its DESeq2 standard errors distort the mse ratio), and the shared-direction
attempts of H39b (OUT/experiments/global_direction.csv). x is always the raw mse ratio: 1 = predicting no change; the mse score is
above 0 only left of the baseline (the context's own mean response given to every target: about 0.985 on A/B/C).

    python experiments/analysis/mse_tradeoff_figures.py

Writes OUT/experiments/mse_tradeoff_figures_*.png (EXPERIMENT_LOG 'Top of the leaderboard and what mse means'). Reads
the H1 scorer outputs in OUT and the table of failed/h39b_global_direction.py (run it first).
Was eda/56_mse_tradeoff.py. Snapshot abc (the A/B/C validation tables). 2 minutes.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

from vcell import config
from vcell.evaluation.panel import Panel

config.use_snapshot("abc")
X = Panel.load()
warnings.filterwarnings("ignore")
MET = ["pds", "nmae", "fid", "reach", "jac"]
TITLE = {"pds": "pds: own profile ranked first", "nmae": "nmae: fold-change error", "fid": "fid: sign fidelity",
         "reach": "reach: real DE genes called", "jac": "jac: overlap of DE gene sets", "overall": "overall"}


def logx(a, ticks):
    a.set_xscale("log")
    a.set_xticks(ticks)
    a.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_minor_formatter(mt.NullFormatter())
    a.grid(alpha=0.25)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_xlabel("raw mse ratio (log; 1 = no change)")


# ---------------------------------------------------------------- leaderboard
LB = [  # rank, overall, pds, mse score, raw mse, jac, nmae, fid, reach (scores, except raw mse)
    (1, 0.473, 0.873, 0.792, 0.236, 0.079, 0.440, 0.185, 0.471),
    (33, 0.3041, 0.816, 0.472, 0.539, -0.004, 0.217, 0.045, 0.278),
    (34, 0.3037, 0.816, 0.472, 0.539, -0.003, 0.232, 0.040, 0.266),
    (306, 0.1968, 0.767, 0.000, 2.587, 0.000, 0.165, 0.019, 0.229),
    (331, 0.1913, 0.759, 0.134, 0.860, 0.001, 0.141, -0.012, 0.125),
    (332, 0.1913, 0.780, 0.090, 0.903, 0.002, 0.157, -0.013, 0.130),
    (333, 0.1911, 0.699, 0.000, 4.517, 0.003, 0.175, 0.036, 0.233),
    (334, 0.1910, 0.670, 0.149, 0.846, -0.004, 0.130, -0.010, 0.212),
    (335, 0.1906, 0.672, 0.131, 0.863, 0.007, 0.164, 0.005, 0.164),
    (421, 0.1716, 0.722, 0.178, 0.818, -0.014, 0.080, -0.053, 0.116),
    (422, 0.1716, 0.645, 0.083, 0.909, -0.014, 0.139, -0.009, 0.185),
    (423, 0.1714, 0.645, 0.083, 0.909, -0.014, 0.138, -0.009, 0.186),
    (424, 0.1711, 0.723, 0.127, 0.867, -0.001, 0.084, -0.018, 0.111),
    (425, 0.1707, 0.723, 0.127, 0.867, -0.001, 0.084, -0.018, 0.109),
    (426, 0.1707, 0.645, 0.083, 0.909, -0.014, 0.138, -0.009, 0.181),
]
lb = pd.DataFrame(LB, columns=["rank", "overall", "pds", "mse", "mse_raw", "jac", "nmae", "fid", "reach"])
ours = pd.DataFrame([("L2", 0.160, 0.568, 0, 3.636, 0.005, 0.165, 0.026, 0.198),
                     ("L5", 0.1705, 0.603, 0, 3.806, 0.002, 0.178, 0.029, 0.212)],
                    columns=["label", "overall", "pds", "mse", "mse_raw", "jac", "nmae", "fid", "reach"])
fig, ax = plt.subplots(1, 6, figsize=(27, 5.2))
pos = lb.mse > 0
for k, m in enumerate(MET + ["overall"]):
    a = ax[k]
    logx(a, [0.25, 0.5, 1, 2, 4])
    a.scatter(lb.mse_raw[pos], lb[m][pos], c="tab:blue", s=45, label="teams with mse score > 0")
    a.scatter(lb.mse_raw[~pos], lb[m][~pos], c="tab:orange", s=45, label="teams with mse score = 0")
    a.scatter(ours.mse_raw, ours[m], c="red", marker="*", s=220, edgecolor="k", label="ours (L2 0.160, L5 0.171)")
    for _, row in lb.iterrows():
        if row["rank"] in (1, 33, 306, 333, 421):
            a.annotate(f"#{int(row['rank'])}", (row.mse_raw, row[m]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    a.axvline(0.985, color="k", ls="--", lw=0.9)
    a.set_title(TITLE[m], fontsize=11)
ax[0].set_ylabel("score (leaderboard scale)")
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Leaderboard A/B/C (rows pasted 2026-10-21): teams that earn mse (blue, left of the dashed baseline 0.985) "
             "score lower on fid and reach than the mse = 0 teams (orange) at the same overall; only #1 and #33 "
             "have both", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(config.results("mse_tradeoff_figures_leaderboard.png"), dpi=110)

# ---------------------------------------------------------------- official H1 scorer runs
H1LAB = {"gen_v1": "generic response", "gen_v3": "pool v3", "gen_v3b": "v3b shrinkage prior", "gen_v3_z15": "z x1.5",
         "gen_v3_z15_s1_cnnone": None, "gen_v3_z15_cn2": None, "gen_v3_z15_cn0.5": None, "gen_v3_z15_s1_cn2": None,
         "gen_v3_z15_pan4": "z x1.5 + Pan H1 x4", "xfer_k562": "real K562", "xfer_hct116": "real HCT116",
         "xfer_pan_h1": "real Pan-H1 alone", "oracle_h1_x1": "true H1 effects", "oracle_h1_x1_floor0": None,
         "oracle_h1_x1_s1": None}
KEY = {"pds": "pds_cosine", "nmae": "de_wilcoxon_lfc_nmae", "fid": "de_wilcoxon_direction_fidelity_yield_raw",
       "reach": "de_wilcoxon_direction_reach_raw", "jac": "de_wilcoxon_sig_jaccard", "mse": "expr_mse_unbiased_capped_norm",
       "overall": "avg_score"}
h1 = []
for tag, lab in H1LAB.items():
    d = config.OUT / f"h1_score_{tag}"          # the official scorer outputs
    if not (d / "scores.csv").exists():
        continue
    sc = pd.read_csv(d / "scores.csv").set_index("metric")["from_replicate"]
    ag = pd.read_csv(d / "aggregates.csv").set_index("metric")["raw_value"]
    h1.append({"tag": tag, "label": lab, "mse_raw": ag[KEY["mse"]], **{m: sc[KEY[m]] for m in MET + ["overall"]}})
h1 = pd.DataFrame(h1)

# ---------------------------------------------------------------- proxy sweep of the effect size
nt = X.not_target


def mse_ratio(eff, hi, rows):
    """Raw mse ratio as failed/h34_mse_structure.py: sum w ((pred - beta)^2 - s^2) / sum w (beta^2 - s^2), after the generator's floor."""
    e = np.where(np.abs(eff) < X.min_effect, 0.0, eff)
    b, s = X.B[hi][rows][:, nt], X.S[hi][rows][:, nt]
    ok = np.isfinite(b) & np.isfinite(s)
    w = np.broadcast_to(X.wts(hi)[nt], b.shape)[ok]
    p = np.nan_to_num(e[rows][:, nt])[ok]
    return float((w * ((p - b[ok]) ** 2 - s[ok] ** 2)).sum() / (w * (b[ok] ** 2 - s[ok] ** 2)).sum())


cache = config.results("mse_tradeoff_figures_sweep.csv")
SCALES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
if cache.exists() and "raw_fid" in pd.read_csv(cache, nrows=1).columns:
    sw = pd.read_csv(cache)
else:
    sw = []
    for hi, name in enumerate(X.lines):
        tr = [i for i in range(len(X.lines)) if i != hi]
        R = X.rungs(tr)
        rows = X.rows_for(hi)
        base_eff = X.baseline_eff(hi, rows)
        base_raw = X.evaluate(base_eff, hi, rows)
        base_mse = mse_ratio(base_eff, hi, rows)
        ref = np.nanmedian(np.where(R["has"], R["sd"], np.nan))
        z = ref * R["v3"] / R["sd"]
        for c in SCALES:
            raw = X.evaluate(c * z, hi, rows)
            sw.append({"line": name, "c": c, "mse_raw": mse_ratio(c * z, hi, rows), "base_mse": base_mse,
                       **{f"raw_{m}": raw[m] for m in MET}, **{f"base_{m}": base_raw[m] for m in MET}})
        print(f"{name}: baseline mse ratio {base_mse:.3f}", flush=True)
    sw = pd.DataFrame(sw)
    sw.to_csv(cache, index=False)
sw = sw[sw.line != "CD4_Rest"]
gd = pd.read_csv(config.results("h39b_global_direction.csv"))

fig, ax = plt.subplots(2, 6, figsize=(27, 10.5))
# row 1: official H1 scorer
for k, m in enumerate(MET + ["overall"]):
    a = ax[0, k]
    logx(a, [0.1, 0.25, 0.5, 1, 2, 4, 8])
    lab, oth = h1[h1.label.notna()], h1[h1.label.isna()]
    col = ["tab:green" if t.startswith("oracle") else "tab:purple" if t.startswith("xfer") else "tab:red" for t in lab.tag]
    a.scatter(oth.mse_raw, oth[m], c="0.65", s=18)
    a.scatter(lab.mse_raw, lab[m], c=col, s=45)
    for _, row in lab.iterrows():
        a.annotate(row.label, (row.mse_raw, row[m]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    a.axvline(0.985, color="k", ls="--", lw=0.9)
    a.set_title(("Official H1 scorer: " if k == 0 else "") + TITLE[m], fontsize=11)
ax[0, 0].set_ylabel("score (from replicate)")
ax[0, 5].text(0.98, 0.97, "red: our models (grey: seed and cell-noise repeats)\npurple: real effects of one other line, "
              "unscaled\ngreen: the true H1 effects", transform=ax[0, 5].transAxes, fontsize=8, va="top", ha="right")

# row 2: proxy sweep (raw values) and the shared-direction attempts
cmap = plt.get_cmap("tab10")
LC = {"K562": cmap(0), "HCT116": cmap(2), "HEK293T": cmap(3), "H1": cmap(4)}
for k, m in enumerate(MET):
    a = ax[1, k]
    logx(a, [0.9, 1, 1.5, 2, 3, 5, 10, 20])
    a.axhline(0, color="none")
    for name, d in sw.groupby("line", sort=False):
        a.plot(d.mse_raw, d[f"raw_{m}"], "-o", ms=3, color=LC[name], label=name)
        for c, mk, s in ((1.5, "*", 160), (0.1, "s", 45)):
            p = d[d.c == c]
            a.scatter(p.mse_raw, p[f"raw_{m}"], marker=mk, s=s, color=LC[name], edgecolor="k", zorder=3)
        a.axvline(d.base_mse.iloc[0], color=LC[name], ls=":", lw=0.9)
        a.axhline(d[f"base_{m}"].iloc[0], color=LC[name], ls=":", lw=0.6, alpha=0.6)
    better = "lower is better" if m == "nmae" else "higher is better"
    a.set_title(("Proxy, effect size z x c: " if k == 0 else "") + f"{m} raw ({better})", fontsize=11)
ax[1, 0].set_ylabel("raw value (proxy, leave one line out)")
ax[1, 0].legend(fontsize=8, loc="lower right", title="held-out line\nstar c = 1.5 (uploaded)\nsquare c = 0.1",
                title_fontsize=8)
a = ax[1, 5]
a.grid(alpha=0.25)
VC = {"z only": "tab:blue", "z + strength x g_src": "tab:orange", "z + strength x S g_src": "tab:red",
      "z + strength x ORACLE g_L": "0.6", "z + ORACLE strength x ORACLE g_L": "0.3"}
for line, d in gd.groupby("line", sort=False):
    d = d.set_index("variant")
    seq = ["z only", "z + strength x g_src", "z + strength x S g_src"]
    a.plot(d.loc[seq, "mse transfer"], d.loc[seq, "est"], "-", color="0.75", lw=1)
    for v, row in d.iterrows():
        a.scatter(row["mse transfer"], row["est"], c=VC[v], s=55, edgecolor="k", label=v if line == "jurkat" else None)
    a.annotate(line, (d.loc["z only", "mse transfer"], d.loc["z only", "est"]), fontsize=8, xytext=(4, 4),
               textcoords="offset points")
a.set_xlabel("raw mse ratio, weights transferred from the other lines")
a.set_ylabel("five-metric estimate (proxy)")
a.set_title("Proxy H39b: adding target strength x a shared\ndirection (essential axis, Jurkat, HepG2, RPE1)", fontsize=11)
a.legend(fontsize=7, loc="lower left")
fig.suptitle("Our attempts. Top: official H1 scorer; every model of ours sits right of the mse baseline (dashed), only the "
             "true H1 effects reach mse. Bottom: shrinking the effect (c down) moves mse towards 1 but never past the "
             "baseline (dotted), while fid, reach and jac fall to their baseline; a shared direction lowers mse only "
             "with weights fitted on the held-out line and costs pds and the estimate", fontsize=11, wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(config.results("mse_tradeoff_figures_attempts.png"), dpi=110)
print("wrote", config.results("mse_tradeoff_figures_leaderboard.png"), "and", config.results("mse_tradeoff_figures_attempts.png"))
