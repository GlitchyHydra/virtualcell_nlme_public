"""F10: is the D/E/F panel about as predictable as the A/B/C panel from other cell types? The leave-one-line-out proxy on
both panels with the same model: each of K562, CD4_Rest, HCT116, HEK293T, H1 held out and predicted from the others
(+ Pan_H1 at 0.25; KOLF left out on both sides, it was still streaming) with the L5 settings.

Result (EXPERIMENT_LOG F10): proxy.est A/B/C -> D/E/F: K562 0.146 -> 0.134, CD4_Rest 0.111 -> 0.120, HCT116 0.155 -> 0.156,
HEK293T 0.148 -> 0.135; mean of the four 0.140 -> 0.136 (-0.004), within the spread between lines. H1 on its own 126
benchmark targets 0.210 -> 0.211: the re-streamed 12,816-gene tables reproduce the old ones.
Was eda/68_proxy_panels.py (run twice: --tables eda/out/val_tables with the A/B/C controls, and on the D/E/F tables).
The pipeline command `python -m vcell evaluate proxy --config ...` runs the same proxy for any config.

    python experiments/analysis/f10_proxy_panels.py [abc|def]
"""
import sys
from dataclasses import replace

from vcell import config
from vcell.evaluation.cli import proxy_lines

if __name__ == "__main__":
    for snap in sys.argv[1:] or ["abc", "def"]:
        config.use_snapshot(snap)
        m = config.load_config("configs/final_def.toml").model
        m = replace(m, line_weights={"Pan_H1": 0.25}, context_weights={})
        df = proxy_lines(m)
        print(f"snapshot {snap}: tables {config.TABLES.name}, panel {config.CONTROLS.name}")
        print(df.round(4).to_string(index=False))
        print(f"mean est over the 4 panel lines (H1 excluded, it scores its own benchmark targets): "
              f"{df[df['held out'] != 'H1'].est.mean():.4f}\n")
        df.to_csv(config.results(f"f10_proxy_panels_{snap}.csv"), index=False)
