"""Command line for the pipeline: python -m vcell <stage> ...

    qc controls [--contexts D E F] [--controls DIR]   control cells: summary, null calibration, per-cell audit
    qc sources [--sources ...]                        cell QC masks of the source screens (not applied, H52)
    qc identity [--controls DIR] [--contexts ...] [--markers] [--hpa]   which cell type is each context
    etl download small|h1|xatlas|more ...             public sources into data/sources/
    etl coverage                                      panel targets per source line
    etl universe [--widen --controls DIR --contexts DEF]   the gene universe (OUT/gene_lines.csv)
    etl stage1 <line> ...                             Stage 1 tables (see vcell/etl/stage1/__init__.py)
    train    --config configs/<run>.toml [--limit N]  effects of every panel target -> OUT/effects_<tag>.npz
    generate --config ... [--effects FILE] [--limit N]   cells -> data/submission/pred_<tag>.h5ad
    build    --config ... [--limit N] [--package]     train + generate (+ check and vcc prep)
    package  --config ... [--pred FILE] [--check-only] [--dry-run]   local checks, then vcc prep -> <tag>.vcc
    evaluate proxy|harness|h1 --config ...            offline scores (see vcell/evaluation/__init__.py)
Heavy steps (Stage 1 streams, build, vcc prep) must run one at a time (32 GB RAM, low-memory reaper).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vcell import config


def _cfg(a):
    over = {"tag": a.tag} if getattr(a, "tag", None) else {}
    cfg = config.load_config(a.config, **over)
    cfg.activate()
    return cfg


def cmd_qc(a, rest) -> None:
    if a.controls:
        config.use(controls=a.controls)
    if a.what == "controls":
        from vcell.qc import controls
        controls.summary(a.contexts)
        controls.cell_audit(a.contexts)
    elif a.what == "sources":
        from vcell.qc import sources
        sources.audit(a.sources or sources.SOURCES)
    else:
        from vcell.qc import identity
        identity.source_match(a.controls, a.contexts, a.out)
        if a.markers or a.hpa:
            identity.marker_panel()
        if a.hpa:
            identity.hpa_match()


def cmd_etl(a, rest) -> int:
    if a.what == "download":
        from vcell.etl import download
        return download.main(rest)
    if a.what == "stage1":
        from vcell.etl import stage1
        stage1.main(rest)
    elif a.what == "coverage":
        from vcell.etl import coverage
        coverage.targets_per_line()
    elif a.what == "universe":
        from vcell.etl import universe
        if a.widen:
            universe.widen(a.controls or "data/controls_final", a.contexts or "DEF")
        else:
            universe.build()
    return 0


def cmd_train(a, rest):
    from vcell.model import effects
    cfg = _cfg(a)
    e = effects.train(cfg, limit=a.limit)
    e.save(a.out)
    return e


def cmd_generate(a, rest, e=None):
    from vcell.model import effects, generate
    cfg = _cfg(a)
    if e is None:
        e = effects.Effects.load(a.effects or config.OUT / f"effects_{cfg.tag}.npz")
    out = Path(a.pred) if getattr(a, "pred", None) else config.SUBMISSION / f"pred_{cfg.tag}.h5ad"
    return generate.write_prediction(e, out, seed=cfg.seed)


def cmd_build(a, rest) -> int:
    from vcell.model import effects, generate
    cfg = _cfg(a)
    e = effects.train(cfg, limit=a.limit)
    if a.save_effects:
        e.save()
    out = generate.write_prediction(e, Path(a.pred) if a.pred else config.SUBMISSION / f"pred_{cfg.tag}.h5ad", seed=cfg.seed)
    if a.package:
        a.pred = str(out)
        return cmd_package(a, rest)
    return 0


def cmd_package(a, rest) -> int:
    from vcell.model import package
    cfg = _cfg(a)
    pred = Path(a.pred) if a.pred else config.SUBMISSION / f"pred_{cfg.tag}.h5ad"
    problems = package.check(pred)
    print(f"local checks of {pred.name}: " + ("passed" if not problems else "; ".join(problems)), flush=True)
    if problems or getattr(a, "check_only", False):
        return 1 if problems else 0
    return package.prep(pred, config.SUBMISSION / f"{cfg.tag}.vcc", dry_run=getattr(a, "dry_run", False),
                        log=config.OUT / f"prep_{cfg.tag}.log")


def cmd_evaluate(a, rest) -> int:
    from vcell.evaluation import cli
    return cli.main(a.what, _cfg(a), rest)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "etl" and argv[1] in ("download", "stage1"):   # their own parsers (and --help)
        from vcell.etl import download, stage1
        return {"download": download.main, "stage1": stage1.main}[argv[1]](argv[2:]) or 0
    ap = argparse.ArgumentParser(prog="python -m vcell", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    q = sub.add_parser("qc")
    q.add_argument("what", choices=["controls", "sources", "identity"])
    q.add_argument("--controls", default=None)
    q.add_argument("--contexts", nargs="+", default=None)
    q.add_argument("--sources", nargs="+", default=None)
    q.add_argument("--out", default=None, help="identity: where the suggested weights go (default OUT/ctx_weights.json)")
    q.add_argument("--markers", action="store_true", help="identity: also the lineage marker panel of A-F")
    q.add_argument("--hpa", action="store_true", help="identity: also the Human Protein Atlas match (implies --markers)")
    e = sub.add_parser("etl")
    e.add_argument("what", choices=["download", "coverage", "universe", "stage1"])
    e.add_argument("--widen", action="store_true", help="universe: add the test contexts (F6)")
    e.add_argument("--controls", default=None)
    e.add_argument("--contexts", default=None)
    for name in ("train", "generate", "build", "package", "evaluate"):
        p = sub.add_parser(name)
        if name == "evaluate":
            p.add_argument("what", choices=["proxy", "harness", "h1"])
        p.add_argument("--config", required=True)
        p.add_argument("--tag", default=None, help="override the config's tag (output names)")
        if name in ("train", "generate", "build"):
            p.add_argument("--limit", type=int, default=None, help="first N panel targets only (a quick test; vcc prep rejects it)")
        if name == "train":
            p.add_argument("--out", default=None)
        if name == "generate":
            p.add_argument("--effects", default=None, help="effects file (default OUT/effects_<tag>.npz)")
        if name in ("generate", "build", "package"):
            p.add_argument("--pred", default=None, help="prediction file (default data/submission/pred_<tag>.h5ad)")
        if name == "build":
            p.add_argument("--package", action="store_true")
            p.add_argument("--save-effects", action="store_true")
        if name in ("build", "package"):
            p.add_argument("--check-only", action="store_true")
            p.add_argument("--dry-run", action="store_true")
    a, rest = ap.parse_known_args(argv)
    if rest and not (a.stage == "etl" and a.what in ("download", "stage1")) and a.stage != "evaluate":
        ap.error(f"unrecognised arguments: {' '.join(rest)}")
    fn = {"qc": cmd_qc, "etl": cmd_etl, "train": cmd_train, "generate": cmd_generate, "build": cmd_build,
          "package": cmd_package, "evaluate": cmd_evaluate}[a.stage]
    r = fn(a, rest)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
