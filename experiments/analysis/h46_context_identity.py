"""H46 and F4: which cell type is each context, and does it match a source line? The pipeline's identity step
(vcell.qc.identity) on the validation contexts A/B/C and on the test contexts D/E/F: on/off features against the source
lines (markers, Y genes, fingerprints, contradictions -> the suggested same-line weights of model.sources), then a broad
lineage marker panel for A-F and the match against the 1,206 cell lines of the Human Protein Atlas.

Result (EXPERIMENT_LOG H46, F4): the features work on known pairs (contradiction 0.00-0.06 for the same cell type across
labs, 0.26-0.74 for different lines). A looks like Jurkat (all 17 Jurkat fingerprint genes on) but our Jurkat screen covers
no panel target; B and C match nothing. HPA: A = Jurkat (0.91), B = HeLa (0.73), C = CAL-33 (0.74), D = MOLM-13 (0.90),
E = U-87 MG (0.81), F = an hTERT fibroblast (0.68). No source line matches D/E/F, so the same-line rule gives nothing; the
U87 table for E (GSE225775) was found by a search afterwards and added by hand (configs/final_def.toml).
Was eda/52_context_match.py, eda/64_identify_contexts.py and eda/65_hpa_match.py. Both snapshots. About 5 minutes.

    python experiments/analysis/h46_context_identity.py [--no-hpa]
"""
import sys

from vcell import config
from vcell.qc import identity

if __name__ == "__main__":
    for snap, ctx in (("abc", ["A", "B", "C"]), ("def", ["D", "E", "F"])):
        config.use_snapshot(snap)
        print(f"\n=== snapshot {snap}: contexts {ctx}")
        identity.source_match(contexts=ctx, out=config.results(f"h46_ctx_weights_{snap}.json"))
    if "--no-hpa" not in sys.argv:
        identity.marker_panel()
        identity.hpa_match()
