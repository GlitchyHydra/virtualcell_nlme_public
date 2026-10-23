"""Virtual Cell Challenge 2026: zero-shot CRISPRi knockdown prediction by pooling public Perturb-seq screens.

The pipeline has four stages, each a subpackage:

    qc          checks of the inputs: the organisers' control cells, cell QC of the source screens, context identity
    etl         downloads, the gene universe, and one Stage 1 table (beta, s) per source line
    model       Stage 2 pooling across lines, the effect recipe, cell generation and packaging
    evaluation  the offline proxies (leave one line out), the 8-line harness and the official H1 scorer

Run the stages with `python -m vcell <stage> ...` (see vcell/__main__.py); settings live in configs/*.toml.
Experiments that use the pipeline are in experiments/ (index: experiments/README.md).
"""
