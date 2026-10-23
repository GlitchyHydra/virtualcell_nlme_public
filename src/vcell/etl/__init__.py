"""ETL: public sources -> one Stage 1 table per source line, all on the same target and gene axis.

    download    resumable, checksum-verified downloads into data/sources/
    universe    the gene universe (genes above 5 CPM in a context) and its expression bins: OUT/gene_lines.csv
    coverage    which panel targets each source line measures
    stage1      the table builders (beta, s per target and gene) and the table format
"""
