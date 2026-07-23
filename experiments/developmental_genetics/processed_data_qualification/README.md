# Processed developmental-data qualification

Question: can processed source tables be identified, joined, counted, and partitioned without
opening protected outcome values?

The code validates archive identity, workbook structure, embryo headers, modality-local joins,
claim-bearing rows, and gene-disjoint partitions. Ambiguous secondary outcome sheets remain
excluded.

Status: processed structure and a 410/137/137 gene partition are qualified for the inspected
source state. Real modeling remains blocked until source identity and processing conditions are
durably resolved.

Implementation: `wormctx.poc.developmental_omix_scaleup` and
`developmental_raw_archive_identity`.

