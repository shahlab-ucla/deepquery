# Matched relationship-model diagnostic

Question: is mismatch between the generating relationship matrix and the fitted
chromosome-excluded model a major source of null inflation?

Four selected cells reuse 400 null phenotype files byte-for-byte and refit them with the
matching whole-panel relationship matrix. Results are paired by cell and replicate.

Status: complete. The mismatch is material, but the matched analysis is conservative and is
not selected as a discovery model. See `results/summary.json`.

Implementation: `wormctx.poc.qtl_matched_relationship_model_diagnostic`.

