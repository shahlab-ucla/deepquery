# WormBase provenance ingestion

Question: can gene, phenotype, developmental-stage, and ontology records be normalized without
losing release, row, evidence, and experimental context?

The bounded adapter validates gene identifiers, phenotype and development GAFs, their OBO
ontologies, qualifiers, negation, stage context, and multi-artifact receipt binding.

Status: seven-file source inventory and rights-safe fixture normalization are qualified. Real
normalization stays blocked where rights or semantic mappings require review. Aggregate source
counts are in `results/source_qualification_summary.json`.

Implementation: `wormctx.adapters.wormbase_ws298` and
`wormctx.poc.wormbase_ws298_source_qualification`.

