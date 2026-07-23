# Experiment contracts and receipts

## Contract contents

A contract should define:

- descriptive analysis identity and schema version;
- input roles and content identities;
- allowed source roots and safe relative paths;
- cohort and ordering rules;
- preprocessing and model settings;
- deterministic seed derivation;
- split, multiplicity, and threshold policies;
- required controls and negative controls;
- terminal output set;
- permitted and prohibited claims.

Unknown fields are rejected. Semantic changes require a schema-version change rather than
silent reinterpretation.

## Freeze sequence

1. Validate the source and rights boundary.
2. Freeze outcome-independent cohort, features, topology, and grouping.
3. Freeze train, validation, and sealed-test partitions.
4. Freeze models, controls, thresholds, and comparison rules.
5. Hash the contract.
6. Materialize a resolved run manifest before accessing protected outcomes.
7. Execute without outcome-dependent early stopping or exclusions unless predeclared.
8. Close checksums and write the terminal marker.
9. Produce a separate interpretation record.

## Receipt hierarchy

Artifact receipts bind individual files. A run manifest binds the contract, source revision,
environment, seeds, and parents. Checksum closure binds the terminal file set. An
interpretation receipt states what the terminal metrics permit and prohibit.

A higher-level receipt cannot override a failed lower-level identity check. For example, a
successful model run cannot repair an unverified source cohort.

## Legacy compatibility

Historical private receipts may retain older identifiers. Public code uses descriptive names.
When compatibility is necessary, use an explicit read-only adapter that maps the old schema to
a new in-memory representation while preserving the original bytes and digest. Never rewrite
an old receipt in place.

