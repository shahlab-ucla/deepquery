# Gene-disjoint outcome prediction

Question: can early developmental features generalize to perturbation genes never seen during
model fitting or tuning?

The experiment freezes gene-disjoint train, validation, and sealed-test partitions, then
compares constant, flat, semantic-cell, and topology-aware predictors under identical outcome
and missingness rules.

Status: synthetic contract and leakage guards are tested. Governed real fitting remains
blocked until overlap, endpoint, and sealed-test gates close.

Implementation: `wormctx.poc.developmental_baselines`.

