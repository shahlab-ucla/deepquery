# Parametric polygenic-null calibration

Question: how often does the full association pipeline cross its declared thresholds when data
are generated from fitted polygenic nulls?

Sixteen cells cross relationship definitions, fixed-effect endpoints, and traits. Each uses
100 deterministic replicates, producing 1,600 maps with cell-specific diagnostics.

Status: complete. Error behavior is strongly relationship-model dependent and does not yet
justify a discovery threshold.

Implementation: `wormctx.poc.qtl_parametric_null`.

