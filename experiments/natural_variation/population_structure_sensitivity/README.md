# Population-structure sensitivity

Question: how do fixed genotype principal components affect the association maps and null
diagnostics?

The same cohort and markers are analyzed at four frozen endpoints: no PCs, 3 PCs, 5 PCs, and
10 PCs. Interval-excluded inflation, QQ quantiles, rank stability, and clumped counts are
retained.

Status: complete. Fixed PCs reduce but do not eliminate inflation.

Implementation: `wormctx.poc.qtl_calibration`.

