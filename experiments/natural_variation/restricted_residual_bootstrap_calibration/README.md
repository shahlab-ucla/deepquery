# Restricted-residual bootstrap calibration

Question: can covariance-preserving restricted-residual permutation calibrate the exact
intended chromosome-excluded model?

The method whitens fitted residuals, permutes only exchangeable restricted coordinates,
recolors them, and refits the complete association model. Labels are never permuted.

Status: smoke qualification complete. A terminal full-run result is not included in the public
tree and no full-calibration claim is made.

Implementation: `wormctx.poc.qtl_restricted_residual_bootstrap`.

