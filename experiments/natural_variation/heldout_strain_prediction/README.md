# Held-out-strain prediction

Question: do state-aware or kernel models predict phenotype for strains excluded from fitting?

The experiment uses nested group-aware cross-validation, training-only standardization, and
training-fold model selection. State, kernel, and simple baselines share identical splits.

Status: synthetic qualification complete. Real evaluation remains blocked until all upstream
state and kernel receipts are available.

Implementation: `wormctx.poc.qtl_heldout_prediction`.

