# Natural-variation calibration

## Separation of questions

The natural-variation work is intentionally decomposed:

1. source and marker identity;
2. conventional association;
3. fixed-covariate sensitivity;
4. relationship-matrix sensitivity;
5. null calibration;
6. regional-state qualification;
7. held-out prediction.

A later stage cannot repair a failed earlier provenance gate by relabeling the analysis.

## Association model

For strain phenotype \(y\), fixed covariates \(X\), marker \(g_j\), and relationship matrix
\(K\), the mixed model is

\[
y=X\beta+g_j\alpha_j+u+\epsilon,\qquad
u\sim N(0,\sigma_g^2K),\quad
\epsilon\sim N(0,\sigma_e^2I).
\]

The code validates sample order, marker count, tool version, command options, convergence, and
result-file identity before summarization. Four traits are kept distinct, and within-trait and
family-wide multiplicity are both recorded.

## Population-structure sensitivity

Principal-component endpoints are frozen before association. The same genotype and phenotype
cohort is mapped with PC0, PC3, PC5, and PC10 fixed effects. Diagnostics include genomic
inflation, interval-excluded inflation, fixed QQ quantiles, rank stability, and clumped hit
counts. This asks whether simple fixed structure terms explain the apparent signal; it is not
a model-selection search.

## Chromosome-excluded relationships

For chromosome \(c\), construct

\[
K_{-c}=\frac{1}{M_{-c}}Z_{-c}Z_{-c}^{T},
\]

using LD-pruned markers outside \(c\). Six matrices are verified against the same ordered
sample roster. Each trait/endpoint pair is fit against all chromosome-specific chunks, and
chunks are reassembled into eight genome-wide maps. Marker lists for relationship estimation
and association are independently hashed.

## Parametric polygenic null

For each frozen cell, a fitted null supplies \(\hat\beta,\hat\sigma_g^2,\hat\sigma_e^2\).
Replicate phenotype vectors are generated as

\[
y^{(b)}=X\hat\beta+Lz^{(b)},\qquad
LL^T=\hat\sigma_g^2K+\hat\sigma_e^2I,
\]

with deterministic seeds derived from a master digest and replicate index. The association
pipeline is replayed without outcome-dependent stopping. Each cell records minimum \(p\),
Bonferroni events, QQ summaries, and inflation metrics.

## Matched relationship diagnostic

Selected null phenotypes are reused byte-for-byte. The analysis relationship is changed from
chromosome-excluded to the whole-panel matrix used by the generator. Paired replicate
differences isolate generator/fitter mismatch. Zero threshold exceedances are interpreted as
conservative behavior, not proof of calibration.

## Restricted-residual bootstrap

For the intended covariance \(\Omega\), compute a Cholesky factor \(L\), whiten residuals, and
project into the residual subspace orthogonal to the whitened design. With basis \(U_1\) and a
permutation matrix \(P\),

\[
y^\* = X\hat\beta_{\mathrm{GLS}}
+LU_1PU_1^TL^{-1}(y-X\hat\beta_{\mathrm{GLS}}).
\]

This permutes exchangeable restricted coordinates, not phenotype or genotype labels. Each
replicate is refit by the exact intended association model. Plus-one empirical tail
probabilities avoid zero estimates:

\[
\hat p=\frac{1+\sum_b I(T_b\ge T_{\mathrm{obs}})}{B+1}.
\]

## Regional states and held-out prediction

Small-variant regional profiles are a sensitivity representation only; they are not relabeled
as pangenome paths or presence/absence calls. Genuine state-aware analysis requires explicit
state semantics, carrier counts, missingness gates, coordinate identity, and qualified
relationships.

Held-out prediction uses nested group-aware cross-validation. Hyperparameters and thresholds
are selected inside training folds, while the held-out strain group remains inaccessible.
Kernel, linear, and state-aware models are compared under identical splits.

## Interpretation boundary

Association recurrence, reduced inflation, or predictive accuracy may motivate further
experiments, but none alone identifies a causal allele or mechanism. Calibration and
prospective validation are separate requirements.

