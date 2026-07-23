# Phenotype-compendium grouped prediction

## Source qualification

Trait tables are first treated as a source census. Eligibility depends only on declared
metadata and minimum panel size, never on outcome rank or model performance. Legacy outputs
are quarantined and cannot select the discovery roster.

The qualified census freezes:

- 1,461 source traits;
- 1,445 eligible traits;
- 1,184 discovery traits;
- 261 locked validation traits.

Source qualification executes no predictive model.

## Genotype and kernel qualification

A 203-strain roster is bound to the genotype source in a fixed order. The implementation
verifies sample identity, marker filtering, relationship dimensions, symmetry, diagonal
behavior, finite values, and positive-semidefinite tolerance. Phenotype values are not inputs
to this stage.

## Grouped prediction

Population groups are derived from genotype principal components with a frozen candidate
range, equal weighting after per-axis standardization, and a minimum group-size gate. Labels
are technical clusters, not asserted ancestry categories.

Prediction uses group-aware folds so close genotype groups do not appear in both fitting and
evaluation partitions. The model ladder contains simple baselines and kernel-aware models
under identical splits. Trait-specific missingness is handled inside each training fold.

## Discovery and validation separation

Discovery traits are used for engineering and model-family qualification. Validation traits
remain locked until the pipeline, exclusions, thresholds, and interpretation rules are frozen.
Traits that fail minimum group coverage are recorded as failures rather than removed after
performance is observed.

## Metrics

Per-trait metrics include prediction correlation or error, baseline-relative improvement,
coverage, and fold completeness. Aggregate summaries report the distribution across traits
and stratify by prespecified metadata. One unusually favorable trait cannot substitute for
compendium-wide behavior.

## Interpretation boundary

This experiment tests whether genotype/context representations generalize across grouped
strains and diverse phenotypes. It does not establish causal variants, trait mechanisms, or
clinical utility.

