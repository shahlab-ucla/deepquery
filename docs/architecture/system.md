# System architecture

## Design objective

DeepQuery separates three responsibilities that are often conflated:

1. representing what an experiment actually measured;
2. performing deterministic scientific and statistical operations;
3. learning how to route, combine, or summarize evidence.

The learned component cannot silently change source identity, study design, statistical
thresholds, or the meaning of a missing observation. Those decisions are enforced in typed
contracts and recorded in receipts.

## Context-graph layer

A source release is declared by a manifest containing immutable release identifiers,
artifact roles, expected byte counts, cryptographic digests, acquisition policy, and rights
status. Fetching writes content-addressed blobs and release receipts. An adapter then maps
source rows into `ContextualObservation` records.

Each observation reifies an evidence event rather than reducing it immediately to a
gene–phenotype edge. It carries:

- subject, object, and predicate;
- organism, life stage, tissue or cell, environment, treatment, and genetic background;
- assay, protocol, comparison group, and measurement;
- evidence direction, interpretation, provenance, and source release;
- explicit missing-context reasons and mapping status.

`wormctx.graph` converts observations to a faithful reified graph. A separate KGX/Biolink
projection emits only associations that have sufficient reviewed exchange metadata. Context
that cannot be represented losslessly is written to a sidecar instead of being discarded.

## Deterministic operator layer

`wormctx.poc.operators` is the trusted scientific core. It exposes a fixed vocabulary of typed
operators for trait derivation, reaction norms, relatedness/state models, developmental state,
perturbation effects, detectability, evidence retrieval, hypothesis updating, and expected
information gain.

Before execution, deterministic guards reject invalid requests such as:

- estimating dominance from a panel without heterozygotes;
- treating a structurally absent locus as an ordinary reference SNP;
- interpreting database absence as a powered negative;
- claiming formal transport without an identified source/target model;
- conditioning on a post-treatment covariate without justification.

Statistical code for association, null simulation, bootstrapping, grouped prediction, and
developmental baselines also lives outside the learned model.

## Learned graph-conditioned layer

`SharedGraphReasoner` receives a bounded episode graph:

1. numeric node features are projected to a shared hidden dimension;
2. node-type embeddings are added;
3. relation embeddings modify messages along typed edges;
4. destination-normalized aggregation updates node states through residual message blocks;
5. a learned query token and node sequence pass through a Transformer encoder;
6. typed heads predict task family, required operators, invalid-design flags, conclusion,
   hypothesis choice, next experiment, and a scalar resolution estimate.

The model is multi-task by construction. It predicts which deterministic operations should be
used; it does not replace those operations. Padded hypothesis and experiment candidates are
masked before cross-entropy is computed.

## Experiment packages

Experiment contracts live under `experiments/`. They specify inputs, parent receipts,
selection rules, split logic, multiplicity, controls, terminal conditions, and claims that
remain prohibited. Implementation modules stay under `src/wormctx` so algorithms can be reused
across experiments.

An experiment may be:

- `complete`: terminal artifacts and the prespecified interpretation exist;
- `qualified`: interfaces and input gates are proven, but no biological result is claimed;
- `active`: the frozen analysis exists but the terminal result is unavailable;
- `blocked`: a named source, rights, calibration, or upstream-evidence condition is unmet.

## Receipt and trust boundaries

Files are treated as immutable once frozen. Writers use staging files followed by atomic
replacement, reject pre-existing destinations, and hash relevant outputs. Terminal success is
valid only when required files, counts, checksums, and semantic fields agree.

The public repository intentionally excludes machine receipts and private inputs. A local or
remote execution materializes those under ignored `artifacts/` and `runs/` paths. This keeps
code and public summaries inspectable without turning a source checkout into a data-release
channel.

## Data flow

```text
release manifest
    -> content-addressed source snapshot
    -> source-specific adapter
    -> contextual observations
    -> faithful reified graph
    -> deterministic operators and experiment contracts
    -> optional graph-conditioned router
    -> locked predictions / statistical outputs
    -> checksum and interpretation receipts
    -> aggregate public summary
```

Every arrow is an explicit interface. Failures remain localized: a rights failure blocks
fetching, a mapping failure blocks normalization, a calibration failure blocks discovery
claims, and a sealed-test failure blocks final evaluation.

