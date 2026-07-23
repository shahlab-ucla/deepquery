# Implementation roadmap

This roadmap is an engineering and scientific-validation sequence. It records what must be
built or measured next without embedding machine inventory, private data locations, or
private strategic planning.

## Current foundation

The repository already provides:

- manifest-bound acquisition and portable receipt migration;
- contextual observation models and a faithful reified graph;
- loss-aware KGX/Biolink projection;
- typed validity guards and deterministic inference operators;
- a relation-aware graph/Transformer feasibility model;
- natural-variation calibration and sensitivity implementations;
- developmental split, stage, baseline, and topology-control implementations;
- compendium qualification and grouped-prediction implementations;
- provider-neutral graph-grounded reasoning build, replay, and scoring contracts;
- synthetic and rights-safe test coverage.

## Immediate engineering sequence

### 1. Freeze a portable software environment

Foundation implemented: cross-platform dependency bounds, an OCI definition, an Apptainer
definition, external mount rules, and content-addressed source bundling are present. The
remaining gate is to build the image on Linux, record its digest and software inventory, and
run the clean-checkout tests inside it.

Deliverables:

- resolved dependency lock for Python 3.11;
- OCI container recipe;
- Apptainer-compatible build path;
- image digest and software/license inventory;
- explicit external-tool mounting rules for tools that cannot be redistributed.

Gate: a clean checkout must run the core tests and rights-safe demo inside the image.

### 2. Replace specialized launchers with one provider-neutral execution layer

Foundation implemented: strict execution manifests now bind trusted Python modules, hashed
inputs, rooted paths, resource requests, declared file or directory outputs, write-once
receipts, and retrieval-side closure verification. The remaining engineering work is signal
forwarding, structured log capture, explicit checkpoint restart, and adapters for direct
remote and scheduler submission.

The launcher should accept:

- experiment contract path;
- source and upstream-artifact roots;
- output root;
- Python executable or container image;
- CPU, memory, GPU, and wall-time limits;
- local, direct-remote, or scheduler backend.

It must create a fresh output directory, write a resolved run manifest before compute, record
environment and accelerator metadata, propagate signals, support restart from explicit
checkpoints, and write `SUCCESS` only after checksum closure.

Gate: local and remote fixture runs produce equivalent semantic receipts.

### 3. Measure the synthetic GPU path

Qualification implemented: the repository can detect a CUDA device without collecting host
identity, enforce a 24-GB-class memory gate, and run a bounded synchronized matrix benchmark.
The model-level small/full, two-seed execution and verified checkpoint comparison still
require an accelerator-capable runtime.

Run the small and full graph-conditioned configurations with at least two seeds. Record:

- wall time and examples per second;
- peak host memory and accelerator memory;
- utilization samples;
- checkpoint and output sizes;
- deterministic replay results;
- metrics for every output head.

Gate: the full configuration fits the target 24 GB device and produces a verified portable
checkpoint without relying on host-specific paths.

### 4. Close terminal statistical-calibration artifacts

The restricted-residual bootstrap must have a complete checkpoint census, no active workers,
an empty scratch area, checksum verification, an aggregate summary, and the prespecified
interpretation. A smoke receipt is not a full result.

Gate: the result can answer whether the intended relationship model controls the declared
trait-wise and family-wide error criteria at the available Monte Carlo resolution.

### 5. Finish the phenotype-compendium run

Resume only from the frozen trait roster and grouped splits. Record failures rather than
changing eligibility. Complete discovery traits before opening locked validation traits.

Gate: every eligible trait has a terminal success/failure record and aggregate metrics are
computed without outcome-ranked exclusions.

### 6. Resolve the developmental-data execution gate

Before real fitting:

- bind source identity and lawful processing/transfer conditions;
- freeze the outcome endpoint and missingness rule;
- prove zero overlap across train, validation, and sealed test;
- freeze topology controls using outcome-blind metadata;
- create a sealed-test access counter.

If those conditions cannot be satisfied, execute the same method on a rights-cleared fallback
and narrow the claim to that source.

Gate: real outcome tensors can be materialized without changing the frozen split or opening the
sealed test.

### 7. Benchmark developmental models

Run a compact factorial benchmark across:

- flat, semantic-cell, and topology-aware models;
- authentic, edge-free, matched, and rewired topology;
- selected hidden sizes and depths;
- at least two seeds.

Record per-gene and per-embryo metrics, uncertainty, runtime, memory, and checkpoint size.

Gate: any topology claim must beat capacity-matched and rewired controls, not only a constant
baseline.

### 8. Construct a small private reasoning benchmark

Curate a bounded set of tasks only after underlying graph and predictor artifacts qualify.
Freeze task authority, gold authority, scorer authority, contamination controls, and paired
arm comparisons before executing a model.

Gate: all tool steps replay deterministically and no provider-side process can read gold.

## Release discipline

Each step should land as a separate commit with:

- descriptive experiment name;
- contract/schema version change when semantics change;
- tests for the new gate;
- public-safe aggregate summary if a result is releasable;
- explicit list of private artifacts required but not committed.

Failures should remain visible. A blocked gate is not replaced with a weaker post hoc label.
