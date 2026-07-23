# Graph-grounded reasoning evaluation

## Objective

The reasoning harness asks whether structured graph retrieval and locked predictors improve
answer validity, evidence fidelity, and appropriate abstention relative to text-only and
ordinary retrieval baselines. It is provider-neutral and does not make external model calls.

## Frozen arms

The six prespecified arms are:

1. text only;
2. ordinary text retrieval;
3. graph retrieval;
4. graph retrieval plus identifiability checks and a qualified locked predictor;
5. graph and identifiability without the predictor;
6. identifiability and predictor without graph retrieval.

These arms isolate graph and predictor contributions while keeping tasks paired.

## Task and authority separation

Release, curator, gold, and scorer authorities are separate write-once bundles. Provider-side
execution cannot read gold or scorer files. Original entity aliases are replaced by
HMAC-derived tokens; the secret is never stored in the release bundle.

Tasks declare required operators, acceptable evidence, numeric tolerance, and conditions under
which abstention is correct. Private prompts and gold answers are intentionally absent from the
public repository.

## Typed programs

A model response includes a constrained inference program. Tool operators are:

- retrieve text;
- retrieve graph;
- check identifiability;
- call a locked predictor.

Pure operators synthesize evidence, emit a prediction, or abstain. Unknown operators,
disallowed tools, malformed dependencies, and execution steps without matching deterministic
receipts invalidate the program.

## Replay

Tool outputs are frozen and replayed locally. The harness checks request identity, argument
hashes, step ordering, and output receipts. Provider text cannot fabricate a successful tool
execution because the scorer requires the corresponding replay receipt.

## Scoring

Dimensions remain separate:

- prediction correctness within task-authorized tolerance;
- program validity and required-operator recall;
- citation precision, recall, and claim support;
- correct abstention, unsafe answering, and over-abstention;
- deterministic execution-trace agreement.

A composite score is prohibited by the current contract. Primary comparisons use paired tasks
by arm and are stratified by task family. Sample-size assumptions and family-wide error control
must be frozen before task release.

## Contamination controls

Prompts and per-task canaries have hash receipts. Model training cutoff and prior-exposure
attestations are required for a confirmatory run. External search is disabled. No model,
decoding, threshold, endpoint, or task exclusion may change after responses are frozen.

## Interpretation boundary

The scaffold proves that a controlled evaluation can be built and replayed. It does not show
that any language model reasons biologically. That requires private curated tasks, completed
arm runs, locked scoring, and an independently interpretable result.

